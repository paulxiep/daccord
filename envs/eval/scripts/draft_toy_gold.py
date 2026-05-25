"""Tier 2A — draft 20-pair toy gold via Groq (Llama) and Google AI Studio (Gemini).

Runs in the `eval` service where groq + google-genai are installed (those SDKs
do not live in the shared root venv; see CLAUDE.md services table):

    docker compose run --rm eval uv run python scripts/draft_toy_gold.py

Produces .draft_{llama,gemini}.jsonl files in <repo>/data/gold/ for human
reconcile + verify. The committed file <repo>/data/gold/toy_v1.jsonl is built
by the human from these drafts; this script does not write it directly.

Both drafters are free-tier; one batched call each (~10k tokens). Cost-tracked
through daccord.costs against per-provider RPD caps.

Schema: validates each emitted pair against daccord.gold.schema.GoldPair (owned
by sister-session 2B). Required for downstream tier-3A baselines and tier-2B
harness.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Literal

from daccord.costs import preflight, record_call
from daccord.gold.schema import GoldPair
from daccord.validation import validated

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "gold"
DEFAULT_ENV_FILE = REPO_ROOT / ".env.local"

DEFAULT_LLAMA_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_GEMINI_MODEL = "gemini-3.1-flash-lite"

FRAMEWORKS: tuple[str, ...] = (
    "gdpr",
    "uk_gdpr",
    "dpa_2018",
    "bdsg",
    "loi_il",
    "pdpa_sg",
    "pdpa_th",
    "dpa_2012_ph",
    "pdpa_my",
)
JURISDICTIONS: tuple[str, ...] = ("eu", "uk", "de", "fr", "sg", "th", "ph", "my")
FRAMEWORK_TO_JURISDICTION: dict[str, str] = {
    "gdpr": "eu",
    "uk_gdpr": "uk",
    "dpa_2018": "uk",
    "bdsg": "de",
    "loi_il": "fr",
    "pdpa_sg": "sg",
    "pdpa_th": "th",
    "dpa_2012_ph": "ph",
    "pdpa_my": "my",
}
# Per-framework primary citation-text language (ISO 639-1).
# pdpa_th/bdsg/loi_il can also be cited in English translations — drafter picks per-pair.
FRAMEWORK_TO_LANGUAGE: dict[str, str] = {
    "gdpr": "en",
    "uk_gdpr": "en",
    "dpa_2018": "en",
    "bdsg": "de",
    "loi_il": "fr",
    "pdpa_sg": "en",
    "pdpa_th": "th",
    "dpa_2012_ph": "en",
    "pdpa_my": "en",
}
CONCEPT_AXES: tuple[str, ...] = (
    "consent / lawful basis for processing",
    "data subject access rights",
    "personal data breach notification",
    "security / technical-and-organisational measures",
    "DPO appointment / data controller duties",
)

PROMPT_TEMPLATE = """You are constructing a small validation set of cross-jurisdiction
data-privacy regulation mapping pairs. Produce exactly {n_pairs} pairs as a single
JSON array. Each pair maps a control from one framework to a structurally analogous
control in another framework.

Framework IDs (use these exact strings for source_framework / target_framework):
  {frameworks}

Jurisdiction codes (use these exact strings for source_jurisdiction / target_jurisdiction):
  {jurisdictions}

Framework -> Jurisdiction (use this when filling jurisdiction codes):
  {framework_jurisdiction_table}

Framework -> primary citation-text language (ISO 639-1; use for source_language /
target_language unless you are explicitly citing an English translation of the
original-language text — then use "en"):
  {framework_language_table}

Each pair has these fields:
  id                   - stable string, e.g. "toy_001", "toy_002", ...
  source_framework     - one of the framework IDs above
  source_jurisdiction  - jurisdiction code matching source_framework
  source_citation_id   - regulator's canonical citation form (e.g. "Art. 6(1)(a)",
                         "Section 13", "Sec. 16", "Paragraph 26")
  source_mechanism     - 1-2 sentences summarising what the source control requires
  source_language      - ISO 639-1 code of the language the citation_id + mechanism
                         text are in (en, de, fr, th, ...)
  target_framework     - one of the framework IDs above (different from source)
  target_jurisdiction  - jurisdiction code matching target_framework
  target_citation_id   - regulator's canonical citation form
  target_mechanism     - 1-2 sentences summarising what the target control requires
  target_language      - ISO 639-1 code, same convention as source_language
  notes                - optional string with edge-case notes; null if not applicable

Concept axes the {n_pairs} pairs should cover (each appearing 3-5 times):
  {concept_axes}

Difficulty mix (guidance for selecting pairs; "difficulty" is NOT a schema field):
  - {n_easy} easy pairs (EU-spine clones: GDPR <-> UK-GDPR; GDPR <-> BDSG; GDPR <-> Loi I+L)
  - {n_medium} medium pairs (GDPR <-> PDPA-SG / PDPA-MY / DPA 2012 PH)
  - {n_hard} hard pairs (PDPA-TH <-> GDPR/PDPA-SG; UK DPA 2018 <-> GDPR/BDSG;
    BDSG <-> Loi I+L/PDPA-SG)

Coverage requirement: every framework ID above MUST appear as source_framework at
least once AND as target_framework at least once.

Return ONLY a JSON object with a single key "pairs" whose value is the array of
{n_pairs} mapping objects. Do not wrap in markdown fences. Do not include
surrounding prose.
"""

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*\n(.*?)\n```\s*$", re.DOTALL)


@validated
def parse_mix(s: str) -> tuple[int, int, int]:
    parts = s.split("/")
    if len(parts) != 3:
        raise ValueError(f"--mix must be 'EASY/MEDIUM/HARD', got {s!r}")
    a, b, c = (int(p) for p in parts)
    return (a, b, c)


@validated
def load_env_file(path: Path) -> None:
    """Set environment variables from KEY=value lines in `path`; never override existing."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@validated
def build_prompt(n_pairs: int, mix: tuple[int, int, int]) -> str:
    n_easy, n_medium, n_hard = mix
    framework_jurisdiction = "; ".join(f"{f} -> {j}" for f, j in FRAMEWORK_TO_JURISDICTION.items())
    framework_language = "; ".join(f"{f} -> {lang}" for f, lang in FRAMEWORK_TO_LANGUAGE.items())
    return PROMPT_TEMPLATE.format(
        n_pairs=n_pairs,
        n_easy=n_easy,
        n_medium=n_medium,
        n_hard=n_hard,
        frameworks=", ".join(FRAMEWORKS),
        jurisdictions=", ".join(JURISDICTIONS),
        framework_jurisdiction_table=framework_jurisdiction,
        framework_language_table=framework_language,
        concept_axes="\n  ".join(f"- {c}" for c in CONCEPT_AXES),
    )


def _strip_json_fences(raw: str) -> str:
    m = _JSON_FENCE_RE.match(raw.strip())
    return m.group(1) if m else raw


@validated
def parse_pairs_list(raw: str) -> list[dict]:
    parsed = json.loads(_strip_json_fences(raw))
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for k in ("pairs", "data", "items", "mapping_pairs"):
            v = parsed.get(k)
            if isinstance(v, list):
                return v
        list_values = [v for v in parsed.values() if isinstance(v, list)]
        if list_values:
            return list_values[0]
        raise ValueError(f"expected list of pairs; got dict with keys {sorted(parsed)}")
    raise ValueError(f"expected list of pairs; got {type(parsed).__name__}")


@validated
def call_groq(prompt: str, model: str) -> tuple[list[dict], int, int]:
    from groq import Groq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        sys.exit("GROQ_API_KEY not set. Add to .env.local or export. See .env.example.")
    client = Groq(api_key=api_key)
    preflight("groq", model, est_input_tokens=4_000, est_output_tokens=6_000)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    pt = int(getattr(usage, "prompt_tokens", 0)) if usage else 0
    ct = int(getattr(usage, "completion_tokens", 0)) if usage else 0
    record_call("groq", model, input_tokens=pt, output_tokens=ct)
    return parse_pairs_list(content), pt, ct


@validated
def call_gemini(prompt: str, model: str) -> tuple[list[dict], int, int]:
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("GOOGLE_API_KEY not set. Add to .env.local or export. See .env.example.")
    client = genai.Client(api_key=api_key)
    preflight("google_gemini", model, est_input_tokens=4_000, est_output_tokens=6_000)
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.3,
        ),
    )
    content = resp.text or ""
    meta = resp.usage_metadata
    pt = int(getattr(meta, "prompt_token_count", 0) or 0) if meta else 0
    ct = int(getattr(meta, "candidates_token_count", 0) or 0) if meta else 0
    record_call("google_gemini", model, input_tokens=pt, output_tokens=ct)
    return parse_pairs_list(content), pt, ct


@validated
def validate_and_write(pairs: list[dict], out_path: Path) -> tuple[int, list[str]]:
    """Validate each pair against GoldPair, write JSONL, return (n_valid, error_msgs)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    n_valid = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, raw in enumerate(pairs):
            try:
                pair = GoldPair.model_validate(raw)
            except Exception as exc:
                errors.append(f"pair {i} ({raw.get('id', '?')}): {exc}")
                continue
            fh.write(pair.model_dump_json() + "\n")
            n_valid += 1
    return n_valid, errors


@validated
def run_drafter(
    drafter: Literal["llama", "gemini"],
    prompt: str,
    model: str,
    out_dir: Path,
) -> int:
    """Call the drafter, validate, write the .draft_*.jsonl, return exit code (0 ok, 1 fail)."""
    print(f"Drafting via {drafter} ({model})...")
    if drafter == "llama":
        pairs, pt, ct = call_groq(prompt, model)
        out_path = out_dir / "toy_v1.draft_llama.jsonl"
    else:
        pairs, pt, ct = call_gemini(prompt, model)
        out_path = out_dir / "toy_v1.draft_gemini.jsonl"
    n_valid, errors = validate_and_write(pairs, out_path)
    print(f"  tokens: in={pt} out={ct}")
    print(f"  wrote {n_valid}/{len(pairs)} valid pairs to {out_path}")
    if errors:
        print(f"  {len(errors)} invalid pair(s):", file=sys.stderr)
        for e in errors:
            print(f"    - {e}", file=sys.stderr)
    if len(pairs) > 0 and n_valid / len(pairs) < 0.5:
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-pairs", type=int, default=20)
    parser.add_argument(
        "--mix",
        type=parse_mix,
        default=(5, 8, 7),
        help="EASY/MEDIUM/HARD pair counts (default 5/8/7)",
    )
    parser.add_argument("--drafter", choices=("llama", "gemini", "both"), default="both")
    parser.add_argument("--llama-model", default=DEFAULT_LLAMA_MODEL)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Path to .env file with API keys (default <repo>/.env.local)",
    )
    args = parser.parse_args(argv)

    load_env_file(args.env_file)
    if sum(args.mix) != args.n_pairs:
        sys.exit(f"--mix {args.mix} sums to {sum(args.mix)}; expected --n-pairs={args.n_pairs}")

    prompt = build_prompt(args.n_pairs, args.mix)

    exit_code = 0
    if args.drafter in ("llama", "both"):
        exit_code |= run_drafter("llama", prompt, args.llama_model, args.out_dir)
    if args.drafter in ("gemini", "both"):
        exit_code |= run_drafter("gemini", prompt, args.gemini_model, args.out_dir)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
