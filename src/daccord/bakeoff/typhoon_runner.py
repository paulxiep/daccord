"""Typhoon-OCR adapter for the parser bake-off.

Uses `scb10x/typhoon-ocr1.5-2b` directly via `transformers.AutoModelForImageText
ToText` — no vllm server, no SCB cloud API key. The model is ~2B params and fits
comfortably on a 16 GB RTX 5080 at fp16. The v1.5 prompt is reused verbatim from
the `typhoon-ocr` package so we don't drift from upstream.
"""

from __future__ import annotations

import importlib
import time
from pathlib import Path

from daccord.validation import ValidatedModel, validated

MODEL_ID = "scb10x/typhoon-ocr1.5-2b"
TARGET_LONGEST_DIM = 1800  # Typhoon v1.5 was trained at this resolution
MAX_NEW_TOKENS = 10000


class TyphoonPageOutput(ValidatedModel):
    page_index: int
    md_path: Path
    markdown: str
    seconds_elapsed: float


@validated
def parser_version() -> str:
    """Return the installed `typhoon-ocr` package version, for MLflow params."""
    pkg = importlib.import_module("typhoon_ocr")
    return getattr(pkg, "__version__", "unknown")


@validated
def model_id() -> str:
    """Return the underlying HF model identifier, for MLflow params."""
    return MODEL_ID


@validated
def parse_pages(png_paths: list[Path], out_dir: Path) -> list[TyphoonPageOutput]:
    """Run Typhoon-OCR v1.5 on each PNG, writing `<out_dir>/page_<n>.md`.

    Model + processor are loaded once on the first call into this function and
    held in module-local globals across the 5 pages of the bake-off run. PIL
    handles the >1800-px downscale (LANCZOS) before the image is encoded.
    """
    transformers = importlib.import_module("transformers")
    pil_image = importlib.import_module("PIL.Image")
    typhoon_prompts = importlib.import_module("typhoon_ocr.ocr_utils").PROMPTS_SYS

    out_dir.mkdir(parents=True, exist_ok=True)
    prompt_text: str = typhoon_prompts["v1.5"](figure_language="Thai")

    model = transformers.AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype="auto", device_map="auto"
    )
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)

    results: list[TyphoonPageOutput] = []
    for png in png_paths:
        page_idx = _page_index_from_stem(png.stem)
        img = pil_image.open(png)
        img = _resize_to_max(img, TARGET_LONGEST_DIM, pil_image)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        t0 = time.perf_counter()
        gen_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        elapsed = time.perf_counter() - t0

        prompt_len = inputs["input_ids"].shape[1]
        out_ids = gen_ids[:, prompt_len:]
        text: str = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        md_path = out_dir / f"page_{page_idx}.md"
        md_path.write_text(text, encoding="utf-8")
        results.append(
            TyphoonPageOutput(
                page_index=page_idx,
                md_path=md_path,
                markdown=text,
                seconds_elapsed=elapsed,
            )
        )
    return results


def _resize_to_max(img, longest_dim: int, pil_image):
    w, h = img.size
    longest = max(w, h)
    if longest <= longest_dim:
        return img
    scale = longest_dim / float(longest)
    return img.resize((int(w * scale), int(h * scale)), pil_image.Resampling.LANCZOS)


@validated
def _page_index_from_stem(stem: str) -> int:
    if not stem.startswith("page_"):
        raise ValueError(f"unexpected per-page PNG stem: {stem!r}")
    return int(stem.removeprefix("page_"))
