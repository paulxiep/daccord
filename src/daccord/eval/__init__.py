"""D'accord eval harness (tier 2B) + ensemble prompt builder (tier 6A).

Public surface for the M0 baseline, M4 results runs, and tier 7A ensemble
generation. See [eval/README.md] for the CSV row contract and the CLI
invocation; see [docs/development_plan.md] tier 6/7 for the ensemble.
"""

from daccord.eval.clients import GeminiClient, GroqClient, ModelClient
from daccord.eval.prompts import build_ensemble_prompt, build_eval_prompt, build_judge_prompt
from daccord.eval.registry import Registry, load_registry
from daccord.eval.schema import CitationCandidate, ModelResponse, PromptMessages

__all__ = [
    "CitationCandidate",
    "GeminiClient",
    "GroqClient",
    "ModelClient",
    "ModelResponse",
    "PromptMessages",
    "Registry",
    "build_ensemble_prompt",
    "build_eval_prompt",
    "build_judge_prompt",
    "load_registry",
]
