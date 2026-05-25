"""D'accord eval harness (tier 2B).

Public surface for the M0 baseline and M4 results runs. See [eval/README.md]
for the CSV row contract and the CLI invocation.
"""

from daccord.eval.clients import GeminiClient, GroqClient, ModelClient
from daccord.eval.prompts import build_eval_prompt, build_judge_prompt
from daccord.eval.schema import CitationCandidate, ModelResponse, PromptMessages

__all__ = [
    "CitationCandidate",
    "GeminiClient",
    "GroqClient",
    "ModelClient",
    "ModelResponse",
    "PromptMessages",
    "build_eval_prompt",
    "build_judge_prompt",
]
