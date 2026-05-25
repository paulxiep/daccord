"""Side-by-side comparison demo for d'accord.

Streamlit app at `app.py` shows three columns per query: retrieval baseline,
fine-tuned d'accord, and base Qwen (or gold answer when input is in eval set).
Each column tagged with provenance. CSV export drops the comparison row into
a compliance-team-importable shape.

Backed at production time by the hybrid SageMaker endpoint
(publish/sagemaker_handler.py); also supports a local-mode demo path using
src/daccord/serving/HybridRouter directly.
"""
