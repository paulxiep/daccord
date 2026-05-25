"""SageMaker packaging + custom inference handler.

  - `sagemaker_handler` — module loaded by SageMaker at endpoint start; wraps
    `daccord.serving.HybridRouter`. Renamed to `code/inference.py` inside
    the `model.tar.gz` per SageMaker convention.
  - `package_model` — CLI that bundles QLoRA adapter + retrieval index +
    embedder snapshot + handler into the SageMaker S3 layout.

Both consumed at tier 14D (`scripts/deploy_endpoint.py` uploads the
archive; the endpoint stands up referencing the S3 URI).
"""
