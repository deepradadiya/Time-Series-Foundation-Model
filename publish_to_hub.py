"""HuggingFace Hub publishing script.

Handles model upload, Space deployment, and retry logic for network operations.
"""

import os
import sys
import time
from typing import Any, Callable

from huggingface_hub import HfApi

from generate_readme import generate_readme


def generate_model_card(
    model_name: str,
    param_count: int,
    domains: dict[str, int],
    metrics: dict[str, dict[str, float]],
    github_url: str,
) -> str:
    """Generate a Model_Card markdown string.

    Sections: model name + architecture summary, pretraining domains,
    benchmark results table, 5-line Python usage example, GitHub link.

    Args:
        model_name: Name of the model (used as the heading).
        param_count: Total number of trainable parameters.
        domains: Mapping of domain_name -> row_count for pretraining data.
        metrics: Mapping of method_name -> {metric_name: value}.
            Expected metric keys: MAE, MSE, MASE, CRPS.
        github_url: URL to the GitHub repository.

    Returns:
        Complete model card as a markdown string.
    """
    lines: list[str] = []

    # Model name heading
    lines.append(f"# {model_name}")
    lines.append("")

    # Architecture summary
    lines.append("## Architecture")
    lines.append("")
    lines.append(
        f"PatchTST transformer encoder with {param_count:,} parameters. "
        "Uses patch-based tokenization (patch length 16) with masked patch "
        "modeling for self-supervised pretraining, followed by a probabilistic "
        "forecast head producing P10/P50/P90 quantile predictions."
    )
    lines.append("")

    # Pretraining domains
    lines.append("## Pretraining Domains")
    lines.append("")
    for domain_name, row_count in domains.items():
        lines.append(f"- {domain_name}: {row_count:,} rows")
    lines.append("")

    # Benchmark results table
    lines.append("## Benchmark Results")
    lines.append("")
    lines.append("| Method | MAE | MSE | MASE | CRPS |")
    lines.append("|--------|-----|-----|------|------|")
    for method_name, method_metrics in metrics.items():
        mae = f"{method_metrics.get('MAE', 0.0):.4f}"
        mse = f"{method_metrics.get('MSE', 0.0):.4f}"
        mase = f"{method_metrics.get('MASE', 0.0):.4f}"
        crps = f"{method_metrics.get('CRPS', 0.0):.4f}"
        lines.append(f"| {method_name} | {mae} | {mse} | {mase} | {crps} |")
    lines.append("")

    # Python usage example (5 lines)
    lines.append("## Usage")
    lines.append("")
    lines.append("```python")
    lines.append("import torch")
    lines.append("from model.patchtst import PatchTSTModel")
    lines.append("from forecasting.inference import run_zero_shot_inference")
    lines.append(f'model = PatchTSTModel.from_pretrained("{model_name}")')
    lines.append("predictions = run_zero_shot_inference(model, input_series, horizon=96)")
    lines.append("```")
    lines.append("")

    # Links
    lines.append("## Links")
    lines.append("")
    lines.append(f"- [GitHub]({github_url})")
    lines.append("")

    return "\n".join(lines)


def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Any:
    """Execute func with exponential backoff retry on errors.

    Retries up to max_retries times. Delay doubles each retry:
    base_delay * 2^k where k is the retry number (0-indexed),
    giving delays of 2s, 4s, 8s for the default base_delay of 2.0.

    Args:
        func: A callable to execute. Called with no arguments.
        max_retries: Maximum number of retry attempts after the initial call.
            Total attempts = max_retries + 1.
        base_delay: Base delay in seconds for the first retry. Each subsequent
            retry doubles the delay.

    Returns:
        The return value of func() on success.

    Raises:
        The last exception encountered if all retries are exhausted.
    """
    last_exception: BaseException | None = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)

    raise last_exception  # type: ignore[misc]


def _ensure_space_requirements(
    requirements_path: str = "requirements.txt",
) -> None:
    """Ensure requirements.txt exists with the necessary Space dependencies.

    If the file already exists, verifies that key dependencies are present.
    If missing dependencies are found, they are appended. If the file doesn't
    exist, it is created with all required dependencies.

    The key dependencies for the HuggingFace Space deployment are:
    - gradio (UI framework)
    - plotly (interactive charts)
    - torch (model inference)
    - numpy (numerical operations)
    - pandas (data handling)

    Args:
        requirements_path: Path to the requirements.txt file.
    """
    required_packages = ["gradio", "plotly", "torch", "numpy", "pandas"]

    if os.path.isfile(requirements_path):
        with open(requirements_path, "r") as f:
            content = f.read()

        # Check which required packages are missing
        missing = []
        for pkg in required_packages:
            # Check if the package name appears at the start of any line
            # (handles cases like "numpy>=1.24.0" or just "numpy")
            found = False
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith(pkg):
                    found = True
                    break
            if not found:
                missing.append(pkg)

        if missing:
            with open(requirements_path, "a") as f:
                for pkg in missing:
                    f.write(f"{pkg}\n")
            print(
                f"Added missing dependencies to {requirements_path}: "
                f"{', '.join(missing)}"
            )
        else:
            print(f"requirements.txt is up-to-date with all Space dependencies.")
    else:
        # Create requirements.txt from scratch
        lines = [
            "# Runtime dependencies for HuggingFace Space deployment",
            "# These are the packages required to run the Gradio forecasting demo app",
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "pandas>=2.0.0",
            "plotly>=5.0.0",
            "gradio>=4.0.0",
            "",
        ]
        with open(requirements_path, "w") as f:
            f.write("\n".join(lines))
        print(f"Created {requirements_path} with Space dependencies.")


def publish_all(
    hf_token: str | None = None,
    pretrained_path: str = "checkpoints/pretrained_patchtst.pt",
    finetuned_path: str = "checkpoints/finetuned_patchtst.pt",
) -> None:
    """Push models and deploy Space to HuggingFace Hub.

    Steps:
    1. Validate HF_TOKEN from env var or parameter
    2. Validate checkpoint files exist
    3. Generate up-to-date README via generate_readme()
    4. Ensure requirements.txt exists for Space deployment
    5. Resolve username via HF Hub API whoami
    6. Push pretrained checkpoint + model card
    7. Push fine-tuned checkpoint + model card
    8. Deploy Gradio Space with app source + requirements.txt

    Args:
        hf_token: HuggingFace API token. If None, reads from HF_TOKEN env var.
        pretrained_path: Path to the pretrained model checkpoint.
        finetuned_path: Path to the fine-tuned model checkpoint.
    """
    # 1. Validate HF_TOKEN
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        print(
            "Error: HF_TOKEN environment variable is not set. "
            "Please set HF_TOKEN to your HuggingFace API token.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 2. Validate checkpoint files exist
    if not os.path.isfile(pretrained_path):
        print(
            f"Error: Pretrained checkpoint not found at '{pretrained_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(finetuned_path):
        print(
            f"Error: Fine-tuned checkpoint not found at '{finetuned_path}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    # 3. Generate up-to-date README before publishing
    print("Generating README.md...")
    generate_readme()
    print("README.md generated successfully.")

    # 4. Ensure requirements.txt exists for Space deployment
    _ensure_space_requirements()

    # 5. Resolve username via HF Hub API
    api = HfApi(token=token)
    print("Resolving HuggingFace username...")
    user_info = retry_with_backoff(lambda: api.whoami())
    username = user_info["name"]
    print(f"Authenticated as: {username}")

    # Repository names
    pretrained_repo = f"{username}/patchtst-foundation-pretrained"
    finetuned_repo = f"{username}/patchtst-etth1-finetuned"
    space_repo = f"{username}/timeseries-foundation-demo"

    # Default model card parameters
    github_url = "https://github.com/your-username/Time-Series-Foundation-Model"
    domains = {"Energy": 50000, "Weather": 50000, "Finance": 50000}
    metrics = {
        "PatchTST (zero-shot)": {"MAE": 0.0, "MSE": 0.0, "MASE": 0.0, "CRPS": 0.0},
        "PatchTST (fine-tuned)": {"MAE": 0.0, "MSE": 0.0, "MASE": 0.0, "CRPS": 0.0},
    }

    # 6. Push pretrained checkpoint + model card
    print(f"Creating repository: {pretrained_repo}...")
    retry_with_backoff(
        lambda: api.create_repo(pretrained_repo, exist_ok=True, repo_type="model")
    )

    pretrained_card = generate_model_card(
        model_name="PatchTST Foundation (Pretrained)",
        param_count=2_500_000,
        domains=domains,
        metrics=metrics,
        github_url=github_url,
    )

    print(f"Uploading pretrained checkpoint to {pretrained_repo}...")
    retry_with_backoff(
        lambda: api.upload_file(
            path_or_fileobj=pretrained_path,
            path_in_repo="pretrained_patchtst.pt",
            repo_id=pretrained_repo,
            repo_type="model",
        )
    )

    print(f"Uploading model card to {pretrained_repo}...")
    retry_with_backoff(
        lambda: api.upload_file(
            path_or_fileobj=pretrained_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=pretrained_repo,
            repo_type="model",
        )
    )

    # 7. Push fine-tuned checkpoint + model card
    print(f"Creating repository: {finetuned_repo}...")
    retry_with_backoff(
        lambda: api.create_repo(finetuned_repo, exist_ok=True, repo_type="model")
    )

    finetuned_card = generate_model_card(
        model_name="PatchTST ETTh1 (Fine-tuned)",
        param_count=2_500_000,
        domains=domains,
        metrics=metrics,
        github_url=github_url,
    )

    print(f"Uploading fine-tuned checkpoint to {finetuned_repo}...")
    retry_with_backoff(
        lambda: api.upload_file(
            path_or_fileobj=finetuned_path,
            path_in_repo="finetuned_patchtst.pt",
            repo_id=finetuned_repo,
            repo_type="model",
        )
    )

    print(f"Uploading model card to {finetuned_repo}...")
    retry_with_backoff(
        lambda: api.upload_file(
            path_or_fileobj=finetuned_card.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=finetuned_repo,
            repo_type="model",
        )
    )

    # 8. Deploy Gradio Space
    print(f"Creating Space: {space_repo}...")
    retry_with_backoff(
        lambda: api.create_repo(
            space_repo, exist_ok=True, repo_type="space", space_sdk="gradio"
        )
    )

    # Upload app source file
    print(f"Uploading app source to {space_repo}...")
    retry_with_backoff(
        lambda: api.upload_file(
            path_or_fileobj="app/gradio_app.py",
            path_in_repo="app.py",
            repo_id=space_repo,
            repo_type="space",
        )
    )

    # Upload requirements.txt
    print(f"Uploading requirements.txt to {space_repo}...")
    retry_with_backoff(
        lambda: api.upload_file(
            path_or_fileobj="requirements.txt",
            path_in_repo="requirements.txt",
            repo_id=space_repo,
            repo_type="space",
        )
    )

    print("\nPublishing complete!")
    print(f"  Pretrained model: https://huggingface.co/{pretrained_repo}")
    print(f"  Fine-tuned model: https://huggingface.co/{finetuned_repo}")
    print(f"  Gradio Space:     https://huggingface.co/spaces/{space_repo}")


if __name__ == "__main__":
    publish_all()
