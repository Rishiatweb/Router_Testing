# Azure AI Inference settings (Azure AI Foundry / Azure OpenAI compatible)
import os
from pathlib import Path


def _load_local_env() -> None:
    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _first_env(*names: str) -> str:
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return ""


_load_local_env()

# Prefer environment variables for secrets and support both local and Azure OpenAI names.
AZURE_AI_ENDPOINT = _first_env("AZURE_AI_ENDPOINT", "AZURE_OPENAI_ENDPOINT")
AZURE_AI_KEY = _first_env("AZURE_AI_KEY", "AZURE_OPENAI_API_KEY")

# Deployment names must match your Azure deployments.
AZURE_MODEL_GENERATOR = _first_env(
    "AZURE_MODEL_GENERATOR",
    "AZURE_OPENAI_DEPLOYMENT_GPT35",
    "AZURE_OPENAI_DEPLOYMENT_GPT4O",
)
AZURE_MODEL_CRITIC = _first_env(
    "AZURE_MODEL_CRITIC",
    "AZURE_OPENAI_DEPLOYMENT_GPT4O",
    "AZURE_OPENAI_DEPLOYMENT_GPT35",
)
