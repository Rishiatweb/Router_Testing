# PDF-Form-ETTC-Azure

A personal, Azure-based evolutionary test-time compute (ETTC) prototype for AI-driven PDF form filling. The system generates candidate field mappings, evaluates them, and iteratively improves them using an LLM loop and lightweight heuristics.

## Overview
This project targets PDF form filling using Azure AI Inference for LLM calls and a simple evolutionary loop to improve field mappings.

## Project Structure

```
pdf-form-ettc-azure/
├── .gitignore
├── app.py
├── core_logic.py
├── intelligent_router.py
├── run_experiment.py
├── run_hybrid_system.py
├── project_secrets.py
├── requirements.txt
├── data/
└── outputs/
```

## Setup

1. Create a venv and install dependencies:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Configure Azure credentials via environment variables:

```bash
AZURE_AI_ENDPOINT="https://<your-resource>.services.ai.azure.com/models"
AZURE_AI_KEY="<your-key>"
AZURE_MODEL_GENERATOR="<your-generator-deployment>"
AZURE_MODEL_CRITIC="<your-critic-deployment>"
```

## Run

### Streamlit App
```bash
streamlit run app.py
```

### CLI Experiment
```bash
python run_experiment.py --pdf "C:\path\to\form.pdf" --data "C:\path\to\data.json"
```

## Notes
- This prototype uses heuristic evaluation when no ground-truth filled PDFs exist.
- For higher accuracy, provide validation cases or expected filled outputs.
- Checkbox fields expect boolean values in your `user_data` JSON; they will be written as `/Yes` or `/Off` in the PDF.
