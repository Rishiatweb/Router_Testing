import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from pypdf import PdfReader, PdfWriter
from pypdf.generic import BooleanObject, NameObject

try:
    import project_secrets
    ENDPOINT = project_secrets.AZURE_AI_ENDPOINT
    KEY = project_secrets.AZURE_AI_KEY
    MODEL_GENERATOR = project_secrets.AZURE_MODEL_GENERATOR
    MODEL_CRITIC = project_secrets.AZURE_MODEL_CRITIC
except Exception:
    ENDPOINT = None
    KEY = None
    MODEL_GENERATOR = None
    MODEL_CRITIC = None

client = None
if ENDPOINT and KEY:
    client = ChatCompletionsClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY)
    )

@dataclass
class Candidate:
    mapping: Dict[str, Any]
    score: float
    notes: str


def extract_pdf_form_fields(pdf_path: str) -> Dict[str, Dict[str, Any]]:
    reader = PdfReader(pdf_path)
    fields = reader.get_fields() or {}
    normalized = {}
    for name, meta in fields.items():
        normalized[name] = {
            "name": name,
            "value": meta.get("/V"),
            "type": str(meta.get("/FT")) if meta else None,
        }
    return normalized


def extract_pdf_text(pdf_path: str, max_pages: int = 2) -> str:
    reader = PdfReader(pdf_path)
    texts = []
    for i, page in enumerate(reader.pages[:max_pages]):
        texts.append(page.extract_text() or "")
    return "\n".join(texts)


def _extract_json_block(text: str) -> str:
    # Extract the first JSON object or array in the response
    match = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    return match.group(1) if match else "{}"


def _parse_json(text: str) -> Dict[str, Any]:
    block = _extract_json_block(text)
    try:
        return json.loads(block)
    except Exception:
        return {}


def call_azure_llm(messages: List[Dict[str, str]], model_name: str, temperature: float = 0.2) -> str:
    if not client:
        return "Error: Azure client not configured"
    response = client.complete(
        messages=[
            SystemMessage(content=messages[0]["content"]),
            UserMessage(content=messages[1]["content"]),
        ],
        model=model_name,
        temperature=temperature,
        max_tokens=1200
    )
    return response.choices[0].message.content.strip()


def generate_candidate_mapping(form_fields: Dict[str, Dict[str, Any]], user_data: Dict[str, Any],
                               form_text: str, model_name: str, temperature: float, seed: int) -> Dict[str, Any]:
    fields_list = list(form_fields.keys())
    system = (
        "You are a precise PDF form-filling assistant."
        " Return ONLY JSON. No commentary."
    )
    user = (
        "Given the PDF form field names and user data keys, produce a mapping JSON.\n"
        "Each entry: field_name -> {source: <user_data_key>, transform: <optional string>}\n"
        "If no suitable data key exists, set source to null.\n\n"
        f"FORM_FIELDS: {fields_list}\n\n"
        f"USER_DATA_KEYS: {list(user_data.keys())}\n\n"
        f"FORM_TEXT_SNIPPET: {form_text[:2000]}\n\n"
        f"SEED_HINT: {seed}"
    )
    raw = call_azure_llm(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model_name=model_name,
        temperature=temperature
    )
    return _parse_json(raw)


def critic_mapping(form_fields: Dict[str, Dict[str, Any]], user_data: Dict[str, Any],
                   mapping: Dict[str, Any], model_name: str) -> str:
    system = "You are a strict reviewer of PDF form field mappings."
    user = (
        "Review the mapping for errors, missing fields, and wrong sources. "
        "Return a short critique list.\n\n"
        f"FORM_FIELDS: {list(form_fields.keys())}\n"
        f"USER_DATA_KEYS: {list(user_data.keys())}\n"
        f"MAPPING_JSON: {json.dumps(mapping, ensure_ascii=True)}"
    )
    raw = call_azure_llm(
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        model_name=model_name,
        temperature=0.1
    )
    return raw


def evaluate_mapping(form_fields: Dict[str, Dict[str, Any]], user_data: Dict[str, Any],
                     mapping: Dict[str, Any]) -> Tuple[float, str]:
    if not mapping:
        return 0.0, "empty mapping"

    total = len(form_fields)
    if total == 0:
        return 0.0, "no form fields"

    hit = 0
    miss = 0
    bad = 0
    for field in form_fields.keys():
        entry = mapping.get(field)
        if entry is None:
            miss += 1
            continue
        if isinstance(entry, dict):
            src = entry.get("source")
        else:
            src = entry
        if src is None:
            miss += 1
        elif src in user_data:
            hit += 1
        else:
            bad += 1

    score = (hit - 0.5 * bad) / max(total, 1)
    score = max(score, 0.0)
    notes = f"hit={hit}, miss={miss}, bad={bad}, total={total}"
    return score, notes


def evolve_mappings(form_fields: Dict[str, Dict[str, Any]], user_data: Dict[str, Any],
                    form_text: str, generations: int = 3, population: int = 5,
                    generator_model: str | None = None, critic_model: str | None = None) -> List[Candidate]:
    gen_model = generator_model or MODEL_GENERATOR
    crit_model = critic_model or MODEL_CRITIC

    if not gen_model or not crit_model:
        raise RuntimeError("Azure model deployments not configured")

    candidates: List[Candidate] = []
    for g in range(generations):
        new_candidates: List[Candidate] = []
        for i in range(population):
            mapping = generate_candidate_mapping(
                form_fields=form_fields,
                user_data=user_data,
                form_text=form_text,
                model_name=gen_model,
                temperature=0.2 + (i * 0.05),
                seed=g * 100 + i
            )
            score, notes = evaluate_mapping(form_fields, user_data, mapping)
            new_candidates.append(Candidate(mapping=mapping, score=score, notes=notes))

        new_candidates.sort(key=lambda c: c.score, reverse=True)
        top = new_candidates[: max(1, population // 2)]
        candidates.extend(new_candidates)

        # Critique the best candidate and let next generation implicitly improve via prompt
        best = top[0]
        critique = critic_mapping(form_fields, user_data, best.mapping, crit_model)
        form_text = form_text + "\n\nCRITIQUE:\n" + critique

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates


def fill_pdf_form(pdf_path: str, output_path: str, mapping: Dict[str, Any], user_data: Dict[str, Any]) -> None:
    reader = PdfReader(pdf_path)
    # Prefer cloning to preserve the full document structure (including AcroForm)
    try:
        writer = PdfWriter(clone_from=reader)
    except Exception:
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        # Ensure AcroForm is present in the writer so update_page_form_field_values works
        try:
            acroform = reader.trailer["/Root"].get("/AcroForm")
            if acroform:
                writer._root_object.update({NameObject("/AcroForm"): acroform})
                # Ask PDF viewers to regenerate appearances for filled fields
                if NameObject("/NeedAppearances") not in acroform:
                    acroform.update({NameObject("/NeedAppearances"): BooleanObject(True)})
        except Exception:
            pass

    field_values = {}
    for field, entry in mapping.items():
        if isinstance(entry, dict):
            src = entry.get("source")
        else:
            src = entry
        if src in user_data:
            val = user_data[src]
            # pypdf expects checkbox values as NameObject like "/Yes" or "/Off"
            if isinstance(val, bool):
                field_values[field] = "/Yes" if val else "/Off"
            else:
                field_values[field] = str(val)

    # Update all pages in the writer
    writer.update_page_form_field_values(None, field_values)

    with open(output_path, "wb") as f:
        writer.write(f)
