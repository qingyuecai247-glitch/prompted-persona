import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DATA = Path(os.environ.get("PERSONA_GENERATION_DATA", REPO_ROOT / "data/raw"))
GENERATION_OUTPUT = Path(os.environ.get("PERSONA_GENERATION_OUTPUT", REPO_ROOT / "outputs/generation"))

import json
import requests
from tqdm import tqdm

INPUT_JSON = os.environ.get("PERSONA_SITUATION_INPUT", str(GENERATION_DATA / "simulate_descriptions_study1.json"))
OUTPUT_JSON = os.environ.get("PERSONA_SITUATION_ENRICHED", str(GENERATION_OUTPUT / "simulate_descriptions_study1_enriched.json"))
LLM_URL = os.environ.get("PERSONA_LLM_URL", "")

SYSTEM_PROMPT = (
    "You will receive a short image description. "
    "Rewrite it into a slightly longer and clearer objective description (more than 20 words). "
    "Do not add interpretations, emotions, intentions, assumptions, or value judgments. "
    "Do not introduce new objects or remove existing objects. "
    "Describe only what can be directly observed in the scene. "
    "Keep it concise and purely factual."
)

def call_llm(desc):
    if not LLM_URL:
        raise RuntimeError("Set PERSONA_LLM_URL before enrichment.")
    payload = {
        "max_tokens": 256,
        "temperature": 0.5,  # lower temp → more objective
        "top_p": 0.9,
        "repetition_penalty": 1.05,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": desc}
        ]
    }
    try:
        resp = requests.post(LLM_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print("ERROR calling LLM:", e, desc)
        return desc

def main():
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    new_data = {}
    for img, desc in tqdm(data.items(), desc="Processing"):
        result = call_llm(desc)
        enriched = result if isinstance(result, str) else desc
        new_data[img] = enriched.strip()

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)

    print("\nSaved to:", OUTPUT_JSON)

if __name__ == "__main__":
    main()