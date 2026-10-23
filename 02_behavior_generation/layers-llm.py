import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DATA = Path(os.environ.get("PERSONA_GENERATION_DATA", REPO_ROOT / "data/raw"))
GENERATION_OUTPUT = Path(os.environ.get("PERSONA_GENERATION_OUTPUT", REPO_ROOT / "outputs/generation"))

import os
import json
import csv
import time
import re
from tqdm import tqdm
from statistics import mean, stdev
from typing import Tuple, Any, Dict
import requests

try:
    from openai import AzureOpenAI
    _AZURE_AVAILABLE = True
except Exception:
    _AZURE_AVAILABLE = False


subscription_key = os.environ.get("AZURE_OPENAI_API_KEY", "")

MODEL_KEY = os.environ.get("PERSONA_MODEL_KEY", "gpt-4o-mini")
MODEL_INFO = {
    "gpt-4o-mini": {
        "name": "gpt-4o-mini",
        "price_per_1M_input": 0.15,
        "price_per_1M_output": 0.60,
        "price_per_1M_cached_input": 0.075
    },
    "gpt-4o": {
        "name": "gpt-4o",
        "price_per_1M_input": 2.5,
        "price_per_1M_output": 10,
        "price_per_1M_cached_input": 1.25
    },
    "gpt-5.2": {
        "name": "gpt-5.2",
        "price_per_1M_input": 1.75,
        "price_per_1M_output": 14,
        "price_per_1M_cached_input": 0.175
    }
}
model_name = os.environ.get("PERSONA_MODEL_NAME", MODEL_INFO[MODEL_KEY]["name"])

if _AZURE_AVAILABLE and subscription_key and os.environ.get("AZURE_OPENAI_ENDPOINT"):
    client = AzureOpenAI(
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=subscription_key,
    )
else:
    client = None

PRICE_INPUT_PER_TOKEN = MODEL_INFO[MODEL_KEY]["price_per_1M_input"] / 1_000_000
PRICE_OUTPUT_PER_TOKEN = MODEL_INFO[MODEL_KEY]["price_per_1M_output"] / 1_000_000
PRICE_CACHED_INPUT_PER_TOKEN = MODEL_INFO[MODEL_KEY]["price_per_1M_cached_input"] / 1_000_000

csv_path = os.environ.get("PERSONA_PROFILE_CSV", str(GENERATION_DATA / "persona_profile_600_sampled.csv"))
big5_questions_path = os.environ.get("PERSONA_BIG5_QUESTIONS", str(GENERATION_DATA / "big_five.txt"))
situation_path = os.environ.get("PERSONA_SITUATION_JSON", str(GENERATION_DATA / "simulate_descriptions_study1.json"))
facet_questions_path = os.environ.get("PERSONA_STATE_QUESTIONS", str(GENERATION_DATA / "state_personality_llm_questions.json"))
llm_final_output_path = os.environ.get("PERSONA_SITUATION_FEATURES", str(GENERATION_DATA / "llm_final_output.csv"))
output_all_file = str(GENERATION_OUTPUT / "simulation_output_all.jsonl")
output_mean_file = str(GENERATION_OUTPUT / "simulation_output_mean.jsonl")
output_flat_file = str(GENERATION_OUTPUT / "simulation_output_flattened.csv")
usage_log_file = str(GENERATION_OUTPUT / "usage_log.csv")

big5_repeat = int(os.environ.get("PERSONA_BIG5_REPEATS", "5"))
facet_repeat = int(os.environ.get("PERSONA_STATE_REPEATS", "5"))

BIG5_DIM_MAP = {
    "O": [5, 10, 15, 20, 25, 30],
    "C": [3, 8, 13, 18, 23, 28, 33],
    "E": [1, 6, 11, 16, 21, 26, 31],
    "A": [2, 7, 12, 17, 22, 27, 32],
    "N": [4, 9, 14, 19, 24, 29, 34],
}
REVERSE_ITEMS = {2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29, 30, 32, 34}

MODEL_SOURCE = os.environ.get("PERSONA_MODEL_SOURCE", "local")
LOCAL_LLM_URL = os.environ.get("PERSONA_LLM_URL", "")
LOCAL_REQUEST_TIMEOUT = 180  # seconds for requests
LOCAL_COUNT_TOKENS = False

# Utility functions
def ensure_result_dir():
    os.makedirs(os.path.dirname(output_all_file), exist_ok=True)

def load_personas(csv_path, max_count=600, start_index=int(os.environ.get("PERSONA_START_INDEX", "540"))):
    personas = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < start_index:
                continue
            if len(personas) >= max_count:
                break
            persona = {
                "name": row["name"],
                "description": row["description"],
                "first_person": row["first_person"],
            }
            personas.append(persona)
    return personas

def load_list(path):
    with open(path,"r",encoding="utf-8") as f:
        return [line.strip() for line in f.read().splitlines() if line.strip()]

def load_json(path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)

def load_llm_final_output(path):
    data = {}
    with open(path,"r",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["id_s"]] = {k: float(row[k]) for k in ["dut","int","adv","mat","pos","neg","dec","soc"]}
    return data

def call_llm_azure(prompt: str, max_tokens=1500, temperature=0.7) -> Tuple[str,int,int]:
    """
    调用 Azure OpenAI，返回 (content, prompt_tokens, completion_tokens)
    """
    if not _AZURE_AVAILABLE or client is None:
        raise RuntimeError("AzureOpenAI client not available in this environment.")
    while True:
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role":"user","content":prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            content = response.choices[0].message.content.strip()
            usage = response.usage
            return content, usage.prompt_tokens, usage.completion_tokens
        except Exception as e:
            print(f"Azure LLM Error: {e}, retrying in 5s...")
            time.sleep(5)

def call_llm_local(prompt: str, max_tokens=1500, temperature=0.7, top_p=0.95, repetition_penalty=1.0) -> Tuple[str,int,int]:
    payload = {
        "messages": [{"role":"user","content":prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "repetition_penalty": repetition_penalty
    }
    headers = {"Content-Type":"application/json"}
    attempt = 0
    while True:
        attempt += 1
        try:
            resp = requests.post(LOCAL_LLM_URL, json=payload, headers=headers, timeout=LOCAL_REQUEST_TIMEOUT)
            if resp.status_code == 200:
                text = ""
                try:
                    parsed = resp.json()
                    if isinstance(parsed, dict):
                        if "choices" in parsed and isinstance(parsed["choices"], list) and len(parsed["choices"])>0:
                            ch0 = parsed["choices"][0]
                            if isinstance(ch0, dict) and "message" in ch0 and isinstance(ch0["message"], dict):
                                text = ch0["message"].get("content","")
                            elif isinstance(ch0, dict) and "text" in ch0:
                                text = ch0.get("text","")
                            else:
                                text = json.dumps(parsed, ensure_ascii=False)
                        else:
                            # fallback to str(parsed)
                            text = str(parsed)
                    else:
                        text = str(parsed)
                except ValueError:
                    text = resp.text

                # trim
                if isinstance(text, bytes):
                    text = text.decode("utf-8", errors="ignore")
                text = text.strip()

                if LOCAL_COUNT_TOKENS:
                    prompt_tokens = 0
                    completion_tokens = 0
                else:
                    prompt_tokens = 0
                    completion_tokens = 0

                return text, prompt_tokens, completion_tokens
            else:
                print(f"Local LLM returned status {resp.status_code}: {resp.text}. Retrying in 5s...")
                time.sleep(5)
        except Exception as e:
            print(f"Local LLM Error (attempt {attempt}): {e}. Retrying in 5s...")
            time.sleep(5)

def call_llm(prompt: str, max_tokens=1500, temperature=0.7) -> Tuple[str,int,int]:
    if MODEL_SOURCE == "azure" and client is None:
        raise RuntimeError("Configure Azure credentials and endpoint before generation.")
    if MODEL_SOURCE == "local" and not LOCAL_LLM_URL:
        raise RuntimeError("Set PERSONA_LLM_URL before generation.")
    if MODEL_SOURCE == "azure":
        return call_llm_azure(prompt, max_tokens=max_tokens, temperature=temperature)
    elif MODEL_SOURCE == "local":
        return call_llm_local(prompt, max_tokens=max_tokens, temperature=temperature)
    else:
        raise ValueError("Unsupported MODEL_SOURCE: " + str(MODEL_SOURCE))

def parse_json_safe(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            return {}
    return {}

def map_big5_scores(answer_json):
    forward_map = {
        "Very Inaccurate": 1,
        "Moderately Inaccurate": 2,
        "Neither Accurate Nor Inaccurate": 3,
        "Moderately Accurate": 4,
        "Very Accurate": 5
    }
    scores = {}
    for q, ans in answer_json.items():
        # scores[q] = forward_map.get(ans, None)
        val = forward_map.get(ans, 3 if ans else 3)  
        scores[q] = val
    return scores

def aggregate_ocean(scores):
    dims = {"O":[], "C":[], "E":[], "A":[], "N":[]}
    for dim, q_nums in BIG5_DIM_MAP.items():
        for qn in q_nums:
            key = f"Q{qn}"
            if key in scores:
                val = scores[key]
                if val is None or not isinstance(val, (int, float)):
                    val = 3
                if qn in REVERSE_ITEMS:
                    val = 6 - val
                dims[dim].append(val)
            else:
                dims[dim].append(3)
    return { "Openness": mean(dims["O"]),
             "Conscientiousness": mean(dims["C"]),
             "Extraversion": mean(dims["E"]),
             "Agreeableness": mean(dims["A"]),
             "Neuroticism": mean(dims["N"]) }

# prompt
def prompt_big5(persona_desc, questions):
    q_block = "\n".join([f"{q}" for i,q in enumerate(questions)])
    return f"""
You are a person described as follows:
"{persona_desc}"

Please complete the following Big Five personality questionnaire in **first-person perspective**.
Answer each question based on your current typical traits. Do not use neutral defaults unless appropriate.

Answer each question using one of the following options:
Very Inaccurate, Moderately Inaccurate, Neither Accurate Nor Inaccurate, Moderately Accurate, Very Accurate

Return directly a JSON dictionary:
{{"Q1": "...", ..., "Q{len(questions)}": "..."}}

Questions:
{q_block}
"""

def prompt_facet(persona_desc, traits, situation_desc, facet_questions):
    prompt_parts = [
        f"You are {persona_desc}.",
        f"You are currently in the following situation:\n{situation_desc}\n",
        "Answer the following questions based on your feelings in this situation. Only return an integer 1-6 in JSON."
    ]
    for domain, items in facet_questions.items():
        for item in items:
            q_text = item["question"]
            options = "\n".join(item["options"])
            prompt_parts.append(f"{domain} - {item['facet']}:\n{q_text}\n{options}")
    prompt_parts.append("\nReturn JSON like:\n{\"Extraversion:Sociability\":5, ...}")
    return "\n".join(prompt_parts)

# simulation
def run_simulation():
    ensure_result_dir()
    personas = load_personas(csv_path)
    big5_questions = load_list(big5_questions_path)
    situations_dict = load_json(situation_path)
    situations = list(situations_dict.items())
    facet_questions = load_json(facet_questions_path)
    scene_features = load_llm_final_output(llm_final_output_path)

    for f in [output_all_file, output_mean_file]:
        if os.path.exists(f):
            os.remove(f)

    # flattened csv header
    with open(output_flat_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id_p","id_s","E_trait","A_trait","C_trait","N_trait","O_trait",
            "dut","int","adv","mat","pos","neg","dec","soc",
            "State_E","State_A","State_C","State_N","State_O"
        ])

    # usage log header
    with open(usage_log_file,"w",newline="",encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["persona_name","big5_total_tokens","big5_total_cost","situation_total_tokens","situation_total_cost","model"])


    for persona in tqdm(personas, desc="Simulating personas"):
        persona_desc = persona.get("first_person","")

        big5_raw_responses = []
        big5_tokens_total = 0
        big5_cost_total = 0.0

        for i in range(big5_repeat):
            prompt = prompt_big5(persona_desc, big5_questions)
            response_text, in_tokens, out_tokens = call_llm(prompt, max_tokens=1800, temperature=0.7)
            parsed = parse_json_safe(response_text)
            scores = map_big5_scores(parsed)
            big5_raw_responses.append({"repeat_index": i+1, "raw_response": response_text, "scores": scores})
            big5_tokens_total += in_tokens + out_tokens
            big5_cost_total += in_tokens * PRICE_INPUT_PER_TOKEN + out_tokens * PRICE_OUTPUT_PER_TOKEN

        ocean_values_per_repeat = [aggregate_ocean(r["scores"]) for r in big5_raw_responses]
        big5_mean_scores = {k: mean([d[k] for d in ocean_values_per_repeat]) for k in ocean_values_per_repeat[0]}
        big5_std_scores  = {k: stdev([d[k] for d in ocean_values_per_repeat]) for k in ocean_values_per_repeat[0]}

        situation_tokens_total = 0
        situation_cost_total = 0.0
        facet_scene_scores = {}

        for image_name, description in situations:
            facet_repeat_scores = []

            for r in range(facet_repeat):
                prompt = prompt_facet(persona_desc, big5_mean_scores, description, facet_questions)
                response_text, in_tokens, out_tokens = call_llm(prompt, max_tokens=1500, temperature=0.7)
                parsed = parse_json_safe(response_text)
                facet_repeat_scores.append(parsed)
                situation_tokens_total += in_tokens + out_tokens
                situation_cost_total += in_tokens * PRICE_INPUT_PER_TOKEN + out_tokens * PRICE_OUTPUT_PER_TOKEN


            keys = set()
            for fr in facet_repeat_scores:
                keys.update(fr.keys())
            if not keys:
                keys = set()

            scene_mean = {}
            scene_std = {}
            for k in keys:
                vals = []
                for fr in facet_repeat_scores:
                    v = fr.get(k)
                    try:
                        vals.append(float(v))
                    except Exception:
                        vals.append(3.0)
                scene_mean[k] = mean(vals)
                scene_std[k] = stdev(vals) if len(vals) > 1 else 0.0

            facet_scene_scores[image_name] = {"mean": scene_mean, "std": scene_std}

            # write simulation_output_all.jsonl
            scene_record = {
                "persona": {"name": persona.get("name",""), "first_person": persona_desc},
                "scene_id": image_name,
                "big5_mean_scores": big5_mean_scores,
                "facet_scores": scene_mean
            }
            with open(output_all_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(scene_record, ensure_ascii=False) + "\n")

            # write flattened CSV (scene-level, immediate)
            flattened_row = [
                persona.get("name",""), image_name,
                big5_mean_scores.get("Extraversion", 3.0),
                big5_mean_scores.get("Agreeableness", 3.0),
                big5_mean_scores.get("Conscientiousness", 3.0),
                big5_mean_scores.get("Neuroticism", 3.0),
                big5_mean_scores.get("Openness", 3.0),
            ]
            feat = scene_features.get(image_name, {k:0 for k in ["dut","int","adv","mat","pos","neg","dec","soc"]})
            flattened_row += [feat.get(k,0) for k in ["dut","int","adv","mat","pos","neg","dec","soc"]]

            def mean_of_keys_containing(keyword):
                vals = [v for kk,v in scene_mean.items() if keyword in kk]
                return mean(vals) if vals else 3.0

            state_values = {
                "State_E": mean_of_keys_containing("Extraversion"),
                "State_A": mean_of_keys_containing("Agreeableness"),
                "State_C": mean_of_keys_containing("Conscientiousness"),
                "State_N": mean_of_keys_containing("Neuroticism"),
                "State_O": mean_of_keys_containing("Openness"),
            }
            flattened_row += [state_values[k] for k in ["State_E","State_A","State_C","State_N","State_O"]]

            with open(output_flat_file,"a",newline="",encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(flattened_row)

        persona_mean = {
            "persona": {"name": persona.get("name",""), "first_person": persona_desc},
            "big5_mean_scores": big5_mean_scores,
            "big5_std_scores": big5_std_scores,
            "scene_facet_scores": facet_scene_scores
        }
        with open(output_mean_file,"a",encoding="utf-8") as f:
            f.write(json.dumps(persona_mean, ensure_ascii=False) + "\n")

        # usage_log
        with open(usage_log_file,"a",newline="",encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([persona.get("name",""), big5_tokens_total, big5_cost_total, situation_tokens_total, situation_cost_total, model_name])

    print("All simulation completed.")


if __name__ == "__main__":
    print(f"MODEL_SOURCE = {MODEL_SOURCE}, LOCAL_LLM_URL = {LOCAL_LLM_URL}")
    run_simulation()