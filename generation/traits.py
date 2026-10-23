import os
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "config/paths.json").is_file())
import sys
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from generation.bfi2 import score_bfi2

GENERATION_DATA = Path(os.environ.get("PERSONA_GENERATION_DATA", REPO_ROOT / "data/raw"))
GENERATION_OUTPUT = Path(os.environ.get("PERSONA_GENERATION_OUTPUT", REPO_ROOT / "outputs/generation"))

import pandas as pd
import requests
import time
from tqdm import tqdm

LLM_API_URL = os.environ.get("PERSONA_LLM_URL", "")
PERSONA_CSV = os.environ.get("PERSONA_PROFILE_CSV", str(GENERATION_DATA / "persona_profile_500_en.csv"))
BIG_FIVE_PATH = str(GENERATION_DATA / "big_five.txt")
OUTPUT_CSV = str(GENERATION_OUTPUT / "big_five_results_test.csv")
REPEAT = int(os.environ.get("PERSONA_BIG5_REPEATS", "5"))



BIG5_TRAIT_CN = {"O": "开放性", "C": "责任心", "E": "外向性", "A": "宜人性", "N": "神经质"}

def load_big_five_questions(path):
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                idx, q = line.split('.', 1)
                questions.append(q.strip())
    if len(questions) != 60:
        raise ValueError("The BFI-2 scorer requires the complete ordered 60-item questionnaire")
    return questions

def call_llm(system_prompt, user_prompt, max_tokens=128, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty
    }
    if not LLM_API_URL:
        raise RuntimeError("Set PERSONA_LLM_URL before generation.")
    for _ in range(3):
        try:
            response = requests.post(LLM_API_URL, json=payload, timeout=60)
            if response.status_code == 200:
                return response.text.strip()
        except Exception as e:
            print(f"Failed: {e}")
            time.sleep(2)
    return "LLM generation failed."

def parse_score(text):
    for line in text.splitlines():
        line = line.strip()
        if line and line[0] in "12345":
            try:
                score = int(line[0])
                if 1 <= score <= 5:
                    return score
            except:
                continue
    for c in text:
        if c in "12345":
            return int(c)
    return None

def calc_big5_traits(scores_dict):
    return {domain: round(value, 2) for domain, value in score_bfi2(scores_dict).items()}

def test_model_bigfive(REPEAT=5):
    questions = load_big_five_questions(BIG_FIVE_PATH)
    model_results = []
    for repeat in range(REPEAT):
        scores = []
        for qidx, question in enumerate(questions):
            user_prompt = (
                f"Please answer the following statement as yourself, using a 1-5 Likert scale (1=Strongly Disagree, 5=Strongly Agree):\n"
                f"{question}\n"
                f"Only output a single number from 1 to 5."
            )
            answer = call_llm("", user_prompt)
            score = parse_score(answer)
            if score is None:
                print(f"Warning: Could not parse score for MODEL, Q{qidx+1}, answer: {answer}")
                score = ""
            scores.append(score)
            time.sleep(0.1)
        model_results.append(scores)
        time.sleep(0.5)

    avg_scores = []
    for i in range(len(questions)):
        valid_scores = [r[i] for r in model_results if isinstance(r[i], int)]
        avg = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else ""
        avg_scores.append(avg)
    avg_scores_dict = {f"Q{i+1}": avg_scores[i] for i in range(len(questions))}
    big5_traits = calc_big5_traits(avg_scores_dict)
    print("模型本身的大五人格测试结果：")
    for trait in ["O", "C", "E", "A", "N"]:
        print(f"{trait}（{BIG5_TRAIT_CN[trait]}）: {big5_traits[trait]}")
    print("各题平均分：", avg_scores)

def main():
    df = pd.read_csv(PERSONA_CSV)
    questions = load_big_five_questions(BIG_FIVE_PATH)
    df = df.head(200)

    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Big Five Testing"):
        name = row['name']
        system_prompt = row['first_person'] if 'first_person' in row else row.iloc[2]
        persona_results = []

        for repeat in range(REPEAT):
            scores = []
            for qidx, question in enumerate(questions):
                user_prompt = (
                    f"Please answer the following statement as yourself, using a 1-5 Likert scale (1=Strongly Disagree, 5=Strongly Agree):\n"
                    f"{question}\n"
                    f"Only output a single number from 1 to 5."
                )
                answer = call_llm(system_prompt, user_prompt)
                score = parse_score(answer)
                if score is None:
                    print(f"Warning: Could not parse score for {name}, Q{qidx+1}, answer: {answer}")
                    score = ""
                scores.append(score)
                time.sleep(0.1)
            persona_results.append(scores)
            time.sleep(0.5)

        avg_scores = []
        for i in range(len(questions)):
            valid_scores = [r[i] for r in persona_results if isinstance(r[i], int)]
            avg = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else ""
            avg_scores.append(avg)

        avg_scores_dict = {f"Q{i+1}": avg_scores[i] for i in range(len(questions))}
        big5_traits = calc_big5_traits(avg_scores_dict)

        results.append({
            "name": name,
            "first_person": system_prompt,
            **{f"Q{i+1}_avg": avg_scores[i] for i in range(len(questions))},
            **{f"Q{i+1}_r{j+1}": persona_results[j][i] for i in range(len(questions)) for j in range(REPEAT)},
            **{f"{trait}_score": big5_traits[trait] for trait in ["O", "C", "E", "A", "N"]}
        })

    columns = (
        ["name", "first_person"] +
        [f"Q{i+1}_avg" for i in range(len(questions))] +
        [f"Q{i+1}_r{j+1}" for i in range(len(questions)) for j in range(REPEAT)] +
        [f"{trait}_score" for trait in ["O", "C", "E", "A", "N"]]
    )
    result_df = pd.DataFrame(results, columns=columns)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    GENERATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    # main()
    test_model_bigfive()