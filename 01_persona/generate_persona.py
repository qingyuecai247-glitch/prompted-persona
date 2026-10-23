import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENERATION_DATA = Path(os.environ.get("PERSONA_GENERATION_DATA", REPO_ROOT / "data/raw"))
GENERATION_OUTPUT = Path(os.environ.get("PERSONA_GENERATION_OUTPUT", REPO_ROOT / "outputs/generation"))

import json
import pandas as pd
import requests
import random
import time
from tqdm import tqdm

LLM_API_URL = os.environ.get("PERSONA_LLM_URL", "")

def call_llm(messages, max_tokens=2048, temperature=0.7, top_p=0.9, repetition_penalty=1.1):
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

def score_big_five(first_person_text):
    scoring_prompt = (
        "Based on the following first-person self-introduction, please rate the Big Five personality dimensions. "
        "For each dimension, choose one label: 'Low', 'Medium', or 'High', according to the following definitions:\n"
        "1. Openness to Experience:\n"
        "   - Low: Prefers familiar things, traditional, pragmatic, little interest in abstract ideas.\n"
        "   - Medium: Balanced between conservatism and exploration.\n"
        "   - High: Open-minded, creative, strong aesthetic sense, enjoys new ideas.\n"
        "2. Conscientiousness:\n"
        "   - Low: Spontaneous, lacks planning, procrastinates easily.\n"
        "   - Medium: Balanced between flexibility and self-discipline.\n"
        "   - High: Organized, responsible, goal-oriented, considered reliable.\n"
        "3. Extraversion:\n"
        "   - Low: Introverted, prefers solitude, low social energy.\n"
        "   - Medium: Can switch freely between socializing and solitude.\n"
        "   - High: Enthusiastic, talkative, enjoys socializing, energetic.\n"
        "4. Agreeableness:\n"
        "   - Low: Distrustful, competitive, rarely compromises.\n"
        "   - Medium: Can cooperate but also maintains independence.\n"
        "   - High: Friendly, trusting, helpful, empathetic.\n"
        "5. Neuroticism:\n"
        "   - Low: Emotionally stable, calm, rarely experiences strong negative emotions.\n"
        "   - Medium: Occasional mood swings but generally well-regulated.\n"
        "   - High: Anxious, emotionally sensitive, easily stressed.\n"
        "Please reply strictly in the following JSON format:\n"
        '{"Openness": "<Low/Medium/High>", "Conscientiousness": "<Low/Medium/High>", "Extraversion": "<Low/Medium/High>", "Agreeableness": "<Low/Medium/High>", "Neuroticism": "<Low/Medium/High>"}\n'
        "Do not output anything else.\n"
        f"Self-introduction:\n{first_person_text}"
    )
    messages = [
        {"role": "system", "content": "You are an expert in psychology and personality assessment."},
        {"role": "user", "content": scoring_prompt}
    ]
    for _ in range(3):
        result = call_llm(messages)
        try:
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            json_str = result[json_start:json_end]
            json_str = json_str.replace('\\"', '"')
            json_str = json_str.replace('\\n', '')
            data = json.loads(json_str)
            return (
                data.get("Openness", ""),
                data.get("Conscientiousness", ""),
                data.get("Extraversion", ""),
                data.get("Agreeableness", ""),
                data.get("Neuroticism", "")
            )
        except Exception as e:
            print(f"Failed: {e}")
            pass
    return ("", "", "", "", "")

def generate_persona_and_first_person(row):
    prompt = (
        f"Given the following information about a famous person:\n"
        f"Occupation: {row['occupation']}\n"
        f"Birthplace: {row['bplace_name']}\n"
        f"Birthdate: {row['birthdate']}\n"
        f"Age: {row['age']}\n"
        f"Please strictly reply in the following JSON format:\n"
        f'{{"description": "<A detailed and specific third-person paragraph for role-playing and Big Five test>", '
        f'"first_person": "<A rich and detailed first-person self-introduction, anonymized, focusing on personality traits and behaviors, without any explicit personal information, suitable for Big Five test and ideal for LLM role-play to simulate this person\'s psychological and behavioral state>"}}.\n'
        f"Do not output anything else.\n"
        f"1. The 'description' field must be a single, detailed and specific paragraph in English, describing this person for role-playing, with explicit behavioral and psychological cues for each of the Big Five personality dimensions (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism). "
        f"Please clearly indicate the person's tendencies and levels in each dimension, using concrete examples of behaviors, thoughts, and emotions.\n"
        f"2. The 'first_person' field must rewrite this description in the first person, strictly anonymizing all personal information (such as name, place, date, or any identifying details), and provide a rich, vivid self-introduction that focuses on personality traits and behavioral tendencies for a Big Five personality test. "
        f"This self-introduction should be suitable for LLM role-play, allowing the model to simulate this person's psychological and behavioral state. "
        f"Do not reveal any explicit personal information, only personality and behavioral characteristics, and make sure to cover all five Big Five dimensions in detail."
    )
    messages = [
        {"role": "system", "content": "You are an expert in psychology and character persona creation for role-playing and personality assessment."},
        {"role": "user", "content": prompt}
    ]
    for _ in range(5):
        result = call_llm(messages)
        try:
            json_start = result.find('{')
            json_end = result.rfind('}') + 1
            json_str = result[json_start:json_end]
            json_str = json_str.replace('\\"', '"')
            json_str = json_str.replace('\\n', '')
            data = json.loads(json_str)
            description = data["description"].strip()
            first_person = data["first_person"].strip()
            big_five_scores = score_big_five(first_person)
            return description, first_person, big_five_scores
        except Exception as e:
            print(f"Failed: {e}")
            pass
    return "", "", ("", "", "", "", "")


if __name__ == "__main__":
    GENERATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(os.environ.get('PERSONA_CANDIDATES_CSV', str(GENERATION_DATA / 'person_2025_update.csv')))
    sampled_df = df.sample(n=int(os.environ.get("PERSONA_SAMPLE_SIZE", "5000")), random_state=42)

    descriptions = []
    first_persons = []
    openness_scores = []
    conscientiousness_scores = []
    extraversion_scores = []
    agreeableness_scores = []
    neuroticism_scores = []

    for _, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0], desc="Generating personas"):
        description, first_person, big_five_scores = generate_persona_and_first_person(row)
        descriptions.append(description)
        first_persons.append(first_person)
        openness_scores.append(big_five_scores[0])
        conscientiousness_scores.append(big_five_scores[1])
        extraversion_scores.append(big_five_scores[2])
        agreeableness_scores.append(big_five_scores[3])
        neuroticism_scores.append(big_five_scores[4])
        

    sampled_df['description'] = descriptions
    sampled_df['first_person'] = first_persons
    sampled_df['Openness'] = openness_scores
    sampled_df['Conscientiousness'] = conscientiousness_scores
    sampled_df['Extraversion'] = extraversion_scores
    sampled_df['Agreeableness'] = agreeableness_scores
    sampled_df['Neuroticism'] = neuroticism_scores

    sampled_df[['name', 'description', 'first_person', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']].to_csv(
        str(GENERATION_OUTPUT / 'persona_profile_5000_en.csv'), index=False
    )
    print("Generated 5000 persona descriptions, anonymized first-person self-introductions, and Big Five dimension scores in English suitable for personality testing.")

    print("\nBig Five Personality Label Distribution:")
    for trait in ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']:
        print(f"\n{trait}:")
        print(sampled_df[trait].value_counts())