#!/usr/bin/env python3
from __future__ import annotations
"""Standalone matched Llama-3.1-8B Base + layer-16 OpenSAE pipeline."""

import sys

# ===== common.py =====

import csv
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAITS = ("E", "A", "C", "N", "O")
TRAIT_NAMES = {
    "E": "Extraversion",
    "A": "Agreeableness",
    "C": "Conscientiousness",
    "N": "Neuroticism",
    "O": "Openness",
}
SITUATION_DIMS = ("dut", "int", "adv", "mat", "pos", "neg", "dec", "soc")



DEFAULT_CONFIG = {
    "project": {
        "seed": 42,
        "output_dir": "./local_assets/llama31_opensae_outputs",
        "fail_score": 3,
    },
    "model": {
        "repo_id": "meta-llama/Meta-Llama-3.1-8B",
        "local_dir": "./local_assets/Meta-Llama-3.1-8B",
        "dtype": "bfloat16",
        "device_map": "auto",
        "max_input_tokens": 2048,
        "expected_hidden_size": 4096,
        "expected_layers": 32,
        "expected_context": 131072,
        "expected_eos_token_id": 128001,
    },
    "sae": {
        "repo_id": "THU-KEG/OpenSAE-LLaMA-3.1-Layer_16",
        "local_dir": "./local_assets/llama3_1_sae_16",
        "layer": 16,
        "expected_hook": "layers.16",
        "expected_d_in": 4096,
        "expected_d_sae": 262144,
        "expected_k": 128,
        "validation_prompts": 200,
        "validation_token_chunk": 32,
    },
    "data": {
        "personas": "./local_assets/code/dataset/persona_profile_600_sampled.csv",
        "big5_questions": "./local_assets/code/dataset/big_five.txt",
        "big5_item_key": "./local_assets/code/dataset/big5_item_key.csv",
        "situations": "./local_assets/code/dataset/simulate_descriptions_study1.json",
        "state_questions": "./local_assets/code/dataset/state_personality_llm_questions.json",
        "situation_features": "./local_assets/code/dataset/situation2.csv",
        "expected_personas": 600,
        "expected_situations": 62,
    },
    "generation": {
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.0,
        "big5_max_new_tokens": 1800,
        "situation_max_new_tokens": 1500,
        "repeats": 5,
        "parse_attempts": 3,
    },
    "discovery": {
        "train_fraction": 0.70,
        "minimum_activation_rate": 0.01,
        "features_per_trait": 20,
        "fdr_alpha": 0.05,
    },
    "intervention": {
        "heldout_personas": 100,
        "calibration_personas": 10,
        "situations": 12,
        "repeats": 3,
        "scales": [0.5, 1.0, 2.0],
        "conditions": [
            "baseline", "positive", "negative",
            "positive_control", "negative_control"
        ],
        "prompt_only": True,
        "batch_size": 4,
        "max_new_tokens": 1500,
        "checkpoint_every_batches": 1,
        "greedy": False,
        "intervention_scope": "final",
    },
}


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix.lower() == ".json":
                cfg = json.load(handle)
            else:
                import yaml
                cfg = yaml.safe_load(handle)
    else:
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        print(f"Config file {path} not found; using the paths embedded at the top of this script.")
    cfg["_config_path"] = str(path)
    cfg["_root"] = str(path.parent)
    return cfg


def project_path(cfg: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(cfg["_root"]) / path


def read_csv_robust(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read study CSVs without modifying them; support UTF-8 and legacy Chinese encodings."""
    path = Path(path)
    last_error = None
    for encoding in ("utf-8-sig", "gb18030", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    raise UnicodeDecodeError(
        last_error.encoding, last_error.object, last_error.start, last_error.end,
        f"Could not decode {path} as UTF-8, GB18030, or GBK",
    )


def persona_id_column(personas: pd.DataFrame) -> str:
    if "id_p" in personas.columns:
        return "id_p"
    if "name" in personas.columns:
        return "name"
    raise ValueError(f"Persona table needs an 'id_p' column; available columns: {list(personas.columns)}")


def ensure_output_dirs(cfg: dict[str, Any]) -> Path:
    output = project_path(cfg, cfg["project"]["output_dir"])
    for name in ("behavior", "sae", "intervention", "manifests"):
        (output / name).mkdir(parents=True, exist_ok=True)
    return output


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(p) for p in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def stable_unit_interval(base_seed: int, *parts: object) -> float:
    return stable_seed(base_seed, *parts) / float(2**32 - 1)


def seed_everything(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted p-values without an extra dependency."""
    values = np.asarray(p_values, dtype=float)
    result = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return result
    raw = values[valid]
    order = np.argsort(raw)
    ranked = raw[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid] = restored
    return result


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


BIG5_LABELS = {
    "very inaccurate": 1.0,
    "moderately inaccurate": 2.0,
    "neither accurate nor inaccurate": 3.0,
    "moderately accurate": 4.0,
    "very accurate": 5.0,
}


def coerce_big5_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and 1 <= float(value) <= 5:
        return float(value)
    if isinstance(value, str):
        normalized = re.sub(r"\s+", " ", value.strip().lower())
        if normalized in BIG5_LABELS:
            return BIG5_LABELS[normalized]
        match = re.search(r"(?<!\d)([1-5])(?:\.0)?(?!\d)", normalized)
        if match:
            return float(match.group(1))
    return None


def coerce_state_value(value: Any) -> float | None:
    if isinstance(value, (int, float)) and 1 <= float(value) <= 6:
        return float(value)
    if isinstance(value, str):
        match = re.search(r"(?<!\d)([1-6])(?:\.0)?(?!\d)", value)
        if match:
            return float(match.group(1))
    return None


def load_big5_questions(path: Path) -> list[str]:
    questions = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            questions.append(re.sub(r"^\d+\.\s*", "", line))
    return questions


def load_item_key(path: Path) -> pd.DataFrame:
    key = pd.read_csv(path)
    required = {"item", "domain", "reverse"}
    if set(key.columns) != required:
        raise ValueError(f"Item key must have exactly {sorted(required)}; got {list(key.columns)}")
    if key["item"].duplicated().any():
        raise ValueError("Item key contains duplicate item IDs")
    if not set(key["domain"]).issubset(TRAITS):
        raise ValueError("Item key contains an unknown Big Five domain")
    return key


def score_big5_repeats(
    parsed_repeats: Iterable[dict[str, Any]], item_key: pd.DataFrame, fail_score: float
) -> dict[str, float]:
    repeat_list = list(parsed_repeats)
    domain_scores: dict[str, list[float]] = {trait: [] for trait in TRAITS}
    for _, row in item_key.iterrows():
        item = str(row["item"])
        values = []
        for parsed in repeat_list:
            value = coerce_big5_value(parsed.get(item))
            values.append(fail_score if value is None else value)
        item_mean = float(np.mean(values)) if values else float(fail_score)
        if int(row["reverse"]):
            item_mean = 6.0 - item_mean
        domain_scores[str(row["domain"])].append(item_mean)
    return {
        trait: float(np.mean(values)) if values else float(fail_score)
        for trait, values in domain_scores.items()
    }


def state_item_spec(state_questions: dict[str, list[dict[str, Any]]]) -> list[tuple[str, str]]:
    result = []
    for domain, items in state_questions.items():
        for item in items:
            result.append((domain, f"{domain}:{item['facet']}"))
    return result


def normalize_state_json(parsed: dict[str, Any]) -> dict[str, Any]:
    """Accept both flat ``Domain:Facet`` and nested ``Domain -> Facet`` JSON."""
    normalized: dict[str, Any] = {}
    for key, value in parsed.items():
        key = str(key).strip()
        if isinstance(value, dict):
            for facet, score in value.items():
                normalized[f"{key}:{str(facet).strip()}"] = score
        else:
            normalized[key] = value
    return normalized


def score_state_repeat(
    parsed: dict[str, Any], state_questions: dict[str, list[dict[str, Any]]], fail_score: float
) -> dict[str, float]:
    by_domain: dict[str, list[float]] = {name: [] for name in TRAIT_NAMES.values()}
    for domain, key in state_item_spec(state_questions):
        value = coerce_state_value(parsed.get(key))
        by_domain[domain].append(float(fail_score if value is None else value))
    return {
        trait: float(np.mean(by_domain[name]))
        for trait, name in TRAIT_NAMES.items()
    }


def build_big5_prompt(persona: str, questions: list[str]) -> str:
    q_block = "\n".join(f"{i}. {question}" for i, question in enumerate(questions, 1))
    return f"""You are adopting the following persona for this research task:
{persona}

Complete the personality questionnaire in the first person. Answer every item
using exactly one of these labels: Very Inaccurate, Moderately Inaccurate,
Neither Accurate Nor Inaccurate, Moderately Accurate, Very Accurate.

Return one JSON object only, with keys Q1 through Q{len(questions)}. Do not add
commentary or Markdown.

Questions:
{q_block}"""


def build_state_prompt(
    persona: str, situation: str, state_questions: dict[str, list[dict[str, Any]]]
) -> str:
    parts = [
        "You are adopting the following persona for this research task:",
        persona,
        "\nCurrent situation:",
        situation,
        "\nRate how you would feel or behave in this situation. Use one integer from 1 to 6 for every item.",
    ]
    keys = []
    for domain, items in state_questions.items():
        for item in items:
            key = f"{domain}:{item['facet']}"
            keys.append(key)
            parts.append(f"\n{key}\n{item['question']}\n" + "\n".join(item["options"]))
    parts.append(
        "\nReturn one JSON object only, without Markdown. Include exactly the "
        "following keys and map every key to an independently selected integer "
        "from 1 to 6. Do not copy a uniform response pattern.\nRequired keys:\n"
        + "\n".join(f"- {key}" for key in keys)
    )
    return "\n".join(parts)


def open_sqlite(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    return connection


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_study_inputs(cfg: dict[str, Any]) -> dict[str, int]:
    personas = read_csv_robust(project_path(cfg, cfg["data"]["personas"]))
    situations = read_json(project_path(cfg, cfg["data"]["situations"]))
    questions = load_big5_questions(project_path(cfg, cfg["data"]["big5_questions"]))
    item_key = load_item_key(project_path(cfg, cfg["data"]["big5_item_key"]))
    state_questions = read_json(project_path(cfg, cfg["data"]["state_questions"]))
    features = read_csv_robust(project_path(cfg, cfg["data"]["situation_features"]))
    id_col = persona_id_column(personas)
    required_persona = {id_col, "first_person"}
    if not required_persona.issubset(personas.columns):
        raise ValueError(f"Persona file lacks columns: {required_persona - set(personas.columns)}")
    if personas[id_col].duplicated().any():
        raise ValueError("Persona IDs must be unique")
    if len(questions) != len(item_key):
        raise ValueError(f"Question count {len(questions)} != item-key count {len(item_key)}")
    missing_situations = set(situations) - set(features["id_s"].astype(str))
    if missing_situations:
        raise ValueError(f"Situation features missing IDs: {sorted(missing_situations)[:5]}")
    return {
        "personas": len(personas),
        "situations": len(situations),
        "big5_items": len(questions),
        "state_items": sum(len(items) for items in state_questions.values()),
    }


def validate_extraction_inputs(cfg: dict[str, Any]) -> dict[str, int]:
    """Validate only the inputs required for SAE feature extraction."""
    personas = read_csv_robust(project_path(cfg, cfg["data"]["personas"]))
    situations = read_json(project_path(cfg, cfg["data"]["situations"]))
    id_col = persona_id_column(personas)
    required_persona = {id_col, "first_person"}
    missing = required_persona - set(personas.columns)
    if missing:
        raise ValueError(f"Persona file lacks columns: {missing}")
    if personas[id_col].isna().any() or personas[id_col].duplicated().any():
        raise ValueError("Persona IDs must be non-missing and unique")
    if not isinstance(situations, dict) or not situations:
        raise ValueError("Situation file must contain a non-empty JSON object")
    expected_personas = int(cfg["data"].get("expected_personas", len(personas)))
    expected_situations = int(cfg["data"].get("expected_situations", len(situations)))
    if len(personas) != expected_personas:
        raise ValueError(f"Expected {expected_personas} personas, found {len(personas)}")
    if len(situations) != expected_situations:
        raise ValueError(f"Expected {expected_situations} situations, found {len(situations)}")
    return {"personas": len(personas), "situations": len(situations)}


def write_csv_rows(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


# ===== progress.py =====

try:
    from tqdm import tqdm as _tqdm
except ImportError:  # data-only validation can run before server dependencies are installed
    def _tqdm(iterable=None, *args, **kwargs):
        return iterable if iterable is not None else range(0)

tqdm = _tqdm


# ===== behavior.py =====

import argparse
import json
import sqlite3
import traceback
from pathlib import Path
from typing import Any

import pandas as pd



BEHAVIOR_SCHEMA = """
CREATE TABLE IF NOT EXISTS completions (
    task_key TEXT PRIMARY KEY,
    persona_name TEXT NOT NULL,
    task_type TEXT NOT NULL,
    situation_id TEXT,
    repeat_index INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    raw_text TEXT NOT NULL,
    parsed_json TEXT NOT NULL,
    success INTEGER NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_completion_lookup
ON completions(persona_name, task_type, situation_id, repeat_index);
"""


def initialize_database(path: Path) -> sqlite3.Connection:
    connection = open_sqlite(path)
    connection.executescript(BEHAVIOR_SCHEMA)
    connection.commit()
    return connection


def normalized_big5(parsed: dict[str, Any], n_items: int) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in parsed.items():
        raw = str(key).strip().upper()
        if raw.startswith("Q"):
            raw = raw[1:]
        if raw.isdigit() and 1 <= int(raw) <= n_items:
            normalized[f"Q{int(raw)}"] = value
    return normalized


def valid_fraction(parsed: dict[str, Any], expected_keys: list[str], task_type: str) -> float:
    converter = coerce_big5_value if task_type == "big5" else coerce_state_value
    valid = sum(converter(parsed.get(key)) is not None for key in expected_keys)
    return valid / max(len(expected_keys), 1)


def insert_completion(connection: sqlite3.Connection, row: dict[str, Any]) -> None:
    connection.execute(
        """INSERT OR REPLACE INTO completions
        (task_key, persona_name, task_type, situation_id, repeat_index, seed,
         raw_text, parsed_json, success, error, created_at)
        VALUES (:task_key, :persona_name, :task_type, :situation_id, :repeat_index,
                :seed, :raw_text, :parsed_json, :success, :error, :created_at)""",
        row,
    )
    connection.commit()


def task_exists(connection: sqlite3.Connection, task_key: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM completions WHERE task_key=?", (task_key,)
    ).fetchone() is not None


def run_one_completion(
    connection: sqlite3.Connection,
    tokenizer,
    model,
    cfg: dict[str, Any],
    persona_name: str,
    task_type: str,
    repeat_index: int,
    prompt: str,
    expected_keys: list[str],
    situation_id: str | None = None,
) -> None:

    task_key = f"{task_type}|{persona_name}|{situation_id or '-'}|{repeat_index}"
    if task_exists(connection, task_key):
        return
    seed = stable_seed(int(cfg["project"]["seed"]), task_key)
    max_new_tokens = int(
        cfg["generation"]["big5_max_new_tokens"]
        if task_type == "big5"
        else cfg["generation"]["situation_max_new_tokens"]
    )
    raw_text = ""
    parsed: dict[str, Any] = {}
    errors = []
    success = False
    for attempt in range(1, int(cfg["generation"]["parse_attempts"]) + 1):
        try:
            raw_text = generate_json_text(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                seed=stable_seed(seed, attempt),
                max_new_tokens=max_new_tokens,
                temperature=float(cfg["generation"]["temperature"]),
                top_p=float(cfg["generation"]["top_p"]),
                repetition_penalty=float(cfg["generation"]["repetition_penalty"]),
                max_input_tokens=int(cfg["model"]["max_input_tokens"]),
            )
            parsed = parse_json_object(raw_text)
            if task_type == "situation":
                parsed = normalize_state_json(parsed)
            if task_type == "big5":
                parsed = normalized_big5(parsed, len(expected_keys))
            fraction = valid_fraction(parsed, expected_keys, task_type)
            if fraction >= 0.90:
                success = True
                break
            errors.append(f"attempt {attempt}: only {fraction:.1%} valid fields")
        except Exception as exc:  # the error is persisted for reproducibility
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
    insert_completion(
        connection,
        {
            "task_key": task_key,
            "persona_name": persona_name,
            "task_type": task_type,
            "situation_id": situation_id,
            "repeat_index": repeat_index,
            "seed": seed,
            "raw_text": raw_text,
            "parsed_json": json.dumps(parsed, ensure_ascii=False),
            "success": int(success),
            "error": " | ".join(errors) if errors else None,
            "created_at": utc_now(),
        },
    )


def load_completion_map(connection: sqlite3.Connection) -> dict[tuple[str, str, str | None, int], dict[str, Any]]:
    rows = connection.execute(
        "SELECT persona_name, task_type, situation_id, repeat_index, parsed_json, success FROM completions"
    ).fetchall()
    return {
        (name, task, situation, repeat): {
            "parsed": json.loads(parsed_json),
            "success": bool(success),
        }
        for name, task, situation, repeat, parsed_json, success in rows
    }


def export_behavior(cfg: dict[str, Any], connection: sqlite3.Connection) -> dict[str, Any]:
    output = ensure_output_dirs(cfg) / "behavior"
    personas = read_csv_robust(project_path(cfg, cfg["data"]["personas"]))
    id_col = persona_id_column(personas)
    situations = read_json(project_path(cfg, cfg["data"]["situations"]))
    state_questions = read_json(project_path(cfg, cfg["data"]["state_questions"]))
    item_key = load_item_key(project_path(cfg, cfg["data"]["big5_item_key"]))
    features = read_csv_robust(project_path(cfg, cfg["data"]["situation_features"]))
    features["id_s"] = features["id_s"].astype(str)
    feature_map = features.set_index("id_s").to_dict("index")
    repeats = int(cfg["generation"]["repeats"])
    fail_score = float(cfg["project"]["fail_score"])
    completion_map = load_completion_map(connection)
    rows = []
    complete_personas = 0
    for _, persona in personas.iterrows():
        name = str(persona[id_col])
        big5_records = [completion_map.get((name, "big5", None, repeat)) for repeat in range(1, repeats + 1)]
        if any(record is None for record in big5_records):
            continue
        traits = score_big5_repeats(
            [record["parsed"] for record in big5_records if record], item_key, fail_score
        )
        persona_complete = True
        for situation_id in situations:
            state_records = [
                completion_map.get((name, "situation", str(situation_id), repeat))
                for repeat in range(1, repeats + 1)
            ]
            if any(record is None for record in state_records):
                persona_complete = False
                continue
            state_by_repeat = [
                score_state_repeat(record["parsed"], state_questions, fail_score)
                for record in state_records
                if record
            ]
            state = {
                trait: float(sum(values[trait] for values in state_by_repeat) / len(state_by_repeat))
                for trait in TRAITS
            }
            scene = feature_map[str(situation_id)]
            row = {
                "id_p": name,
                "id_s": str(situation_id),
                **{f"{trait}_trait": traits[trait] for trait in TRAITS},
                **{dim: float(scene[dim]) for dim in SITUATION_DIMS},
                **{f"State_{trait}": state[trait] for trait in TRAITS},
            }
            rows.append(row)
        if persona_complete:
            complete_personas += 1
    fields = [
        "id_p", "id_s", "E_trait", "A_trait", "C_trait", "N_trait", "O_trait",
        *SITUATION_DIMS, "State_E", "State_A", "State_C", "State_N", "State_O",
    ]
    output_csv = output / "simulation_output_flattened-llama-3.1-8b-base.csv"
    write_csv_rows(output_csv, fields, rows)
    failures = pd.read_sql_query(
        "SELECT * FROM completions WHERE success=0 ORDER BY persona_name, task_type, situation_id, repeat_index",
        connection,
    )
    failures.to_csv(output / "generation_failures.csv", index=False, encoding="utf-8-sig")
    counts = pd.read_sql_query(
        "SELECT task_type, success, COUNT(*) AS n FROM completions GROUP BY task_type, success",
        connection,
    ).to_dict("records")
    summary = {
        "created_at": utc_now(),
        "output_csv": str(output_csv),
        "rows_exported": len(rows),
        "complete_personas": complete_personas,
        "expected_personas": int(cfg["data"]["expected_personas"]),
        "expected_situations": int(cfg["data"]["expected_situations"]),
        "completion_counts": counts,
        "note": "Missing/invalid fields are scored as the configured fail_score; failures remain auditable in SQLite and CSV.",
    }
    write_json(output / "behavior_run_summary.json", summary)
    return summary


def behavior_main() -> None:
    parser = argparse.ArgumentParser(description="Resumable Llama-3.1-8B Base behaviour generation")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--only", choices=("all", "big5", "situations"), default="all")
    parser.add_argument("--max-personas", type=int)
    parser.add_argument("--max-situations", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    inventory = validate_study_inputs(cfg)
    print("Validated study inputs:", inventory)
    if args.validate_only:
        return

    output = ensure_output_dirs(cfg)
    database = output / "behavior" / "behavior_responses.sqlite"
    connection = initialize_database(database)

    tokenizer, model, source = load_language_model(cfg)
    print(f"Loaded exact behaviour model from: {source}")

    personas = read_csv_robust(project_path(cfg, cfg["data"]["personas"]))
    id_col = persona_id_column(personas)
    if args.max_personas:
        personas = personas.head(args.max_personas)
    situations = read_json(project_path(cfg, cfg["data"]["situations"]))
    if args.max_situations:
        situations = dict(list(situations.items())[: args.max_situations])
    questions = load_big5_questions(project_path(cfg, cfg["data"]["big5_questions"]))
    state_questions = read_json(project_path(cfg, cfg["data"]["state_questions"]))
    big5_keys = [f"Q{i}" for i in range(1, len(questions) + 1)]
    state_keys = [key for _, key in state_item_spec(state_questions)]
    repeats = int(cfg["generation"]["repeats"])

    for _, persona in tqdm(personas.iterrows(), total=len(personas), desc="Personas"):
        name = str(persona[id_col])
        persona_text = str(persona["first_person"])
        if args.only in ("all", "big5"):
            prompt = build_big5_prompt(persona_text, questions)
            for repeat in range(1, repeats + 1):
                run_one_completion(
                    connection, tokenizer, model, cfg, name, "big5", repeat,
                    prompt, big5_keys,
                )
        if args.only in ("all", "situations"):
            for situation_id, situation in situations.items():
                description = situation["description"] if isinstance(situation, dict) else str(situation)
                prompt = build_state_prompt(persona_text, description, state_questions)
                for repeat in range(1, repeats + 1):
                    run_one_completion(
                        connection, tokenizer, model, cfg, name, "situation", repeat,
                        prompt, state_keys, str(situation_id),
                    )
    summary = export_behavior(cfg, connection)
    print(json.dumps(summary, ensure_ascii=False, indent=2))




# ===== modeling.py =====

from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList,
)



def _dtype(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The full Llama/OpenSAE pipeline requires an NVIDIA CUDA GPU.")


def resolve_source(cfg: dict[str, Any], section: str) -> str:
    local = project_path(cfg, cfg[section]["local_dir"])
    if local.is_dir() and (local / "config.json").exists():
        return str(local)
    return str(cfg[section]["repo_id"])


def load_language_model(cfg: dict[str, Any]):
    require_cuda()
    source = resolve_source(cfg, "model")
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True)
    if tokenizer.chat_template:
        raise RuntimeError("An instruction/chat tokenizer was loaded; the matched SAE requires Llama-3.1-8B Base.")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=_dtype(cfg["model"]["dtype"]),
        device_map=cfg["model"]["device_map"],
        low_cpu_mem_usage=True,
    )
    model.eval()
    expected = {
        "model_type": "llama",
        "hidden_size": int(cfg["model"]["expected_hidden_size"]),
        "num_hidden_layers": int(cfg["model"]["expected_layers"]),
        "max_position_embeddings": int(cfg["model"]["expected_context"]),
        "eos_token_id": int(cfg["model"]["expected_eos_token_id"]),
    }
    bad = [f"{k}={getattr(model.config, k, None)!r}, expected {v!r}" for k, v in expected.items()
           if getattr(model.config, k, None) != v]
    if bad:
        raise RuntimeError("Base-model identity check failed: " + "; ".join(bad))
    return tokenizer, model, source


def load_sae(cfg: dict[str, Any], model):
    from opensae import OpenSae

    source = resolve_source(cfg, "sae")
    layer = int(cfg["sae"]["layer"])
    device = next(model.model.layers[layer].parameters()).device
    sae = OpenSae.from_pretrained(source, torch_dtype=torch.float32).to(device)
    sae.eval()
    c = sae.config
    expected = {
        "model_name": "meta-llama/meta-llama-3.1-8b",
        "input_hookpoint": cfg["sae"]["expected_hook"],
        "hidden_size": int(cfg["sae"]["expected_d_in"]),
        "feature_size": int(cfg["sae"]["expected_d_sae"]),
        "k": int(cfg["sae"]["expected_k"]),
        "input_normalize": True,
    }
    bad = [f"{k}={getattr(c, k, None)!r}, expected {v!r}" for k, v in expected.items()
           if getattr(c, k, None) != v]
    if bad:
        raise RuntimeError("OpenSAE identity check failed: " + "; ".join(bad))
    identity = {k: getattr(c, k, None) for k in expected}
    identity["device"] = str(device)
    return sae, identity, source


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def plain_prompt(prompt: str) -> str:
    return "Research task:\n" + prompt.strip() + "\n\nResponse:\n"


def tokenize(tokenizer, model, prompt: str, max_input_tokens: int):
    return tokenizer(
        plain_prompt(prompt), return_tensors="pt", truncation=True,
        max_length=max_input_tokens, add_special_tokens=True,
    ).to(input_device(model))


@torch.inference_mode()
def generate_json_text(tokenizer, model, prompt, seed, max_new_tokens, temperature,
                       top_p, repetition_penalty, max_input_tokens):
    seed_everything(seed)
    inputs = tokenize(tokenizer, model, prompt, max_input_tokens)
    out = model.generate(
        **inputs, do_sample=True, temperature=temperature, top_p=top_p,
        repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def sae_encode(sae, hidden: torch.Tensor):
    out = sae.encode(hidden.reshape(-1, hidden.shape[-1]).float())
    return out


def sae_decode(sae, encoded):
    return sae.decode(
        encoded.sparse_feature_indices,
        encoded.sparse_feature_activations,
        input_mean=encoded.input_mean,
        input_std=encoded.input_std,
    ).sae_output


# ===== sae_tools.py =====

from contextlib import contextmanager
from typing import Callable

import numpy as np
import torch



def sae_prompt(persona: str, situation: str) -> str:
    return (
        "Persona description:\n" + persona.strip() +
        "\n\nSituation:\n" + situation.strip() +
        "\n\nDescribe in the first person how this person would feel and behave."
    )


class ActivationCapture:
    def __init__(self):
        self.value = None

    def __call__(self, _module, inputs):
        self.value = inputs[0].detach()


@contextmanager
def temporary_pre_hook(layer, hook: Callable):
    handle = layer.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def reconstruct_pre_hook(sae, mode: str):
    if mode not in {"reconstruct", "zero"}:
        raise ValueError(mode)

    def hook(_module, inputs):
        hidden = inputs[0]
        if mode == "zero":
            replacement = torch.zeros_like(hidden)
        else:
            encoded = sae_encode(sae, hidden)
            replacement = sae_decode(sae, encoded).reshape_as(hidden).to(hidden.dtype)
        return (replacement, *inputs[1:])
    return hook


@torch.inference_mode()
def encode_last_token(model, sae, input_ids, attention_mask, layer_index: int):
    capture = ActivationCapture()
    layer = model.model.layers[layer_index]
    with temporary_pre_hook(layer, capture):
        model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    if capture.value is None:
        raise RuntimeError("The layer-input hook captured no activation")
    last = attention_mask.sum(dim=1) - 1
    rows = torch.arange(input_ids.shape[0], device=input_ids.device)
    activation = capture.value[rows, last].to(next(sae.parameters()).device)
    encoded = sae_encode(sae, activation)
    reconstruction = sae_decode(sae, encoded)
    return activation, encoded, reconstruction


def topk_sparse(encoded) -> tuple[np.ndarray, np.ndarray]:
    return (
        encoded.sparse_feature_indices.detach().cpu().numpy().astype(np.int32),
        encoded.sparse_feature_activations.detach().cpu().numpy().astype(np.float32),
    )


def make_feature_intervention_hook(
    sae, feature_ids, feature_deltas, prompt_only=True, audit=None,
    scope="final", reference_feature_ids=None, reference_feature_deltas=None,
):
    device = next(sae.parameters()).device
    ids = feature_ids.to(device=device, dtype=torch.long)
    deltas = feature_deltas.to(device=device, dtype=torch.float32)
    ref_ids = None if reference_feature_ids is None else reference_feature_ids.to(device=device, dtype=torch.long)
    ref_deltas = None if reference_feature_deltas is None else reference_feature_deltas.to(device=device, dtype=torch.float32)

    def hook(_module, inputs):
        hidden = inputs[0]
        if prompt_only and hidden.shape[1] == 1:
            return None
        width = 1 if scope == "final" else 4
        width = min(width, hidden.shape[1])
        target_3d = hidden[:, -width:, :].to(device)
        target = target_3d.reshape(-1, target_3d.shape[-1])
        encoded = sae_encode(sae, target)
        baseline = sae_decode(sae, encoded)
        batch_ids = ids.unsqueeze(0).expand(target.shape[0], -1)
        requested_delta = deltas.unsqueeze(0).expand(target.shape[0], -1)
        matches = encoded.sparse_feature_indices.unsqueeze(-1) == batch_ids.unsqueeze(1)
        current = (
            encoded.sparse_feature_activations.unsqueeze(-1) * matches.to(encoded.sparse_feature_activations.dtype)
        ).sum(dim=1)
        changed_selected = torch.clamp(current + requested_delta, min=0.0)
        effective_delta = changed_selected - current
        # Duplicate indices are intentional: sparse decoding is linear, so appending
        # the effective delta implements set-to-clamped(current + delta).
        changed_ids = torch.cat([encoded.sparse_feature_indices, batch_ids], dim=-1)
        changed_values = torch.cat([encoded.sparse_feature_activations, effective_delta], dim=-1)
        changed = sae.decode(
            changed_ids, changed_values,
            input_mean=encoded.input_mean, input_std=encoded.input_std,
        ).sae_output
        residual_delta = changed - baseline
        norm_match_scale = torch.ones(target.shape[0], device=device)
        if ref_ids is not None and ref_deltas is not None:
            reference_ids = ref_ids.unsqueeze(0).expand(target.shape[0], -1)
            reference_requested = ref_deltas.unsqueeze(0).expand(target.shape[0], -1)
            reference_matches = encoded.sparse_feature_indices.unsqueeze(-1) == reference_ids.unsqueeze(1)
            reference_current = (
                encoded.sparse_feature_activations.unsqueeze(-1)
                * reference_matches.to(encoded.sparse_feature_activations.dtype)
            ).sum(dim=1)
            reference_effective = torch.clamp(
                reference_current + reference_requested, min=0.0
            ) - reference_current
            reference_changed = sae.decode(
                torch.cat([encoded.sparse_feature_indices, reference_ids], dim=-1),
                torch.cat([encoded.sparse_feature_activations, reference_effective], dim=-1),
                input_mean=encoded.input_mean, input_std=encoded.input_std,
            ).sae_output
            reference_norm = (reference_changed - baseline).norm(dim=1)
            control_norm = residual_delta.norm(dim=1)
            norm_match_scale = torch.where(
                control_norm > 1e-12, reference_norm / control_norm, torch.zeros_like(control_norm)
            )
            effective_delta = effective_delta * norm_match_scale.unsqueeze(1)
            changed = sae.decode(
                torch.cat([encoded.sparse_feature_indices, batch_ids], dim=-1),
                torch.cat([encoded.sparse_feature_activations, effective_delta], dim=-1),
                input_mean=encoded.input_mean, input_std=encoded.input_std,
            ).sae_output
            residual_delta = changed - baseline
        if audit is not None:
            batch = hidden.shape[0]
            def per_prompt(values):
                return values.reshape(batch, width).mean(1).detach().cpu().tolist()
            audit["selected_active_fraction"] = per_prompt((current > 0).float().mean(1))
            audit["requested_feature_delta_l2"] = per_prompt(requested_delta.norm(dim=1))
            audit["effective_feature_delta_l2"] = per_prompt(effective_delta.norm(dim=1))
            audit["residual_delta_l2"] = per_prompt(residual_delta.norm(dim=1))
            audit["norm_match_scale"] = per_prompt(norm_match_scale)
        modified = hidden.clone()
        modified[:, -width:, :] = (target_3d + residual_delta.reshape_as(target_3d)).to(hidden.dtype)
        return (modified, *inputs[1:])
    return hook


# ===== download_models.py =====

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download



def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_snapshot(repo_id: str, local_dir: Path, allow_patterns):
    if (local_dir / "config.json").exists() and any(local_dir.glob("*.safetensors")):
        return "existing-local-copy"
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, local_dir=local_dir, allow_patterns=allow_patterns)
    return "downloaded"


def download_main():
    parser = argparse.ArgumentParser(description="Prepare matched Llama-3.1-8B Base and layer-16 OpenSAE")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_dir = project_path(cfg, cfg["model"]["local_dir"])
    sae_dir = project_path(cfg, cfg["sae"]["local_dir"])
    model_status = ensure_snapshot(
        cfg["model"]["repo_id"], model_dir,
        ["*.json", "*.model", "*.safetensors", "*.txt", "LICENSE*"],
    )
    sae_status = ensure_snapshot(
        cfg["sae"]["repo_id"], sae_dir,
        ["config.json", "*.safetensors", "model.safetensors.index.json", "README.md"],
    )
    manifest = {
        "created_at": utc_now(), "model_repo": cfg["model"]["repo_id"],
        "model_local_dir": str(model_dir), "model_status": model_status,
        "sae_repo": cfg["sae"]["repo_id"], "sae_local_dir": str(sae_dir),
        "sae_status": sae_status,
        "config_hashes": {
            "model": sha256(model_dir / "config.json"),
            "sae": sha256(sae_dir / "config.json"),
        },
    }
    write_json(ensure_output_dirs(cfg) / "manifests" / "download_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))




# ===== sae_validate.py =====

import argparse
import json

import numpy as np
import pandas as pd
import torch



def bootstrap_mean_ci(values: np.ndarray, seed: int, repetitions: int = 2000) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(repetitions, len(values)), replace=True).mean(axis=1)
    return tuple(np.quantile(samples, [0.025, 0.975]).tolist())


@torch.inference_mode()
def validate_prompt(tokenizer, model, sae, cfg, prompt: str) -> dict[str, float]:
    inputs = tokenize(tokenizer, model, prompt, int(cfg["model"]["max_input_tokens"]))
    layer = model.model.layers[int(cfg["sae"]["layer"])]
    capture = ActivationCapture()
    with temporary_pre_hook(layer, capture):
        baseline_loss = float(model(**inputs, labels=inputs["input_ids"], use_cache=False).loss)
    if capture.value is None:
        raise RuntimeError("Failed to capture baseline activation")
    hidden = capture.value[:, 1:, :].to(next(sae.parameters()).device)
    encoded = sae_encode(sae, hidden)
    reconstructed = sae_decode(sae, encoded).reshape_as(hidden)
    error = reconstructed.float() - hidden.float()
    mse = float(error.square().mean())
    centered = hidden.float() - hidden.float().mean(dim=(0, 1), keepdim=True)
    fvu = float(error.square().sum() / centered.square().sum().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(hidden.float(), reconstructed.float(), dim=-1).mean())
    l0 = float((encoded.sparse_feature_activations > 0).sum(dim=-1).float().mean())
    with temporary_pre_hook(layer, reconstruct_pre_hook(sae, "reconstruct")):
        reconstructed_loss = float(model(**inputs, labels=inputs["input_ids"], use_cache=False).loss)
    with temporary_pre_hook(layer, reconstruct_pre_hook(sae, "zero")):
        zero_loss = float(model(**inputs, labels=inputs["input_ids"], use_cache=False).loss)
    denominator = zero_loss - baseline_loss
    ce_recovered = (
        1.0 - (reconstructed_loss - baseline_loss) / denominator
        if abs(denominator) > 1e-12
        else float("nan")
    )
    return {
        "tokens": int(inputs["input_ids"].shape[1]),
        "mse": mse,
        "fvu": fvu,
        "cosine": cosine,
        "l0": l0,
        "baseline_ce": baseline_loss,
        "reconstructed_ce": reconstructed_loss,
        "zero_ablation_ce": zero_loss,
        "ce_loss_recovered": ce_recovered,
    }


def validate_sae_main() -> None:
    parser = argparse.ArgumentParser(description="Validate matched Llama-3.1-8B Base/OpenSAE")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    personas = read_csv_robust(project_path(cfg, cfg["data"]["personas"]))
    situations = read_json(project_path(cfg, cfg["data"]["situations"]))
    id_col = persona_id_column(personas)
    print("Validated SAE inputs:", {"personas": len(personas), "situations": len(situations)})
    if args.validate_only:
        return
    tokenizer, model, source = load_language_model(cfg)
    sae, sae_identity, sae_source = load_sae(cfg, model)
    candidates = []
    base_seed = int(cfg["project"]["seed"])
    for _, persona in personas.iterrows():
        for situation_id, value in situations.items():
            description = value["description"] if isinstance(value, dict) else str(value)
            candidates.append(
                (
                    stable_seed(base_seed, "sae_validation", persona[id_col], situation_id),
                    str(persona[id_col]),
                    str(situation_id),
                    sae_prompt(str(persona["first_person"]), description),
                )
            )
    candidates.sort(key=lambda item: item[0])
    limit = args.max_prompts or int(cfg["sae"]["validation_prompts"])
    selected = candidates[:limit]
    rows = []
    for _, persona_name, situation_id, prompt in tqdm(selected, desc="SAE validation"):
        metrics = validate_prompt(tokenizer, model, sae, cfg, prompt)
        rows.append({"id_p": persona_name, "id_s": situation_id, **metrics})
    output = ensure_output_dirs(cfg) / "sae"
    table = pd.DataFrame(rows)
    table.to_csv(output / "sae_reconstruction_validation.csv", index=False, encoding="utf-8-sig")
    summary = {
        "created_at": utc_now(),
        "model_source": source,
        "model_repo_id": cfg["model"]["repo_id"],
        # Local OpenSAE checkpoints do not necessarily expose a `release`
        # field.  Record the configured repository and the checkpoint that was
        # actually loaded instead of requiring hub-release metadata.
        "sae_repo_id": cfg["sae"].get("repo_id"),
        "sae_release": cfg["sae"].get(
            "release", Path(cfg["sae"]["local_dir"]).name
        ),
        "sae_identity": sae_identity,
        "sae_source": sae_source,
        "activation_runtime": "Hugging Face layer-16 forward-pre-hook; official OpenSAE encode/decode",
        "n_prompts": len(table),
        "metrics": {},
    }
    for column in (
        "mse", "fvu", "cosine", "l0", "baseline_ce", "reconstructed_ce",
        "zero_ablation_ce", "ce_loss_recovered",
    ):
        values = table[column].to_numpy(dtype=float)
        low, high = bootstrap_mean_ci(values, stable_seed(base_seed, column))
        summary["metrics"][column] = {
            "mean": float(np.nanmean(values)),
            "bootstrap_ci_95": [low, high],
        }
    write_json(output / "sae_reconstruction_validation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))




# ===== sae_extract.py =====

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
import torch



FEATURE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sparse_features (
    id_p TEXT NOT NULL,
    id_s TEXT NOT NULL,
    split TEXT NOT NULL,
    indices BLOB NOT NULL,
    values_blob BLOB NOT NULL,
    k INTEGER NOT NULL,
    mse REAL NOT NULL,
    fvu REAL NOT NULL,
    cosine REAL NOT NULL,
    l0 INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (id_p, id_s)
);
CREATE INDEX IF NOT EXISTS idx_sparse_split ON sparse_features(split, id_p);
"""


def initialize_feature_database(path):
    connection = open_sqlite(path)
    connection.executescript(FEATURE_SCHEMA)
    connection.commit()
    return connection


@torch.inference_mode()
def extract_one(tokenizer, model, sae, cfg, prompt: str):
    inputs = tokenize(tokenizer, model, prompt, int(cfg["model"]["max_input_tokens"]))
    activation, encoded, reconstruction = encode_last_token(
        model, sae, inputs["input_ids"], inputs["attention_mask"], int(cfg["sae"]["layer"])
    )
    error = reconstruction.float() - activation.float()
    centered = activation.float() - activation.float().mean(dim=-1, keepdim=True)
    mse = float(error.square().mean())
    fvu = float(error.square().sum() / centered.square().sum().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(activation.float(), reconstruction.float(), dim=-1).mean())
    indices, values = topk_sparse(encoded)
    return indices[0], values[0], mse, fvu, cosine, int((values[0] > 0).sum())


def extract_main() -> None:
    parser = argparse.ArgumentParser(description="Extract resumable Llama/OpenSAE persona-situation features")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print("Validated extraction inputs:", validate_extraction_inputs(cfg))
    if args.validate_only:
        return
    output = ensure_output_dirs(cfg)
    personas = read_csv_robust(project_path(cfg, cfg["data"]["personas"]))
    id_col = persona_id_column(personas)
    personas = personas.set_index(id_col)
    situations = read_json(project_path(cfg, cfg["data"]["situations"]))
    tokenizer, model, source = load_language_model(cfg)
    sae, sae_identity, sae_source = load_sae(cfg, model)
    database_path = output / "sae" / "persona_situation_sparse_features.sqlite"
    connection = initialize_feature_database(database_path)
    existing = {
        (row[0], row[1]) for row in connection.execute("SELECT id_p, id_s FROM sparse_features")
    }
    records = [(str(p), str(s)) for p in personas.index for s in situations]
    if args.max_records:
        records = records[: args.max_records]
    base_seed = int(cfg["project"]["seed"])
    train_fraction = float(cfg["discovery"]["train_fraction"])
    for persona_name, situation_id in tqdm(records, desc="Sparse feature extraction"):
        persona_name, situation_id = str(persona_name), str(situation_id)
        if (persona_name, situation_id) in existing:
            continue
        if persona_name not in personas.index or situation_id not in situations:
            raise KeyError(f"Input mismatch: {persona_name!r}, {situation_id!r}")
        situation_value = situations[situation_id]
        description = situation_value["description"] if isinstance(situation_value, dict) else str(situation_value)
        prompt = sae_prompt(str(personas.loc[persona_name, "first_person"]), description)
        indices, values, mse, fvu, cosine, l0 = extract_one(tokenizer, model, sae, cfg, prompt)
        split = "discovery" if stable_unit_interval(base_seed, "persona_split", persona_name) < train_fraction else "heldout"
        connection.execute(
            """INSERT INTO sparse_features
            (id_p, id_s, split, indices, values_blob, k, mse, fvu, cosine, l0, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                persona_name, situation_id, split, indices.tobytes(), values.tobytes(),
                len(indices), mse, fvu, cosine, l0, utc_now(),
            ),
        )
        connection.commit()
    counts = [
        {"split": split, "n": n}
        for split, n in connection.execute("SELECT split, COUNT(*) FROM sparse_features GROUP BY split")
    ]
    summary = {
        "created_at": utc_now(),
        "database": str(database_path),
        "model_source": source,
        "sae_identity": sae_identity,
        "sae_source": sae_source,
        "activation_runtime": "Hugging Face layer-16 forward-pre-hook; official OpenSAE encode/decode",
        "counts": counts,
        "array_format": {"indices": "int32", "values_blob": "float32", "length": int(cfg["sae"]["expected_k"])},
    }
    write_json(output / "sae" / "feature_extraction_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))




# ===== sae_discover.py =====

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
from scipy import stats



def decode_sparse(indices_blob: bytes, values_blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    return np.frombuffer(indices_blob, dtype=np.int32), np.frombuffer(values_blob, dtype=np.float32)


def finalize_persona(
    activation_sum: np.ndarray,
    n_situations: int,
    targets: np.ndarray,
    accumulators: dict[str, np.ndarray],
) -> None:
    if n_situations == 0:
        return
    x = activation_sum / float(n_situations)
    active = np.flatnonzero(x > 0)
    values = x[active].astype(np.float64)
    accumulators["sum_x"][active] += values
    accumulators["sum_x2"][active] += values * values
    accumulators["presence"][active] += 1
    accumulators["sum_xy"][:, active] += targets[:, None] * values[None, :]
    accumulators["sum_y"] += targets
    accumulators["sum_y2"] += targets * targets
    accumulators["n"][0] += 1


def discover_main() -> None:
    parser = argparse.ArgumentParser(description="Discover persona-associated SAE features on discovery personas only")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ensure_output_dirs(cfg)
    feature_path = output / "sae" / "persona_situation_sparse_features.sqlite"
    if not feature_path.exists():
        raise FileNotFoundError("Run SAE extraction before discovery")
    personas = read_csv_robust(project_path(cfg, cfg["data"]["personas"]))
    id_col = persona_id_column(personas)
    trait_table = personas[[id_col, *[f"{trait}_trait" for trait in TRAITS]]].drop_duplicates(id_col).set_index(id_col)
    d_sae = int(cfg["sae"]["expected_d_sae"])
    accumulators = {
        "sum_x": np.zeros(d_sae, dtype=np.float64),
        "sum_x2": np.zeros(d_sae, dtype=np.float64),
        "presence": np.zeros(d_sae, dtype=np.int32),
        "sum_xy": np.zeros((len(TRAITS), d_sae), dtype=np.float64),
        "sum_y": np.zeros(len(TRAITS), dtype=np.float64),
        "sum_y2": np.zeros(len(TRAITS), dtype=np.float64),
        "n": np.zeros(1, dtype=np.int64),
    }
    connection = sqlite3.connect(feature_path)
    cursor = connection.execute(
        "SELECT id_p, indices, values_blob FROM sparse_features WHERE split='discovery' ORDER BY id_p, id_s"
    )
    current_persona = None
    activation_sum = np.zeros(d_sae, dtype=np.float32)
    n_situations = 0
    targets = None
    for persona, indices_blob, values_blob in tqdm(cursor, desc="Discovery-persona aggregation"):
        if persona != current_persona:
            if current_persona is not None and targets is not None:
                finalize_persona(activation_sum, n_situations, targets, accumulators)
            current_persona = persona
            activation_sum.fill(0)
            n_situations = 0
            if persona not in trait_table.index:
                raise KeyError(f"No behaviour scores for {persona}")
            targets = trait_table.loc[persona, [f"{trait}_trait" for trait in TRAITS]].to_numpy(dtype=np.float64)
        indices, values = decode_sparse(indices_blob, values_blob)
        positive = values > 0
        np.add.at(activation_sum, indices[positive], values[positive])
        n_situations += 1
    if current_persona is not None and targets is not None:
        finalize_persona(activation_sum, n_situations, targets, accumulators)
    connection.close()

    n = int(accumulators["n"][0])
    if n < 20:
        raise RuntimeError(f"Only {n} discovery personas; feature statistics would be unstable")
    sum_x, sum_x2 = accumulators["sum_x"], accumulators["sum_x2"]
    numerator = n * accumulators["sum_xy"] - accumulators["sum_y"][:, None] * sum_x[None, :]
    x_ss = n * sum_x2 - sum_x * sum_x
    y_ss = n * accumulators["sum_y2"] - accumulators["sum_y"] ** 2
    denominator = np.sqrt(np.maximum(y_ss[:, None] * x_ss[None, :], 0.0))
    correlations = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    correlations = np.clip(correlations, -1, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_values = correlations * np.sqrt((n - 2) / np.maximum(1 - correlations**2, 1e-15))
    p_values = 2 * stats.t.sf(np.abs(t_values), df=n - 2)
    activation_rate = accumulators["presence"] / float(n)
    means = sum_x / float(n)
    variances = np.maximum(sum_x2 / float(n) - means**2, 0.0)
    stds = np.sqrt(variances)
    q_values = np.full_like(p_values, np.nan)
    minimum_rate = float(cfg["discovery"]["minimum_activation_rate"])
    eligible = (activation_rate >= minimum_rate) & (stds > 0)
    selected_rows = []
    feature_limit = int(cfg["discovery"]["features_per_trait"])
    fdr_alpha = float(cfg["discovery"]["fdr_alpha"])
    for trait_index, trait in enumerate(TRAITS):
        valid = eligible & np.isfinite(p_values[trait_index])
        if valid.any():
            q_values[trait_index, valid] = benjamini_hochberg(p_values[trait_index, valid])
        candidates = np.flatnonzero(valid & (q_values[trait_index] <= fdr_alpha))
        if len(candidates) < feature_limit:
            candidates = np.flatnonzero(valid)
        order = candidates[np.argsort(-np.abs(correlations[trait_index, candidates]))][:feature_limit]
        for rank, feature_id in enumerate(order, 1):
            correlation = float(correlations[trait_index, feature_id])
            selected_rows.append(
                {
                    "trait": trait,
                    "rank": rank,
                    "feature_id": int(feature_id),
                    "correlation": correlation,
                    "direction_sign": 1 if correlation >= 0 else -1,
                    "p_value": float(p_values[trait_index, feature_id]),
                    "q_value": float(q_values[trait_index, feature_id]),
                    "activation_rate": float(activation_rate[feature_id]),
                    "mean_activation": float(means[feature_id]),
                    "sd_activation": float(stds[feature_id]),
                }
            )
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(output / "sae" / "selected_persona_features.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output / "sae" / "feature_population_statistics.npz",
        traits=np.array(TRAITS),
        correlations=correlations.astype(np.float32),
        p_values=p_values.astype(np.float32),
        q_values=q_values.astype(np.float32),
        activation_rate=activation_rate.astype(np.float32),
        mean_activation=means.astype(np.float32),
        sd_activation=stds.astype(np.float32),
        n_personas=np.array([n], dtype=np.int32),
    )
    summary = {
        "created_at": utc_now(),
        "discovery_personas": n,
        "split_unit": "persona",
        "aggregation": "mean SAE activation across all situations within persona",
        "multiple_testing": "Benjamini-Hochberg FDR within trait",
        "fdr_alpha": fdr_alpha,
        "minimum_activation_rate": minimum_rate,
        "selected_features": selected.groupby("trait").size().to_dict(),
    }
    write_json(output / "sae" / "feature_discovery_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))




# ===== sae_intervene.py =====

import argparse
import json
import sqlite3

import numpy as np
import pandas as pd
import torch



INTERVENTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS intervention_responses (
    task_key TEXT PRIMARY KEY,
    id_p TEXT NOT NULL, id_s TEXT NOT NULL,
    target_trait TEXT NOT NULL, condition TEXT NOT NULL,
    scale REAL NOT NULL, repeat_index INTEGER NOT NULL, seed INTEGER NOT NULL,
    feature_ids TEXT NOT NULL, raw_text TEXT NOT NULL,
    parsed_json TEXT NOT NULL, state_scores TEXT NOT NULL,
    success INTEGER NOT NULL, error TEXT, created_at TEXT NOT NULL,
    valid_fraction REAL NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_seconds REAL NOT NULL DEFAULT 0,
    word_count INTEGER NOT NULL DEFAULT 0,
    repetition_ratio REAL NOT NULL DEFAULT 0,
    shard_index INTEGER NOT NULL DEFAULT 0,
    feature_deltas TEXT NOT NULL DEFAULT '[]',
    selected_active_fraction REAL NOT NULL DEFAULT 0,
    requested_feature_delta_l2 REAL NOT NULL DEFAULT 0,
    effective_feature_delta_l2 REAL NOT NULL DEFAULT 0,
    residual_delta_l2 REAL NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_intervention_design
ON intervention_responses(id_p,id_s,target_trait,condition,scale,repeat_index);
"""


def initialize_intervention_database(path):
    connection = open_sqlite(path)
    connection.executescript(INTERVENTION_SCHEMA)
    existing = {row[1] for row in connection.execute("PRAGMA table_info(intervention_responses)")}
    migrations = {
        "valid_fraction": "REAL NOT NULL DEFAULT 0",
        "prompt_tokens": "INTEGER NOT NULL DEFAULT 0",
        "completion_tokens": "INTEGER NOT NULL DEFAULT 0",
        "latency_seconds": "REAL NOT NULL DEFAULT 0",
        "word_count": "INTEGER NOT NULL DEFAULT 0",
        "repetition_ratio": "REAL NOT NULL DEFAULT 0",
        "shard_index": "INTEGER NOT NULL DEFAULT 0",
        "feature_deltas": "TEXT NOT NULL DEFAULT '[]'",
        "selected_active_fraction": "REAL NOT NULL DEFAULT 0",
        "requested_feature_delta_l2": "REAL NOT NULL DEFAULT 0",
        "effective_feature_delta_l2": "REAL NOT NULL DEFAULT 0",
        "residual_delta_l2": "REAL NOT NULL DEFAULT 0",
    }
    for column, definition in migrations.items():
        if column not in existing:
            connection.execute(
                f"ALTER TABLE intervention_responses ADD COLUMN {column} {definition}"
            )
    connection.commit()
    return connection


def choose_matched_controls(
    selected: pd.DataFrame, population: dict[str, np.ndarray], seed: int
) -> dict[str, list[int]]:
    rates = population["activation_rate"]
    stds = population["sd_activation"]
    all_selected = set(selected["feature_id"].astype(int))
    eligible = np.flatnonzero((rates >= 0.01) & (stds > 0))
    result: dict[str, list[int]] = {}
    for trait in TRAITS:
        trait_rows = selected[selected["trait"] == trait].sort_values("rank")
        controls = []
        for _, row in trait_rows.iterrows():
            feature_id = int(row["feature_id"])
            distance = np.abs(rates[eligible] - rates[feature_id]) + np.abs(
                np.log1p(stds[eligible]) - np.log1p(stds[feature_id])
            )
            order = eligible[np.argsort(distance)]
            pool = [int(x) for x in order[:500] if int(x) not in all_selected and int(x) not in controls]
            if not pool:
                raise RuntimeError("Could not construct matched random-feature controls")
            rng = np.random.default_rng(stable_seed(seed, "control", trait, feature_id))
            controls.append(int(rng.choice(pool)))
        result[trait] = controls
    return result


def intervention_spec(
    trait: str,
    condition: str,
    scale: float,
    selected: pd.DataFrame,
    controls: dict[str, list[int]],
    population: dict[str, np.ndarray],
    seed: int,
) -> tuple[list[int], np.ndarray]:
    trait_rows = selected[selected["trait"] == trait].sort_values("rank")
    if condition in {"positive", "negative"}:
        ids = trait_rows["feature_id"].astype(int).tolist()
        signs = trait_rows["direction_sign"].to_numpy(dtype=float)
        if condition == "negative":
            signs *= -1
    elif condition in {"positive_control", "negative_control"}:
        ids = controls[trait]
        # Match the target feature directions as well as activation rate and SD.
        signs = trait_rows["direction_sign"].to_numpy(dtype=float)
        if condition == "negative_control":
            signs *= -1
    else:
        return [], np.array([], dtype=np.float32)
    magnitudes = population["sd_activation"][np.asarray(ids, dtype=int)]
    deltas = signs * float(scale) * magnitudes
    return ids, deltas.astype(np.float32)


def _repetition_ratio(text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    return float(1.0 - len(set(words)) / len(words))


class CompleteJsonBatchStoppingCriteria(StoppingCriteria):
    """Stop once every sequence contains a sufficiently complete JSON object."""
    def __init__(self, tokenizer, prompt_width: int, expected_keys: list[str], check_every: int = 8):
        self.tokenizer = tokenizer
        self.prompt_width = prompt_width
        self.expected_keys = expected_keys
        self.check_every = check_every

    def __call__(self, input_ids, scores, **kwargs):
        generated = input_ids.shape[1] - self.prompt_width
        if generated < 24 or generated % self.check_every:
            return False
        texts = self.tokenizer.batch_decode(
            input_ids[:, self.prompt_width:], skip_special_tokens=True
        )
        return all(
            valid_fraction(
                normalize_state_json(parse_json_object(text)),
                self.expected_keys, "situation"
            ) >= 0.90
            for text in texts
        )


@torch.inference_mode()
def generate_json_text_batch(tokenizer, model, prompts, seed, max_new_tokens,
                             temperature, top_p, repetition_penalty, max_input_tokens,
                             expected_keys, greedy=False):
    """Generate a fixed prompt batch; paired conditions reuse the same seed."""
    seed_everything(seed)
    tokenizer.padding_side = "left"
    encoded = tokenizer(
        [plain_prompt(prompt) for prompt in prompts], return_tensors="pt",
        padding=True, truncation=True, max_length=max_input_tokens,
        add_special_tokens=True,
    ).to(input_device(model))
    prompt_lengths = encoded["attention_mask"].sum(dim=1).detach().cpu().numpy()
    started = time.perf_counter()
    stopping = CompleteJsonBatchStoppingCriteria(
        tokenizer, encoded["input_ids"].shape[1], expected_keys
    )
    sampling = ({"do_sample": False} if greedy else {
        "do_sample": True, "temperature": temperature, "top_p": top_p
    })
    generated = model.generate(
        **encoded, **sampling, repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        use_cache=True, stopping_criteria=StoppingCriteriaList([stopping]),
    )
    latency = time.perf_counter() - started
    prefix = encoded["input_ids"].shape[1]
    texts, completion_lengths = [], []
    for row in range(len(prompts)):
        tokens = generated[row, prefix:]
        if tokenizer.eos_token_id is not None:
            eos = (tokens == tokenizer.eos_token_id).nonzero(as_tuple=False)
            if len(eos):
                tokens = tokens[: int(eos[0, 0])]
        completion_lengths.append(int(len(tokens)))
        texts.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
    return texts, prompt_lengths, np.asarray(completion_lengths), latency


def _task_key(unit, trait, condition, scale, repeat):
    return f"v7|{unit['design_id']}|{unit['id_p']}|{unit['id_s']}|{trait}|{condition}|{scale:g}|{repeat}"


def run_generation_batch(connection, tokenizer, model, sae, cfg, units,
                         expected_keys, trait, condition, scale, repeat,
                         feature_ids, deltas, state_questions, paired_seed,
                         shard_index, reference_feature_ids=None, reference_deltas=None):
    keys = [_task_key(unit, trait, condition, scale, repeat) for unit in units]
    existing = {
        row[0] for row in connection.execute(
            f"SELECT task_key FROM intervention_responses WHERE task_key IN ({','.join('?' for _ in keys)})",
            keys,
        )
    }
    if len(existing) == len(keys):
        return 0
    prompts = [unit["prompt"] for unit in units]
    context = None
    audit = {}
    if condition != "baseline":
        hook = make_feature_intervention_hook(
            sae, torch.as_tensor(feature_ids), torch.as_tensor(deltas),
            prompt_only=bool(cfg["intervention"]["prompt_only"]), audit=audit,
            scope=str(cfg["intervention"].get("intervention_scope", "final")),
            reference_feature_ids=(None if reference_feature_ids is None else torch.as_tensor(reference_feature_ids)),
            reference_feature_deltas=(None if reference_deltas is None else torch.as_tensor(reference_deltas)),
        )
        context = temporary_pre_hook(model.model.layers[int(cfg["sae"]["layer"])], hook)
    try:
        if context is None:
            texts, prompt_tokens, completion_tokens, latency = generate_json_text_batch(
                tokenizer, model, prompts, paired_seed,
                int(cfg["intervention"]["max_new_tokens"]),
                float(cfg["generation"]["temperature"]),
                float(cfg["generation"]["top_p"]),
                float(cfg["generation"]["repetition_penalty"]),
                int(cfg["model"]["max_input_tokens"]),
                expected_keys,
                bool(cfg["intervention"].get("greedy", False)),
            )
        else:
            with context:
                texts, prompt_tokens, completion_tokens, latency = generate_json_text_batch(
                    tokenizer, model, prompts, paired_seed,
                    int(cfg["intervention"]["max_new_tokens"]),
                    float(cfg["generation"]["temperature"]),
                    float(cfg["generation"]["top_p"]),
                    float(cfg["generation"]["repetition_penalty"]),
                    int(cfg["model"]["max_input_tokens"]),
                    expected_keys,
                    bool(cfg["intervention"].get("greedy", False)),
                )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(units) == 1:
            raise
        midpoint = len(units) // 2
        return run_generation_batch(
            connection, tokenizer, model, sae, cfg, units[:midpoint], expected_keys,
            trait, condition, scale, repeat, feature_ids, deltas, state_questions,
            paired_seed, shard_index, reference_feature_ids, reference_deltas,
        ) + run_generation_batch(
            connection, tokenizer, model, sae, cfg, units[midpoint:], expected_keys,
            trait, condition, scale, repeat, feature_ids, deltas, state_questions,
            paired_seed, shard_index, reference_feature_ids, reference_deltas,
        )
    rows = []
    per_item_latency = float(latency / max(len(units), 1))
    for index, (unit, task_key, raw) in enumerate(zip(units, keys, texts)):
        if task_key in existing:
            continue
        parsed = normalize_state_json(parse_json_object(raw))
        fraction = valid_fraction(parsed, expected_keys, "situation")
        success = fraction >= 0.90
        error = None if success else f"only {fraction:.1%} valid fields"
        scores = (
            score_state_repeat(parsed, state_questions, float(cfg["project"]["fail_score"]))
            if success else {trait_name: None for trait_name in TRAITS}
        )
        def audit_value(name):
            values = audit.get(name, [])
            return float(values[index]) if index < len(values) else 0.0
        rows.append((
            task_key, unit["id_p"], unit["id_s"], trait, condition,
            float(scale), repeat, int(paired_seed), json.dumps(feature_ids), raw,
            json.dumps(parsed, ensure_ascii=False), json.dumps(scores), int(success),
            error, utc_now(), float(fraction), int(prompt_tokens[index]),
            int(completion_tokens[index]), per_item_latency,
            len(re.findall(r"\w+", raw)), _repetition_ratio(raw), int(shard_index),
            json.dumps(np.asarray(deltas, dtype=float).tolist()),
            audit_value("selected_active_fraction"),
            audit_value("requested_feature_delta_l2"),
            audit_value("effective_feature_delta_l2"),
            audit_value("residual_delta_l2"),
        ))
    connection.executemany(
        """INSERT OR REPLACE INTO intervention_responses
        (task_key,id_p,id_s,target_trait,condition,scale,repeat_index,seed,
         feature_ids,raw_text,parsed_json,state_scores,success,error,created_at,
         valid_fraction,prompt_tokens,completion_tokens,latency_seconds,
         word_count,repetition_ratio,shard_index,feature_deltas,
         selected_active_fraction,requested_feature_delta_l2,
         effective_feature_delta_l2,residual_delta_l2)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows,
    )
    connection.commit()
    return len(rows)


def export_results(connection, output, filename="intervention_results_long.csv"):
    rows = []
    for row in connection.execute(
        """SELECT id_p,id_s,target_trait,condition,scale,repeat_index,state_scores,
        success,error,valid_fraction,prompt_tokens,completion_tokens,latency_seconds,
        word_count,repetition_ratio,shard_index,feature_ids,feature_deltas,
        selected_active_fraction,requested_feature_delta_l2,
        effective_feature_delta_l2,residual_delta_l2 FROM intervention_responses"""
    ):
        (id_p, id_s, trait, condition, scale, repeat, scores_json, success, error,
         fraction, prompt_tokens, completion_tokens, latency, word_count,
         repetition_ratio, shard_index, feature_ids, feature_deltas,
         active_fraction, requested_l2, effective_l2, residual_l2) = row
        scores = json.loads(scores_json)
        rows.append(
            {
                "id_p": id_p, "id_s": id_s, "target_trait": trait,
                "condition": condition, "scale": scale, "repeat_index": repeat,
                **{f"State_{key}": value for key, value in scores.items()},
                "success": success, "error": error, "valid_fraction": fraction,
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "latency_seconds": latency, "word_count": word_count,
                "repetition_ratio": repetition_ratio, "shard_index": shard_index,
                "feature_ids": feature_ids, "feature_deltas": feature_deltas,
                "selected_active_fraction": active_fraction,
                "requested_feature_delta_l2": requested_l2,
                "effective_feature_delta_l2": effective_l2,
                "residual_delta_l2": residual_l2,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output / filename, index=False, encoding="utf-8-sig")
    return table


def intervene_main() -> None:
    parser = argparse.ArgumentParser(description="Held-out causal Llama/OpenSAE interventions")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--max-personas", type=int)
    parser.add_argument("--max-situations", type=int)
    parser.add_argument("--mode", choices=("smoke", "pilot", "technical", "confirmatory", "full"), default="full")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-traits", type=int)
    parser.add_argument("--max-features-per-trait", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--greedy", action="store_true",
                        help="Use deterministic greedy decoding for causal calibration")
    parser.add_argument("--intervention-scope", choices=("final", "last4"), default="final")
    parser.add_argument("--scales", nargs="+", type=float,
                        help="Override intervention doses, e.g. --scales 2 4 8")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["intervention"]["greedy"] = bool(args.greedy)
    cfg["intervention"]["intervention_scope"] = args.intervention_scope
    output = ensure_output_dirs(cfg)
    intervention_output = output / "intervention"
    if args.merge_only:
        files = sorted(intervention_output.glob("intervention_results_long_v5_full_sampled_final_shard*.csv"))
        if not files:
            raise FileNotFoundError("No shard CSV files to merge")
        merged = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
        merged = merged.drop_duplicates(
            ["id_p", "id_s", "target_trait", "condition", "scale", "repeat_index"],
            keep="last",
        )
        merged.to_csv(intervention_output / "intervention_results_long.csv", index=False,
                      encoding="utf-8-sig")
        print(f"Merged {len(files)} shards into {len(merged):,} rows")
        return
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require 0 <= shard-index < num-shards")
    if args.max_new_tokens:
        cfg["intervention"]["max_new_tokens"] = args.max_new_tokens
    selected_path = output / "sae" / "selected_persona_features.csv"
    population_path = output / "sae" / "feature_population_statistics.npz"
    feature_db = output / "sae" / "persona_situation_sparse_features.sqlite"
    for path in (selected_path, population_path, feature_db):
        if not path.exists():
            raise FileNotFoundError(f"Missing prerequisite: {path}")
    selected = pd.read_csv(selected_path)
    if args.max_features_per_trait:
        selected = selected.sort_values(["trait", "rank"]).groupby("trait").head(
            args.max_features_per_trait
        )
    population_npz = np.load(population_path)
    population = {key: population_npz[key] for key in population_npz.files}
    base_seed = int(cfg["project"]["seed"])
    feature_connection = sqlite3.connect(feature_db)
    heldout_personas = [
        row[0] for row in feature_connection.execute(
            "SELECT DISTINCT id_p FROM sparse_features WHERE split='heldout' ORDER BY id_p"
        )
    ]
    heldout_personas.sort(key=lambda name: stable_seed(base_seed, "intervention_persona", name))
    calibration_count = int(cfg["intervention"].get("calibration_personas", 10))
    calibration_persona_ids = heldout_personas[:calibration_count]
    if args.mode in {"full", "confirmatory"}:
        heldout_personas = heldout_personas[calibration_count:]
    else:
        heldout_personas = calibration_persona_ids
    if args.mode == "smoke":
        args.max_personas, args.max_situations = 1, 1
        args.max_traits, args.max_features_per_trait = 1, 1
        selected = selected.sort_values(["trait", "rank"]).groupby("trait").head(1)
        cfg["intervention"]["repeats"] = 1
        cfg["intervention"]["scales"] = [1.0]
        cfg["intervention"]["max_new_tokens"] = min(
            int(cfg["intervention"]["max_new_tokens"]), 384
        )
    elif args.mode == "pilot":
        args.max_personas = args.max_personas or 10
        args.max_situations = args.max_situations or 3
        cfg["intervention"]["repeats"] = 1
        cfg["intervention"]["scales"] = [0.5, 1.0, 2.0]
    elif args.mode == "technical":
        args.max_personas = args.max_personas or 5
        args.max_situations = args.max_situations or 2
        cfg["intervention"]["repeats"] = 1
        cfg["intervention"]["scales"] = [2.0, 4.0, 8.0]
        cfg["intervention"]["greedy"] = True
    elif args.mode == "confirmatory":
        # Frozen after the v6 technical gate: do not tune these values on results.
        if args.max_personas not in {None, 100} or args.max_situations not in {None, 12}:
            raise ValueError("Confirmatory mode is frozen at 100 personas and 12 situations")
        if args.max_traits is not None or args.max_features_per_trait is not None:
            raise ValueError("Confirmatory mode requires all 5 traits and all 20 selected features")
        if args.scales is not None:
            raise ValueError("Confirmatory mode is frozen at the validated 8-SD dose")
        args.max_personas = 100
        args.max_situations = 12
        cfg["intervention"]["repeats"] = 1
        cfg["intervention"]["scales"] = [8.0]
        cfg["intervention"]["greedy"] = True
        cfg["intervention"]["intervention_scope"] = "last4"
    if args.scales and args.mode != "confirmatory":
        if any(value <= 0 for value in args.scales):
            raise ValueError("All intervention scales must be positive")
        cfg["intervention"]["scales"] = [float(value) for value in args.scales]
    controls = choose_matched_controls(selected, population, base_seed)
    persona_limit = args.max_personas or int(cfg["intervention"]["heldout_personas"])
    heldout_personas = heldout_personas[:persona_limit]
    heldout_personas = [
        value for index, value in enumerate(heldout_personas)
        if index % args.num_shards == args.shard_index
    ]
    personas = read_csv_robust(project_path(cfg, cfg["data"]["personas"]))
    id_col = persona_id_column(personas)
    personas = personas.set_index(id_col)
    situations = read_json(project_path(cfg, cfg["data"]["situations"]))
    situation_ids = list(situations)
    situation_ids.sort(key=lambda sid: stable_seed(base_seed, "intervention_situation", sid))
    situation_limit = args.max_situations or int(cfg["intervention"]["situations"])
    situation_ids = situation_ids[:situation_limit]
    state_questions = read_json(project_path(cfg, cfg["data"]["state_questions"]))
    expected_keys = [key for _, key in state_item_spec(state_questions)]
    tokenizer, model, source = load_language_model(cfg)
    sae, sae_identity, sae_source = load_sae(cfg, model)
    suffix = f"_shard{args.shard_index:02d}" if args.num_shards > 1 else ""
    decoder_tag = "greedy" if cfg["intervention"]["greedy"] else "sampled"
    scope_tag = str(cfg["intervention"]["intervention_scope"])
    scale_tag = "-".join(f"{value:g}" for value in cfg["intervention"]["scales"])
    database_path = intervention_output / f"intervention_responses_v7_{args.mode}_{decoder_tag}_{scope_tag}_doses-{scale_tag}{suffix}.sqlite"
    connection = initialize_intervention_database(database_path)
    repeats = int(cfg["intervention"]["repeats"])
    scales = [float(value) for value in cfg["intervention"]["scales"]]
    traits = list(TRAITS[: args.max_traits]) if args.max_traits else list(TRAITS)
    design_payload = {
        "pipeline_version": 7, "prompt_version": 3, "mode": args.mode, "traits": traits,
        "frozen_confirmatory_contract": bool(args.mode == "confirmatory"),
        "greedy": cfg["intervention"]["greedy"],
        "intervention_scope": cfg["intervention"]["intervention_scope"],
        "scales": scales, "repeats": repeats,
        "selected": selected[["trait", "feature_id", "direction_sign"]].to_dict("records"),
        "controls": controls,
    }
    design_id = hashlib.sha256(
        json.dumps(design_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    units = []
    for id_p in heldout_personas:
        persona_text = str(personas.loc[id_p, "first_person"])
        for id_s in situation_ids:
            value = situations[id_s]
            description = value["description"] if isinstance(value, dict) else str(value)
            prompt = build_state_prompt(persona_text, description, state_questions)
            units.append({
                "id_p": str(id_p), "id_s": str(id_s), "prompt": prompt,
                "design_id": design_id,
            })
    batch_size = args.batch_size or int(cfg["intervention"]["batch_size"])
    batches = [units[index:index + batch_size] for index in range(0, len(units), batch_size)]
    conditions_per_unit = repeats * (1 + len(traits) * len(scales) * 4)
    total = len(units) * conditions_per_unit
    completed = int(connection.execute(
        "SELECT COUNT(*) FROM intervention_responses WHERE task_key LIKE ?",
        (f"v7|{design_id}|%",),
    ).fetchone()[0])
    bar = tqdm(total=total, initial=min(completed, total), desc=f"Interventions shard {args.shard_index}", unit="gen")
    for batch_index, batch in enumerate(batches):
        for repeat in range(1, repeats + 1):
            paired_seed = stable_seed(base_seed, "paired", args.shard_index, batch_index, repeat)
            inserted = run_generation_batch(
                connection, tokenizer, model, sae, cfg, batch, expected_keys,
                "all", "baseline", 0.0, repeat, [], np.array([]), state_questions,
                paired_seed, args.shard_index,
            )
            bar.update(inserted)
            for trait in traits:
                for scale in scales:
                    for condition in ("positive", "negative", "positive_control", "negative_control"):
                        ids, deltas = intervention_spec(
                            trait, condition, scale, selected, controls, population, base_seed
                        )
                        reference_ids = reference_deltas = None
                        if condition in {"positive_control", "negative_control"}:
                            reference_condition = "positive" if condition == "positive_control" else "negative"
                            reference_ids, reference_deltas = intervention_spec(
                                trait, reference_condition, scale, selected, controls, population, base_seed
                            )
                        inserted = run_generation_batch(
                            connection, tokenizer, model, sae, cfg, batch, expected_keys,
                            trait, condition, scale, repeat, ids, deltas, state_questions,
                            paired_seed, args.shard_index, reference_ids, reference_deltas,
                        )
                        bar.update(inserted)
    bar.close()
    if args.num_shards > 1:
        result_filename = f"intervention_results_long_v7_{args.mode}_{decoder_tag}_{scope_tag}_doses-{scale_tag}_shard{args.shard_index:02d}.csv"
    elif args.mode == "full":
        result_filename = "intervention_results_long.csv"
    else:
        result_filename = f"intervention_results_long_v7_{args.mode}_{decoder_tag}_{scope_tag}_doses-{scale_tag}.csv"
    table = export_results(connection, intervention_output, result_filename)
    manifest = {
        "created_at": utc_now(),
        "model_source": source,
        "sae_identity": sae_identity,
        "sae_source": sae_source,
        "split": "heldout personas only",
        "calibration_policy": (
            "first seeded held-out calibration personas; excluded from full"
            if args.mode != "full" else
            f"excluded {calibration_count} calibration personas before confirmatory sampling"
        ),
        "persona_ids": heldout_personas,
        "situation_ids": situation_ids,
        "matched_random_features": controls,
        "conditions": ["baseline", "positive", "negative", "positive_control", "negative_control"],
        "scales": scales,
        "rows": len(table),
        "mode": args.mode, "batch_size": batch_size,
        "decoding": decoder_tag, "intervention_scope": scope_tag,
        "design_id": design_id,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "paired_sampling_seed": "shared within batch and repeat across baseline, target and controls",
        "intervention_rule": "Add decoded SAE-feature difference to the original residual stream at the final prompt token; preserve reconstruction residual.",
        "activation_runtime": "Hugging Face layer-16 forward-pre-hook; official OpenSAE encode/decode",
    }
    manifest_name = (
        f"intervention_manifest_v7_{args.mode}_{decoder_tag}_{scope_tag}_doses-{scale_tag}_shard{args.shard_index:02d}.json"
        if args.num_shards > 1 else f"intervention_manifest_v7_{args.mode}_{decoder_tag}_{scope_tag}_doses-{scale_tag}.json"
    )
    write_json(intervention_output / manifest_name, manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key not in {"persona_ids", "matched_random_features"}}, indent=2))




# ===== intervention_analyze.py =====

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon



def cluster_bootstrap_mean(data: pd.DataFrame, value: str, cluster: str, seed: int, repetitions: int = 2000):
    clusters = data[cluster].drop_duplicates().to_numpy()
    if len(clusters) < 2:
        return float("nan"), float("nan")
    grouped = {key: group[value].to_numpy(dtype=float) for key, group in data.groupby(cluster)}
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        estimates[index] = np.mean(np.concatenate([grouped[key] for key in sampled]))
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def analyze_main() -> None:
    parser = argparse.ArgumentParser(description="Summarize held-out SAE interventions")
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ensure_output_dirs(cfg) / "intervention"
    source = output / "intervention_results_long.csv"
    if not source.exists():
        shard_files = sorted(output.glob("intervention_results_long_v5_full_sampled_final_shard*.csv"))
        if not shard_files:
            raise FileNotFoundError(f"Missing {source} and shard CSVs")
        combined = pd.concat([pd.read_csv(path) for path in shard_files], ignore_index=True)
        combined = combined.drop_duplicates(
            ["id_p", "id_s", "target_trait", "condition", "scale", "repeat_index"],
            keep="last",
        )
        combined.to_csv(source, index=False, encoding="utf-8-sig")
    table = pd.read_csv(source)
    quality = table.groupby(["condition", "target_trait", "scale"], as_index=False).agg(
        n=("success", "size"), success_rate=("success", "mean"),
        mean_valid_fraction=("valid_fraction", "mean"),
        mean_completion_tokens=("completion_tokens", "mean"),
        mean_latency_seconds=("latency_seconds", "mean"),
        mean_word_count=("word_count", "mean"),
        mean_repetition_ratio=("repetition_ratio", "mean"),
        mean_selected_active_fraction=("selected_active_fraction", "mean"),
        mean_requested_feature_delta_l2=("requested_feature_delta_l2", "mean"),
        mean_effective_feature_delta_l2=("effective_feature_delta_l2", "mean"),
        mean_residual_delta_l2=("residual_delta_l2", "mean"),
    )
    quality.to_csv(output / "intervention_output_quality.csv", index=False, encoding="utf-8-sig")
    table = table[table["success"] == 1].copy()
    state_columns = [f"State_{trait}" for trait in TRAITS]
    averaged = table.groupby(
        ["id_p", "id_s", "target_trait", "condition", "scale"], as_index=False
    )[state_columns].mean()
    baseline = averaged[averaged["condition"] == "baseline"][["id_p", "id_s", *state_columns]].copy()
    baseline = baseline.rename(columns={column: f"baseline_{column}" for column in state_columns})
    changed = averaged[averaged["condition"] != "baseline"].merge(baseline, on=["id_p", "id_s"], how="inner")
    changed["target_delta"] = [
        getattr(row, f"State_{row.target_trait}") - getattr(row, f"baseline_State_{row.target_trait}")
        for row in changed.itertuples(index=False)
    ]
    non_target_deltas = []
    for row in changed.itertuples(index=False):
        values = [
            getattr(row, f"State_{trait}") - getattr(row, f"baseline_State_{trait}")
            for trait in TRAITS if trait != row.target_trait
        ]
        non_target_deltas.append(float(np.mean(values)))
    changed["non_target_delta"] = non_target_deltas
    changed["specificity"] = changed["target_delta"] - changed["non_target_delta"]
    changed.to_csv(output / "intervention_paired_deltas.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for (trait, condition, scale), group in changed.groupby(["target_trait", "condition", "scale"]):
        for outcome in ("target_delta", "non_target_delta", "specificity"):
            values = group[outcome].to_numpy(dtype=float)
            low, high = cluster_bootstrap_mean(
                group, outcome, "id_p", stable_seed(int(cfg["project"]["seed"]), trait, condition, scale, outcome)
            )
            try:
                p_value = float(wilcoxon(values).pvalue) if np.any(values != 0) else 1.0
            except ValueError:
                p_value = float("nan")
            summary_rows.append(
                {
                    "target_trait": trait, "condition": condition, "scale": scale,
                    "outcome": outcome, "n_pairs": len(group), "n_personas": group["id_p"].nunique(),
                    "mean": float(np.mean(values)), "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high, "wilcoxon_p": p_value,
                }
            )
    summary = pd.DataFrame(summary_rows)
    valid = summary["wilcoxon_p"].notna()
    summary.loc[valid, "fdr_q"] = benjamini_hochberg(summary.loc[valid, "wilcoxon_p"].to_numpy())
    summary.to_csv(output / "intervention_summary.csv", index=False, encoding="utf-8-sig")

    target = changed[changed["condition"].isin(["positive", "negative"])].copy()
    target["matched_control_condition"] = target["condition"].map(
        {"positive": "positive_control", "negative": "negative_control"}
    )
    controls = changed[changed["condition"].isin(
        ["positive_control", "negative_control"]
    )][[
        "id_p", "id_s", "target_trait", "scale", "condition",
        "target_delta", "specificity"
    ]].rename(columns={
        "condition": "matched_control_condition",
        "target_delta": "control_target_delta",
        "specificity": "control_specificity",
    })
    causal = target.merge(
        controls,
        on=["id_p", "id_s", "target_trait", "scale", "matched_control_condition"],
        how="inner",
    )
    causal["target_vs_control"] = causal["target_delta"] - causal["control_target_delta"]
    causal["specificity_vs_control"] = causal["specificity"] - causal["control_specificity"]
    contrast_rows = []
    for (trait, condition, scale), group in causal.groupby(["target_trait", "condition", "scale"]):
        for outcome in ("target_vs_control", "specificity_vs_control"):
            low, high = cluster_bootstrap_mean(
                group, outcome, "id_p", stable_seed(int(cfg["project"]["seed"]), "contrast", trait, condition, scale, outcome)
            )
            values = group[outcome].to_numpy(dtype=float)
            try:
                p_value = float(wilcoxon(values).pvalue) if np.any(values != 0) else 1.0
            except ValueError:
                p_value = float("nan")
            contrast_rows.append(
                {
                    "target_trait": trait, "condition": condition, "scale": scale,
                    "outcome": outcome, "n_pairs": len(group), "mean": float(values.mean()),
                    "bootstrap_ci_low": low, "bootstrap_ci_high": high, "wilcoxon_p": p_value,
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    valid = contrasts["wilcoxon_p"].notna()
    contrasts.loc[valid, "fdr_q"] = benjamini_hochberg(contrasts.loc[valid, "wilcoxon_p"].to_numpy())
    contrasts.to_csv(output / "intervention_random_control_contrasts.csv", index=False, encoding="utf-8-sig")

    dose_rows = []
    target_changed = changed[changed["condition"].isin(["positive", "negative"])]
    for keys, group in target_changed.groupby(
        ["id_p", "id_s", "target_trait", "condition"]
    ):
        group = group.sort_values("scale")
        if group["scale"].nunique() < 2:
            continue
        direction = 1.0 if keys[3] == "positive" else -1.0
        directed = direction * group["target_delta"].to_numpy(float)
        dose_rows.append({
            "id_p": keys[0], "id_s": keys[1], "target_trait": keys[2],
            "condition": keys[3],
            "intended_direction_slope": float(np.polyfit(group["scale"], directed, 1)[0]),
            "monotonic": bool(np.all(np.diff(directed) >= 0)),
        })
    dose_table = pd.DataFrame(dose_rows)
    dose_table.to_csv(output / "intervention_dose_response_by_unit.csv", index=False, encoding="utf-8-sig")
    dose_summary_rows = []
    if not dose_table.empty:
        for (trait, condition), group in dose_table.groupby(["target_trait", "condition"]):
            values = group["intended_direction_slope"].to_numpy(float)
            low, high = cluster_bootstrap_mean(
                group, "intended_direction_slope", "id_p",
                stable_seed(int(cfg["project"]["seed"]), "dose", trait, condition),
            )
            dose_summary_rows.append({
                "target_trait": trait, "condition": condition,
                "n_units": len(group), "n_personas": group["id_p"].nunique(),
                "mean_slope": float(values.mean()), "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "monotonic_fraction": float(group["monotonic"].mean()),
                "wilcoxon_p_greater": float(wilcoxon(values, alternative="greater").pvalue)
                if np.any(values != 0) else 1.0,
            })
    dose_summary = pd.DataFrame(dose_summary_rows)
    if not dose_summary.empty:
        dose_summary["fdr_q"] = benjamini_hochberg(
            dose_summary["wilcoxon_p_greater"].to_numpy()
        )
    dose_summary.to_csv(output / "intervention_dose_response_summary.csv", index=False, encoding="utf-8-sig")

    differentiation_rows = []
    for (situation, trait, condition, scale), group in target_changed.groupby(
        ["id_s", "target_trait", "condition", "scale"]
    ):
        if len(group) < 3:
            continue
        before = float(group[f"baseline_State_{trait}"].var(ddof=1))
        after = float(group[f"State_{trait}"].var(ddof=1))
        differentiation_rows.append({
            "id_s": situation, "target_trait": trait, "condition": condition,
            "scale": scale, "n_personas": len(group),
            "baseline_variance": before, "intervention_variance": after,
            "variance_ratio": after / max(before, 1e-12),
        })
    differentiation = pd.DataFrame(differentiation_rows)
    differentiation.to_csv(
        output / "intervention_persona_differentiation.csv",
        index=False, encoding="utf-8-sig",
    )
    report = {
        "created_at": utc_now(),
        "analysis_unit": "persona-situation pair after averaging repeats",
        "uncertainty": "2,000-replicate persona-cluster bootstrap",
        "tests": "Wilcoxon signed-rank with Benjamini-Hochberg correction",
        "successful_raw_rows": len(table),
        "paired_rows": len(changed),
        "quality_file": "intervention_output_quality.csv",
        "dose_response_file": "intervention_dose_response_summary.csv",
        "persona_differentiation_file": "intervention_persona_differentiation.csv",
    }
    write_json(output / "intervention_analysis_summary.json", report)
    print(json.dumps(report, indent=2))




COMMANDS = {
    "validate-inputs": behavior_main,
    "download": download_main,
    "validate-sae": validate_sae_main,
    "extract": extract_main,
    "discover": discover_main,
    "intervene": intervene_main,
    "analyze": analyze_main,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print("Usage: python llama31_opensae_single.py {" + ",".join(COMMANDS) + "} [options]")
        raise SystemExit(2)
    command = sys.argv.pop(1)
    COMMANDS[command]()
