from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent / "results"

STATE_LABELS = {
    "State_E": "Extraversion",
    "State_A": "Agreeableness",
    "State_C": "Conscientiousness",
    "State_N": "Neuroticism",
    "State_O": "Openness",
}
CONTEXT_LABELS = {
    "dut": "Duty",
    "int": "Intellect",
    "adv": "Adversity",
    "mat": "Mating",
    "pos": "Positivity",
    "neg": "Negativity",
    "dec": "Deception",
    "soc": "Sociality",
}
SOURCE_LABELS = {
    "human": "Human",
    "gpt4omini": "GPT-4o mini",
    "gpt52": "GPT-5.2",
    "deepseek": "DeepSeek-3.2",
    "grok3": "Grok-3",
    "llama": "Llama-3.1-8B-Instruct",
    "qwen7": "Qwen2.5-7B-Instruct",
    "qwen14": "Qwen2.5-14B-Instruct",
    "qwen32": "Qwen2.5-32B-Instruct",
    "phimini": "Phi-3-mini-4k-instruct",
    "phismall": "Phi-3-small-8k-instruct",
    "phimedium": "Phi-3-medium-4k-instruct",
}
SOURCE_ORDER = list(SOURCE_LABELS)


def p_tex(value: float) -> str:
    if value == 0 or value < 1e-300:
        return r"$<10^{-300}$"
    coefficient, exponent = f"{value:.2e}".split("e")
    return rf"${coefficient}\times10^{{{int(exponent)}}}$"


def write_lrt_table() -> None:
    data = pd.read_csv(ROOT / "mixed_model_lrt_results.csv")
    state_order = list(STATE_LABELS)
    context_order = list(CONTEXT_LABELS)
    data["state"] = pd.Categorical(data["state"], state_order, ordered=True)
    data["context"] = pd.Categorical(data["context"], context_order, ordered=True)
    data = data.sort_values(["state", "context"])

    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{p{0.18\textwidth}p{0.16\textwidth}rrrr}",
        r"\caption{Omnibus source-by-situation interaction tests for all 40 state--situation combinations.}\label{tab:mixedlm_lrt_complete}\\",
        r"\hline",
        r"State & Situation & $\chi^2$ & df & BH-adjusted $p$ & Holm-adjusted $p$ \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"State & Situation & $\chi^2$ & df & BH-adjusted $p$ & Holm-adjusted $p$ \\",
        r"\hline",
        r"\endhead",
    ]
    for row in data.itertuples():
        lines.append(
            f"{STATE_LABELS[str(row.state)]} & {CONTEXT_LABELS[str(row.context)]} & "
            f"{row.lr_statistic:.2f} & {int(row.df)} & "
            f"{p_tex(row.lrt_family_p_fdr_bh)} & {p_tex(row.lrt_family_p_holm)} \\\\"
        )
    lines.extend([r"\hline", r"\end{longtable}", r"\normalsize", ""])
    (ROOT / "mixedlm-lrt-table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_sample_table() -> None:
    data = pd.read_csv(ROOT / "mixedlm_sample_summary_by_source.csv")
    data["source"] = pd.Categorical(data["source"], SOURCE_ORDER, ordered=True)
    data = data.sort_values("source")
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{p{0.42\textwidth}rrrr}",
        r"\caption{Analysis-level sample sizes for each source.}\label{tab:mixedlm_sample_sizes}\\",
        r"\hline",
        r"Source & Observations & Groups & Min. per group & Median per group \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"Source & Observations & Groups & Min. per group & Median per group \\",
        r"\hline",
        r"\endhead",
    ]
    for row in data.itertuples():
        assert row.n_observations_min == row.n_observations_max
        assert row.n_groups_min == row.n_groups_max
        lines.append(
            f"{SOURCE_LABELS[str(row.source)]} & {int(row.n_observations_min):,} & "
            f"{int(row.n_groups_min):,} & {int(row.obs_per_group_min)} & "
            f"{row.obs_per_group_median:.0f} \\\\"
        )
    lines.extend([r"\hline", r"\end{longtable}", r"\normalsize", ""])
    (ROOT / "mixedlm-sample-table.tex").write_text("\n".join(lines), encoding="utf-8")


def write_interaction_summary() -> None:
    data = pd.read_csv(ROOT / "mixedlm_interaction_summary_by_source.csv")
    data["source"] = pd.Categorical(data["source"], SOURCE_ORDER, ordered=True)
    data = data.sort_values("source")
    lines = [
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{longtable}{p{0.52\textwidth}rrr}",
        r"\caption{Number of source-specific interaction coefficients that differed from the human reference slope.}\label{tab:mixedlm_interaction_summary}\\",
        r"\hline",
        r"LLM source & Tested & BH significant & Holm significant \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"LLM source & Tested & BH significant & Holm significant \\",
        r"\hline",
        r"\endhead",
    ]
    for row in data.itertuples():
        lines.append(
            f"{SOURCE_LABELS[str(row.source)]} & {int(row.tests)} & "
            f"{int(row.fdr_significant)} & {int(row.holm_significant)} \\\\"
        )
    lines.extend([r"\hline", r"\end{longtable}", r"\normalsize", ""])
    (ROOT / "mixedlm-interaction-summary-table.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    write_lrt_table()
    write_sample_table()
    write_interaction_summary()
    print(f"Wrote appendix tables to {ROOT}")
