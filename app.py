"""
LLM Evaluation Web UI  (Gradio)

Launch:
    cd interface
    python app.py

Then open http://localhost:6006 in your browser.

Required environment variables (set before launching):
    FIREWORKS_API_KEY   — Fireworks AI key (for Qwen2-7B, Qwen2.5-14B, Mistral-7B)
    OPENAI_API_KEY      — OpenAI key (for GPT-4o and Whisper transcription)
    GOOGLE_API_KEY      — Google AI key (for gemini-3.1-pro-preview)
    ANTHROPIC_API_KEY   — Anthropic key (for Claude)
"""

import csv as csv_module
import math
import os
import queue as queue_module
import re
import sys

from dotenv import load_dotenv
load_dotenv()  # loads .env from the current working directory
import threading
import traceback
from itertools import combinations
from pathlib import Path

import gradio as gr
import pandas as pd
import research_statistics
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.llm_eval.dataset_cleaner import (
    analyze as _dc_analyze,
    clean as _dc_clean,
    pairs_to_dataframe as _dc_to_df,
    save_cleaned as _dc_save,
)
from src.llm_eval.metrics.evaluator import Evaluator
from src.llm_eval.models import load_models_from_config
from src.llm_eval.output.exporter import export_csv, export_summary_csv
from src.llm_eval.runner import (
    EvalRunner,
    _AUDIO_FILE_ALIASES,
    _QUESTION_ALIASES,
    _REFERENCE_ALIASES,
    _find_col,
)

CONFIG_PATH = "config/models.yaml"
RESULTS_DIR = "results"
DATA_DIR    = "data"
MULTITURN_SCENARIO_DIR = Path(DATA_DIR) / "multi_turn"
VOICE_SAMPLES_DIR = Path(DATA_DIR) / "voice_samples"
SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

def _get_voice_sample_folders() -> list[str]:
    """Return language/voice folders stored under data/voice_samples/."""
    if not VOICE_SAMPLES_DIR.exists():
        return []
    return sorted(folder.name for folder in VOICE_SAMPLES_DIR.iterdir() if folder.is_dir())

def _get_voice_files(folder_name: str) -> list[str]:
    """Return supported audio filenames from one selected language folder."""
    if not folder_name:
        return []
    folder = VOICE_SAMPLES_DIR / folder_name
    if not folder.exists():
        return []
    return sorted(
        file.name for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in SUPPORTED_AUDIO
    )

def _build_audio_files_map(folder_name: str, selected_files: list[str]) -> dict[str, str]:
    """Build the filename -> local path mapping for only the selected recordings."""
    if not folder_name or not selected_files:
        return {}
    folder = VOICE_SAMPLES_DIR / folder_name
    return {
        filename: str(folder / filename)
        for filename in selected_files
        if (folder / filename).is_file()
    }


def _read_dataset_audio_filenames(dataset_path: str) -> list[str]:
    """Read audio filenames declared by a CSV, JSON, or JSONL QA dataset."""
    if not dataset_path:
        return []

    path = Path(str(dataset_path))
    ext = path.suffix.lower()

    try:
        if ext == ".csv":
            with open(path, encoding="utf-8-sig", newline="") as f:
                reader = csv_module.DictReader(f)
                headers = list(reader.fieldnames or [])
                audio_col = _find_col(headers, _AUDIO_FILE_ALIASES)
                if not audio_col:
                    return []
                return [
                    Path((row.get(audio_col) or "").strip()).name
                    for row in reader
                    if (row.get(audio_col) or "").strip()
                ]

        if ext in {".json", ".jsonl"}:
            import json as json_module
            with open(path, encoding="utf-8-sig") as f:
                if ext == ".json":
                    data = json_module.load(f)
                    rows = data if isinstance(data, list) else [data]
                else:
                    rows = [json_module.loads(line) for line in f if line.strip()]

            rows = [row for row in rows if isinstance(row, dict)]
            if not rows:
                return []
            headers = list(rows[0].keys())
            audio_col = _find_col(headers, _AUDIO_FILE_ALIASES)
            if not audio_col:
                return []
            return [
                Path(str(row.get(audio_col) or "").strip()).name
                for row in rows
                if str(row.get(audio_col) or "").strip()
            ]
    except Exception:
        return []

    return []


def _match_dataset_audio(dataset_path: str):
    """Find the voice-sample folder with the largest real filename overlap."""
    csv_files = _read_dataset_audio_filenames(dataset_path)
    if not csv_files:
        return None, [], 0

    wanted = set(csv_files)
    best_folder = None
    best_matches = []

    for folder_name in _get_voice_sample_folders():
        available = set(_get_voice_files(folder_name))
        matches = sorted(wanted & available)
        if len(matches) > len(best_matches):
            best_folder = folder_name
            best_matches = matches

    return best_folder, best_matches, len(wanted)


def _voice_folder_display_name(folder_name: str) -> str:
    """Return a clear UI name for a repository voice folder."""
    if not folder_name:
        return ""
    language_names = {
        "EN": "English", "DE": "German", "HE": "Hebrew", "HU": "Hungarian",
        "ZH": "Chinese", "CN": "Chinese", "ES": "Spanish", "FR": "French",
        "IT": "Italian", "PT": "Portuguese", "JA": "Japanese", "JP": "Japanese",
        "KO": "Korean", "AR": "Arabic", "RU": "Russian", "UK": "Ukrainian",
    }
    code = folder_name.upper().split("-")[-1]
    return f"{language_names.get(code, code)} ({folder_name})"


def _audio_auto_status(folder_name, matches, total):
    if total == 0:
        return _status_err(
            "Audio mode requires a CSV with an audio filename column "
            f"({', '.join(sorted(_AUDIO_FILE_ALIASES))})."
        )
    if not folder_name or not matches:
        return _status_err(
            f"0 / {total} dataset audio files were found in data/voice_samples/."
        )

    missing = total - len(matches)
    if missing:
        return _status_warn(
            f"Matched automatically: <b>{_voice_folder_display_name(folder_name)}</b> — "
            f"{len(matches)} / {total} recordings available ({missing} missing)."
        )
    return _status_ok(
        f"Matched automatically: <b>{_voice_folder_display_name(folder_name)}</b> — "
        f"{len(matches)} / {total} recordings available."
    )


def _auto_match_audio(dataset_path):
    """Populate the audio selector directly from the selected dataset."""
    folder, matches, total = _match_dataset_audio(dataset_path)
    return (
        folder or "",
        gr.update(choices=matches, value=matches),
        _audio_selection_preview(matches),
        _audio_auto_status(folder, matches, total),
    )


def _select_all_matched_voice_files(folder_name):
    """Select all valid CSV-matched files, not every file in the folder."""
    # The checkbox choices are already restricted to the CSV/repository intersection.
    # This callback is replaced below with a dataset-aware callback.
    files = _get_voice_files(folder_name) if folder_name else []
    return gr.update(value=files), _audio_selection_preview(files)


def _select_all_for_dataset(dataset_path):
    folder, matches, _ = _match_dataset_audio(dataset_path)
    return gr.update(choices=matches, value=matches), _audio_selection_preview(matches)


def _on_voice_folder_change(folder_name):
    """Refresh the multi-select when a different language folder is chosen."""
    files = _get_voice_files(folder_name)
    return gr.update(choices=files, value=[]), _audio_selection_preview([])


def _audio_selection_preview(selected_files):
    """Compact summary: count + first four selected filenames."""
    import html

    files = list(selected_files or [])
    count = len(files)

    if not files:
        return (
            "<div style='padding:10px 12px;border:1px solid #e5e7eb;border-radius:8px;"
            "background:#fafafa;color:#6b7280;font-size:0.92em;'>"
            "<b>0 selected</b> &nbsp;·&nbsp; Open the selector below or choose Select all."
            "</div>"
        )

    tags = "".join(
        "<span style='display:inline-block;padding:5px 9px;margin:3px 4px 3px 0;"
        "border:1px solid #d9dce3;border-radius:6px;background:white;"
        "font-size:0.92em;white-space:nowrap;'>"
        f"{html.escape(name)}</span>"
        for name in files[:4]
    )
    remainder = count - 4
    more = (
        f"<span style='display:inline-block;margin-left:4px;color:#6b7280;"
        f"font-size:0.9em;'>+{remainder} more</span>"
        if remainder > 0 else ""
    )

    return (
        "<div style='padding:9px 12px;border:1px solid #e5e7eb;border-radius:8px;"
        "background:#fafafa;'>"
        f"<div style='font-weight:600;margin-bottom:4px;'>{count} audio file"
        f"{'s' if count != 1 else ''} selected</div>"
        f"<div>{tags}{more}</div>"
        "</div>"
    )

def _clear_voice_files():
    """Clear all selected audio files."""
    return gr.update(value=[]), _audio_selection_preview([])


def _select_all_voice_files(folder_name):
    """Select all audio files in the currently selected voice folder."""
    files = _get_voice_files(folder_name)
    return gr.update(choices=files, value=files), _audio_selection_preview(files)

_AUDIO_CAPABLE_TYPES = {"openai", "gemini"}

# Matches any CJK Unified Ideograph block (covers Simplified + Traditional)
_CHINESE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]")

# ---------------------------------------------------------------------------
# Status-box HTML helpers
# ---------------------------------------------------------------------------

def _status_ok(msg: str) -> str:
    return (
        f'<div style="color:#155724;border:1px solid #28a745;padding:6px 12px;'
        f'border-radius:5px;background:#d4edda;margin-top:4px;">✓ {msg}</div>'
    )

def _status_warn(msg: str) -> str:
    return (
        f'<div style="color:#856404;border:1px solid #ffc107;padding:6px 12px;'
        f'border-radius:5px;background:#fff3cd;margin-top:4px;">⚠ {msg}</div>'
    )

def _status_err(msg: str) -> str:
    return (
        f'<div style="color:#721c24;border:1px solid #cc0000;padding:6px 12px;'
        f'border-radius:5px;background:#fff5f5;margin-top:4px;">❌ {msg}</div>'
    )


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _read_model_names() -> list[str]:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return [m["name"] for m in raw.get("models", [])]
    except Exception:
        return []


def _read_model_types() -> dict[str, str]:
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return {m["name"]: m["type"] for m in raw.get("models", [])}
    except Exception:
        return {}


def _ensure_claude_model_config() -> None:
    """Add a usable default Claude entry when the config has none."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        models = raw.setdefault("models", [])
        if any(m.get("type") in {"claude", "anthropic"} for m in models):
            return

        models.append({
            "name": "Claude-Sonnet",
            "type": "claude",
            "model_id": "claude-sonnet-5",
            "api_key": "${ANTHROPIC_API_KEY}",
        })
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(raw, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    except Exception as exc:
        print(f"WARNING: could not add the default Claude model to {CONFIG_PATH}: {exc}")


def _existing_datasets() -> list[str]:
    exts = {".json", ".jsonl", ".csv", ".txt"}
    p = Path(DATA_DIR)
    if not p.exists():
        return []
    return sorted(
        str(f) for f in p.iterdir()
        if f.suffix.lower() in exts and not _looks_like_scenario_file(f)
    )


def _looks_like_scenario_file(path) -> bool:
    """Return True when a supported file follows the scenario-turn schema."""
    try:
        EvalRunner([], Evaluator()).load_scenario(str(path))
        return True
    except Exception:
        return False


def _existing_scenarios() -> list[str]:
    """Load benchmark scenarios only from the project's data/multi_turn folder."""
    root = MULTITURN_SCENARIO_DIR
    if not root.exists():
        return []
    candidates = (
        f for f in root.rglob("*")
        if f.is_file() and f.suffix.lower() in {".csv", ".json", ".jsonl"}
    )
    return sorted(str(f) for f in candidates if _looks_like_scenario_file(f))


# ---------------------------------------------------------------------------
# Metric summary builder
# ---------------------------------------------------------------------------

def _build_summary(df: pd.DataFrame, use_bertscore: bool) -> pd.DataFrame:
    wanted = [
        "rouge1", "rouge1_p", "rouge1_r",
        "rouge2", "rougeL",
        "bleu", "meteor", "f1",
        "response_length", "latency_seconds",
    ]
    if use_bertscore:
        wanted.append("bertscore_f1")
    # Include audio quality metrics when present (audio mode)
    for aq_col in ("snr_db", "speech_ratio", "clarity_score", "whisper_confidence"):
        if aq_col in df.columns and df[aq_col].any():
            wanted.append(aq_col)
    cols = [c for c in wanted if c in df.columns]

    summary = (
        df[df["error"].isna()]
        .groupby("model_name")[cols]
        .mean()
        .round(4)
        .reset_index()
        .rename(columns={
            "model_name":          "Model",
            "rouge1":              "ROUGE-1 F",
            "rouge1_p":            "ROUGE-1 P",
            "rouge1_r":            "ROUGE-1 R",
            "rouge2":              "ROUGE-2",
            "rougeL":              "ROUGE-L",
            "bleu":                "BLEU",
            "meteor":              "METEOR",
            "f1":                  "Token-F1",
            "response_length":     "Avg Length(w)",
            "bertscore_f1":        "BERTScore-F1",
            "latency_seconds":     "Avg Latency(s)",
            "snr_db":              "SNR(dB)",
            "speech_ratio":        "Speech Ratio",
            "clarity_score":       "Clarity Score",
            "whisper_confidence":  "Whisper Conf.",
        })
    )
    sort_col = (
        "BERTScore-F1" if (use_bertscore and "BERTScore-F1" in summary)
        else "Token-F1"
    )
    if sort_col in summary.columns:
        summary = summary.sort_values(sort_col, ascending=False)
    return summary.reset_index(drop=True)


def _scored_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Return only validly executed and validly judged scenario runs."""
    if df.empty:
        return df.copy()
    if "evaluation_status" in df:
        return df[df["evaluation_status"] == "scored"].copy()
    system_ok = df.get("system_error", pd.Series(None, index=df.index)).isna()
    judge_ok = df.get("judge_error", pd.Series(None, index=df.index)).isna()
    return df[system_ok & judge_ok].copy()


def _format_binary_rate(values, *, denominator: int | None = None) -> str:
    """Descriptive numerator/denominator only; no independence-based run-level CI."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    n = len(series) if denominator is None else denominator
    return f"{int(series.eq(1).sum())}/{n} ({series.eq(1).sum()/n*100:.1f}%)" if n else "—"


def _macro_scenario_mean(scored: pd.DataFrame, column: str) -> float | None:
    """Average scenario-level means so longer/repeated scenarios do not dominate."""
    if scored.empty or column not in scored:
        return None
    work = scored[["scenario_id", column]].copy()
    work[column] = pd.to_numeric(work[column], errors="coerce")
    by_scenario = work.dropna().groupby("scenario_id")[column].mean()
    return None if by_scenario.empty else float(by_scenario.mean())


def _build_scenario_validity_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise run validity without mixing infrastructure and behaviour."""
    columns = [
        "Model", "Attempts", "Execution success", "Examiner success",
        "Valid evaluated runs", "Timeouts", "Execution errors", "Judge errors",
        "Median latency (ms)", "P95 latency (ms)",
    ]
    if df.empty or "model_name" not in df:
        return pd.DataFrame(columns=columns)
    rows = []
    for model_name, model_df in df.groupby("model_name", sort=False):
        attempts = len(model_df)
        execution = pd.to_numeric(model_df.get("execution_success"), errors="coerce")
        judge_attempted = pd.to_numeric(model_df.get("judge_attempted"), errors="coerce")
        judge_success = pd.to_numeric(model_df.get("judge_success"), errors="coerce")
        valid = pd.to_numeric(
            model_df.get("valid_evaluated_run", judge_success), errors="coerce"
        )
        submitted = int((judge_attempted == 1).sum())
        latency = pd.to_numeric(model_df.get("mean_latency_ms"), errors="coerce").dropna()
        error_categories = model_df.get(
            "error_category", pd.Series("", index=model_df.index)
        ).fillna("").astype(str)
        rows.append({
            "Model": model_name,
            "Attempts": attempts,
            "Execution success": _format_binary_rate(execution, denominator=attempts),
            "Examiner success": _format_binary_rate(
                judge_success[judge_attempted == 1], denominator=submitted
            ),
            "Valid evaluated runs": _format_binary_rate(valid, denominator=attempts),
            "Timeouts": int(error_categories.eq("timeout").sum()),
            "Execution errors": int((execution == 0).sum()),
            "Judge errors": int(((judge_attempted == 1) & (judge_success == 0)).sum()),
            "Median latency (ms)": round(float(latency.median()), 1) if not latency.empty else "—",
            "P95 latency (ms)": round(float(latency.quantile(0.95)), 1) if not latency.empty else "—",
        })
    return pd.DataFrame(rows, columns=columns)


def _build_scenario_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Audited, scenario-weighted estimates; valid runs only."""
    return research_statistics.model_summary(df)


def _build_scenario_run_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Show outcome and process measures separately for every scenario run."""
    columns = [
        "Model", "Scenario", "Rep", "Status", "Execution", "Examiner",
        "Goal", "Safe", "Terminal", "Checkpoints", "Critical checkpoints",
        "Critical failure", "Turns", "Latency (ms)", "Error category",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)

    def binary_label(value) -> str:
        if value is None or pd.isna(value):
            return "—"
        try:
            return "Yes" if int(value) == 1 else "No"
        except (TypeError, ValueError):
            return "—"

    rows = []
    for _, row in df.sort_values(["model_name", "scenario_id", "repetition"], kind="stable").iterrows():
        goal = row.get("goal_achieved", row.get("task_success"))
        completion = pd.to_numeric(row.get("checkpoint_completion"), errors="coerce")
        quality = pd.to_numeric(row.get("checkpoint_quality"), errors="coerce")
        critical_completion = pd.to_numeric(
            row.get("critical_checkpoint_completion"), errors="coerce"
        )
        latency = pd.to_numeric(row.get("mean_latency_ms"), errors="coerce")
        rows.append({
            "Model": row.get("model_name", ""),
            "Scenario": row.get("scenario_title") or row.get("scenario_id", ""),
            "Rep": row.get("repetition", ""),
            "Status": row.get("evaluation_status", "scored"),
            "Execution": binary_label(row.get("execution_success", 1)),
            "Examiner": binary_label(row.get("judge_success")),
            "Goal": binary_label(goal),
            "Safe": binary_label(row.get("safe_completion")),
            "Terminal": binary_label(row.get("terminal_state_valid")),
            "Checkpoints": f"{float(completion) * 100:.1f}%" if pd.notna(completion) else "—",
            "Critical checkpoints": f"{float(critical_completion) * 100:.1f}%" if pd.notna(critical_completion) else "N/A",
            "Critical failure": binary_label(row.get("critical_failure")),
            "Turns": f"{row.get('actual_turns', '—')}/{row.get('target_turns', '—')}",
            "Latency (ms)": round(float(latency), 1) if pd.notna(latency) else "—",
            "Error category": row.get("error_category") or "",
        })
    return pd.DataFrame(rows, columns=columns)


def _build_scenario_dimension_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise ordinal dimensions without treating N/A as zero."""
    dimensions = {
        "information_gathering": "Information gathering",
        "instruction_following": "Instruction following",
        "state_constraint_tracking": "State and constraint tracking",
        "correction_recovery": "Correction and recovery",
        "domain_accuracy": "Domain accuracy",
        "relevance_efficiency": "Relevance and efficiency",
        "clarity_actionability": "Clarity and actionability",
        "safety_cost_awareness": "Safety and cost awareness",
        "conversational_coherence": "Conversational coherence",
    }
    if df.empty or "model_name" not in df.columns:
        return pd.DataFrame(columns=["Model", "Dimension", "Median", "IQR", "Mean ± SD", "N"])
    valid = _scored_runs(df)
    rows = []
    for model_name, model_df in valid.groupby("model_name", sort=False):
        for column, label in dimensions.items():
            if column not in model_df:
                continue
            values = pd.to_numeric(model_df[column], errors="coerce").dropna()
            if values.empty:
                continue
            rows.append({
                "Model": model_name,
                "Dimension": label,
                "Median": round(float(values.median()), 2),
                "IQR": f"{values.quantile(0.25):.2f}–{values.quantile(0.75):.2f}",
                "Mean ± SD": (
                    f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"
                    if len(values) > 1 else f"{values.mean():.2f} ± N/A"
                ),
                "N": len(values),
            })
    return pd.DataFrame(
        rows, columns=["Model", "Dimension", "Median", "IQR", "Mean ± SD", "N"]
    )


def _build_scenario_stability_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Within-scenario repetition summary, with failures retained in denominators."""
    return research_statistics.repetition_summary(df)



def _build_paired_model_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    """Scenario-paired descriptive differences; no unplanned significance tests."""
    return research_statistics.paired_comparison_table(df)


def _non_audio_models(selected: list[str], model_types: dict[str, str]) -> list[str]:
    return [
        name for name in selected
        if model_types.get(name, "") not in _AUDIO_CAPABLE_TYPES
    ]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_dataset_file(path) -> dict:
    """
    Validate a dataset file and return a gr.update() for the status HTML box.
    Also emits gr.Warning() toasts for non-fatal issues.
    Called on dataset picker / existing-box change.
    """
    if not path:
        return gr.update(visible=False, value="")

    path = str(path)
    try:
        runner = EvalRunner([], Evaluator())
        pairs = runner.load_dataset(path)
    except Exception as e:
        gr.Warning(f"Dataset parse error: {e}")
        return gr.update(visible=True, value=_status_err(f"Parse error: {e}"))

    name = Path(path).name

    if len(pairs) == 0:
        gr.Warning("Dataset is empty (0 rows found).")
        return gr.update(visible=True, value=_status_warn("Dataset is empty."))

    no_ref = sum(1 for p in pairs if not p.reference_answer)
    no_q   = sum(1 for p in pairs if not p.question)

    issues = []
    if no_q:
        issues.append(f"{no_q} rows missing a question column.")
        gr.Warning(f"{no_q} rows are missing a question — check your column names (expected: {sorted(_QUESTION_ALIASES)}).")
    if no_ref == len(pairs):
        issues.append("No reference answer column found — ROUGE/BLEU/METEOR will score 0.")
        gr.Warning("No reference answer column detected. Evaluation metrics will score 0.")
    elif no_ref > 0:
        issues.append(f"{no_ref}/{len(pairs)} rows missing reference answers.")
        gr.Warning(f"{no_ref}/{len(pairs)} rows are missing reference answers.")

    if issues:
        body = "<br>".join(f"⚠ {i}" for i in issues)
        return gr.update(visible=True, value=_status_warn(f"{len(pairs)} questions loaded from <b>{name}</b><br>{body}"))

    return gr.update(
        visible=True,
        value=_status_ok(f"{len(pairs)} questions loaded from <b>{name}</b>"),
    )


def _auto_skip_bertscore_for_dataset(path):
    """Force BERTScore off when reference answers are unavailable."""
    if not path:
        return gr.update()

    try:
        runner = EvalRunner([], Evaluator())
        pairs = runner.load_dataset(str(path))
    except Exception:
        # Dataset validation reports parse errors separately; do not change the
        # user's metric choice when the file cannot be inspected.
        return gr.update()

    if not pairs:
        return gr.update()

    references_missing = any(not (p.reference_answer or "").strip() for p in pairs)
    if references_missing:
        return gr.update(value=True, interactive=False)
    return gr.update(value=False, interactive=True)


def _dataset_requires_bertscore_skip(path) -> bool:
    """Backend guard matching the UI lock for missing references."""
    if not path:
        return False
    try:
        pairs = EvalRunner([], Evaluator()).load_dataset(str(path))
    except Exception:
        return False
    return bool(pairs) and any(
        not (pair.reference_answer or "").strip() for pair in pairs
    )


def _auto_skip_bertscore_for_source(source, uploaded, existing):
    """Apply the same auto-setting when switching dataset source."""
    path = uploaded if source == "Upload file" else existing
    return _auto_skip_bertscore_for_dataset(path)


def _validate_audio_csv(file) -> dict:
    """
    Validate audio metadata CSV column names.
    Returns gr.update() for the audio-CSV status HTML box.
    """
    if file is None:
        return gr.update(visible=False, value="")

    path = file if isinstance(file, str) else str(file)
    try:
        with open(path, encoding="utf-8-sig") as f:
            reader = csv_module.DictReader(f)
            headers = list(reader.fieldnames or [])
            rows = list(reader)
    except Exception as e:
        gr.Warning(f"CSV read error: {e}")
        return gr.update(visible=True, value=_status_err(f"Read error: {e}"))

    af_key = _find_col(headers, _AUDIO_FILE_ALIASES)
    r_key  = _find_col(headers, _REFERENCE_ALIASES)

    issues = []
    if not af_key:
        msg = (f"No audio filename column found. "
               f"Expected one of: {sorted(_AUDIO_FILE_ALIASES)}. "
               f"Found columns: {headers}")
        issues.append(msg)
        gr.Warning(msg)
    if not r_key:
        msg = (f"No reference answer column found. "
               f"Expected one of: {sorted(_REFERENCE_ALIASES)}. "
               f"Found columns: {headers}")
        issues.append(msg)
        gr.Warning(msg)

    if issues:
        body = "<br>".join(f"⚠ {i}" for i in issues)
        return gr.update(visible=True, value=_status_err(f"Column issues:<br>{body}"))

    return gr.update(
        visible=True,
        value=_status_ok(
            f"{len(rows)} rows — audio column: <b>{af_key}</b>, reference column: <b>{r_key}</b>"
        ),
    )


def _check_api_keys(selected_models: list[str], model_types: dict[str, str]) -> list[str]:
    """Return list of warning strings for missing API keys."""
    _KEY_MAP = {
        "fireworks": ("FIREWORKS_API_KEY", "Fireworks AI"),
        "openai":    ("OPENAI_API_KEY",    "OpenAI"),
        "gemini":    ("GOOGLE_API_KEY",    "Google Gemini"),
        "claude":     ("ANTHROPIC_API_KEY", "Anthropic Claude"),
        "anthropic":  ("ANTHROPIC_API_KEY", "Anthropic Claude"),
    }
    seen = set()
    warnings = []
    for name in selected_models:
        mtype = model_types.get(name, "")
        if mtype not in _KEY_MAP:
            continue
        env_var, provider = _KEY_MAP[mtype]
        if env_var in seen:
            continue
        seen.add(env_var)
        if not os.environ.get(env_var):
            warnings.append(
                f"{env_var} not set ({provider} models: {name})"
            )
    return warnings


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def run_evaluation(
    dataset_source,
    selected_dataset,
    existing_path,
    selected_models,
    max_tokens,
    temperature,
    limit_text,
    no_bertscore,
    input_mode,
    voice_folder,
    selected_audio_files,
    use_whisper,
):
    # ---- resolve dataset path -------------------------------------------
    # Text and Audio modes use the same selected metadata dataset.
    if dataset_source == "Upload file":
        dataset_path = selected_dataset
    else:
        dataset_path = existing_path

    if not dataset_path:
        yield "Please provide a dataset file.", None, None, gr.update(interactive=True)
        return

    if not selected_models:
        yield "Please select at least one model.", None, None, gr.update(interactive=True)
        return

    limit = None
    if str(limit_text).strip():
        try:
            limit = int(limit_text)
        except ValueError:
            yield "Limit must be an integer (or leave blank).", None, None, gr.update(interactive=True)
            return

    # ---- API key pre-check (non-blocking warning) -----------------------
    type_map = _read_model_types()
    key_warnings = _check_api_keys(selected_models, type_map)
    for w in key_warnings:
        gr.Warning(f"⚠ Missing API key: {w}")

    # Do not allow a stale or manipulated UI value to run BERTScore without
    # reference answers; the same rule is enforced by the disabled checkbox.
    use_bertscore = not no_bertscore and not _dataset_requires_bertscore_skip(dataset_path)

    audio_files_map: dict = {}
    if input_mode == "Audio":
        # Resolve the audio folder from the CSV automatically. There is no
        # user-selectable language/folder dropdown, so mismatched combinations
        # cannot be created in the UI.
        matched_folder, matched_files, csv_audio_count = _match_dataset_audio(dataset_path)

        if csv_audio_count == 0:
            yield (
                "Audio mode requires a dataset containing an audio filename field/column."
            ), None, None, gr.update(interactive=True)
            return

        if not matched_folder or not matched_files:
            yield (
                "No matching repository audio was found for this CSV under "
                "data/voice_samples/."
            ), None, None, gr.update(interactive=True)
            return

        valid = set(matched_files)
        chosen = [name for name in (selected_audio_files or []) if name in valid]
        # If the user has not chosen a subset, use all valid matched recordings.
        chosen = chosen or matched_files

        audio_files_map = _build_audio_files_map(matched_folder, chosen)
        if not audio_files_map:
            yield "No valid matched audio files are available.", None, None, gr.update(interactive=True)
            return

    # ---- run in background thread ---------------------------------------
    q: queue_module.Queue = queue_module.Queue()

    def _thread():
        try:
            all_models = load_models_from_config(CONFIG_PATH)
            models = [m for m in all_models if m.name in selected_models]
            if not models:
                q.put(("error", "None of the selected models were found in config."))
                return

            applied_token_caps = {}
            for m in models:
                if hasattr(m, "configure_generation"):
                    applied = m.configure_generation(int(max_tokens), float(temperature))
                else:
                    applied = min(max(1, int(max_tokens)), 4096)
                    m.config.max_tokens = applied
                    m.config.temperature = float(temperature)
                applied_token_caps[m.name] = applied

            whisper_fn = None
            if input_mode == "Audio" and use_whisper:
                try:
                    from llm_eval.transcriber import make_whisper_transcriber
                    whisper_fn = make_whisper_transcriber()
                    q.put(("log", "Whisper transcriber ready (non-audio models will transcribe first)."))
                except Exception as e:
                    q.put(("log", f"⚠ Whisper init failed: {e} — non-audio models will be skipped."))

            evaluator = Evaluator()
            runner    = EvalRunner(models, evaluator)

            if input_mode == "Audio":
                dataset = runner.load_audio_dataset(dataset_path, audio_files_map)
            else:
                dataset = runner.load_dataset(dataset_path)

            if limit:
                dataset = dataset[:limit]

            mode_str = f"Audio ({len(audio_files_map)} files)" if input_mode == "Audio" else "Text"
            q.put(("log", f"Dataset  : {dataset_path}  ({len(dataset)} questions)"))
            q.put(("log", f"Mode     : {mode_str}"))
            q.put(("log", f"Models   : {[m.name for m in models]}"))
            q.put(("log", f"Token caps: {applied_token_caps}"))
            q.put(("log", f"Metrics  : BLEU · METEOR · Token-F1 · ROUGE" +
                          (" · BERTScore" if use_bertscore else " (BERTScore skipped)")))
            q.put(("log", "-" * 60))

            def on_progress(fraction: float, desc: str):
                q.put(("progress", fraction, desc))

            df = runner.run(
                dataset,
                use_bertscore=use_bertscore,
                input_mode=input_mode.lower(),
                whisper_fn=whisper_fn,
                on_progress=on_progress,
            )

            os.makedirs(RESULTS_DIR, exist_ok=True)
            csv_path = os.path.join(RESULTS_DIR, "eval_results.csv")
            export_csv(df, csv_path)
            export_summary_csv(df, csv_path)
            q.put(("done", df, csv_path))

        except Exception:
            q.put(("error", traceback.format_exc()))

    thread = threading.Thread(target=_thread, daemon=True)
    thread.start()

    log_lines: list[str] = []
    yield "Starting…", None, None, gr.update(interactive=False)

    while True:
        try:
            item = q.get(timeout=0.3)
        except queue_module.Empty:
            if not thread.is_alive():
                break
            yield "\n".join(log_lines) or "Running…", None, None, gr.update(interactive=False)
            continue

        kind = item[0]
        if kind in ("log", "progress"):
            msg = item[1] if kind == "log" else item[2]
            log_lines.append(msg)
            yield "\n".join(log_lines[-60:]), None, None, gr.update(interactive=False)
        elif kind == "done":
            _, df, csv_path = item
            log_lines.append("")
            log_lines.append("✅  Evaluation complete!")
            summary = _build_summary(df, use_bertscore)
            yield "\n".join(log_lines), summary, csv_path, gr.update(interactive=True)
            return
        elif kind == "error":
            log_lines.append(f"\n❌  Error:\n{item[1]}")
            yield "\n".join(log_lines), None, None, gr.update(interactive=True)
            return

    yield "\n".join(log_lines), None, None, gr.update(interactive=True)


def run_scenario_evaluation(
    scenario_source,
    uploaded_scenarios,
    selected_existing_scenarios,
    selected_models,
    judge_model_name,
    response_max_tokens,
    judge_max_tokens,
    temperature,
    repetitions,
):
    """Run reusable scripted multi-turn scenarios and judge full transcripts."""
    def empty_output(message: str, interactive: bool = True):
        return (
            message, None, None, None, None, None, None, None,
            gr.update(interactive=interactive), None,
        )

    if scenario_source == "Upload files":
        raw_paths = uploaded_scenarios or []
    else:
        raw_paths = selected_existing_scenarios or []
    if not isinstance(raw_paths, list):
        raw_paths = [raw_paths]
    scenario_paths = [str(path) for path in raw_paths if path]

    if not scenario_paths:
        yield empty_output("Please select or upload at least one scenario.")
        return
    if not selected_models:
        yield empty_output("Please select at least one model to evaluate.")
        return
    if not judge_model_name:
        yield empty_output("Please select a judge model.")
        return
    if judge_model_name in selected_models:
        yield empty_output(
            "The judge model must be different from every model being evaluated. "
            "Select an independent judge to avoid self-evaluation bias."
        )
        return

    try:
        repetitions = int(repetitions)
        if repetitions < 3:
            raise ValueError
    except (TypeError, ValueError):
        yield empty_output(
            "Use at least three repetitions per model–scenario combination "
            "to estimate within-scenario stability."
        )
        return
    try:
        response_max_tokens = int(response_max_tokens)
        judge_max_tokens = int(judge_max_tokens)
        if not 64 <= response_max_tokens <= 16384:
            raise ValueError
        if not 512 <= judge_max_tokens <= 16384:
            raise ValueError
    except (TypeError, ValueError):
        yield empty_output(
            "Response tokens must be 64–16384 and judge tokens 512–16384."
        )
        return

    type_map = _read_model_types()
    for warning in _check_api_keys(
        list(dict.fromkeys(list(selected_models) + [judge_model_name])), type_map
    ):
        gr.Warning(f"⚠ Missing API key: {warning}")
    q: queue_module.Queue = queue_module.Queue()

    def _thread():
        try:
            all_models = load_models_from_config(CONFIG_PATH)
            by_name = {model.name: model for model in all_models}
            missing = [name for name in selected_models if name not in by_name]
            if judge_model_name not in by_name:
                missing.append(judge_model_name)
            if missing:
                q.put(("error", f"Models not found in config: {sorted(set(missing))}"))
                return

            examinees = [by_name[name] for name in selected_models]
            judge = by_name[judge_model_name]
            response_generation_settings = {}
            for model in examinees:
                if hasattr(model, "configure_generation"):
                    applied = model.configure_generation(
                        response_max_tokens, float(temperature)
                    )
                else:
                    applied = response_max_tokens
                    model.config.max_tokens = applied
                    model.config.temperature = float(temperature)
                extra = getattr(model.config, "extra", {})
                if not isinstance(extra, dict):
                    extra = {}
                response_generation_settings[model.name] = {
                    "max_output_tokens": applied,
                    "temperature": float(model.config.temperature),
                    "thinking_level": extra.get("thinking_level"),
                    "effort": extra.get("effort"),
                }
                if applied < response_max_tokens:
                    raise RuntimeError(
                        f"{model.name}: requested {response_max_tokens} response "
                        f"tokens, but the configured model cap applied {applied}. "
                        "The run was stopped before data collection. Update "
                        "extra.max_output_tokens_limit."
                    )
            if hasattr(judge, "configure_generation"):
                applied_judge_tokens = judge.configure_generation(
                    judge_max_tokens, 0.0
                )
            else:
                applied_judge_tokens = judge_max_tokens
                judge.config.max_tokens = applied_judge_tokens
                judge.config.temperature = 0.0
            judge_extra = getattr(judge.config, "extra", {})
            if not isinstance(judge_extra, dict):
                judge_extra = {}
            judge_generation_settings = {
                "max_output_tokens": applied_judge_tokens,
                "temperature": float(judge.config.temperature),
                "thinking_level": judge_extra.get("thinking_level"),
                "effort": judge_extra.get("effort"),
            }
            if applied_judge_tokens < judge_max_tokens:
                raise RuntimeError(
                    f"{judge.name}: requested {judge_max_tokens} judge tokens, "
                    f"but the configured model cap applied {applied_judge_tokens}. "
                    "The run was stopped before data collection. Update "
                    "extra.structured_max_output_tokens_limit."
                )

            runner = EvalRunner(examinees, Evaluator())
            scenarios = runner.load_scenarios(scenario_paths)
            q.put(("log", f"Scenarios : {[scenario.scenario_id for scenario in scenarios]}"))
            q.put(("log", f"Examinees : {[model.name for model in examinees]}"))
            q.put(("log", f"Judge     : {judge.name}"))
            q.put(("log", f"Repetitions: {repetitions}"))
            q.put(("log", f"Response generation: {response_generation_settings}"))
            q.put(("log", f"Judge generation: {judge_generation_settings}"))
            q.put(("log", "Mode      : scripted text multi-turn"))
            q.put(("log", "-" * 60))

            def on_progress(fraction: float, description: str):
                q.put(("progress", fraction, description))

            df = runner.run_scenarios(
                scenarios,
                judge_model=judge,
                repetitions=repetitions,
                on_progress=on_progress,
            )
            os.makedirs(RESULTS_DIR, exist_ok=True)
            csv_path = os.path.join(RESULTS_DIR, "scenario_eval_results.csv")
            df.to_csv(csv_path, index=False)
            q.put(("done", df, csv_path))
        except Exception:
            q.put(("error", traceback.format_exc()))

    thread = threading.Thread(target=_thread, daemon=True)
    thread.start()
    log_lines: list[str] = []
    yield empty_output("Starting scenario evaluation…", interactive=False)

    while True:
        try:
            item = q.get(timeout=0.3)
        except queue_module.Empty:
            if not thread.is_alive():
                break
            yield empty_output("\n".join(log_lines) or "Running…", interactive=False)
            continue

        kind = item[0]
        if kind in ("log", "progress"):
            message = item[1] if kind == "log" else item[2]
            log_lines.append(message)
            yield empty_output("\n".join(log_lines[-80:]), interactive=False)
        elif kind == "done":
            _, df, csv_path = item
            if "evaluation_status" in df:
                scored_mask = df["evaluation_status"] == "scored"
                system_mask = df["evaluation_status"] == "system_error"
                judge_error_mask = df["evaluation_status"] == "judge_error"
            else:
                system_mask = df["system_error"].notna()
                judge_error_mask = df["judge_error"].notna() & ~system_mask
                scored_mask = ~(system_mask | judge_error_mask)
            judged = int(scored_mask.sum())
            system_failures = int(system_mask.sum())
            judge_failures = int(judge_error_mask.sum())
            log_lines.extend([
                "",
                f"Scenario evaluation complete: {judged} scored, "
                f"{system_failures} execution failures, {judge_failures} judge failures "
                f"out of {len(df)} runs.",
            ])
            if system_failures:
                log_lines.append("⚠ Model execution errors (excluded from behavioural scores):")
                for _, row in df[system_mask].head(5).iterrows():
                    log_lines.append(
                        f"- {row['model_name']} · {row['scenario_id']} · "
                        f"rep {row['repetition']}: {row['system_error']}"
                    )
            if judge_failures:
                log_lines.append("⚠ Judge errors:")
                failed = df[judge_error_mask]
                for _, row in failed.head(5).iterrows():
                    log_lines.append(
                        f"- {row['model_name']} · {row['scenario_id']}: "
                        f"{row['judge_error']}"
                    )
                log_lines.append(
                    "The full CSV contains the raw judge response for diagnosis."
                )
            if not system_failures and not judge_failures:
                log_lines.append("✅ All scenario runs were scored successfully.")
            yield (
                "\n".join(log_lines),
                _build_scenario_validity_summary(df),
                _build_scenario_summary(df),
                _build_scenario_run_summary(df),
                _build_scenario_dimension_summary(df),
                _build_scenario_stability_summary(df),
                _build_paired_model_comparisons(df),
                csv_path,
                gr.update(interactive=True),
                research_statistics.checkpoint_summary_table(df),
            )
            return
        elif kind == "error":
            log_lines.append(f"\n❌ Error:\n{item[1]}")
            yield empty_output("\n".join(log_lines))
            return

    yield empty_output("\n".join(log_lines))


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------

def _toggle_source(choice):
    return (
        gr.update(visible=(choice == "Upload file")),
        gr.update(visible=(choice == "Use existing file")),
    )


def _on_files_uploaded(files):
    """Populate dataset picker when multiple files uploaded."""
    if not files:
        return gr.update(choices=[], value=None, visible=False)
    paths = files if isinstance(files, list) else [files]
    names = [Path(p).name for p in paths]
    choices = list(zip(names, paths))
    visible = len(paths) > 1
    return gr.update(choices=choices, value=paths[0], visible=visible)


def _update_audio_warning(input_mode, selected_models, model_types_state, use_whisper):
    """Red HTML warning when Audio mode + non-audio models selected."""
    if input_mode != "Audio" or not selected_models:
        return gr.update(visible=False, value="")
    bad = _non_audio_models(selected_models, model_types_state)
    if not bad:
        return gr.update(visible=False, value="")
    names = ", ".join(bad)
    if use_whisper:
        msg = (
            f'<div style="color:#cc0000;font-weight:bold;padding:8px 12px;'
            f'border:1.5px solid #cc0000;border-radius:6px;background:#fff5f5;">'
            f'⚠ {names} do not support audio input directly. '
            f'Whisper transcription is enabled — audio will be transcribed to text before inference.'
            f'<br>Note: Whisper outputs Traditional Chinese for Mandarin audio. Use the Script Converter tool if you need Simplified.'
            f'</div>'
        )
    else:
        msg = (
            f'<div style="color:#cc0000;font-weight:bold;padding:8px 12px;'
            f'border:1.5px solid #cc0000;border-radius:6px;background:#fff5f5;">'
            f'⚠ {names} do not support audio input and Whisper transcription is disabled — these models will be skipped.'
            f'</div>'
        )
    return gr.update(visible=True, value=msg)


def _toggle_audio_section(input_mode):
    return gr.update(visible=(input_mode == "Audio"))


_VALID_TYPES = {"fireworks", "openai", "gemini", "claude", "hf_local"}


def _load_config_df() -> pd.DataFrame:
    """Read models.yaml and return a DataFrame with columns [name, type, model_id, api_key]."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        rows = [
            {
                "name":     m.get("name", ""),
                "type":     m.get("type", ""),
                "model_id": m.get("model_id", ""),
                "api_key":  m.get("api_key", ""),
            }
            for m in raw.get("models", [])
        ]
    except Exception:
        rows = []
    return pd.DataFrame(rows, columns=["name", "type", "model_id", "api_key"])


def _save_config(df: pd.DataFrame):
    """
    Validate and write the edited DataFrame back to models.yaml.
    Returns (config_status_html, model_select_update).
    """
    try:
        # Drop rows where all three key fields are empty
        df = df.dropna(how="all").reset_index(drop=True)
        df = df[
            (df["name"].astype(str).str.strip() != "") |
            (df["type"].astype(str).str.strip() != "") |
            (df["model_id"].astype(str).str.strip() != "")
        ].reset_index(drop=True)

        errors = []
        for i, row in df.iterrows():
            name     = str(row.get("name", "")).strip()
            typ      = str(row.get("type", "")).strip()
            model_id = str(row.get("model_id", "")).strip()
            if not name:
                errors.append(f"Row {i+1}: name is empty.")
            if not typ:
                errors.append(f"Row {i+1} ({name}): type is empty.")
            elif typ not in _VALID_TYPES:
                errors.append(f"Row {i+1} ({name}): invalid type '{typ}'. Must be one of {sorted(_VALID_TYPES)}.")
            if not model_id:
                errors.append(f"Row {i+1} ({name}): model_id is empty.")
        if errors:
            return _status_err("<br>".join(errors)), gr.update()

        # Preserve non-models sections from existing YAML
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
        except Exception:
            existing = {}

        models_list = []
        for _, row in df.iterrows():
            entry = {
                "name":     str(row.get("name", "")).strip(),
                "type":     str(row.get("type", "")).strip(),
                "model_id": str(row.get("model_id", "")).strip(),
                "api_key":  str(row.get("api_key", "")).strip(),
            }
            models_list.append(entry)

        existing["models"] = models_list
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        new_names = [m["name"] for m in models_list]
        return (
            _status_ok(f"Saved {len(models_list)} model(s) to {CONFIG_PATH}"),
            gr.update(choices=new_names, value=new_names[:1] if new_names else []),
        )
    except Exception as e:
        return _status_err(f"Save failed: {e}"), gr.update()


def _add_model_row(df: pd.DataFrame, name: str, type_: str, model_id: str, api_key: str):
    """Append a new model row to the editor DataFrame."""
    name     = (name or "").strip()
    type_    = (type_ or "").strip()
    model_id = (model_id or "").strip()
    api_key  = (api_key or "").strip()

    if not name or not type_ or not model_id:
        return df, _status_warn("Please fill in Name, Type, and Model ID before adding.")

    new_row = pd.DataFrame([{"name": name, "type": type_, "model_id": model_id, "api_key": api_key}])
    updated = pd.concat([df, new_row], ignore_index=True)
    return updated, _status_ok(f"Added '{name}'. Click Save Config to persist.")


def _convert_dataset(file, direction: str):
    """
    Convert all string values in a dataset file between Traditional and Simplified Chinese.
    Emits gr.Warning if no Chinese characters are detected.
    """
    import json as json_module
    import tempfile
    import zhconv

    if file is None:
        gr.Warning("Please upload a file first.")
        return "Please upload a file first.", None

    target = "zh-hans" if direction == "t2s" else "zh-hant"

    def conv(s: str) -> str:
        return zhconv.convert(s, target) if isinstance(s, str) else s

    def conv_obj(obj):
        if isinstance(obj, dict):
            return {k: conv_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [conv_obj(i) for i in obj]
        if isinstance(obj, str):
            return conv(obj)
        return obj

    src = file if isinstance(file, str) else str(file)
    ext = Path(src).suffix.lower()

    # detect encoding from BOM
    with open(src, "rb") as rb:
        bom = rb.read(4)
    if bom[:2] in (b"\xff\xfe", b"\xfe\xff"):
        enc = "utf-16"
    elif bom[:3] == b"\xef\xbb\xbf":
        enc = "utf-8-sig"
    else:
        enc = "utf-8"

    # ---- Chinese content detection --------------------------------------
    try:
        with open(src, encoding=enc, errors="replace") as f:
            sample = f.read(2000)
        if not _CHINESE_RE.search(sample):
            gr.Warning(
                "No Chinese characters detected in this file. "
                "Conversion has no effect on non-Chinese text."
            )
    except Exception:
        pass  # encoding issue during sample — continue anyway

    suffix   = "_simplified" if direction == "t2s" else "_traditional"
    out_name = Path(src).stem + suffix + ext
    out_path = str(Path(tempfile.gettempdir()) / out_name)

    try:
        if ext == ".jsonl":
            with open(src, encoding=enc) as f:
                lines = [json_module.loads(l) for l in f if l.strip()]
            converted = [conv_obj(l) for l in lines]
            with open(out_path, "w", encoding="utf-8") as f:
                for obj in converted:
                    f.write(json_module.dumps(obj, ensure_ascii=False) + "\n")
            count = len(converted)

        elif ext == ".json":
            with open(src, encoding=enc) as f:
                data = json_module.load(f)
            converted = conv_obj(data)
            with open(out_path, "w", encoding="utf-8") as f:
                json_module.dump(converted, f, ensure_ascii=False, indent=2)
            count = len(converted) if isinstance(converted, list) else 1

        elif ext == ".csv":
            with open(src, encoding=enc, newline="") as f:
                reader = csv_module.DictReader(f)
                rows = list(reader)
                fieldnames = reader.fieldnames or []
            converted = [conv_obj(row) for row in rows]
            with open(out_path, "w", encoding="utf-8", newline="") as f:
                writer = csv_module.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(converted)
            count = len(converted)

        elif ext == ".txt":
            with open(src, encoding=enc) as f:
                lines = f.readlines()
            converted = [conv(l) for l in lines]
            with open(out_path, "w", encoding="utf-8") as f:
                f.writelines(converted)
            count = len([l for l in converted if l.strip()])

        else:
            return f"Unsupported format: {ext}. Use .json / .jsonl / .csv / .txt", None

        label = "Traditional → Simplified" if direction == "t2s" else "Simplified → Traditional"
        return f"Done ({label}): {count} records converted. File ready to download.", out_path

    except Exception as e:
        return f"Error: {e}", None


# ---------------------------------------------------------------------------
# Dataset Cleaner handlers
# ---------------------------------------------------------------------------

def _analyze_dataset(file):
    """Analyse an uploaded dataset and return an HTML quality report."""
    if file is None:
        gr.Warning("Please upload a file first.")
        return gr.update(visible=False, value=""), gr.update(interactive=False)

    src = file if isinstance(file, str) else str(file)
    fname = Path(src).name

    try:
        runner = EvalRunner([], Evaluator())
        pairs = runner.load_dataset(src)
    except Exception as exc:
        return (
            gr.update(visible=True, value=_status_err(f"Parse error: {exc}")),
            gr.update(interactive=False),
        )

    if not pairs:
        return (
            gr.update(visible=True, value=_status_warn("Dataset is empty (0 rows found).")),
            gr.update(interactive=False),
        )

    report = _dc_analyze(pairs)
    total = report["total"]
    clean_count = report["clean_count"]
    summary_lines = report["summary_lines"]
    has_issues = any(len(report["issues"][k]) > 0 for k in report["issues"])

    # Build HTML table
    rows_html = ""
    for label, count, indices in summary_lines:
        colour = "#721c24" if count > 0 else "#155724"
        rows_html += (
            f"<tr><td>{label}</td>"
            f"<td style='text-align:center;color:{colour};font-weight:bold'>{count}</td>"
            f"<td style='color:#555;font-size:0.9em'>{indices}</td></tr>"
        )

    table_html = (
        "<table style='width:100%;border-collapse:collapse;margin:8px 0'>"
        "<thead><tr style='background:#f0f0f0'>"
        "<th style='text-align:left;padding:4px 8px'>Issue</th>"
        "<th style='padding:4px 8px'>Count</th>"
        "<th style='text-align:left;padding:4px 8px'>Row #s (first 10)</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )

    footer = (
        f"<p style='margin:4px 0'>Clean rows: <b>{clean_count}</b> / {total}</p>"
    )

    title = f"Dataset Quality Report — <b>{fname}</b>&nbsp;({total} rows)"

    if not has_issues:
        html = _status_ok(
            f"{title}<br>"
            f"<p style='margin:4px 0'>No issues found. Dataset is ready for evaluation.</p>"
        )
        return gr.update(visible=True, value=html), gr.update(interactive=False)

    html = _status_warn(f"{title}{table_html}{footer}")
    return gr.update(visible=True, value=html), gr.update(interactive=True)


def _clean_dataset(file):
    """Clean the uploaded dataset and return preview + download path."""
    if file is None:
        gr.Warning("Please upload a file first.")
        return (
            gr.update(visible=False, value=""),
            gr.update(visible=False, value=None),
            gr.update(visible=False),
            None,
        )

    src = file if isinstance(file, str) else str(file)
    fname = Path(src).name

    try:
        runner = EvalRunner([], Evaluator())
        pairs = runner.load_dataset(src)
    except Exception as exc:
        return (
            gr.update(visible=True, value=_status_err(f"Parse error: {exc}")),
            gr.update(visible=False, value=None),
            gr.update(visible=False),
            None,
        )

    before_count = len(pairs)
    cleaned_pairs, change_log = _dc_clean(pairs)
    dropped = before_count - len(cleaned_pairs)
    fixed = len(set(e["row"] for e in change_log))

    out_path = _dc_save(cleaned_pairs, src)
    preview_df = _dc_to_df(cleaned_pairs)

    summary = (
        f"Cleaning complete — <b>{fname}</b><br>"
        f"Original rows: <b>{before_count}</b> &nbsp;·&nbsp; "
        f"Rows fixed: <b>{fixed}</b> &nbsp;·&nbsp; "
        f"Rows dropped (empty/duplicate): <b>{dropped}</b> &nbsp;·&nbsp; "
        f"Retained: <b>{len(cleaned_pairs)}</b>"
    )
    html = _status_ok(summary)

    return (
        gr.update(visible=True, value=html),
        gr.update(visible=True, value=preview_df),
        gr.update(visible=True),
        out_path,
    )


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------


APP_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 0 28px 48px !important;
}

/* Header */
.app-hero {
    padding: 28px 0 20px;
    margin-bottom: 22px;
    border-bottom: 1px solid var(--border-color-primary);
}
.app-hero h1 {
    margin: 0 0 6px !important;
    font-size: 1.8rem !important;
    font-weight: 680 !important;
    letter-spacing: -0.03em;
}
.app-hero p {
    margin: 0 !important;
    max-width: 800px;
    opacity: .66;
    font-size: .96rem;
    line-height: 1.55;
}

/* Main research workflow */
.step-card, .run-card {
    border: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 10px 0 22px !important;
}
.step-title {
    margin-bottom: 14px;
    padding-bottom: 9px;
    border-bottom: 1px solid var(--border-color-primary);
}
.step-title h3 {
    margin: 0 0 3px !important;
    font-size: 1.02rem !important;
    font-weight: 660 !important;
    letter-spacing: -0.012em;
}
.step-title p {
    margin: 0 !important;
    opacity: .61;
    font-size: .86rem;
    line-height: 1.45;
}

/* Audio is visually subordinate to modality, not another full card */
.audio-card {
    border: 0 !important;
    border-left: 3px solid var(--primary-500) !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 10px 4px 10px 18px !important;
    margin: 10px 0 4px;
}
.audio-summary {
    margin: 8px 0 10px;
}

/* Parameter area */
.run-card {
    border-top: 1px solid var(--border-color-primary) !important;
    margin-top: 12px;
    padding-top: 22px !important;
}
.settings-row {
    gap: 12px !important;
    margin: 4px 0 16px !important;
}
.run-button {
    min-height: 48px !important;
    font-size: 1rem !important;
    font-weight: 670 !important;
    border-radius: 7px !important;
}

/* Output */
.output-section {
    margin-top: 14px !important;
    padding-top: 10px !important;
    border-top: 1px solid var(--border-color-primary);
}
.output-heading h2 {
    margin-bottom: 3px !important;
    font-size: 1.2rem !important;
}
.output-heading p {
    margin-top: 0 !important;
    opacity: .6;
    font-size: .86rem;
}

.output-section > div {
    gap: 8px !important;
}
.output-heading {
    margin: 0 !important;
    padding: 2px 0 4px !important;
}
.output-heading h2 {
    margin: 0 0 2px !important;
}
.output-heading p {
    margin: 0 !important;
}

#evaluation-log textarea {
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
    font-size: .84rem !important;
    line-height: 1.5 !important;
}
#research-results {
    margin-top: 8px;
}
#research-results table {
    font-size: .9rem;
}
footer { opacity: .5; }
"""


_ensure_claude_model_config()
model_names    = _read_model_names()
model_types    = _read_model_types()
existing_files = _existing_datasets()
existing_scenario_files = _existing_scenarios()

# Resolve audio for the initially selected existing dataset immediately.
# Gradio does not fire .change() just because a Dropdown starts with a value.
_initial_dataset = existing_files[0] if existing_files else None
_initial_bertscore_lock = (
    _dataset_requires_bertscore_skip(_initial_dataset)
    if _initial_dataset else False
)
_initial_audio_folder, _initial_audio_files, _initial_audio_total = (
    _match_dataset_audio(_initial_dataset) if _initial_dataset else (None, [], 0)
)

with gr.Blocks(title="LLM Evaluation", theme=gr.themes.Soft(), css=APP_CSS) as demo:

    model_types_state = gr.State(model_types)

    with gr.Tabs():

        # ================================================================
        # Tab 1: Evaluation
        # ================================================================
        with gr.Tab("Evaluation"):

            gr.HTML(
                """
                <div class="app-hero">
                    <h1>LLM Evaluation</h1>
                    <p>Comparative evaluation of language models across text and audio question-answering datasets, with consistent scoring and reproducible run parameters.</p>
                </div>
                """
            )

            with gr.Accordion("Environment & API configuration", open=False):
                gr.Markdown(
                    """
                    Set the required environment variables before launching the app:
                    `FIREWORKS_API_KEY` · `OPENAI_API_KEY` · `GOOGLE_API_KEY` · `ANTHROPIC_API_KEY`

                    `OPENAI_API_KEY` is also used for Whisper transcription.
                    """
                )

            # ---- Step 1: Dataset + Models --------------------------------
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes=["step-card"]):
                    gr.HTML(
                        '<div class="step-title"><h3>1. Evaluation dataset</h3>'
                        '<p>Select the dataset that defines the evaluation questions and reference answers.</p></div>'
                    )
                    dataset_source = gr.Radio(
                        choices=["Use existing file", "Upload file"],
                        value="Use existing file" if existing_files else "Upload file",
                        label="Dataset source",
                    )
                    with gr.Group(visible=not existing_files) as upload_group:
                        upload_box = gr.File(
                            label="Upload dataset",
                            file_types=[".json", ".jsonl", ".csv", ".txt"],
                            file_count="multiple",
                        )
                        dataset_picker = gr.Dropdown(
                            choices=[],
                            label="Dataset",
                            visible=False,
                        )
                    existing_box = gr.Dropdown(
                        choices=existing_files,
                        value=existing_files[0] if existing_files else None,
                        label="Dataset",
                        info="Files found in data/",
                        visible=bool(existing_files),
                    )
                    dataset_status = gr.HTML(visible=False)

                    dataset_source.change(
                        _toggle_source,
                        inputs=dataset_source,
                        outputs=[upload_group, existing_box],
                    )
                    upload_box.upload(
                        _on_files_uploaded,
                        inputs=upload_box,
                        outputs=dataset_picker,
                    )
                    dataset_picker.change(
                        _validate_dataset_file,
                        inputs=dataset_picker,
                        outputs=dataset_status,
                    )
                    existing_box.change(
                        _validate_dataset_file,
                        inputs=existing_box,
                        outputs=dataset_status,
                    )

                with gr.Column(scale=1, elem_classes=["step-card"]):
                    gr.HTML(
                        '<div class="step-title"><h3>2. Model cohort</h3>'
                        '<p>Select the models included in this experimental run.</p></div>'
                    )
                    if model_names:
                        model_select = gr.CheckboxGroup(
                            choices=model_names,
                            value=[model_names[0]],
                            label="Models included in evaluation",
                        )
                    else:
                        gr.Markdown(
                            "_No models found in `config/models.yaml`. Add a model in the Models tab._"
                        )
                        model_select = gr.CheckboxGroup(choices=[], label="Models")

            # ---- Step 3: Input -------------------------------------------
            with gr.Group(elem_classes=["step-card"]):
                gr.HTML(
                    '<div class="step-title"><h3>3. Evaluation modality</h3>'
                    '<p>Choose text or speech input. Audio recordings are matched to the dataset automatically.</p></div>'
                )
                input_mode = gr.Radio(
                    choices=["Text", "Audio"],
                    value="Text",
                    label="Modality",
                )
                audio_warning = gr.HTML(visible=False)

                with gr.Group(visible=False, elem_classes=["audio-card"]) as audio_section:
                    gr.Markdown("### Audio sample selection")
                    gr.Markdown(
                        "Recordings are matched automatically to the selected dataset. "
                        "All valid matched samples are included by default."
                    )

                    voice_folder_box = gr.State(_initial_audio_folder or "")
                    audio_match_status = gr.HTML(
                        value=(
                            _audio_auto_status(
                                _initial_audio_folder,
                                _initial_audio_files,
                                _initial_audio_total,
                            )
                            if _initial_dataset
                            else _status_warn("Select a dataset with audio_file values to match audio automatically.")
                        )
                    )
                    initial_voice_files = _initial_audio_files

                    audio_selection_preview = gr.HTML(
                        value=_audio_selection_preview(initial_voice_files),
                        elem_classes=["audio-summary"],
                    )

                    with gr.Row():
                        select_all_audio_btn = gr.Button("Select all", size="sm")
                        clear_audio_btn = gr.Button("Clear", size="sm")

                    with gr.Accordion("Review / modify audio samples", open=False):
                        voice_files_box = gr.Dropdown(
                            choices=initial_voice_files,
                            value=initial_voice_files,
                            multiselect=True,
                            label="Recordings",
                            info="Only recordings matched to the active dataset are available.",
                            filterable=True,
                        )

                    use_whisper = gr.Checkbox(
                        value=True,
                        label="Enable Whisper fallback for text-only models",
                        info="Audio → Whisper transcription → text model",
                    )

                    with gr.Accordion("Audio notes", open=False):
                        gr.Markdown(
                            "For Mandarin audio, Whisper may return Traditional Chinese. "
                            "Use **Tools → Script Converter** when Simplified Chinese is required."
                        )

                    voice_files_box.change(
                        fn=_audio_selection_preview,
                        inputs=voice_files_box,
                        outputs=audio_selection_preview,
                    )
                    select_all_audio_btn.click(
                        fn=lambda source, uploaded, existing: _select_all_for_dataset(
                            uploaded if source == "Upload file" else existing
                        ),
                        inputs=[dataset_source, dataset_picker, existing_box],
                        outputs=[voice_files_box, audio_selection_preview],
                    )
                    clear_audio_btn.click(
                        fn=_clear_voice_files,
                        outputs=[voice_files_box, audio_selection_preview],
                    )

                    existing_box.change(
                        fn=_auto_match_audio,
                        inputs=existing_box,
                        outputs=[
                            voice_folder_box,
                            voice_files_box,
                            audio_selection_preview,
                            audio_match_status,
                        ],
                    )
                    dataset_picker.change(
                        fn=_auto_match_audio,
                        inputs=dataset_picker,
                        outputs=[
                            voice_folder_box,
                            voice_files_box,
                            audio_selection_preview,
                            audio_match_status,
                        ],
                    )

            input_mode.change(_toggle_audio_section, inputs=input_mode, outputs=audio_section)
            for _comp in [input_mode, model_select]:
                _comp.change(
                    _update_audio_warning,
                    inputs=[input_mode, model_select, model_types_state, use_whisper],
                    outputs=audio_warning,
                )
            use_whisper.change(
                _update_audio_warning,
                inputs=[input_mode, model_select, model_types_state, use_whisper],
                outputs=audio_warning,
            )

            # ---- Step 4: Settings + Run ---------------------------------
            with gr.Group(elem_classes=["run-card"]):
                gr.HTML(
                    '<div class="step-title"><h3>4. Evaluation parameters</h3>'
                    '<p>Set generation, sample-limit, and scoring parameters before starting the run.</p></div>'
                )
                with gr.Row(elem_classes=["settings-row"]):
                    max_tokens_box = gr.Number(
                        value=512, label="Max tokens", precision=0,
                        minimum=1, maximum=4096,
                        info="Provider/model output limits are enforced automatically.",
                    )
                    temperature_box = gr.Number(
                        value=0.0, label="Temperature", precision=2, minimum=0.0
                    )
                    limit_box = gr.Textbox(
                        value="",
                        label="Question limit",
                        placeholder="Blank = all questions",
                    )
                    no_bertscore_box = gr.Checkbox(
                        value=True,
                        label="Skip BERTScore",
                        info="Recommended for faster runs",
                        interactive=not _initial_bertscore_lock,
                    )

                # Keep the metric choice aligned with the selected dataset.
                # Datasets with missing references force BERTScore off and lock
                # the checkbox; reference-answer datasets leave it editable.
                dataset_picker.change(
                    _auto_skip_bertscore_for_dataset,
                    inputs=dataset_picker,
                    outputs=no_bertscore_box,
                )
                existing_box.change(
                    _auto_skip_bertscore_for_dataset,
                    inputs=existing_box,
                    outputs=no_bertscore_box,
                )
                dataset_source.change(
                    _auto_skip_bertscore_for_source,
                    inputs=[dataset_source, dataset_picker, existing_box],
                    outputs=no_bertscore_box,
                )

                run_btn = gr.Button(
                    "Run evaluation",
                    variant="primary",
                    size="lg",
                    elem_classes=["run-button"],
                )

            with gr.Group(elem_classes=["output-section"]):
                gr.HTML(
                    '<div class="output-heading"><h2>Execution</h2>'
                    '<p>Live run status and model-processing messages.</p></div>'
                )
                log_box = gr.Textbox(
                    label="Run log",
                    lines=8,
                    max_lines=15,
                    interactive=False,
                    elem_id="evaluation-log",
                )

            with gr.Group(elem_classes=["output-section"]):
                gr.HTML(
                    '<div class="output-heading"><h2>Evaluation results</h2>'
                    '<p>Mean scores across successfully evaluated samples.</p></div>'
                )
                results_table = gr.Dataframe(
                    label="Comparative model summary",
                    interactive=False,
                    wrap=True,
                    elem_id="research-results",
                )
                download_btn = gr.File(
                    label="Export full results (CSV)",
                    interactive=False,
                )

            run_btn.click(
                fn=run_evaluation,
                inputs=[
                    dataset_source, dataset_picker, existing_box,
                    model_select, max_tokens_box, temperature_box,
                    limit_box, no_bertscore_box,
                    input_mode, voice_folder_box, voice_files_box, use_whisper,
                ],
                outputs=[log_box, results_table, download_btn, run_btn],
            )

        # ================================================================
        # Tab 2: Scenario Benchmark
        # ================================================================
        with gr.Tab("Scenario Benchmark"):
            gr.Markdown(
                """
                ## Scripted multi-turn scenario evaluation

                Run any CSV, JSON, or JSONL scenario that defines a sequence of
                **user messages** and **expected assistant actions**. The assistant
                receives the complete dialogue history, while a separately selected
                judge scores the finished transcript.

                This evaluates semantic multi-turn performance. Streaming audio-only
                measures such as interruption handling and speech naturalness remain
                `N/A` until a real-time duplex audio adapter is used.
                """
            )

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    scenario_source = gr.Radio(
                        choices=["Use existing files", "Upload files"],
                        value=("Use existing files" if existing_scenario_files else "Upload files"),
                        label="Scenario source",
                    )
                    with gr.Group(visible=not existing_scenario_files) as scenario_upload_group:
                        scenario_uploads = gr.File(
                            label="Scenario files",
                            file_types=[".csv", ".json", ".jsonl"],
                            file_count="multiple",
                        )
                    with gr.Group(visible=bool(existing_scenario_files)) as scenario_existing_group:
                        scenario_existing = gr.CheckboxGroup(
                            choices=existing_scenario_files,
                            value=existing_scenario_files,
                            label="Scenarios",
                            info="Valid scenario files found under data/multi_turn/",
                        )

                with gr.Column(scale=1):
                    scenario_models = gr.CheckboxGroup(
                        choices=model_names,
                        value=[model_names[0]] if model_names else [],
                        label="Models to evaluate",
                    )
                    scenario_judge = gr.Dropdown(
                        choices=model_names,
                        value=(
                            model_names[1] if len(model_names) > 1
                            else (model_names[0] if model_names else None)
                        ),
                        label="Judge model",
                        info="Must be a capable model that is not one of the examinees.",
                    )

            scenario_source.change(
                fn=lambda choice: (
                    gr.update(visible=(choice == "Upload files")),
                    gr.update(visible=(choice == "Use existing files")),
                ),
                inputs=scenario_source,
                outputs=[scenario_upload_group, scenario_existing_group],
            )

            with gr.Row():
                scenario_max_tokens = gr.Number(
                    value=4096, precision=0, minimum=64, maximum=16384,
                    label="Response token limit",
                    info=(
                        "Applied per assistant turn; provider/model caps are enforced. "
                        "Thinking models may require a larger total output budget."
                    ),
                )
                scenario_judge_max_tokens = gr.Number(
                    value=8192, precision=0, minimum=512, maximum=16384,
                    label="Judge token limit",
                    info="Separate deterministic budget for the structured evaluation JSON.",
                )
                scenario_temperature = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.0, step=0.1,
                    label="Temperature",
                    info="Use 0 for the primary reproducible experiment.",
                )
                scenario_repetitions = gr.Number(
                    value=3, precision=0, minimum=3, label="Repetitions",
                    info="Repeated runs estimate stability; they are not independent scenarios.",
                )

            scenario_run_btn = gr.Button(
                "Run scenario evaluation", variant="primary", size="lg"
            )
            scenario_log = gr.Textbox(
                label="Run log", lines=9, max_lines=18, interactive=False
            )
            gr.Markdown(
                "Results are separated into **run validity**, **primary outcomes**, "
                "and **diagnostic measures**. Behavioural scores use only validly "
                "executed and judged runs; end-to-end success retains execution failures."
            )
            scenario_validity_results = gr.Dataframe(
                label="1. Run validity", interactive=False, wrap=True
            )
            scenario_results = gr.Dataframe(
                label="2. Primary outcomes", interactive=False, wrap=True
            )
            with gr.Accordion("3. Per-run diagnostics", open=False):
                scenario_run_results = gr.Dataframe(
                    label="Individual runs — grouped by model, then scenario, then repetition", interactive=False, wrap=False
                )
            with gr.Accordion("4. Behavioural dimensions (1–5)", open=False):
                scenario_dimension_results = gr.Dataframe(
                    label="Ordinal score summaries",
                    interactive=False,
                    wrap=True,
                )
            with gr.Accordion("5. Repeated-run stability", open=False):
                scenario_stability_results = gr.Dataframe(
                    label="Within-scenario stability",
                    interactive=False,
                    wrap=True,
                )
            with gr.Accordion("6. Model comparisons on the same scenarios (descriptive only)", open=False):
                gr.Markdown("Each model is first averaged across its valid repetitions **within each scenario**. Only scenarios with scores for both models are included. The table then averages these scenario scores equally. The difference is Model A minus Model B, in **percentage points**; it is an observed difference, not a significance test or a prediction. Different numbers of valid repetitions can contribute to the two model averages.")
                scenario_comparison_results = gr.Dataframe(
                    label="Observed model differences across the same scenarios",
                    interactive=False,
                    wrap=True,
                )
            with gr.Accordion("7. Individual checkpoint consistency across repetitions", open=False):
                gr.Markdown(
                    "**One row per checkpoint for each model and scenario.** "
                    "The 'Met / valid repetitions' column counts how often that exact requirement "
                    "was satisfied when the same scenario was repeated. For example, 2 / 3 means "
                    "the checkpoint was met twice in three valid runs (66.67%). "
                    "'Outcome varies' is Yes if the checkpoint was met in some runs and missed in others. "
                    "A No can mean either consistently met or consistently missed; check the count. "
                    "Failed or unjudged runs are excluded from the checkpoint denominator. "
                    "Use Section 3 for the overall result of each run; this table reveals "
                    "*which specific requirements* are consistently met or missed."
                )
                scenario_checkpoint_summary = gr.Dataframe(
                    label="Checkpoint success by model and scenario (valid repetitions only)",
                    interactive=False, wrap=True,
                )
            scenario_download = gr.File(
                label="Export full scenario results (CSV)", interactive=False
            )

            scenario_run_btn.click(
                fn=run_scenario_evaluation,
                inputs=[
                    scenario_source, scenario_uploads, scenario_existing,
                    scenario_models, scenario_judge, scenario_max_tokens,
                    scenario_judge_max_tokens, scenario_temperature,
                    scenario_repetitions,
                ],
                outputs=[
                    scenario_log, scenario_validity_results, scenario_results,
                    scenario_run_results, scenario_dimension_results,
                    scenario_stability_results, scenario_comparison_results,
                    scenario_download, scenario_run_btn,
                    scenario_checkpoint_summary,
                ],
            )

        # ================================================================
        # Tab 3: Noise Robustness
        # ================================================================
        with gr.Tab("Noise Robustness"):
            gr.Markdown(
                """
                ## Noise Robustness Evaluation
                Tests how model accuracy degrades as audio quality decreases.

                **How it works:**
                1. Upload the same audio dataset used in the Evaluation tab.
                2. Select noise levels — the system automatically adds calibrated white noise
                   at each SNR level (Clean → 20 dB → 10 dB → 5 dB).
                3. Runs full evaluation at every noise level and reports metric degradation.

                **Metric to watch:** BLEU and Token-F1 should decrease as SNR drops.
                A model that degrades slowly is more **noise-robust**.
                """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Audio Dataset")
                    nr_audio_csv = gr.File(
                        label="Audio metadata CSV (same format as Evaluation tab)",
                        file_types=[".csv"],
                        file_count="single",
                    )
                    nr_audio_files = gr.File(
                        label="Audio files (.wav / .mp3 / .m4a)",
                        file_types=[".wav", ".mp3", ".m4a", ".ogg", ".flac"],
                        file_count="multiple",
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### Models & Settings")
                    nr_model_select = gr.CheckboxGroup(
                        choices=model_names,
                        value=[model_names[0]] if model_names else [],
                        label="Models to evaluate",
                    )
                    nr_noise_levels = gr.CheckboxGroup(
                        choices=["Clean", "SNR 20 dB", "SNR 10 dB", "SNR 5 dB"],
                        value=["Clean", "SNR 20 dB", "SNR 10 dB", "SNR 5 dB"],
                        label="Noise levels to test",
                    )
                    nr_use_whisper = gr.Checkbox(
                        value=True,
                        label="Enable Whisper transcription for non-audio models",
                    )
                    nr_max_tokens = gr.Number(
                        value=512, label="Max tokens", precision=0,
                        minimum=1, maximum=4096,
                    )
                    nr_temperature = gr.Number(value=0.0, label="Temperature", precision=2, minimum=0.0)

            nr_run_btn = gr.Button("▶  Run Noise Robustness Test", variant="primary", size="lg")

            gr.Markdown("### Progress")
            nr_log_box = gr.Textbox(label="Log", lines=12, max_lines=12, interactive=False)

            gr.Markdown("### Results — Metric Degradation by Noise Level")
            nr_results_table = gr.Dataframe(
                label="Mean BLEU / Token-F1 per model × noise level",
                interactive=False,
                wrap=True,
            )
            nr_download_btn = gr.File(label="Download full results CSV", interactive=False)

            def _run_noise_robustness(
                nr_csv, nr_files, nr_models, nr_levels, nr_whisper,
                nr_max_tok, nr_temp,
            ):
                import queue as _queue
                import tempfile
                import threading as _threading
                import traceback as _traceback

                if not nr_csv:
                    yield "Please upload an audio metadata CSV.", None, None
                    return
                if not nr_files:
                    yield "Please upload audio files.", None, None
                    return
                if not nr_models:
                    yield "Please select at least one model.", None, None
                    return
                if not nr_levels:
                    yield "Please select at least one noise level.", None, None
                    return

                # Map label → snr_db (None = clean)
                _LEVEL_MAP = {
                    "Clean": None,
                    "SNR 20 dB": 20.0,
                    "SNR 10 dB": 10.0,
                    "SNR 5 dB":  5.0,
                }

                q = _queue.Queue()

                def _thread():
                    try:
                        from src.llm_eval.models import load_models_from_config
                        from src.llm_eval.metrics.evaluator import Evaluator
                        from src.llm_eval.runner import EvalRunner
                        from src.llm_eval.noise_augment import generate_noise_variants
                        from src.llm_eval.output.exporter import export_csv

                        all_models = load_models_from_config(CONFIG_PATH)
                        models = [m for m in all_models if m.name in nr_models]
                        if not models:
                            q.put(("error", "None of the selected models found in config."))
                            return

                        applied_token_caps = {}
                        for m in models:
                            if hasattr(m, "configure_generation"):
                                applied = m.configure_generation(
                                    int(nr_max_tok), float(nr_temp)
                                )
                            else:
                                applied = min(max(1, int(nr_max_tok)), 4096)
                                m.config.max_tokens = applied
                                m.config.temperature = float(nr_temp)
                            applied_token_caps[m.name] = applied

                        uploads = nr_files if isinstance(nr_files, list) else [nr_files]
                        audio_files_map = {Path(p).name: p for p in uploads if p}

                        whisper_fn = None
                        if nr_whisper:
                            try:
                                from src.llm_eval.transcriber import make_whisper_transcriber
                                whisper_fn = make_whisper_transcriber()
                                q.put(("log", "Whisper transcriber ready."))
                            except Exception as e:
                                q.put(("log", f"⚠ Whisper init failed: {e}"))

                        evaluator = Evaluator()
                        runner = EvalRunner(models, evaluator)
                        dataset = runner.load_audio_dataset(str(nr_csv), audio_files_map)
                        q.put(("log", f"Dataset: {len(dataset)} audio files loaded."))
                        q.put(("log", f"Token caps: {applied_token_caps}"))

                        noise_dir = str(Path(tempfile.gettempdir()) / "llm_eval_noise")
                        all_dfs = []

                        for level_label in nr_levels:
                            snr = _LEVEL_MAP.get(level_label)
                            q.put(("log", f"\n--- Testing noise level: {level_label} ---"))

                            # Build dataset with noisy audio
                            if snr is None:
                                level_dataset = dataset
                            else:
                                from src.llm_eval.runner import QAPair
                                level_dataset = []
                                for qa in dataset:
                                    if qa.audio_path:
                                        try:
                                            variants = generate_noise_variants(
                                                qa.audio_path, [snr], noise_dir
                                            )
                                            noisy_path = variants.get(f"snr_{int(snr)}", qa.audio_path)
                                        except Exception:
                                            noisy_path = qa.audio_path
                                        level_dataset.append(QAPair(
                                            question=qa.question,
                                            reference_answer=qa.reference_answer,
                                            audio_path=noisy_path,
                                        ))
                                    else:
                                        level_dataset.append(qa)

                            def _progress(frac, desc, ll=level_label):
                                q.put(("log", f"  [{ll}] {desc}"))

                            df = runner.run(
                                level_dataset,
                                use_bertscore=False,
                                input_mode="audio",
                                whisper_fn=whisper_fn,
                                on_progress=_progress,
                                noise_level_db=snr,
                            )
                            df["noise_level"] = level_label
                            all_dfs.append(df)

                        combined = pd.concat(all_dfs, ignore_index=True)

                        os.makedirs(RESULTS_DIR, exist_ok=True)
                        csv_path = os.path.join(RESULTS_DIR, "noise_robustness_results.csv")
                        combined.to_csv(csv_path, index=False)

                        # Build pivot summary
                        pivot_cols = [c for c in ("bleu", "f1", "rouge1", "latency_seconds") if c in combined.columns]
                        summary = (
                            combined[combined["error"].isna()]
                            .groupby(["model_name", "noise_level"])[pivot_cols]
                            .mean()
                            .round(4)
                            .reset_index()
                            .rename(columns={
                                "model_name":      "Model",
                                "noise_level":     "Noise Level",
                                "bleu":            "BLEU",
                                "f1":              "Token-F1",
                                "rouge1":          "ROUGE-1",
                                "latency_seconds": "Latency(s)",
                            })
                        )
                        q.put(("done", summary, csv_path))
                    except Exception:
                        q.put(("error", _traceback.format_exc()))

                t = _threading.Thread(target=_thread, daemon=True)
                t.start()

                log_lines = []
                yield "Starting noise robustness test…", None, None

                while True:
                    try:
                        item = q.get(timeout=0.3)
                    except _queue.Empty:
                        if not t.is_alive():
                            break
                        yield "\n".join(log_lines) or "Running…", None, None
                        continue

                    kind = item[0]
                    if kind == "log":
                        log_lines.append(item[1])
                        yield "\n".join(log_lines[-80:]), None, None
                    elif kind == "done":
                        _, summary_df, csv_path = item
                        log_lines.append("\n✅  Noise robustness test complete!")
                        yield "\n".join(log_lines), summary_df, csv_path
                        return
                    elif kind == "error":
                        log_lines.append(f"\n❌  Error:\n{item[1]}")
                        yield "\n".join(log_lines), None, None
                        return

                yield "\n".join(log_lines), None, None

            nr_run_btn.click(
                fn=_run_noise_robustness,
                inputs=[
                    nr_audio_csv, nr_audio_files,
                    nr_model_select, nr_noise_levels, nr_use_whisper,
                    nr_max_tokens, nr_temperature,
                ],
                outputs=[nr_log_box, nr_results_table, nr_download_btn],
            )

        # ================================================================
        # Tab 3: Tools
        # ================================================================
        with gr.Tab("Tools"):
            with gr.Tabs():

                # ---- Sub-tab 1: Model Config --------------------------------
                with gr.Tab("Model Config"):
                    gr.Markdown("## Model Configuration Editor")
                    gr.Markdown(
                        "Edit model names, types, IDs, and API keys below. "
                        "Click **Save Config** to write changes to `config/models.yaml`. "
                        "The model list in the Evaluation tab will refresh automatically."
                    )

                    config_df = gr.Dataframe(
                        value=_load_config_df(),
                        headers=["name", "type", "model_id", "api_key"],
                        datatype=["str", "str", "str", "str"],
                        col_count=(4, "fixed"),
                        interactive=True,
                        label="Models (edit cells directly)",
                    )

                    with gr.Accordion("Add New Model", open=False):
                        with gr.Row():
                            new_name     = gr.Textbox(label="Name", placeholder="e.g. My-Qwen")
                            new_type     = gr.Dropdown(
                                choices=sorted(_VALID_TYPES),
                                label="Type",
                                value="fireworks",
                            )
                            new_model_id = gr.Textbox(
                                label="Model ID",
                                placeholder="e.g. claude-sonnet-5",
                            )
                            new_api_key  = gr.Textbox(
                                label="API Key",
                                placeholder="e.g. ${ANTHROPIC_API_KEY}",
                            )
                        add_model_btn = gr.Button("+ Add Model", variant="secondary")

                    with gr.Row():
                        save_config_btn   = gr.Button("Save Config", variant="primary")
                        reload_config_btn = gr.Button("Reload from File")

                    config_status = gr.HTML(visible=False)

                    add_model_btn.click(
                        _add_model_row,
                        inputs=[config_df, new_name, new_type, new_model_id, new_api_key],
                        outputs=[config_df, config_status],
                    ).then(lambda: gr.update(visible=True), outputs=config_status)

                    save_config_btn.click(
                        _save_config,
                        inputs=config_df,
                        outputs=[config_status, model_select],
                    ).then(lambda: gr.update(visible=True), outputs=config_status)

                    reload_config_btn.click(
                        _load_config_df,
                        outputs=config_df,
                    )

                # ---- Sub-tab 2: Script Converter ----------------------------
                with gr.Tab("Script Converter"):
                    gr.Markdown(
                        """
                        ## Chinese Script Converter
                        Upload a dataset file (.json / .jsonl / .csv / .txt) to convert **all text fields**
                        between Traditional and Simplified Chinese. The converted file can be downloaded and
                        used directly in the Evaluation tab.

                        > Useful when Whisper transcribes Mandarin audio as Traditional Chinese — convert to Simplified before evaluation.
                        """
                    )

                    conv_file = gr.File(
                        label="Upload Dataset",
                        file_types=[".json", ".jsonl", ".csv", ".txt"],
                    )

                    with gr.Row():
                        t2s_btn = gr.Button("Traditional → Simplified", variant="primary")
                        s2t_btn = gr.Button("Simplified → Traditional")

                    conv_status   = gr.Textbox(label="Status", interactive=False, lines=2)
                    conv_download = gr.File(label="Download Converted Dataset", interactive=False)

                    t2s_btn.click(
                        lambda f: _convert_dataset(f, "t2s"),
                        inputs=conv_file,
                        outputs=[conv_status, conv_download],
                    )
                    s2t_btn.click(
                        lambda f: _convert_dataset(f, "s2t"),
                        inputs=conv_file,
                        outputs=[conv_status, conv_download],
                    )

                    gr.Markdown(
                        """
                        ---
                        **Notes**
                        - Supports .json / .jsonl / .csv / .txt — all string fields are converted.
                        - Output is UTF-8. Filename gets a `_simplified` or `_traditional` suffix.
                        - Based on the `zhconv` dictionary. Proper nouns may need manual review.
                        """
                    )

                with gr.Tab("Dataset Cleaner"):
                    gr.Markdown(
                        """
                        ## Dataset Quality Manager
                        Upload a dataset to **detect issues** (garbled characters, missing fields, meaningless
                        symbols, excess whitespace, duplicates) and optionally **auto-clean** it before evaluation.
                        """
                    )

                    cleaner_upload = gr.File(
                        label="Upload Dataset",
                        file_types=[".json", ".jsonl", ".csv", ".txt"],
                    )

                    with gr.Row():
                        analyze_btn = gr.Button("🔍 Analyze", variant="primary")
                        clean_btn   = gr.Button("🧹 Clean & Fix", interactive=False)

                    cleaner_status = gr.HTML(visible=False)

                    cleaner_preview_md = gr.Markdown("### Cleaned Dataset Preview", visible=False)
                    cleaner_preview    = gr.Dataframe(
                        label="Cleaned rows (scroll to inspect)",
                        visible=False,
                        wrap=True,
                        max_height=400,
                    )
                    cleaner_download = gr.File(
                        label="Download Cleaned Dataset",
                        interactive=False,
                        visible=False,
                    )

                    # ── Analyze ──────────────────────────────────────────
                    analyze_btn.click(
                        _analyze_dataset,
                        inputs=cleaner_upload,
                        outputs=[cleaner_status, clean_btn],
                    )

                    # ── Clean & Fix ───────────────────────────────────────
                    clean_btn.click(
                        _clean_dataset,
                        inputs=cleaner_upload,
                        outputs=[cleaner_status, cleaner_preview, cleaner_preview_md, cleaner_download],
                    )

                    # Reset preview visibility when a new file is uploaded
                    cleaner_upload.change(
                        lambda _: (
                            gr.update(visible=False, value=""),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(interactive=False),
                            None,
                        ),
                        inputs=cleaner_upload,
                        outputs=[cleaner_status, cleaner_preview, cleaner_preview_md, clean_btn, cleaner_download],
                    )

                    gr.Markdown(
                        """
                        ---
                        **Checks performed**
                        | Issue | Detection rule |
                        |---|---|
                        | Garbled characters | Unicode replacement char `\\ufffd`, null bytes, control characters |
                        | Meaningless content | Non-empty but contains no alphanumeric characters |
                        | Missing question | `question` field is empty |
                        | Missing reference | `reference_answer` field is empty |
                        | Excess whitespace | Leading/trailing spaces or internal runs of 2+ spaces |
                        | Duplicate rows | Identical (question, reference) pair appearing more than once |

                        **Cleaning applied**
                        - Remove null bytes and control characters
                        - Remove Unicode replacement chars (`\\ufffd`)
                        - Strip leading/trailing whitespace
                        - Collapse internal whitespace runs to single space
                        - Drop rows where both fields are empty after cleaning
                        - Deduplicate — keep first occurrence of each row
                        """
                    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=6006,
        share=False,
    )
