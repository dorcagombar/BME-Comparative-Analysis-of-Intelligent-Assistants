"""
LLM Evaluation Web UI  (Gradio)

Launch:
    cd interface
    python app.py

Then open http://localhost:6006 in your browser.

Required environment variables (set before launching):
    FIREWORKS_API_KEY   — Fireworks AI key (for Qwen2-7B, Qwen2.5-14B, Mistral-7B)
    OPENAI_API_KEY      — OpenAI key (for GPT-4o and Whisper transcription)
    GOOGLE_API_KEY      — Google AI key (for Gemini-2.5-Pro)
    ANTHROPIC_API_KEY   — Anthropic key (for Claude)
"""

import csv as csv_module
import os
import queue as queue_module
import re
import sys

from dotenv import load_dotenv
load_dotenv()  # loads .env from the current working directory
import threading
import traceback
from pathlib import Path

import gradio as gr
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from llm_eval.dataset_cleaner import (
    analyze as _dc_analyze,
    clean as _dc_clean,
    pairs_to_dataframe as _dc_to_df,
    save_cleaned as _dc_save,
)
from llm_eval.metrics.evaluator import Evaluator
from llm_eval.models import load_models_from_config
from llm_eval.output.exporter import export_csv, export_summary_csv
from llm_eval.runner import (
    EvalRunner,
    _AUDIO_FILE_ALIASES,
    _QUESTION_ALIASES,
    _REFERENCE_ALIASES,
    _find_col,
)

CONFIG_PATH = "config/models.yaml"
RESULTS_DIR = "results"
DATA_DIR    = "data"

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
    return sorted(str(f) for f in p.iterdir() if f.suffix in exts)


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
    audio_csv_file,
    audio_uploads,
    use_whisper,
):
    # ---- resolve dataset path -------------------------------------------
    if dataset_source == "Upload file":
        dataset_path = audio_csv_file if input_mode == "Audio" else selected_dataset
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

    use_bertscore = not no_bertscore

    audio_files_map: dict = {}
    if input_mode == "Audio" and audio_uploads:
        uploads = audio_uploads if isinstance(audio_uploads, list) else [audio_uploads]
        audio_files_map = {Path(p).name: p for p in uploads if p}

    # ---- run in background thread ---------------------------------------
    q: queue_module.Queue = queue_module.Queue()

    def _thread():
        try:
            all_models = load_models_from_config(CONFIG_PATH)
            models = [m for m in all_models if m.name in selected_models]
            if not models:
                q.put(("error", "None of the selected models were found in config."))
                return

            for m in models:
                m.config.max_tokens = int(max_tokens)
                m.config.temperature = float(temperature)

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

_ensure_claude_model_config()
model_names    = _read_model_names()
model_types    = _read_model_types()
existing_files = _existing_datasets()

with gr.Blocks(title="LLM Evaluation", theme=gr.themes.Soft()) as demo:

    model_types_state = gr.State(model_types)

    with gr.Tabs():

        # ================================================================
        # Tab 1: Evaluation
        # ================================================================
        with gr.Tab("Evaluation"):

            gr.Markdown(
                """
                # LLM Evaluation Dashboard
                Evaluate cloud-hosted models on a QA dataset and compare results side-by-side.
                Results include **BLEU · METEOR · Token-F1 · ROUGE · BERTScore** and response latency.
                In **Audio** mode, additional **Speech Clarity** metrics are computed: SNR(dB), Speech Ratio, Clarity Score, and Whisper Confidence.

                **Supported models:** Qwen2-7B · Qwen2.5-14B · Mistral-7B (Fireworks AI) · GPT-4o (OpenAI) · Gemini-2.5-Pro (Google) · Claude (Anthropic)

                > **Voice Assistant Proxy Note:** Due to the absence of public programmatic APIs for
                > proprietary voice assistants (Siri, Cortana, Bixby), this system evaluates their
                > equivalent foundation models as proxies — **GPT-4o** serves as the OpenAI Voice
                > Assistant proxy, and **Gemini-2.5-Pro** serves as the Google Assistant proxy.
                > This follows standard practice in comparative NLP evaluation research when
                > direct system access is unavailable.
                """
            )

            with gr.Accordion("API Key Setup", open=False):
                gr.Markdown(
                    """
                    Set these environment variables **before** launching `app.py`:

                    **Windows (Anaconda Prompt):**
                    ```
                    set FIREWORKS_API_KEY=fw_xxx
                    set OPENAI_API_KEY=sk-xxx
                    set GOOGLE_API_KEY=AIza-xxx
                    set ANTHROPIC_API_KEY=sk-ant-xxx
                    ```
                    `OPENAI_API_KEY` is also used for Whisper audio transcription.
                    """
                )

            # ---- Dataset + Model selection ------------------------------
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Dataset")
                    dataset_source = gr.Radio(
                        choices=["Upload file", "Use existing file"],
                        value="Upload file" if not existing_files else "Use existing file",
                        label="Source",
                    )
                    with gr.Group(visible=not existing_files) as upload_group:
                        upload_box = gr.File(
                            label="Upload dataset(s) (.json / .jsonl / .csv / .txt)",
                            file_types=[".json", ".jsonl", ".csv", ".txt"],
                            file_count="multiple",
                        )
                        dataset_picker = gr.Dropdown(
                            choices=[],
                            label="Select dataset to run (shown when multiple files uploaded)",
                            visible=False,
                        )
                    existing_box = gr.Dropdown(
                        choices=existing_files,
                        value=existing_files[0] if existing_files else None,
                        label="Select from data/ folder",
                        visible=bool(existing_files),
                    )
                    # Dataset status indicator (green / yellow / red)
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
                    # Validate on picker change (covers both single-file auto-select
                    # and manual multi-file selection)
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

                with gr.Column(scale=1):
                    gr.Markdown("### Models")
                    if model_names:
                        model_select = gr.CheckboxGroup(
                            choices=model_names,
                            value=[model_names[0]],
                            label="Select models to evaluate",
                        )
                    else:
                        gr.Markdown(
                            "_No models found in `config/models.yaml`. "
                            "Add entries with `type: fireworks`, `type: openai`, `type: gemini`, or `type: claude`._"
                        )
                        model_select = gr.CheckboxGroup(choices=[], label="Models")

            # ---- Input mode --------------------------------------------
            with gr.Row():
                input_mode = gr.Radio(
                    choices=["Text", "Audio"],
                    value="Text",
                    label="Input mode",
                )

            audio_warning = gr.HTML(visible=False)

            # ---- Audio upload section ----------------------------------
            with gr.Group(visible=False) as audio_section:
                gr.Markdown("### Audio Dataset")
                gr.Markdown(
                    "Upload a **metadata CSV** (columns: `audio_file`, `reference_answer`) "
                    "and the matching **audio files** below."
                )
                use_whisper = gr.Checkbox(
                    value=True,
                    label="Enable Whisper transcription for non-audio models (Qwen, Mistral → Whisper → text → model)",
                )
                gr.HTML(
                    '<div style="color:#856404;background:#fff3cd;border:1px solid #ffc107;'
                    'border-radius:6px;padding:8px 12px;margin-top:4px;">'
                    '⚠ <b>Chinese audio note:</b> Whisper outputs <b>Traditional Chinese</b> for Mandarin audio. '
                    'Use the Script Converter in the Tools tab to convert to Simplified Chinese.'
                    '</div>'
                )
                with gr.Row():
                    with gr.Column():
                        audio_csv_box = gr.File(
                            label="Audio metadata CSV",
                            file_types=[".csv"],
                            file_count="single",
                        )
                        audio_csv_status = gr.HTML(visible=False)
                    audio_files_box = gr.File(
                        label="Audio files (.wav / .mp3 / .m4a)",
                        file_types=[".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"],
                        file_count="multiple",
                    )

                audio_csv_box.upload(
                    _validate_audio_csv,
                    inputs=audio_csv_box,
                    outputs=audio_csv_status,
                )

            # Wire audio section + warning
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

            # ---- Generation / evaluation settings ----------------------
            with gr.Row():
                max_tokens_box   = gr.Number(value=512,  label="Max tokens",  precision=0, minimum=1)
                temperature_box  = gr.Number(value=0.0,  label="Temperature", precision=2, minimum=0.0)
                limit_box        = gr.Textbox(value="",  label="Question limit (blank = all)",
                                              placeholder="e.g. 10")
                no_bertscore_box = gr.Checkbox(value=True, label="Skip BERTScore (faster)")

            run_btn = gr.Button("▶  Run Evaluation", variant="primary", size="lg")

            gr.Markdown("### Progress")
            log_box = gr.Textbox(
                label="Log", lines=15, max_lines=15,
                interactive=False,
            )

            gr.Markdown("### Results")
            results_table = gr.Dataframe(
                label="Per-model summary (mean scores)",
                interactive=False, wrap=True,
            )
            download_btn = gr.File(label="Download full results CSV", interactive=False)

            run_btn.click(
                fn=run_evaluation,
                inputs=[
                    dataset_source, dataset_picker, existing_box,
                    model_select, max_tokens_box, temperature_box,
                    limit_box, no_bertscore_box,
                    input_mode, audio_csv_box, audio_files_box, use_whisper,
                ],
                outputs=[log_box, results_table, download_btn, run_btn],
            )

        # ================================================================
        # Tab 2: Noise Robustness
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
                    nr_max_tokens = gr.Number(value=512, label="Max tokens", precision=0, minimum=1)
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
                        from llm_eval.models import load_models_from_config
                        from llm_eval.metrics.evaluator import Evaluator
                        from llm_eval.runner import EvalRunner
                        from llm_eval.noise_augment import generate_noise_variants
                        from llm_eval.output.exporter import export_csv

                        all_models = load_models_from_config(CONFIG_PATH)
                        models = [m for m in all_models if m.name in nr_models]
                        if not models:
                            q.put(("error", "None of the selected models found in config."))
                            return

                        for m in models:
                            m.config.max_tokens = int(nr_max_tok)
                            m.config.temperature = float(nr_temp)

                        uploads = nr_files if isinstance(nr_files, list) else [nr_files]
                        audio_files_map = {Path(p).name: p for p in uploads if p}

                        whisper_fn = None
                        if nr_whisper:
                            try:
                                from llm_eval.transcriber import make_whisper_transcriber
                                whisper_fn = make_whisper_transcriber()
                                q.put(("log", "Whisper transcriber ready."))
                            except Exception as e:
                                q.put(("log", f"⚠ Whisper init failed: {e}"))

                        evaluator = Evaluator()
                        runner = EvalRunner(models, evaluator)
                        dataset = runner.load_audio_dataset(str(nr_csv), audio_files_map)
                        q.put(("log", f"Dataset: {len(dataset)} audio files loaded."))

                        noise_dir = str(Path(tempfile.gettempdir()) / "llm_eval_noise")
                        all_dfs = []

                        for level_label in nr_levels:
                            snr = _LEVEL_MAP.get(level_label)
                            q.put(("log", f"\n--- Testing noise level: {level_label} ---"))

                            # Build dataset with noisy audio
                            if snr is None:
                                level_dataset = dataset
                            else:
                                from llm_eval.runner import QAPair
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
