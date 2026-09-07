"""
Evaluation orchestrator — loads one model at a time to avoid GPU OOM,
runs it over the full QA dataset, then moves to the next.
"""

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .metrics.evaluator import Evaluator
from .metrics.audio_quality import compute_audio_clarity
from .models.base import BaseModel, ModelResponse

# ---------------------------------------------------------------------------
# Column-name aliases for flexible dataset parsing
# ---------------------------------------------------------------------------

_QUESTION_ALIASES = {"question", "prompt", "input", "text", "q"}
_REFERENCE_ALIASES = {
    "reference_answer", "reference", "answer", "output",
    "expected", "label", "ground_truth", "a",
}
_AUDIO_FILE_ALIASES = {"audio_file", "audio", "file", "filename", "audio_path"}


def _find_col(keys: List[str], aliases: set) -> Optional[str]:
    """Return the first key whose lowercased name is in aliases, else None."""
    for k in keys:
        if k.lower().strip() in aliases:
            return k
    return None


def _extract_qa(row: dict) -> tuple:
    """
    Extract (question, reference_answer) from a dict using column aliases.
    Returns ("", "") if neither column is found.
    """
    keys = list(row.keys())
    q_key = _find_col(keys, _QUESTION_ALIASES)
    r_key = _find_col(keys, _REFERENCE_ALIASES)
    q = row[q_key].strip() if q_key else ""
    r = row[r_key].strip() if r_key else ""
    return q, r


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class QAPair:
    question: str
    reference_answer: str
    audio_path: Optional[str] = None   # set for audio-mode datasets


@dataclass
class EvalRecord:
    model_name: str
    question: str
    reference: str
    prediction: str
    latency_seconds: float
    error: Optional[str]
    rouge1:             float = 0.0
    rouge1_p:           float = 0.0
    rouge1_r:           float = 0.0
    rouge2:             float = 0.0
    rougeL:             float = 0.0
    bleu:               float = 0.0
    meteor:             float = 0.0
    f1:                 float = 0.0
    response_length:    int   = 0
    bertscore_p:        float = 0.0
    bertscore_r:        float = 0.0
    bertscore_f1:       float = 0.0
    # Audio quality / Speech Clarity metrics
    snr_db:             float = 0.0
    speech_ratio:       float = 0.0
    clarity_score:      float = 0.0
    whisper_confidence: float = 0.0
    # Noise robustness: None = clean audio, numeric = noise SNR in dB
    noise_level_db: Optional[float] = None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class EvalRunner:
    def __init__(self, models: List[BaseModel], evaluator: Evaluator):
        self.models = models
        self.evaluator = evaluator

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(self, path: str) -> List[QAPair]:
        """
        Load QA pairs from JSON, JSONL, CSV, or TXT.

        JSON/JSONL: dicts with flexible key names (see _QUESTION_ALIASES / _REFERENCE_ALIASES)
        CSV:        flexible column headers — same aliases apply
        TXT:        tab-separated lines: question<TAB>reference_answer
                    Lines without a tab are treated as question-only (reference = "")
        """
        path = str(path)

        # Detect encoding from BOM; fall back to utf-8-sig (handles utf-8 BOM too)
        def _open(p):
            raw = open(p, "rb").read(4)
            if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
                enc = "utf-16"
            elif raw[:3] == b'\xef\xbb\xbf':
                enc = "utf-8-sig"
            else:
                enc = "utf-8"
            return open(p, encoding=enc)

        if path.endswith(".jsonl"):
            with _open(path) as f:
                data = [json.loads(line) for line in f if line.strip()]
            return [QAPair(*_extract_qa(d)) for d in data]

        elif path.endswith(".json"):
            with _open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                data = next(iter(data.values()))
            return [QAPair(*_extract_qa(d)) for d in data]

        elif path.endswith(".csv"):
            pairs = []
            with _open(path) as f:
                for row in csv.DictReader(f):
                    pairs.append(QAPair(*_extract_qa(row)))
            return pairs

        elif path.endswith(".txt"):
            pairs = []
            with _open(path) as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line.strip():
                        continue
                    if "\t" in line:
                        q, r = line.split("\t", 1)
                        pairs.append(QAPair(question=q.strip(), reference_answer=r.strip()))
                    else:
                        pairs.append(QAPair(question=line.strip(), reference_answer=""))
            return pairs

        else:
            raise ValueError(
                f"Unsupported dataset format: '{path}'. "
                "Use .json, .jsonl, .csv, or .txt"
            )

    def load_audio_dataset(
        self,
        csv_path: str,
        audio_files: Dict[str, str],
    ) -> List[QAPair]:
        """
        Load an audio dataset.

        csv_path:    path to a CSV with flexible column names:
                       - audio filename column (aliases: audio_file, audio, file, filename, audio_path)
                       - reference answer column (aliases: same as _REFERENCE_ALIASES)
        audio_files: {filename: full_path} mapping built from uploaded files
        """
        pairs: List[QAPair] = []
        with open(csv_path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                keys = list(row.keys())
                af_key = _find_col(keys, _AUDIO_FILE_ALIASES)
                r_key  = _find_col(keys, _REFERENCE_ALIASES)

                filename = row[af_key].strip() if af_key else ""
                ref      = row[r_key].strip()  if r_key  else ""
                full_path = audio_files.get(filename, audio_files.get(Path(filename).name, ""))

                pairs.append(QAPair(
                    question=filename,
                    reference_answer=ref,
                    audio_path=full_path or None,
                ))
        return pairs

    # ------------------------------------------------------------------
    # Main evaluation loop
    # ------------------------------------------------------------------

    def run(
        self,
        dataset: List[QAPair],
        use_bertscore: bool = True,
        input_mode: str = "text",          # "text" | "audio"
        whisper_fn: Optional[Callable[[str], Tuple[str, float]]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
        noise_level_db: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Evaluate all QA pairs across all models.

        on_progress(fraction, description) is called after each question.
        If None, a rich progress bar is shown in the terminal instead.
        input_mode: "text" uses generate(); "audio" uses generate_audio()
                    for capable models and records an error for others.
        """
        all_records: List[EvalRecord] = []
        total_steps = len(self.models) * len(dataset)
        done = 0

        for model in self.models:
            # --- notify: loading ---
            if on_progress:
                on_progress(done / max(total_steps, 1), f"Loading {model.name}…")
            else:
                print(f"\n[{model.name}]")

            try:
                model.load()
            except Exception as e:
                msg = f"ERROR loading {model.name}: {e}"
                if on_progress:
                    on_progress(done / max(total_steps, 1), msg)
                else:
                    print(msg)
                done += len(dataset)
                continue

            responses: List[ModelResponse] = []
            # Per-response whisper confidence (parallel list to responses)
            whisper_confidences: List[float] = []

            def _call(qa: QAPair) -> tuple:
                """
                Dispatch to text or audio generation, with optional Whisper fallback.
                Returns (ModelResponse, whisper_confidence: float).
                whisper_confidence is 0.0 when Whisper is not used.
                """
                if input_mode == "audio":
                    if model.supports_audio() and qa.audio_path:
                        # Model natively handles audio — no Whisper confidence available
                        return model.generate_audio(qa.audio_path), 0.0
                    elif not model.supports_audio() and qa.audio_path and whisper_fn:
                        # Transcribe with Whisper, then send as text
                        try:
                            transcription, wconf = whisper_fn(qa.audio_path)
                        except Exception as e:
                            return ModelResponse(
                                model_name=model.name,
                                question=qa.question,
                                prediction="",
                                latency_seconds=0.0,
                                error=f"Whisper transcription failed: {e}",
                            ), 0.0
                        resp = model.generate(transcription)
                        return resp, wconf
                    elif not model.supports_audio():
                        return ModelResponse(
                            model_name=model.name,
                            question=qa.question,
                            prediction="",
                            latency_seconds=0.0,
                            error=f"{model.name} does not support audio (enable Whisper to transcribe)",
                        ), 0.0
                    else:
                        return ModelResponse(
                            model_name=model.name,
                            question=qa.question,
                            prediction="",
                            latency_seconds=0.0,
                            error=f"Audio file not found: {qa.question}",
                        ), 0.0
                return model.generate(qa.question), 0.0

            if on_progress:
                # --- Gradio path: plain loop + callback ---
                for j, qa in enumerate(dataset):
                    response, wconf = _call(qa)
                    responses.append(response)
                    whisper_confidences.append(wconf)
                    done += 1
                    on_progress(
                        done / total_steps,
                        f"[{model.name}]  question {j + 1}/{len(dataset)}  "
                        f"({response.latency_seconds:.1f}s)"
                        + (f"  ⚠ {response.error}" if response.error else ""),
                    )
            else:
                # --- CLI path: rich progress bar ---
                from rich.progress import (
                    BarColumn, Progress, SpinnerColumn, TimeElapsedColumn
                )
                with Progress(
                    SpinnerColumn(),
                    "[progress.description]{task.description}",
                    BarColumn(),
                    "[progress.percentage]{task.percentage:>3.0f}%",
                    TimeElapsedColumn(),
                ) as progress:
                    task_id = progress.add_task(
                        f"  {model.name} — {len(dataset)} questions",
                        total=len(dataset),
                    )
                    for qa in dataset:
                        response, wconf = _call(qa)
                        responses.append(response)
                        whisper_confidences.append(wconf)
                        done += 1
                        progress.advance(task_id)

            model.unload()
            if on_progress:
                on_progress(done / total_steps, f"{model.name} unloaded.")

            # --- BERTScore in one batch for this model ---
            bs_map: dict = {}
            if use_bertscore:
                valid_indices = [
                    i for i, r in enumerate(responses)
                    if not r.error and r.prediction
                ]
                if valid_indices:
                    if on_progress:
                        on_progress(
                            done / total_steps,
                            f"[{model.name}] Computing BERTScore for "
                            f"{len(valid_indices)} predictions…"
                        )
                    valid_preds = [responses[i].prediction for i in valid_indices]
                    valid_refs  = [dataset[i].reference_answer for i in valid_indices]
                    try:
                        bs_results = self.evaluator.compute_bertscore_batch(
                            valid_preds, valid_refs
                        )
                        bs_map = dict(zip(valid_indices, bs_results))
                    except ImportError as exc:
                        raise RuntimeError(str(exc)) from exc

            # --- Build records with metrics ---
            zero_rb = {
                "rouge1": 0.0, "rouge1_p": 0.0, "rouge1_r": 0.0,
                "rouge2": 0.0, "rougeL": 0.0,
                "bleu": 0.0, "meteor": 0.0, "f1": 0.0, "response_length": 0,
            }
            zero_bs = {"bertscore_p": 0.0, "bertscore_r": 0.0, "bertscore_f1": 0.0}

            for i, (response, qa) in enumerate(zip(responses, dataset)):
                rb = dict(zero_rb)
                bs = dict(zero_bs)

                if not response.error and response.prediction:
                    rb = self.evaluator.compute_rouge_bleu_f1(
                        response.prediction, qa.reference_answer
                    ) if qa.reference_answer else {
                        **zero_rb, "response_length": len(response.prediction.split())
                    }
                    if use_bertscore and qa.reference_answer:
                        bs = bs_map.get(i, zero_bs)

                # --- Audio quality / Speech Clarity ---
                wconf = whisper_confidences[i] if i < len(whisper_confidences) else 0.0
                aq = {"snr_db": 0.0, "speech_ratio": 0.0, "clarity_score": 0.0}
                if input_mode == "audio" and qa.audio_path:
                    aq = compute_audio_clarity(qa.audio_path, whisper_confidence=wconf or None)

                all_records.append(
                    EvalRecord(
                        model_name=response.model_name,
                        question=qa.question,
                        reference=qa.reference_answer,
                        prediction=response.prediction,
                        latency_seconds=response.latency_seconds,
                        error=response.error,
                        **rb,
                        **bs,
                        snr_db=aq["snr_db"],
                        speech_ratio=aq["speech_ratio"],
                        clarity_score=aq["clarity_score"],
                        whisper_confidence=wconf,
                        noise_level_db=noise_level_db,
                    )
                )

        return pd.DataFrame([vars(r) for r in all_records])
