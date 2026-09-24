"""
Evaluation orchestrator — loads one model at a time to avoid GPU OOM,
runs it over the full QA dataset, then moves to the next.
"""

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
_SCENARIO_TURN_ALIASES = {"turn", "turn_id", "step", "step_id"}
_SCENARIO_USER_ALIASES = {
    "user message", "user_message", "user", "examiner message",
    "examiner_message", "prompt",
}
_SCENARIO_EXPECTED_ALIASES = {
    "expected assistant action", "expected_assistant_action",
    "expected action", "expected_action", "checkpoint", "success criterion",
}
_SCENARIO_CRITICAL_ALIASES = {
    "critical failure", "critical_failure", "failure condition",
    "failure_condition",
}
_SCENARIO_CRITICAL_CHECKPOINT_ALIASES = {
    "critical checkpoint", "critical_checkpoint", "mandatory checkpoint",
    "mandatory_checkpoint",
}
_SCENARIO_DIMENSION_ALIASES = {
    "applicable dimensions", "applicable_dimensions", "dimensions",
}


def _find_col(keys: List[str], aliases: set) -> Optional[str]:
    """Return the first key whose lowercased name is in aliases, else None."""
    for k in keys:
        # csv.DictReader can produce a None key for malformed/extra columns.
        # Ignore non-string keys instead of crashing on .lower().
        if isinstance(k, str) and k.lower().strip() in aliases:
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
    q = (row.get(q_key) or "").strip() if q_key else ""
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
class ScenarioTurn:
    turn: int
    user_message: str
    expected_assistant_action: str
    critical_failure: str = ""
    critical_checkpoint: bool = False
    applicable_dimensions: List[str] = field(default_factory=list)


@dataclass
class Scenario:
    scenario_id: str
    title: str
    turns: List[ScenarioTurn]
    domain: str = "telecommunications"
    language: str = "unspecified"
    difficulty: str = "unspecified"
    maximum_turns: Optional[int] = None
    source_path: str = ""


@dataclass
class ScenarioEvalRecord:
    schema_version: str
    evaluation_mode: str
    evaluation_status: str
    model_name: str
    judge_model: str
    scenario_id: str
    scenario_title: str
    scenario_domain: str
    scenario_language: str
    scenario_difficulty: str
    repetition: int
    execution_success: int
    judge_attempted: int
    judge_success: int
    valid_evaluated_run: int
    end_to_end_success: Optional[int]
    failed_turn: Optional[int]
    error_category: Optional[str]
    response_max_tokens: int
    judge_max_tokens: int
    response_temperature: float
    judge_temperature: float
    response_thinking_level: Optional[str]
    judge_thinking_level: Optional[str]
    response_effort: Optional[str]
    judge_effort: Optional[str]
    goal_achieved: Optional[int]
    goal_rationale: str
    terminal_state_valid: Optional[int]
    safe_completion: Optional[int]
    all_checkpoints_met: Optional[int]
    task_success: Optional[int]
    checkpoint_completion: Optional[float]
    checkpoint_quality: Optional[float]
    critical_checkpoint_completion: Optional[float]
    critical_failure: Optional[int]
    critical_failure_type: Optional[str]
    information_gathering: Optional[int]
    instruction_following: Optional[int]
    state_constraint_tracking: Optional[int]
    correction_recovery: Optional[int]
    domain_accuracy: Optional[int]
    relevance_efficiency: Optional[int]
    clarity_actionability: Optional[int]
    safety_cost_awareness: Optional[int]
    conversational_coherence: Optional[int]
    turn_taking_fluency: Optional[int]
    interruption_handling: Optional[int]
    response_timing_appropriateness: Optional[int]
    speech_intelligibility: Optional[int]
    speech_naturalness: Optional[int]
    actual_turns: int
    target_turns: int
    maximum_turns: int
    maximum_turn_violation: int
    mean_latency_ms: float
    total_latency_ms: float
    repeated_questions: Optional[int]
    unsupported_assumptions: Optional[int]
    state_constraint_errors: Optional[int]
    correction_failures: Optional[int]
    successful_recoveries: Optional[int]
    premature_termination: Optional[int]
    excess_turns: Optional[int]
    examiner_redirections: Optional[int]
    asr_errors: Optional[int]
    system_error: Optional[str]
    judge_error: Optional[str]
    checkpoint_results_json: str
    transcript_json: str
    judge_response: str
    judge_notes: str


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
        dataset_path: str,
        audio_files: Dict[str, str],
    ) -> List[QAPair]:
        """
        Load an audio QA dataset from JSON, JSONL, or CSV.

        Each row/object may contain flexible aliases for the audio filename,
        question, and reference answer. ``audio_files`` maps repository
        filenames to their resolved local paths. Rows whose audio file was not
        selected/found are skipped.
        """
        dataset_path = str(dataset_path)

        def _open(p):
            raw = open(p, "rb").read(4)
            if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
                enc = "utf-16"
            elif raw[:3] == b'\xef\xbb\xbf':
                enc = "utf-8-sig"
            else:
                enc = "utf-8"
            return open(p, encoding=enc, newline="")

        suffix = Path(dataset_path).suffix.lower()
        if suffix == ".json":
            with _open(dataset_path) as f:
                rows = json.load(f)
            if isinstance(rows, dict):
                # Support common wrapped forms such as {"data": [...]} while
                # retaining compatibility with the text dataset loader.
                rows = next((v for v in rows.values() if isinstance(v, list)), [])
        elif suffix == ".jsonl":
            with _open(dataset_path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
        elif suffix == ".csv":
            with _open(dataset_path) as f:
                rows = list(csv.DictReader(f))
        else:
            raise ValueError(
                f"Unsupported audio dataset format: '{dataset_path}'. "
                "Use .json, .jsonl, or .csv"
            )

        if not isinstance(rows, list):
            raise ValueError("Audio dataset must contain a list of row objects.")

        pairs: List[QAPair] = []
        for row in rows:
            if not isinstance(row, dict):
                continue

            keys = list(row.keys())
            af_key = _find_col(keys, _AUDIO_FILE_ALIASES)
            if not af_key:
                continue

            raw_filename = row.get(af_key)
            filename = str(raw_filename or "").strip()
            if not filename:
                continue

            basename = Path(filename).name
            full_path = audio_files.get(filename) or audio_files.get(basename, "")
            if not full_path:
                continue

            question, ref = _extract_qa(row)
            pairs.append(QAPair(
                question=question or basename,
                reference_answer=ref,
                audio_path=full_path,
            ))

        return pairs

    def load_scenario(self, path: str) -> Scenario:
        """Load one reusable multi-turn scenario from CSV, JSON, or JSONL.

        A scenario file contains one row/object per user turn. Required fields
        are a user message and the expected assistant action. ``turn`` is
        optional and defaults to row order. An optional critical-failure field
        can define scenario-specific prohibited behaviour.
        """
        path = str(path)

        def _open(p):
            raw = open(p, "rb").read(4)
            if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
                enc = "utf-16"
            elif raw[:3] == b'\xef\xbb\xbf':
                enc = "utf-8-sig"
            else:
                enc = "utf-8"
            return open(p, encoding=enc, newline="")

        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            with _open(path) as f:
                rows = list(csv.DictReader(f))
        elif suffix == ".json":
            with _open(path) as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                rows = next(
                    (value for value in payload.values() if isinstance(value, list)),
                    [],
                )
            else:
                rows = payload
        elif suffix == ".jsonl":
            with _open(path) as f:
                rows = [json.loads(line) for line in f if line.strip()]
        else:
            raise ValueError(
                f"Unsupported scenario format: '{path}'. Use .csv, .json, or .jsonl"
            )

        if not isinstance(rows, list) or not rows:
            raise ValueError(f"Scenario file is empty: {path}")

        turns: List[ScenarioTurn] = []
        scenario_id = Path(path).stem
        title = scenario_id.replace("_", " ").strip().title()
        domain = "telecommunications"
        language = "unspecified"
        difficulty = "unspecified"
        maximum_turns = None

        for row_number, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            keys = list(row.keys())
            turn_key = _find_col(keys, _SCENARIO_TURN_ALIASES)
            user_key = _find_col(keys, _SCENARIO_USER_ALIASES)
            expected_key = _find_col(keys, _SCENARIO_EXPECTED_ALIASES)
            critical_key = _find_col(keys, _SCENARIO_CRITICAL_ALIASES)
            critical_checkpoint_key = _find_col(
                keys, _SCENARIO_CRITICAL_CHECKPOINT_ALIASES
            )
            dimensions_key = _find_col(keys, _SCENARIO_DIMENSION_ALIASES)

            if not user_key or not expected_key:
                raise ValueError(
                    f"Scenario row {row_number} in {Path(path).name} must contain "
                    "'User message' and 'Expected assistant action' columns."
                )

            user_message = str(row.get(user_key) or "").strip()
            expected_action = str(row.get(expected_key) or "").strip()
            critical_failure = (
                str(row.get(critical_key) or "").strip() if critical_key else ""
            )
            raw_critical_checkpoint = (
                str(row.get(critical_checkpoint_key) or "").strip().lower()
                if critical_checkpoint_key else ""
            )
            critical_checkpoint = raw_critical_checkpoint in {
                "1", "true", "yes", "y", "required", "critical",
            }
            raw_dimensions = (
                str(row.get(dimensions_key) or "").strip()
                if dimensions_key else ""
            )
            applicable_dimensions = [
                value.strip()
                for value in re.split(r"[;,|]", raw_dimensions)
                if value.strip()
            ]
            if not user_message or not expected_action:
                raise ValueError(
                    f"Scenario row {row_number} in {Path(path).name} has an "
                    "empty user message or expected assistant action."
                )

            raw_turn = row.get(turn_key) if turn_key else row_number
            try:
                turn_number = int(str(raw_turn).strip())
            except (TypeError, ValueError):
                turn_number = row_number

            turns.append(ScenarioTurn(
                turn=turn_number,
                user_message=user_message,
                expected_assistant_action=expected_action,
                critical_failure=critical_failure,
                critical_checkpoint=critical_checkpoint,
                applicable_dimensions=applicable_dimensions,
            ))

            raw_id = row.get("scenario_id") or row.get("Scenario ID")
            raw_title = row.get("scenario_title") or row.get("Scenario title")
            raw_domain = row.get("domain") or row.get("Domain")
            raw_language = row.get("language") or row.get("Language")
            raw_difficulty = row.get("difficulty") or row.get("Difficulty")
            raw_maximum_turns = (
                row.get("maximum_turns") or row.get("Maximum turns")
                or row.get("max_turns") or row.get("Max turns")
            )
            if raw_id:
                scenario_id = str(raw_id).strip()
            if raw_title:
                title = str(raw_title).strip()
            if raw_domain:
                domain = str(raw_domain).strip()
            if raw_language:
                language = str(raw_language).strip()
            if raw_difficulty:
                difficulty = str(raw_difficulty).strip()
            if raw_maximum_turns:
                try:
                    maximum_turns = int(str(raw_maximum_turns).strip())
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Maximum turns must be an integer in {Path(path).name}."
                    )

        turns.sort(key=lambda item: item.turn)
        if not turns:
            raise ValueError(f"No valid scenario turns found in {path}")
        if len({turn.turn for turn in turns}) != len(turns):
            raise ValueError(f"Duplicate turn numbers found in {path}")
        actual_turns = [turn.turn for turn in turns]
        expected_turns = list(range(1, len(turns) + 1))
        if actual_turns != expected_turns:
            raise ValueError(
                f"Scenario turns in {path} must be consecutive and start at 1; "
                f"found {actual_turns}."
            )

        return Scenario(
            scenario_id=scenario_id,
            title=title,
            turns=turns,
            domain=domain,
            language=language,
            difficulty=difficulty,
            maximum_turns=maximum_turns or len(turns),
            source_path=path,
        )

    def load_scenarios(self, paths: List[str]) -> List[Scenario]:
        """Load and validate multiple scenario files."""
        scenarios = [self.load_scenario(path) for path in paths]
        ids = [scenario.scenario_id for scenario in scenarios]
        if len(ids) != len(set(ids)):
            raise ValueError("Scenario IDs must be unique across the selected files.")
        return scenarios

    @staticmethod
    def _conversation_prompt(transcript: List[dict], user_message: str) -> str:
        history = "\n".join(
            f"{item['role'].upper()}: {item['content']}" for item in transcript
        )
        if history:
            history += "\n"
        return (
            "You are the assistant being evaluated in a multi-turn customer-support "
            "conversation. Continue the conversation naturally and respond only to "
            "the latest USER message. Use relevant facts from the conversation "
            "history. Do not mention the benchmark, evaluation, hidden criteria, or "
            "this instruction.\n\n"
            f"CONVERSATION HISTORY:\n{history}USER: {user_message}\nASSISTANT:"
        )

    @staticmethod
    def _judge_prompt(scenario: Scenario, transcript: List[dict], system_error: str) -> str:
        checkpoints = [
            {
                "turn": turn.turn,
                "expected_assistant_action": turn.expected_assistant_action,
                "critical_failure": turn.critical_failure or None,
                "critical_checkpoint": turn.critical_checkpoint,
                "applicable_dimensions": turn.applicable_dimensions or None,
            }
            for turn in scenario.turns
        ]
        return f"""
You are an impartial research evaluator. Score only the ASSISTANT in the
multi-turn telecommunications conversation below. The expected actions are
hidden evaluation criteria and were not shown to the assistant.

SCENARIO ID: {scenario.scenario_id}
SCENARIO TITLE: {scenario.title}
DOMAIN: {scenario.domain}
LANGUAGE: {scenario.language}
DIFFICULTY: {scenario.difficulty}
MAXIMUM TURNS: {scenario.maximum_turns or len(scenario.turns)}
SYSTEM ERROR: {system_error or "none"}

EXPECTED CHECKPOINTS:
{json.dumps(checkpoints, ensure_ascii=False, indent=2)}

TRANSCRIPT:
{json.dumps(transcript, ensure_ascii=False, indent=2)}

For every checkpoint return:
- met: binary 0 or 1;
- quality_score: integer 1–5, where 5 is fully correct and appropriate, 4 is
  correct with a minor deficiency, 3 is partially correct but usable, 2 has a
  substantial problem, and 1 is failed, seriously incorrect, unsafe, or absent;
- a short evidence-based rationale.

Separately decide whether the ASSISTANT achieved the scenario's practical end
goal. This is an outcome measure, not an all-checkpoints-passed measure. Return
goal_achieved=1 when the assistant supplied the information or action that
caused, or would reasonably cause, the user's problem to be resolved safely.
A run may achieve its goal even when some procedural checkpoints were missed.
Conversely, do not mark the goal achieved merely because a fixed scripted user
message later claims success: the assistant's preceding response must have
materially enabled that outcome. Give a short evidence-based goal_rationale.

Return terminal_state_valid=1 only when the dialogue finishes in a state that
is consistent with the scenario outcome and the assistant does not incorrectly
claim resolution, terminate prematurely, or leave a required final action
unaddressed.

Score these nine non-overlapping global dimensions from 1–5 using the same
anchors. Use JSON null only when the capability was genuinely not tested:
information_gathering, instruction_following, state_constraint_tracking,
correction_recovery, domain_accuracy, relevance_efficiency,
clarity_actionability, safety_cost_awareness, conversational_coherence.

State and constraint tracking covers entities, settings, prior decisions and
explicit constraints. Correction and recovery is scored across the whole
conversation when the user corrects earlier information. Safety and cost
awareness includes privacy, destructive actions and avoidable charges.

This runner is scripted text multi-turn rather than streaming audio, so return
null for turn_taking_fluency, interruption_handling,
response_timing_appropriateness, speech_intelligibility, and
speech_naturalness. Do not infer audio or interactive timing performance from
text. Latency is measured separately by the runner.

Critical failure is binary. Mark it 1 only for a material unsafe, destructive,
privacy-compromising, charge-inducing, or explicitly scenario-prohibited
instruction. Use the scenario's predefined failure conditions; do not invent a
new failure after seeing the model identity. Count the requested observable
events as non-negative integers. Examiner redirections are not applicable
because the scripted user turns do not adapt to assistant quality.

Return exactly one JSON object with this schema and no surrounding prose:
{{
  "goal_achieved": 0,
  "goal_rationale": "...",
  "terminal_state_valid": 0,
  "checkpoint_results": [
    {{"turn": 1, "met": 0, "quality_score": 1, "rationale": "..."}}
  ],
  "critical_failure": 0,
  "critical_failure_type": null,
  "scores": {{
    "information_gathering": 1,
    "instruction_following": 1,
    "state_constraint_tracking": 1,
    "correction_recovery": null,
    "domain_accuracy": 1,
    "relevance_efficiency": 1,
    "clarity_actionability": 1,
    "safety_cost_awareness": 1,
    "conversational_coherence": 1,
    "turn_taking_fluency": null,
    "interruption_handling": null,
    "response_timing_appropriateness": null,
    "speech_intelligibility": null,
    "speech_naturalness": null
  }},
  "counts": {{
    "repeated_questions": 0,
    "unsupported_assumptions": 0,
    "state_constraint_errors": 0,
    "correction_failures": 0,
    "successful_recoveries": 0,
    "premature_termination": 0
  }},
  "notes": "..."
}}
""".strip()

    @staticmethod
    def _parse_judge_json(text: str) -> dict:
        """Extract the first valid JSON object from a judge response."""
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1] if "\n" in raw else raw
            raw = raw.rsplit("```", 1)[0].strip()
        decoder = json.JSONDecoder()
        last_error: Optional[Exception] = None
        for start, character in enumerate(raw):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(raw[start:])
            except json.JSONDecodeError as exc:
                last_error = exc
                continue
            if isinstance(parsed, dict):
                return parsed
        if last_error:
            raise ValueError(f"Judge returned invalid or truncated JSON: {last_error}")
        raise ValueError("Judge did not return a JSON object.")

    @staticmethod
    def _judge_repair_prompt(
        original_prompt: str,
        previous_response: str,
        parse_error: str,
    ) -> str:
        """Ask once for a corrected JSON-only evaluation after parse failure."""
        return (
            f"{original_prompt}\n\n"
            "Your previous response could not be parsed. Return the complete JSON "
            "object again, with valid JSON syntax, double-quoted property names, no "
            "comments, no markdown fences, and no text before or after the object. "
            "Do not omit any checkpoint or score.\n"
            f"PARSER ERROR: {parse_error}\n"
            "PREVIOUS RESPONSE:\n"
            f"{(previous_response or '')[-6000:]}"
        )

    @classmethod
    def _validate_judge_payload(cls, payload: dict, scenario: Scenario) -> None:
        """Reject structurally incomplete judge output before scoring it.

        Without this validation, an incidental JSON object (or a partial object
        from a truncated response) can be mistaken for a completed evaluation,
        turning missing fields into false checkpoint failures.
        """
        required_top_level = {
            "goal_achieved", "goal_rationale",
            "terminal_state_valid",
            "checkpoint_results", "critical_failure",
            "critical_failure_type", "scores", "counts", "notes",
        }
        missing_top = sorted(required_top_level - set(payload))
        if missing_top:
            raise ValueError(
                "Judge JSON is missing top-level fields: " + ", ".join(missing_top)
            )

        if payload.get("goal_achieved") not in (
            0, 1, False, True, "0", "1", "false", "true", "False", "True"
        ):
            raise ValueError("goal_achieved must be binary 0 or 1.")
        if not str(payload.get("goal_rationale") or "").strip():
            raise ValueError("goal_rationale must be a non-empty string.")
        if payload.get("terminal_state_valid") not in (
            0, 1, False, True, "0", "1", "false", "true", "False", "True"
        ):
            raise ValueError("terminal_state_valid must be binary 0 or 1.")

        checkpoint_results = payload.get("checkpoint_results")
        if not isinstance(checkpoint_results, list):
            raise ValueError("checkpoint_results must be a JSON array.")
        expected_turns = {turn.turn for turn in scenario.turns}
        returned_turns = set()
        for item in checkpoint_results:
            if not isinstance(item, dict):
                raise ValueError("Every checkpoint result must be a JSON object.")
            turn = cls._non_negative_int(item.get("turn"))
            if turn is None:
                raise ValueError("Every checkpoint result must contain a valid turn.")
            if turn in returned_turns:
                raise ValueError(f"Checkpoint turn {turn} was returned more than once.")
            returned_turns.add(turn)
            if item.get("met") not in (0, 1, False, True, "0", "1", "false", "true", "False", "True"):
                raise ValueError(f"Checkpoint {turn} has an invalid binary met value.")
            if cls._score_1_to_5(item.get("quality_score")) is None:
                raise ValueError(f"Checkpoint {turn} is missing a 1–5 quality_score.")
            if not str(item.get("rationale") or "").strip():
                raise ValueError(f"Checkpoint {turn} is missing its rationale.")
        if returned_turns != expected_turns:
            missing = sorted(expected_turns - returned_turns)
            extra = sorted(returned_turns - expected_turns)
            raise ValueError(
                f"Checkpoint turn mismatch; missing={missing}, unexpected={extra}."
            )

        if payload.get("critical_failure") not in (
            0, 1, False, True, "0", "1", "false", "true", "False", "True"
        ):
            raise ValueError("critical_failure must be binary 0 or 1.")
        critical_is_true = payload.get("critical_failure") in (
            1, True, "1", "true", "True"
        )
        if critical_is_true and not str(
            payload.get("critical_failure_type") or ""
        ).strip():
            raise ValueError(
                "critical_failure_type is required when critical_failure is 1."
            )

        required_scores = {
            "information_gathering", "instruction_following",
            "state_constraint_tracking", "correction_recovery",
            "domain_accuracy", "relevance_efficiency",
            "clarity_actionability", "safety_cost_awareness",
            "conversational_coherence",
            "turn_taking_fluency", "interruption_handling",
            "response_timing_appropriateness", "speech_intelligibility",
            "speech_naturalness",
        }
        scores = payload.get("scores")
        if not isinstance(scores, dict):
            raise ValueError("scores must be a JSON object.")
        missing_scores = sorted(required_scores - set(scores))
        if missing_scores:
            raise ValueError(
                "Judge JSON is missing score fields: " + ", ".join(missing_scores)
            )
        for key in required_scores:
            value = scores.get(key)
            if value is not None and cls._score_1_to_5(value) is None:
                raise ValueError(f"Score {key} must be null or an integer from 1 to 5.")

        required_counts = {
            "repeated_questions", "unsupported_assumptions",
            "state_constraint_errors", "correction_failures",
            "successful_recoveries", "premature_termination",
        }
        counts = payload.get("counts")
        if not isinstance(counts, dict):
            raise ValueError("counts must be a JSON object.")
        missing_counts = sorted(required_counts - set(counts))
        if missing_counts:
            raise ValueError(
                "Judge JSON is missing count fields: " + ", ".join(missing_counts)
            )
        for key in required_counts:
            if cls._non_negative_int(counts.get(key)) is None:
                raise ValueError(f"Count {key} must be a non-negative integer.")

    @staticmethod
    def _score_1_to_5(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            score = int(value)
        except (TypeError, ValueError):
            return None
        return score if 1 <= score <= 5 else None

    @staticmethod
    def _non_negative_int(value: Any) -> Optional[int]:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        return max(0, number)

    @staticmethod
    def _error_category(message: str) -> Optional[str]:
        """Map provider text to stable research-reporting categories."""
        text = (message or "").lower()
        if not text:
            return None
        if "model load failed" in text:
            return "model_load_error"
        if (
            "truncated" in text
            or "finish_reason=max_tokens" in text
            or "finish_reason=length" in text
        ):
            return "truncated_output"
        if "max_tokens" in text or "max output" in text or "token limit" in text:
            return "invalid_generation_config"
        if "rate limit" in text or "429" in text:
            return "rate_limit"
        if "timeout" in text or "timed out" in text:
            return "timeout"
        if "api key" in text or "unauthorized" in text or "401" in text:
            return "authentication_error"
        if "not found" in text or "inaccessible" in text or "404" in text:
            return "model_unavailable"
        if "empty" in text or "no final text" in text or "no text returned" in text:
            return "empty_output"
        return "generation_error"

    @staticmethod
    def _mean_latency_ms(latencies: List[float]) -> Tuple[float, float]:
        valid = []
        for value in latencies:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number >= 0:
                valid.append(number)
        total = sum(valid) * 1000.0
        mean = total / len(valid) if valid else 0.0
        return round(mean, 3), round(total, 3)

    @classmethod
    def _unscored_scenario_record(
        cls,
        run: dict,
        judge_model_name: str,
        judge_max_tokens: int,
        judge_temperature: float,
        judge_thinking_level: Optional[str],
        judge_effort: Optional[str],
        *,
        evaluation_status: str,
        judge_attempted: int,
        judge_error: Optional[str] = None,
        judge_response: str = "",
    ) -> ScenarioEvalRecord:
        """Represent infrastructure failures without fabricating quality scores."""
        scenario = run["scenario"]
        transcript = run["transcript"]
        system_error = run.get("system_error") or ""
        mean_latency_ms, total_latency_ms = cls._mean_latency_ms(run["latencies"])
        execution_success = int(not system_error)
        return ScenarioEvalRecord(
            schema_version="3.0",
            evaluation_mode="scripted_text_multiturn",
            evaluation_status=evaluation_status,
            model_name=run["model_name"],
            judge_model=judge_model_name,
            scenario_id=scenario.scenario_id,
            scenario_title=scenario.title,
            scenario_domain=scenario.domain,
            scenario_language=scenario.language,
            scenario_difficulty=scenario.difficulty,
            repetition=run["repetition"],
            execution_success=execution_success,
            judge_attempted=judge_attempted,
            judge_success=0,
            valid_evaluated_run=0,
            end_to_end_success=0 if not execution_success else None,
            failed_turn=run.get("failed_turn"),
            error_category=(
                cls._error_category(system_error)
                if system_error else "judge_error"
            ),
            response_max_tokens=int(run.get("response_max_tokens") or 0),
            judge_max_tokens=int(judge_max_tokens or 0),
            response_temperature=float(run.get("response_temperature") or 0.0),
            judge_temperature=float(judge_temperature),
            response_thinking_level=run.get("response_thinking_level"),
            judge_thinking_level=judge_thinking_level,
            response_effort=run.get("response_effort"),
            judge_effort=judge_effort,
            goal_achieved=None,
            goal_rationale="",
            terminal_state_valid=None,
            safe_completion=None,
            all_checkpoints_met=None,
            task_success=0 if not execution_success else None,
            checkpoint_completion=None,
            checkpoint_quality=None,
            critical_checkpoint_completion=None,
            critical_failure=None,
            critical_failure_type=None,
            information_gathering=None,
            instruction_following=None,
            state_constraint_tracking=None,
            correction_recovery=None,
            domain_accuracy=None,
            relevance_efficiency=None,
            clarity_actionability=None,
            safety_cost_awareness=None,
            conversational_coherence=None,
            turn_taking_fluency=None,
            interruption_handling=None,
            response_timing_appropriateness=None,
            speech_intelligibility=None,
            speech_naturalness=None,
            actual_turns=sum(1 for item in transcript if item.get("role") == "assistant"),
            target_turns=len(scenario.turns),
            maximum_turns=int(scenario.maximum_turns or len(scenario.turns)),
            maximum_turn_violation=int(
                sum(1 for item in transcript if item.get("role") == "assistant")
                > int(scenario.maximum_turns or len(scenario.turns))
            ),
            mean_latency_ms=mean_latency_ms,
            total_latency_ms=total_latency_ms,
            repeated_questions=None,
            unsupported_assumptions=None,
            state_constraint_errors=None,
            correction_failures=None,
            successful_recoveries=None,
            premature_termination=None,
            excess_turns=None,
            examiner_redirections=None,
            asr_errors=None,
            system_error=system_error or None,
            judge_error=judge_error,
            checkpoint_results_json="[]",
            transcript_json=json.dumps(transcript, ensure_ascii=False),
            judge_response=judge_response,
            judge_notes=(
                "Behavioural scores are unavailable because model execution failed."
                if system_error else
                "Behavioural scores are unavailable because judging failed."
            ),
        )

    def run_scenarios(
        self,
        scenarios: List[Scenario],
        judge_model: BaseModel,
        repetitions: int = 1,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> pd.DataFrame:
        """Run scripted multi-turn scenarios and score them with an LLM judge.

        The examinee receives the complete dialogue history at each turn, but
        never sees expected actions or scoring criteria. Models are executed
        one at a time, unloaded, and only then is the judge loaded.
        """
        if not scenarios:
            return pd.DataFrame(columns=list(ScenarioEvalRecord.__dataclass_fields__))
        repetitions = max(1, int(repetitions))
        run_count = len(self.models) * len(scenarios) * repetitions
        total_steps = run_count + run_count
        completed_steps = 0
        pending_runs: List[dict] = []

        for model in self.models:
            model_config = getattr(model, "config", None)
            model_extra = getattr(model_config, "extra", {})
            if not isinstance(model_extra, dict):
                model_extra = {}
            response_temperature = float(
                getattr(model_config, "temperature", 0.0) or 0.0
            )
            response_thinking_level = str(
                model_extra.get("thinking_level") or ""
            ).strip().lower() or None
            response_effort = str(
                model_extra.get("effort") or ""
            ).strip().lower() or None
            if on_progress:
                on_progress(
                    completed_steps / max(total_steps, 1),
                    f"Loading examinee {model.name}…",
                )
            try:
                model.load()
            except Exception as exc:
                for repetition in range(1, repetitions + 1):
                    for scenario in scenarios:
                        pending_runs.append({
                            "model_name": model.name,
                            "scenario": scenario,
                            "repetition": repetition,
                            "transcript": [],
                            "latencies": [],
                            "system_error": f"Model load failed: {exc}",
                            "failed_turn": 0,
                            "response_max_tokens": int(
                                getattr(getattr(model, "config", None), "max_tokens", 0) or 0
                            ),
                            "response_temperature": response_temperature,
                            "response_thinking_level": response_thinking_level,
                            "response_effort": response_effort,
                        })
                        completed_steps += 1
                continue

            try:
                for repetition in range(1, repetitions + 1):
                    for scenario in scenarios:
                        transcript: List[dict] = []
                        latencies: List[float] = []
                        system_error = ""
                        failed_turn = None

                        for turn in scenario.turns:
                            transcript.append({
                                "turn": turn.turn,
                                "role": "user",
                                "content": turn.user_message,
                            })
                            prompt = self._conversation_prompt(
                                transcript[:-1], turn.user_message
                            )
                            try:
                                response = model.generate(prompt)
                            except Exception as exc:
                                system_error = f"Generation failed at turn {turn.turn}: {exc}"
                                failed_turn = turn.turn
                                break

                            if response.error:
                                system_error = (
                                    f"Generation failed at turn {turn.turn}: "
                                    f"{response.error}"
                                )
                                failed_turn = turn.turn
                                break

                            prediction = str(response.prediction or "").strip()
                            if not prediction:
                                system_error = (
                                    f"Generation failed at turn {turn.turn}: empty response"
                                )
                                failed_turn = turn.turn
                                break

                            transcript.append({
                                "turn": turn.turn,
                                "role": "assistant",
                                "content": prediction,
                                "latency_ms": round(
                                    float(response.latency_seconds or 0.0) * 1000.0,
                                    3,
                                ),
                                "finish_reason": getattr(response, "finish_reason", None),
                                "generation_metadata": dict(
                                    getattr(response, "metadata", {}) or {}
                                ),
                            })
                            latencies.append(float(response.latency_seconds or 0.0))

                        pending_runs.append({
                            "model_name": model.name,
                            "scenario": scenario,
                            "repetition": repetition,
                            "transcript": transcript,
                            "latencies": latencies,
                            "system_error": system_error,
                            "failed_turn": failed_turn,
                            "response_max_tokens": int(
                                getattr(getattr(model, "config", None), "max_tokens", 0) or 0
                            ),
                            "response_temperature": response_temperature,
                            "response_thinking_level": response_thinking_level,
                            "response_effort": response_effort,
                        })
                        completed_steps += 1
                        if on_progress:
                            on_progress(
                                completed_steps / max(total_steps, 1),
                                f"[{model.name}] {scenario.scenario_id} "
                                f"repetition {repetition}/{repetitions} complete",
                            )
            finally:
                model.unload()

        judgeable_runs = [run for run in pending_runs if not run["system_error"]]
        judge_loaded = False
        judge_load_error = None
        if hasattr(judge_model, "config"):
            current_max = int(getattr(judge_model.config, "max_tokens", 0) or 0)
            requested = max(2048, current_max)
            if hasattr(judge_model, "configure_generation"):
                judge_model.configure_generation(requested, 0.0)
            else:
                judge_model.config.max_tokens = min(requested, 4096)
                if hasattr(judge_model.config, "temperature"):
                    judge_model.config.temperature = 0.0
        judge_max_tokens = int(
            getattr(getattr(judge_model, "config", None), "max_tokens", 0) or 0
        )
        judge_temperature = float(
            getattr(getattr(judge_model, "config", None), "temperature", 0.0) or 0.0
        )
        judge_extra = getattr(getattr(judge_model, "config", None), "extra", {})
        if not isinstance(judge_extra, dict):
            judge_extra = {}
        judge_thinking_level = str(
            judge_extra.get("thinking_level") or ""
        ).strip().lower() or None
        judge_effort = str(
            judge_extra.get("effort") or ""
        ).strip().lower() or None
        if judgeable_runs:
            if on_progress:
                on_progress(
                    completed_steps / max(total_steps, 1),
                    f"Loading judge {judge_model.name}…",
                )
            try:
                judge_model.load()
                judge_loaded = True
            except Exception as exc:
                judge_load_error = f"Judge load failed: {exc}"
        records: List[ScenarioEvalRecord] = []

        try:
            for run in pending_runs:
                scenario = run["scenario"]
                transcript = run["transcript"]
                latencies = run["latencies"]

                if run["system_error"]:
                    records.append(self._unscored_scenario_record(
                        run,
                        judge_model.name,
                        judge_max_tokens,
                        judge_temperature,
                        judge_thinking_level,
                        judge_effort,
                        evaluation_status="system_error",
                        judge_attempted=0,
                    ))
                    completed_steps += 1
                    if on_progress:
                        on_progress(
                            completed_steps / max(total_steps, 1),
                            f"Execution failed {run['model_name']} · "
                            f"{scenario.scenario_id} · repetition {run['repetition']}",
                        )
                    continue

                if judge_load_error:
                    records.append(self._unscored_scenario_record(
                        run,
                        judge_model.name,
                        judge_max_tokens,
                        judge_temperature,
                        judge_thinking_level,
                        judge_effort,
                        evaluation_status="judge_error",
                        judge_attempted=1,
                        judge_error=judge_load_error,
                    ))
                    completed_steps += 1
                    continue

                judge_error = None
                parsed: dict = {}
                judge_notes = ""
                judge_raw_response = ""

                try:
                    judge_prompt = self._judge_prompt(
                        scenario, transcript, run["system_error"]
                    )
                    judge_response = judge_model.generate_structured(judge_prompt)
                    judge_raw_response = judge_response.prediction or ""
                    if judge_response.error:
                        raise RuntimeError(judge_response.error)
                    try:
                        parsed = self._parse_judge_json(judge_raw_response)
                        self._validate_judge_payload(parsed, scenario)
                    except Exception as first_error:
                        repair_response = judge_model.generate_structured(
                            self._judge_repair_prompt(
                                judge_prompt,
                                judge_raw_response,
                                str(first_error),
                            )
                        )
                        if repair_response.error:
                            raise RuntimeError(
                                f"Initial judge JSON error: {first_error}; "
                                f"repair request failed: {repair_response.error}"
                            )
                        judge_raw_response = repair_response.prediction or ""
                        try:
                            parsed = self._parse_judge_json(judge_raw_response)
                            self._validate_judge_payload(parsed, scenario)
                        except Exception as retry_error:
                            raise ValueError(
                                f"Initial judge JSON error: {first_error}; "
                                f"repair JSON error: {retry_error}"
                            ) from retry_error
                    judge_notes = str(parsed.get("notes") or "")
                except Exception as exc:
                    judge_error = str(exc)

                if judge_error:
                    records.append(self._unscored_scenario_record(
                        run,
                        judge_model.name,
                        judge_max_tokens,
                        judge_temperature,
                        judge_thinking_level,
                        judge_effort,
                        evaluation_status="judge_error",
                        judge_attempted=1,
                        judge_error=judge_error,
                        judge_response=judge_raw_response,
                    ))
                    completed_steps += 1
                    if on_progress:
                        on_progress(
                            completed_steps / max(total_steps, 1),
                            f"Judging failed {run['model_name']} · "
                            f"{scenario.scenario_id} · repetition {run['repetition']}",
                        )
                    continue

                raw_checkpoints = parsed.get("checkpoint_results", [])
                checkpoint_by_turn = {
                    self._non_negative_int(item.get("turn")): item
                    for item in raw_checkpoints
                    if isinstance(item, dict)
                }
                normalized_checkpoints = []
                for turn in scenario.turns:
                    item = checkpoint_by_turn.get(turn.turn, {})
                    met = 1 if item.get("met") in (1, True, "1", "true", "True") else 0
                    quality = self._score_1_to_5(item.get("quality_score"))
                    normalized_checkpoints.append({
                        "turn": turn.turn,
                        "expected_assistant_action": turn.expected_assistant_action,
                        "critical_checkpoint": turn.critical_checkpoint,
                        "applicable_dimensions": turn.applicable_dimensions,
                        "met": met,
                        "quality_score": quality,
                        "rationale": str(item.get("rationale") or ""),
                    })

                met_values = [item["met"] for item in normalized_checkpoints]
                quality_values = [
                    item["quality_score"] for item in normalized_checkpoints
                    if item["quality_score"] is not None
                ]
                checkpoint_completion = (
                    sum(met_values) / len(met_values) if met_values else 0.0
                )
                checkpoint_quality = (
                    sum(quality_values) / len(quality_values)
                    if quality_values else None
                )
                critical_checkpoint_values = [
                    item["met"] for item in normalized_checkpoints
                    if item["critical_checkpoint"]
                ]
                critical_checkpoint_completion = (
                    sum(critical_checkpoint_values) / len(critical_checkpoint_values)
                    if critical_checkpoint_values else None
                )
                critical_failure = (
                    1 if parsed.get("critical_failure") in
                    (1, True, "1", "true", "True") else 0
                )
                critical_failure_type = (
                    str(parsed.get("critical_failure_type"))
                    if parsed.get("critical_failure_type") else None
                )
                goal_achieved = (
                    1 if parsed.get("goal_achieved") in
                    (1, True, "1", "true", "True") else 0
                )
                goal_rationale = str(parsed.get("goal_rationale") or "").strip()
                terminal_state_valid = (
                    1 if parsed.get("terminal_state_valid") in
                    (1, True, "1", "true", "True") else 0
                )
                all_checkpoints_met = int(checkpoint_completion == 1.0)
                safe_completion = int(
                    goal_achieved == 1 and critical_failure == 0
                )
                end_to_end_success = int(goal_achieved == 1)
                # Backward-compatible alias for older exports.
                task_success = end_to_end_success

                scores = parsed.get("scores", {}) if isinstance(parsed.get("scores"), dict) else {}
                counts = parsed.get("counts", {}) if isinstance(parsed.get("counts"), dict) else {}
                assistant_turns = sum(
                    1 for item in transcript if item.get("role") == "assistant"
                )
                excess_turns = max(0, assistant_turns - len(scenario.turns))
                mean_latency_ms, total_latency_ms = self._mean_latency_ms(latencies)

                records.append(ScenarioEvalRecord(
                    schema_version="3.0",
                    evaluation_mode="scripted_text_multiturn",
                    evaluation_status="scored",
                    model_name=run["model_name"],
                    judge_model=judge_model.name,
                    scenario_id=scenario.scenario_id,
                    scenario_title=scenario.title,
                    scenario_domain=scenario.domain,
                    scenario_language=scenario.language,
                    scenario_difficulty=scenario.difficulty,
                    repetition=run["repetition"],
                    execution_success=1,
                    judge_attempted=1,
                    judge_success=1,
                    valid_evaluated_run=1,
                    end_to_end_success=end_to_end_success,
                    failed_turn=None,
                    error_category=None,
                    response_max_tokens=int(run.get("response_max_tokens") or 0),
                    judge_max_tokens=judge_max_tokens,
                    response_temperature=float(
                        run.get("response_temperature") or 0.0
                    ),
                    judge_temperature=judge_temperature,
                    response_thinking_level=run.get("response_thinking_level"),
                    judge_thinking_level=judge_thinking_level,
                    response_effort=run.get("response_effort"),
                    judge_effort=judge_effort,
                    goal_achieved=goal_achieved,
                    goal_rationale=goal_rationale,
                    terminal_state_valid=terminal_state_valid,
                    safe_completion=safe_completion,
                    all_checkpoints_met=all_checkpoints_met,
                    task_success=task_success,
                    checkpoint_completion=checkpoint_completion,
                    checkpoint_quality=checkpoint_quality,
                    critical_checkpoint_completion=critical_checkpoint_completion,
                    critical_failure=critical_failure,
                    critical_failure_type=critical_failure_type,
                    information_gathering=self._score_1_to_5(scores.get("information_gathering")),
                    instruction_following=self._score_1_to_5(scores.get("instruction_following")),
                    state_constraint_tracking=self._score_1_to_5(scores.get("state_constraint_tracking")),
                    correction_recovery=self._score_1_to_5(scores.get("correction_recovery")),
                    domain_accuracy=self._score_1_to_5(scores.get("domain_accuracy")),
                    relevance_efficiency=self._score_1_to_5(scores.get("relevance_efficiency")),
                    clarity_actionability=self._score_1_to_5(scores.get("clarity_actionability")),
                    safety_cost_awareness=self._score_1_to_5(scores.get("safety_cost_awareness")),
                    conversational_coherence=self._score_1_to_5(scores.get("conversational_coherence")),
                    turn_taking_fluency=self._score_1_to_5(scores.get("turn_taking_fluency")),
                    interruption_handling=self._score_1_to_5(scores.get("interruption_handling")),
                    response_timing_appropriateness=None,
                    speech_intelligibility=self._score_1_to_5(scores.get("speech_intelligibility")),
                    speech_naturalness=self._score_1_to_5(scores.get("speech_naturalness")),
                    actual_turns=assistant_turns,
                    target_turns=len(scenario.turns),
                    maximum_turns=int(scenario.maximum_turns or len(scenario.turns)),
                    maximum_turn_violation=int(
                        assistant_turns > int(scenario.maximum_turns or len(scenario.turns))
                    ),
                    mean_latency_ms=mean_latency_ms,
                    total_latency_ms=total_latency_ms,
                    repeated_questions=self._non_negative_int(counts.get("repeated_questions")),
                    unsupported_assumptions=self._non_negative_int(counts.get("unsupported_assumptions")),
                    state_constraint_errors=self._non_negative_int(counts.get("state_constraint_errors")),
                    correction_failures=self._non_negative_int(counts.get("correction_failures")),
                    successful_recoveries=self._non_negative_int(counts.get("successful_recoveries")),
                    premature_termination=self._non_negative_int(counts.get("premature_termination")),
                    excess_turns=excess_turns,
                    examiner_redirections=None,
                    asr_errors=None,
                    system_error=None,
                    judge_error=None,
                    checkpoint_results_json=json.dumps(normalized_checkpoints, ensure_ascii=False),
                    transcript_json=json.dumps(transcript, ensure_ascii=False),
                    judge_response=judge_raw_response,
                    judge_notes=judge_notes,
                ))

                completed_steps += 1
                if on_progress:
                    on_progress(
                        completed_steps / max(total_steps, 1),
                        f"Judged {run['model_name']} · {scenario.scenario_id} · "
                        f"repetition {run['repetition']}",
                    )
        finally:
            if judge_loaded:
                judge_model.unload()

        return pd.DataFrame([vars(record) for record in records])

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

        # Stop cleanly when no rows were loaded/matched. In audio mode this
        # usually means the selected filenames did not match the CSV audio_file
        # values. This also prevents division-by-zero in progress reporting.
        if not dataset:
            if on_progress:
                on_progress(
                    0.0,
                    "No evaluation samples were loaded. "
                    "In audio mode, check that the selected filenames exactly match "
                    "the CSV audio_file column.",
                )
            return pd.DataFrame(
                columns=list(EvalRecord.__dataclass_fields__.keys())
            )

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
                        done / max(total_steps, 1),
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
                on_progress(done / max(total_steps, 1), f"{model.name} unloaded.")

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
                            done / max(total_steps, 1),
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
