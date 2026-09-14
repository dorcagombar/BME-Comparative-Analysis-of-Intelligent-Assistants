from pathlib import Path


# Project root:
# BME-Comparative-Analysis-of-Intelligent-Assistants/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

VOICE_SAMPLES_DIR = PROJECT_ROOT / "data" / "voice_samples"


def resolve_audio_path(audio_file: str) -> Path:
    """
    Resolve an audio filename from the CSV to its actual file.

    Example:
        A-EN-Q014.m4a
            ->
        data/voice_samples/A-EN/A-EN-Q014.m4a

        B-DE-Q003.m4a
            ->
        data/voice_samples/B-DE/B-DE-Q003.m4a
    """

    if not audio_file:
        raise ValueError("No audio filename provided.")

    audio_file = str(audio_file).strip()

    parts = audio_file.split("-")

    if len(parts) < 3:
        raise ValueError(
            f"Unexpected audio filename format: {audio_file}"
        )

    # A-EN-Q014.m4a -> A-EN
    folder_name = f"{parts[0]}-{parts[1]}"

    audio_path = VOICE_SAMPLES_DIR / folder_name / audio_file

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio sample does not exist: {audio_path}"
        )

    return audio_path
