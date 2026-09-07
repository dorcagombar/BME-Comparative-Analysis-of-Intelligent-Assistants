# LLM Evaluation Interface

A web-based interface for evaluating multiple LLM models on QA datasets. Supports cloud API models (OpenAI GPT-4o, Google Gemini, Fireworks AI) and optional local HuggingFace models. Features include dataset cleaning, audio transcription, noise robustness testing, and automated metric evaluation (ROUGE, BLEU, BERTScore).

---
## Data Availability

The full audio dataset used in this project is available on Hugging Face:

Cnyuan/modeleval_interface
https://huggingface.co/datasets/Cnyuan/modeleval_interface/tree/main

Due to the large size of the audio files, the complete audio data is not stored directly in this GitHub repository. The `data/` folder only contains the text-based version of the dataset, including the English and Chinese question files and reference information used for evaluation.

Users who want to reproduce the full audio-based experiments should download the audio files from the Hugging Face dataset page and place them in the corresponding local audio directories before running the evaluation scripts.

## Requirements

- Python 3.10+
- [Conda](https://docs.conda.io/en/latest/) (recommended)
- API keys for the models you want to use

---

## Setup

### 1. Create and activate a conda environment

```bash
conda create -n llm-eval python=3.10 -y
conda activate llm-eval
```

### 2. Clone the repository

```bash
git clone https://github.com/XXVM24/modeleval-interface.git
cd interface
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root:

```bash
FIREWORKS_API_KEY=fw_xxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxx
GOOGLE_API_KEY=AIzaxxxxxxxxxx
```

> You only need keys for the models you plan to use. Leave unused keys blank.
> To get OpenAI API key:please visit:https://platform.openai.com/home
> Gemini API key:https://aistudio.google.com/prompts/new_chat
> firework AI:https://fireworks.ai/ 
---

## Launch

```bash
conda activate llm-eval
python app.py
```

Then open **http://localhost:6006** in your browser.

---

## Models Supported

| Model | Provider | Key Required |
|---|---|---|
| GPT-4o (Audio) | OpenAI | `OPENAI_API_KEY` |
| Gemini-2.5-Pro | Google | `GOOGLE_API_KEY` |
| Qwen2-7B | Fireworks AI | `FIREWORKS_API_KEY` |
| Qwen2.5-14B | Fireworks AI | `FIREWORKS_API_KEY` |
| Mistral-7B | Fireworks AI | `FIREWORKS_API_KEY` |
| Local HF models | Local (GPU) | — |

---

## Dataset Format

Upload a CSV or JSON file with the following columns:

| Column | Description |
|---|---|
| `question` | The question to ask the model |
| `reference` | The expected (ground-truth) answer |
| `audio_file` *(optional)* | Path to audio file for speech-based evaluation |

A sample dataset is provided at `data/sample_qa.json`.

---

## Features

- **Dataset Cleaner** — detects and removes malformed rows before evaluation
- **Noise Robustness** — adds background noise at controlled SNR levels to test model robustness
- **Audio Transcription** — transcribes audio using OpenAI Whisper before sending to models
- **Metrics** — ROUGE-1/2/L, BLEU, audio quality score (SNR, speech ratio, clarity)
- **Export** — results saved to CSV in the `results/` folder

---

## Optional: BERTScore

BERTScore requires PyTorch and is disabled by default. To enable:

```bash
# Install CPU-only PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install bert-score>=0.3.13
```

Then uncomment the `bert-score` line in `requirements.txt`.

---

## Optional: Local HuggingFace Models

GPU recommended. Uncomment the relevant model block in `config/models.yaml` and set the local model path.

```bash
pip install torch transformers accelerate sentencepiece
```
