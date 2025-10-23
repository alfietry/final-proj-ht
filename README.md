# LLM t-test demo

This is a tiny demo that compares OpenAI's GPT-4 and a local Ollama gemma3:4b model on a one-sample t-test task.

What it does
- Generates 30 data points from N(mean=10, sd=5) with a fixed seed
- Computes the ground-truth one-sample t-test p-value using scipy (~0.62 for the generated sample)
- Prompts each model to perform the t-test and return a JSON with p_value and conclusion
- Parses the model outputs and logs everything to Weights & Biases in offline mode

Requirements
- Python 3.8+
- Install packages from `requirements.txt`

Environment variables
- `OPENAI_API_KEY` (optional). If not set, the OpenAI step is skipped.
- Ollama must be running locally at `http://localhost:11434` with model name `gemma3:4b` to use the Ollama step. If not available, that step is skipped.

Run
1. (Optional) Set your OpenAI key:

```pwsh
$env:OPENAI_API_KEY = 'sk-...'
```

2. Install deps and run:

```pwsh
python -m pip install -r requirements.txt
python demo_ttest.py
```

Notes
- W&B logs are stored locally in offline mode. You can set `WANDB_MODE` to `online` and provide `WANDB_API_KEY` to upload logs.
