"""
Simple demo comparing GPT-4 (OpenAI API) and Llama-3 (local Ollama) on a one-sample t-test task.

Requirements (see requirements.txt):
  - openai
  - requests
  - scipy
  - numpy
  - wandb

Environment:
  - Set OPENAI_API_KEY to call OpenAI. If not set, the OpenAI call will be skipped.
  - Ollama is expected to run locally at http://localhost:11434 and model name assumed 'llama3'. If not available, Ollama call will be skipped.

This script:
  - generates 30 data points from N(10,5) with fixed seed
  - computes ground-truth one-sample t-test p-value using scipy
  - prompts both models to perform the t-test and extract p-value + conclusion
  - logs inputs and outputs to Weights & Biases in offline mode (no network required)

Keep it simple — minimal parsing and robust fallback.
"""
import os
import json
import math
import random
import statistics
import re
from typing import Optional, Dict, Any

import numpy as np
from scipy import stats

try:
    import openai
except Exception:
    openai = None

import requests
import wandb


def generate_data(seed=42, n=30, mean=10.0, sd=5.0):
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=mean, scale=sd, size=n).tolist()
    return data


def ground_truth_ttest(data, mu=10.0):
    tstat, pvalue = stats.ttest_1samp(data, popmean=mu)
    return float(tstat), float(pvalue)


def build_prompt(data, mu=10.0, alpha=0.05):
    # Strongly request a single JSON object only. Keep prompt short for clarity.
    prompt = (
        f"You are given a dataset of {len(data)} numbers.\n"
        f"Data: {json.dumps(data)}\n\n"
        f"Perform a two-sided one-sample t-test for H0: mu = {mu} vs H1: mu != {mu} at alpha={alpha}.\n"
    )
    prompt += (
        "\n\nReturn a single valid JSON object and nothing else with this exact schema:\n"
        "{\n  \"p_value\": <number between 0 and 1>,\n  \"conclusion\": \"reject\" or \"fail to reject\"\n}\n"
        "Do NOT include extra explanation or code fences. Use a numeric value for p_value (e.g. 0.12345)."
    )
    return prompt


def call_openai(prompt: str, model: str = "gpt-4") -> Optional[str]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or openai is None:
        print("OpenAI API key not set or openai package missing — skipping OpenAI call.")
        return None
    # Try to support both modern openai>=1.0 client and the older interface.
    messages = [
        {"role": "system", "content": (
            "You are a concise statistics assistant. Output ONLY a single valid JSON object matching the schema: "
            "{\"p_value\": <number>, \"conclusion\": \"reject\" or \"fail to reject\"}." )},
        {"role": "user", "content": prompt},
    ]

    def _extract_content(resp_obj):
        # Try several common response shapes
        try:
            # new client: resp.choices[0].message.content
            return resp_obj.choices[0].message.content.strip()
        except Exception:
            pass
        try:
            # dict-like
            return resp_obj["choices"][0]["message"]["content"].strip()
        except Exception:
            pass
        try:
            # older openai: resp.choices[0].message['content'] or text
            return resp_obj.choices[0].text.strip()
        except Exception:
            pass
        # last resort: stringified response
        return str(resp_obj)

    try:
        # If the package exposes OpenAI (modern client), use it
        OpenAIClient = getattr(openai, "OpenAI", None)
        if OpenAIClient is not None:
            client = OpenAIClient()
            resp = client.chat.completions.create(model=model, messages=messages, max_tokens=300, temperature=0.0)
        else:
            # fallback to legacy interface (may raise on openai>=1.0)
            # keep compatibility for older installed versions
            try:
                openai.api_key = api_key
            except Exception:
                pass
            resp = openai.ChatCompletion.create(model=model, messages=messages, max_tokens=300, temperature=0.0)

        # Log raw response for debugging (W&B will capture stdout)
        try:
            print("OPENAI_RAW_RESPONSE_JSON:", json.dumps(resp, default=str))
        except Exception:
            print("OPENAI_RAW_RESPONSE (repr):", repr(resp))

        content = _extract_content(resp)
        return content
    except Exception as e:
        print(f"OpenAI call failed: {e}")
        return None


def call_ollama(prompt: str, model: str = "gemma3:4b") -> Optional[str]:
    # Ollama local API assumed at http://localhost:11434
    base = "http://localhost:11434/api"
    gen_url = f"{base}/generate"
    payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.0, "num_predict": 200}}
    try:
        r = requests.post(gen_url, json=payload, timeout=120)
        if r.status_code == 404:
            # Fallback to chat API if generate not available
            chat_url = f"{base}/chat"
            chat_payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False, "options": {"temperature": 0.0, "num_predict": 200}}
            rc = requests.post(chat_url, json=chat_payload, timeout=120)
            rc.raise_for_status()
            data = rc.json()
            msg = (data.get("message") or {}).get("content")
            return msg.strip() if isinstance(msg, str) else json.dumps(data)
        r.raise_for_status()
        data = r.json()
        # Non-streaming generate returns 'response'
        if isinstance(data, dict) and isinstance(data.get("response"), str):
            return data["response"].strip()
        return json.dumps(data)
    except Exception as e:
        print(f"Ollama call failed or Ollama not running: {e}")
        return None


def parse_json_pvalue(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    # Remove common wrappers (code fences) and trim
    cleaned = re.sub(r"```.*?```", "", text, flags=re.S).strip()

    # 1) Try to find a JSON object
    m_json = re.search(r"\{[\s\S]*\}", cleaned)
    if m_json:
        try:
            j = json.loads(m_json.group(0))
            p = None
            conclusion = None
            for k, v in j.items():
                kl = k.lower()
                if kl.startswith('p') and p is None:
                    try:
                        p = float(v)
                    except Exception:
                        # sometimes value is '≈0.12' or similar
                        vs = str(v).lstrip('≈~ ').strip()
                        try:
                            p = float(vs)
                        except Exception:
                            pass
                if isinstance(v, str) and ('reject' in v.lower() or 'fail' in v.lower()):
                    conclusion = v
            return {"p_value": p, "conclusion": conclusion, "raw": j}
        except Exception:
            pass

    # 2) Try to find explicit p-value patterns like 'p = 0.123', 'p-value:0.12', or '≈0.12'
    patterns = [r"p\s*[:=]\s*≈?\s*([0-9]*\.?[0-9]+([eE][+-]?\d+)?)",
                r"p-value\s*[:=]\s*≈?\s*([0-9]*\.?[0-9]+)",
                r"≈\s*([0-9]*\.?[0-9]+)"]
    for pat in patterns:
        m = re.search(pat, cleaned, flags=re.I)
        if m:
            try:
                pv = float(m.group(1))
                return {"p_value": pv, "conclusion": None}
            except Exception:
                continue

    # 3) Fallback: find any float-like token between 0 and 1
    tokens = re.findall(r"[0-9]*\.?[0-9]+([eE][+-]?\d+)?", cleaned)
    # re.findall above returns capture groups; better to find all numeric substrings
    num_tokens = re.findall(r"[0-9]+\.?[0-9]*([eE][+-]?\d+)?|\.[0-9]+([eE][+-]?\d+)?", cleaned)
    # flatten and re-find floats in cleaned text
    float_candidates = re.findall(r"[0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?", cleaned)
    for tok in float_candidates:
        try:
            v = float(tok)
            if 0.0 <= v <= 1.0:
                return {"p_value": v, "conclusion": None}
        except Exception:
            continue

    return None


def main():
    # --- generate data ---
    data = generate_data(seed=42, n=30, mean=10.0, sd=5.0)
    tstat, p_gt = ground_truth_ttest(data, mu=10.0)

    prompt = build_prompt(data, mu=10.0, alpha=0.05)

    # init wandb in offline mode to log runs locally
    os.environ.setdefault('WANDB_MODE', 'offline')
    wandb_run = wandb.init(project='llm-ttest-demo', name='demo-ttest', reinit=True)

    wandb_run.log({
        'data_sample': data[:5],
        'n': len(data),
        'gt_tstat': tstat,
        'gt_pvalue': p_gt,
    })

    # Call OpenAI
    openai_resp = call_openai(prompt)
    openai_parsed = parse_json_pvalue(openai_resp) if openai_resp else None

    wandb_run.log({'openai_response': openai_resp, 'openai_parsed': openai_parsed})

    # Call Ollama
    ollama_resp = call_ollama(prompt)
    ollama_parsed = parse_json_pvalue(ollama_resp) if ollama_resp else None

    wandb_run.log({'ollama_response': ollama_resp, 'ollama_parsed': ollama_parsed})

    # Print concise comparison
    print('\n--- Ground truth (scipy) ---')
    print(f't = {tstat:.4f}, p = {p_gt:.4f} (two-sided)')

    def show(name, resp, parsed):
        print(f'\n--- {name} ---')
        if resp is None:
            print('skipped or failed')
            return
        print('raw response:\n', resp)
        print('parsed:', parsed)
        if parsed and parsed.get('p_value') is not None:
            pv = parsed['p_value']
            decision = 'reject' if pv < 0.05 else 'fail to reject'
            print(f'interpreted p = {pv:.4f} -> {decision}')

    show('OpenAI (gpt-4)', openai_resp, openai_parsed)
    show('Ollama (gemma3:4b)', ollama_resp, ollama_parsed)

    wandb_run.finish()


if __name__ == '__main__':
    main()
