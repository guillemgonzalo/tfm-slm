#!/usr/bin/env python3
"""
Lanza una tanda de preguntas al API del chat tfm-slm y muestra las respuestas.
Requiere el servicio expuesto (p.ej. `kubectl port-forward svc/tfm-slm-chat-service -n tfm-slm 8000:8000`).

Uso:
  uv run python app/ask_chat.py
  CHAT_URL=http://localhost:8000 uv run python app/ask_chat.py
"""

import json
import os
import urllib.request

CHAT_URL = os.getenv("CHAT_URL", "http://localhost:8000")

PROMPTS = [
    "Give me three tips to study better.",
    "Write a short poem about the sea.",
    "Explain what a neural network is in one sentence.",
    "List three healthy breakfast ideas.",
    "What is the capital of France?",
    "Continue this story: Once upon a time, in a distant land,",
    "Translate 'good morning' to Spanish.",
    "Give me a recipe for a simple sandwich.",
    "What are the benefits of exercise?",
    "Write a motivational quote about learning.",
]


def ask(prompt: str, max_tokens: int = 64, temperature: float = 0.6, top_k: int = 30) -> str:
    payload = json.dumps(
        {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature, "top_k": top_k}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{CHAT_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())["response"]


def main() -> None:
    print(f"Enviando {len(PROMPTS)} preguntas a {CHAT_URL}\n")
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"[{i}/{len(PROMPTS)}] Q: {prompt}")
        try:
            answer = ask(prompt)
            print(f"      A: {answer}\n")
        except Exception as e:
            print(f"      ERROR: {e}\n")
    print("Listo. Historial persistido en .output/chat_history.jsonl del servidor.")


if __name__ == "__main__":
    main()
