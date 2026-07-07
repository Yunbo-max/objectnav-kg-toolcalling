#!/usr/bin/env python3
"""Minimal OpenAI-compatible image chat server for local Qwen2.5-VL smoke tests."""

import argparse
import base64
import io
import json
import re
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


MODEL = None
PROCESSOR = None
MODEL_NAME = "qwen2.5-vl"


def _decode_image(url):
    if not url:
        return None
    if url.startswith("data:"):
        _, data = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")
    return Image.open(url).convert("RGB")


def _extract_prompt_and_image(messages):
    texts = []
    image = None
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            texts.append(content)
            continue
        for item in content:
            if item.get("type") == "text":
                texts.append(item.get("text", ""))
            elif item.get("type") == "image_url":
                image_url = item.get("image_url", {})
                image = _decode_image(image_url.get("url"))
    return "\n\n".join(t for t in texts if t), image


def _options_from_probability_request(messages):
    req = None
    for msg in reversed(messages):
        req = msg.get("return_string_probabilities")
        if req:
            break
    if not req:
        return None
    return re.findall(r"[A-Za-z]+", req)


def _score_options(options, content):
    if not options:
        return ""
    lowered = content.lower()
    scores = []
    for option in options:
        scores.append(1.0 if re.search(rf"\b{re.escape(option.lower())}\b", lowered) else 0.0)
    if not any(scores):
        scores = [1.0 / len(options)] * len(options)
    else:
        total = sum(scores)
        scores = [s / total for s in scores]
    return scores


def generate(messages, max_new_tokens):
    prompt, image = _extract_prompt_and_image(messages)
    if image is None:
        image = Image.new("RGB", (32, 32), color=(255, 255, 255))

    chat_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    text = PROCESSOR.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
    inputs = PROCESSOR(text=[text], images=[image], padding=True, return_tensors="pt").to(MODEL.device)

    with torch.no_grad():
        output_ids = MODEL.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    input_len = inputs["input_ids"].shape[1]
    return PROCESSOR.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


class Handler(BaseHTTPRequestHandler):
    def _json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": MODEL_NAME, "object": "model"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        messages = payload.get("messages", [])
        max_tokens = int(payload.get("max_tokens", 128))
        max_new_tokens = max(1, min(max_tokens, 128))
        content = generate(messages, max_new_tokens)
        scores = _score_options(_options_from_probability_request(messages), content)
        self._json(
            200,
            {
                "id": "chatcmpl-local-qwen-vl",
                "object": "chat.completion",
                "model": MODEL_NAME,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content, "scores": scores},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            },
        )

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31511)
    args = parser.parse_args()

    global MODEL, PROCESSOR, MODEL_NAME
    MODEL_NAME = args.model_path
    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    print(f"Loading {args.model_path} on {args.device}", flush=True)
    MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map={"": args.device},
    )
    PROCESSOR = AutoProcessor.from_pretrained(args.model_path)
    print(f"Serving on http://{args.host}:{args.port}", flush=True)
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
