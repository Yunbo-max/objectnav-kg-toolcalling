"""
Local VLM adapter for Co-NavGPT.
Replaces OpenAI API calls with local Qwen2.5 models on RTX 4090.
Supports both text-only (Qwen2.5-3B-Instruct) and vision (Qwen2.5-VL-3B/7B) modes.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
import os
import time

# Singleton model holder
_model = None
_tokenizer = None
_processor = None
_model_name = None


def load_model(model_path, device="cuda:0", model_type="text"):
    """Load a local model. Call once at startup."""
    global _model, _tokenizer, _processor, _model_name

    if _model is not None and _model_name == model_path:
        return  # Already loaded

    print(f"Loading model from {model_path} on {device}...")
    start = time.time()
    torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    device_map = {"": device}

    if model_type == "vl":
        # Vision-Language model (Qwen2.5-VL)
        from transformers import Qwen2_5_VLForConditionalGeneration
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        _processor = AutoProcessor.from_pretrained(model_path)
        _tokenizer = None
    else:
        # Text-only model (Qwen2.5-Instruct)
        _tokenizer = AutoTokenizer.from_pretrained(model_path)
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        _processor = None

    _model_name = model_path
    print(f"Model loaded in {time.time() - start:.1f}s")


def generate_response(system_prompt, user_prompt, max_new_tokens=256, temperature=0.01):
    """
    Generate response from local model.
    Returns string in same format as OpenAI API response content.
    """
    global _model, _tokenizer, _processor

    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    if _tokenizer is not None:
        # Text-only path
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    else:
        # VL path — combine system+user into single user message (VL models handle system role poorly)
        combined = system_prompt.strip() + "\n\n" + user_prompt.strip()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": combined}]},
        ]
        text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = _processor(text=[text], return_tensors="pt", padding=True).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    response = (_tokenizer or _processor).decode(
        output_ids[0][input_len:], skip_special_tokens=True
    )

    return response.strip()


def generate_response_with_image(system_prompt, user_prompt, image, max_new_tokens=256, temperature=0.01):
    """
    Generate response from VL model WITH an image input.
    `image` can be a PIL Image or file path.
    """
    global _model, _processor

    if _processor is None:
        raise RuntimeError("VL model not loaded. Call load_model() with model_type='vl'.")

    from PIL import Image as PILImage
    if isinstance(image, str):
        image = PILImage.open(image)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_prompt},
        ]},
    ]

    text = _processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _processor(
        text=[text], images=[image], return_tensors="pt", padding=True
    ).to(_model.device)

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
        )

    input_len = inputs["input_ids"].shape[1]
    response = _processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return response.strip()


# OpenAI-compatible wrapper for drop-in replacement
class LocalLLMResponse:
    """Mimics OpenAI API response structure."""
    def __init__(self, content):
        self.choices = [type('obj', (object,), {
            'message': type('obj', (object,), {'content': content})()
        })()]
        self.usage = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }

    def __getitem__(self, key):
        if key == 'choices':
            return [{'message': {'content': self.choices[0].message.content}}]
        if key == 'usage':
            return self.usage
        raise KeyError(key)


def chat_completion_create(model, messages, temperature=0, **kwargs):
    """
    Drop-in replacement for openai.ChatCompletion.create().
    Ignores the `model` parameter — uses the locally loaded model.
    """
    system_prompt = ""
    user_prompt = ""
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] == "user":
            user_prompt = msg["content"]

    max_new_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 256))
    content = generate_response(
        system_prompt,
        user_prompt,
        max_new_tokens=max_new_tokens,
        temperature=max(temperature, 0.01),
    )
    return LocalLLMResponse(content)


if __name__ == "__main__":
    # Quick test
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "/tf/notebooks/models/Qwen2.5-3B-Instruct"
    device = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"

    load_model(model_path, device=device, model_type="text")

    resp = generate_response(
        "You are a navigation assistant.",
        "Given robot_0 at (240,240) and frontier_0 at (180,175), frontier_1 at (195,280), assign each robot a frontier.\n[output:]"
    )
    print(f"Response:\n{resp}")
