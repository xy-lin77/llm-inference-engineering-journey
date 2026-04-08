import argparse
import time
from typing import Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_MAP: Dict[str, str] = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen72b": "Qwen/Qwen2.5-72B-Instruct",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FP32 / BF16 / AMP inference for Qwen models."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen7b", "qwen72b"],
        required=True,
        help="Which model to load.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16", "amp"],
        required=True,
        help="Precision mode: fp32 / bf16 / amp.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="用一句话解释什么是 Transformer。",
        help="User prompt.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate.",
    )
    return parser.parse_args()

def get_device_type() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_model_and_tokenizer(model_name: str, precision: str):
    model_id = MODEL_MAP[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if precision == "fp32":
        dtype = torch.float32
    elif precision in ["bf16", "amp"]:
        dtype = torch.bfloat16
    else:
        raise ValueError("Unknown precision")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    print("cuda device count:", torch.cuda.device_count())
    print("hf_device_map:", model.hf_device_map)
    return tokenizer, model, model_id

def build_inputs(tokenizer, model, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt")
    first_param_device = next(model.parameters()).device
    inputs = {k: v.to(first_param_device) for k, v in inputs.items()}
    return inputs

def run_generation(model, inputs, precision: str, max_new_tokens: int):
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        if precision == "amp":
            if not use_cuda:
                raise RuntimeError("amp needs CUDA GPU, but CUDA not available")
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        else:
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    if use_cuda:
        torch.cuda.synchronize()
    end_time = time.time()
    peak_memory_mb = None
    if use_cuda:
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    return outputs, end_time - start_time, peak_memory_mb

def main():
    args = parse_args()
    print(f"Loading model: {args.model} -> {MODEL_MAP[args.model]}")
    print(f"Precision mode: {args.precision}")
    tokenizer, model, model_id = load_model_and_tokenizer(args.model, args.precision)
    inputs = build_inputs(tokenizer, model, args.prompt)
    outputs, elapsed_time, peak_memory_mb = run_generation(
        model=model,
        inputs=inputs,
        precision=args.precision,
        max_new_tokens=args.max_new_tokens,
    )
    decoded_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n===== Result =====")
    print(decoded_text)
    print("\n===== Stats =====")
    print(f"Model ID: {model_id}")
    print(f"Precision: {args.precision}")
    print(f"Time: {elapsed_time:.4f} s")
    if peak_memory_mb is not None:
        print(f"Peak GPU memory: {peak_memory_mb:.2f} MB")
    else:
        print("Peak GPU memory: N/A (CUDA not available)")

if __name__ == "__main__":
    main()
