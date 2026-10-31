"""Pick the compute device so the kit runs on laptops without an NVIDIA GPU.

Order with --device auto (the default): NVIDIA CUDA (also AMD ROCm on Linux, which
PyTorch exposes as "cuda"), then Apple Silicon MPS, then CPU. DirectML (AMD/Intel GPUs
on Windows) is opt-in with --device dml, because T5 training on it is slow and
some ops fall back to CPU; it is mainly useful for inference.

Override from the shell with SAGE_DEVICE=cpu|cuda|mps|dml.
"""
import contextlib
import os

import torch


def pick(name="auto"):
    name = (os.environ.get("SAGE_DEVICE") or name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "dml":
        import torch_directml  # pip install torch-directml (Windows only)
        return torch_directml.device()
    if name == "mps":
        # Let ops MPS lacks run on CPU instead of crashing.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if name == "cpu":
        # Use the physical cores; PyTorch's default can oversubscribe on laptops.
        n = os.cpu_count() or 4
        torch.set_num_threads(max(1, n // 2) if n > 4 else n)
    return torch.device(name)


def kind(dev):
    return dev.type if isinstance(dev, torch.device) else str(dev).split(":")[0]


def bf16_ok(dev):
    """bf16 autocast only on CUDA GPUs that support it. T5 is unstable in fp16, so everything else stays fp32."""
    return kind(dev) == "cuda" and torch.cuda.is_bf16_supported()


def autocast(dev, enabled):
    return torch.autocast("cuda", dtype=torch.bfloat16) if enabled and kind(dev) == "cuda" else contextlib.nullcontext()


def describe(dev):
    k = kind(dev)
    if k == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    if k == "cpu":
        return f"cpu ({torch.get_num_threads()} threads)"
    return str(dev)


def load_causal_lm(model_id, load_4bit=False, device="auto", trust_remote_code=False):
    """Load a Hugging Face causal LM on whatever this machine has.

    CUDA: bf16/fp16 with device_map="auto", optional 4-bit (bitsandbytes).
    Anything else: 4-bit is ignored (bitsandbytes needs NVIDIA); fp16 on MPS/DirectML,
    fp32 on CPU. A 7B model then needs ~28 GB RAM on CPU, so use a 0.5B-1.5B model locally.
    """
    from transformers import AutoModelForCausalLM
    dev = pick(device)
    k = kind(dev)
    if k == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kw = {"dtype": dtype, "device_map": "auto", "trust_remote_code": trust_remote_code}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
        return AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    if load_4bit:
        print("note: --load-4bit needs an NVIDIA GPU; loading unquantised on", describe(dev), flush=True)
    dtype = torch.float32 if k == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, trust_remote_code=trust_remote_code)
    print("device:", describe(dev), flush=True)
    return model.to(dev).eval()
