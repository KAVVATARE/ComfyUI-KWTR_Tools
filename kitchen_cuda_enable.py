"""
Re-enable comfy-kitchen's CUDA backend on a cu128 torch.

comfy/quant_ops.py calls ck.registry.disable("cuda") whenever torch.version.cuda
is below 13, because comfy-kitchen's CUDA kernels are built with CUDA 13.0.
On this Paperspace box the host driver is 550 (CUDA 12.4), but
comfyui_start.sh installs cuda-compat-13-0 and prepends its directory to
LD_LIBRARY_PATH (CUDA Forward Compatibility), so the loaded user-mode driver
reports 13000 and those kernels run fine. Without this patch the core
Model Sparse Attention node (sol-attn / sla) silently falls back to dense
("no compiled sol_attn kernel for this GPU") and int8 linears skip the
CUDA kernels.

Only re-enables when the loaded driver API is >= 13.0 and a tiny sol_attn +
int8_linear smoke test passes; otherwise the backend stays disabled.
Opt out with KWTR_KITCHEN_CUDA=0.
"""
import ctypes
import logging
import os

import torch

_TAG = "[KWTR kitchen_cuda_enable]"


def _driver_version():
    try:
        lib = ctypes.CDLL("libcuda.so.1")
        v = ctypes.c_int(0)
        if lib.cuDriverGetVersion(ctypes.byref(v)) != 0:
            return None
        return v.value
    except Exception:
        return None


def _smoke_test(ck):
    dev = torch.device("cuda")
    q, k, v = (torch.randn(1, 512, 2, 128, device=dev, dtype=torch.bfloat16) for _ in range(3))
    ck.sol_attn(q, k, v, tau=1.0)
    x = torch.randn(64, 256, device=dev, dtype=torch.bfloat16)
    w = torch.randint(-127, 127, (256, 256), device=dev, dtype=torch.int8)
    s = torch.full((256,), 0.01, device=dev, dtype=torch.float32)
    ck.int8_linear(x, w, s, out_dtype=torch.bfloat16, convrot=True)
    torch.cuda.synchronize()


def _enable_kitchen_cuda():
    if os.environ.get("KWTR_KITCHEN_CUDA", "1") == "0":
        return
    try:
        import comfy_kitchen as ck
        import comfy.quant_ops  # noqa: F401  (make sure core has made its decision first)
    except Exception:
        return
    if torch.version.cuda is None or not torch.cuda.is_available():
        return
    if tuple(map(int, str(torch.version.cuda).split("."))) >= (13,):
        return  # core already leaves the CUDA backend enabled
    if "cuda" not in ck.registry._disabled:
        return
    drv = _driver_version()
    if drv is None or drv < 13000:
        logging.info(f"{_TAG} driver API {drv} < 13000 (no forward compat); leaving comfy-kitchen CUDA backend disabled")
        return
    ck.registry.enable("cuda")
    try:
        _smoke_test(ck)
    except Exception as e:
        ck.registry.disable("cuda")
        logging.warning(f"{_TAG} smoke test failed, comfy-kitchen CUDA backend stays disabled: {e}")
        return
    # attention.py caches this at import (before custom nodes load), so the
    # "comfy kitchen attention" option of Model Attention Backend would stay off.
    try:
        import comfy.ldm.modules.attention as attention
        attention.COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE = ck.int8_attention_is_available()
    except Exception:
        pass
    logging.info(f"{_TAG} driver API {drv}: re-enabled comfy-kitchen CUDA backend "
                 f"(sol_attn available: {ck.sol_attn_is_available(torch.device('cuda'))})")


_enable_kitchen_cuda()
