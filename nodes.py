from __future__ import annotations

import datetime
import gc
import math
import os
import re
import shutil
import subprocess
import torch
import folder_paths
import comfy.samplers
import comfy.clip_vision
import nodes

from . import minimax_progress_patch  # noqa: F401  (applies its patch on import)
from comfy_extras.nodes_resolution import AspectRatio, ASPECT_RATIOS


class AnyType(str):
    def __ne__(self, other):
        return False


any_type = AnyType("*")


class DualPhaseKSampler:
    """
    2つのKSampler (Advanced) を「共有ノイズスケジュール」で合体させた
    デュアルフェーズサンプラー。SDXL Base+Refinerと同じ考え方です。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # --- 接続スロット ---
                "model_1": ("MODEL",),
                "positive_1": ("CONDITIONING",),
                "negative_1": ("CONDITIONING",),
                "model_2": ("MODEL",),
                "positive_2": ("CONDITIONING",),
                "negative_2": ("CONDITIONING",),
                "latent_image": ("LATENT",),

                # --- ウィジェット ---
                "add_noise_1": (["enable", "disable"], {"default": "enable"}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "total_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "shift_model_at_step": ("INT", {"default": 10, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "sampler_name_1": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_1": (comfy.samplers.KSampler.SCHEDULERS,),
                "sampler_name_2": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_2": (comfy.samplers.KSampler.SCHEDULERS,),

                # --- cfgは指定順の対象外だったため末尾に追加 ---
                "cfg_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cfg_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom"

    def sample(
        self,
        model_1,
        positive_1,
        negative_1,
        model_2,
        positive_2,
        negative_2,
        latent_image,
        add_noise_1,
        noise_seed,
        total_steps,
        start_at_step,
        shift_model_at_step,
        end_at_step,
        sampler_name_1,
        scheduler_1,
        sampler_name_2,
        scheduler_2,
        cfg_1,
        cfg_2,
    ):
        ksampler_adv = nodes.KSamplerAdvanced()

        # --- フェーズ1: start_at_step 〜 shift_model_at_step ---
        # return_with_leftover_noise は disable にして、この区間で完全に
        # デノイズを完了させます（model_1とmodel_2はアーキテクチャ／スケジュールが
        # 異なりうるため、生のノイズスケジュールをそのまま引き継ぐと
        # sigmaの不一致でノイズが残留することがあります）。
        phase1_latent = ksampler_adv.sample(
            model_1,
            add_noise_1,
            noise_seed,
            total_steps,
            cfg_1,
            sampler_name_1,
            scheduler_1,
            positive_1,
            negative_1,
            latent_image,
            start_at_step,
            shift_model_at_step,
            "disable",  # return_with_leftover_noise
        )[0]

        # --- フェーズ2: shift_model_at_step 〜 end_at_step ---
        # add_noise は enable にして、model_2自身のスケジュールに合わせて
        # ノイズを乗せ直してからリファインします（Hires fix / img2img的な考え方）。
        phase2_latent = ksampler_adv.sample(
            model_2,
            "enable",
            noise_seed,
            total_steps,
            cfg_2,
            sampler_name_2,
            scheduler_2,
            positive_2,
            negative_2,
            phase1_latent,
            shift_model_at_step,
            end_at_step,
            "disable",  # return_with_leftover_noise
        )[0]

        return (phase2_latent,)


def _latent_upscale_model_choices():
    """Return available H3 latent-upscaler checkpoints without hard depending on load order."""
    try:
        choices = list(folder_paths.get_filename_list("latent_upscale_models"))
        if choices:
            return choices
    except Exception:
        pass

    # Fallback for cases where the upscaler custom node is loaded after KWTR_Tools.
    try:
        from pathlib import Path as _Path
        model_dir = _Path(folder_paths.models_dir) / "latent_upscale_models"
        if model_dir.is_dir():
            choices = sorted(
                x.name for x in model_dir.iterdir()
                if x.is_file() and x.suffix.lower() in {".safetensors", ".pth", ".pt"}
            )
            if choices:
                return choices
    except Exception:
        pass

    return ["(no latent upscaler model found)"]


def _run_registered_node(node_name, *args):
    """Execute another registered ComfyUI node at runtime.

    Runtime lookup avoids import-order problems between custom-node packages.
    This helper targets the classic FUNCTION-based ComfyUI node API used by
    the current KWTR_Tools nodes and the H3 latent-upscaler workflow.
    """
    cls = nodes.NODE_CLASS_MAPPINGS.get(node_name)
    if cls is None:
        raise RuntimeError(
            f"Required node '{node_name}' is not registered. "
            "Install/update ComfyUI and Comfyui_Minimax_h3_latent_Upscaler, then restart ComfyUI."
        )

    obj = cls()
    function_name = getattr(cls, "FUNCTION", None) or getattr(obj, "FUNCTION", None)
    if not function_name:
        raise RuntimeError(f"Node '{node_name}' does not expose the classic FUNCTION API.")

    fn = getattr(obj, function_name)
    result = fn(*args)

    # Classic nodes return tuples. Keep a small compatibility path for wrappers
    # that expose their values through a `result` attribute.
    if isinstance(result, tuple):
        return result
    if isinstance(result, list):
        return tuple(result)
    if hasattr(result, "result"):
        value = result.result
        return tuple(value) if isinstance(value, (tuple, list)) else (value,)
    return (result,)


def _safe_vram_cleanup():
    """Best-effort cache cleanup without forcing the active H3 model to unload."""
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    try:
        import comfy.model_management as model_management
        if hasattr(model_management, "soft_empty_cache"):
            try:
                model_management.soft_empty_cache()
            except TypeError:
                model_management.soft_empty_cache(True)
    except Exception:
        pass


def _is_nested_tensor(value):
    """NestedTensor check that works across current ComfyUI implementations."""
    try:
        import comfy.nested_tensor
        if isinstance(value, comfy.nested_tensor.NestedTensor):
            return True
    except Exception:
        pass
    return bool(getattr(value, "is_nested", False))


def _nested_members(value):
    """Return NestedTensor members as a list, or None if value is not nested."""
    if not _is_nested_tensor(value):
        return None

    if hasattr(value, "unbind"):
        try:
            return list(value.unbind())
        except Exception:
            pass

    tensors = getattr(value, "tensors", None)
    if tensors is not None:
        try:
            return list(tensors)
        except Exception:
            pass

    return None


def _extract_h3_video_tensor(latent):
    """Extract the MiniMax-H3 video tensor from a LATENT dict."""
    if not isinstance(latent, dict):
        return None

    samples = latent.get("samples")

    if isinstance(samples, torch.Tensor):
        return samples if samples.ndim >= 4 else None

    members = _nested_members(samples)
    if members:
        video = members[0]
        if isinstance(video, torch.Tensor) and video.ndim >= 4:
            return video

    if isinstance(samples, (list, tuple)):
        for item in samples:
            if isinstance(item, torch.Tensor) and item.ndim >= 4:
                return item

    return None


def _pixel_size_from_h3_latent(latent, fallback=None, spatial_factor=16):
    """Return (width, height) in pixel space from an H3 video latent."""
    video = _extract_h3_video_tensor(latent)
    if video is None:
        return fallback

    latent_h = int(video.shape[-2])
    latent_w = int(video.shape[-1])
    if latent_h <= 0 or latent_w <= 0:
        return fallback

    return latent_w * spatial_factor, latent_h * spatial_factor


def _round_to_multiple(value, multiple):
    multiple = max(1, int(multiple))
    return max(multiple, int(round(float(value) / multiple)) * multiple)


def _target_resolution_for_megapixels(
    src_width, src_height, target_megapixels, latent_align, spatial_factor=16
):
    """Compute explicit pixel W/H while preserving aspect ratio.

    `latent_align` is the grid used by H3LatentUpscalerNodeResolution in latent
    space. Therefore its equivalent pixel grid is latent_align * spatial_factor.
    """
    src_width = int(src_width)
    src_height = int(src_height)
    if src_width <= 0 or src_height <= 0:
        raise ValueError(f"Invalid source resolution: {src_width}x{src_height}")

    target_pixels = max(1.0, float(target_megapixels) * 1_000_000.0)
    aspect = float(src_width) / float(src_height)

    width = math.sqrt(target_pixels * aspect)
    height = math.sqrt(target_pixels / aspect)

    pixel_align = max(1, int(latent_align)) * int(spatial_factor)
    width = _round_to_multiple(width, pixel_align)
    height = _round_to_multiple(height, pixel_align)

    return int(width), int(height), int(pixel_align)


def _format_mp(width, height):
    return (float(width) * float(height)) / 1_000_000.0


class LatentUpscaleKSampler:
    """
    MiniMax H3 two-stage sampler with neural latent upscaling between passes.

    Internal flow:
      BasicScheduler -> SplitSigmas -> SamplerCustomAdvanced (low-res, denoised output)
      -> H3 Latent Upscaler (Resolution, aspect-locked) -> H3 Latent Cond Sync
      -> ManualSigmas -> SamplerCustomAdvanced (high-res refine)

    This is intentionally modeled after the user's working H3 latent-upscale
    workflow so the surrounding graph can be collapsed into one node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # --- Connections ---
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),

                # --- Sampling ---
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "total_steps": ("INT", {"default": 8, "min": 2, "max": 10000}),
                "upscale_at_step": ("INT", {"default": 4, "min": 1, "max": 9999}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),

                # --- H3 latent upscale ---
                "upscale_model_name": (_latent_upscale_model_choices(),),
                "target_megapixels": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1}),
                # NOTE: this is a LATENT-space grid, not a pixel-space grid.
                # MiniMax-H3 uses a 16x spatial factor, so align=2 == 32px grid.
                "align": ("INT", {"default": 2, "min": 2, "max": 32, "step": 2}),
                "upscale_device": (["cuda", "cpu"], {"default": "cuda"}),
                "upscale_precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),

                # --- High-resolution refinement ---
                "refine_sigmas": ("STRING", {
                    "default": "0.8500, 0.6316, 0.3158, 0.0000",
                    "multiline": False,
                }),
                "cfg_refine": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "cleanup_between_passes": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "bypass_upscale": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Skip the latent upscale + high-res refine entirely and return "
                               "only the first-phase result. Runs pass 1 with the FULL sigma "
                               "schedule (total_steps, all the way to sigma 0) rather than the "
                               "upscale_at_step-truncated split, so the output is a properly "
                               "finished low-res sample, not an intermediate x0 estimate.",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("LATENT", "positive_hi", "negative_hi")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom"
    DESCRIPTION = (
        "MiniMax H3 two-stage sampler: low-res sampling -> aspect-locked learned latent upscale -> "
        "conditioning sync -> short high-res refinement. Requires "
        "LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler. Set bypass_upscale=True to run only the "
        "first phase (full-schedule low-res sample) and skip upscale + refine entirely. "
        "positive_hi/negative_hi are the conditioning actually used for the (possibly upscaled) "
        "output latent -- wire these into any downstream sampler (e.g. H3 Audio Refine Sampler) "
        "that consumes this node's LATENT output, instead of the original positive/negative, or "
        "MiniMax-H3's image/keyframe conditioning will still be sized for the pre-upscale "
        "resolution and crash with a PackedLayout shape mismatch."
    )

    def sample(
        self,
        model,
        positive,
        negative,
        latent_image,
        noise_seed,
        total_steps,
        upscale_at_step,
        sampler_name,
        scheduler,
        upscale_model_name,
        target_megapixels,
        align,
        upscale_device,
        upscale_precision,
        refine_sigmas,
        cfg_refine,
        cleanup_between_passes,
        bypass_upscale=False,
    ):
        # Reuse one deterministic noise object and one sampler across both passes,
        # matching the original graph's shared RandomNoise/KSamplerSelect wiring.
        noise = _run_registered_node("RandomNoise", noise_seed)[0]
        sampler = _run_registered_node("KSamplerSelect", sampler_name)[0]

        if bypass_upscale:
            # Full schedule, all the way to sigma 0 -- a properly finished low-res
            # sample, not the upscale_at_step-truncated x0 estimate pass1 normally
            # produces. Skips the upscale-checkpoint requirement entirely.
            print(f"[KWTR LatentUpscale] bypass_upscale=True, running phase 1 only "
                  f"({total_steps} steps, full schedule).")
            full_sigmas = _run_registered_node(
                "BasicScheduler", model, scheduler, total_steps, 1.0
            )[0]
            guider_1 = _run_registered_node("BasicGuider", model, positive)[0]
            pass1_only = _run_registered_node(
                "SamplerCustomAdvanced", noise, guider_1, sampler, full_sigmas, latent_image
            )
            return (pass1_only[0], positive, negative)

        if upscale_at_step >= total_steps:
            raise ValueError(
                f"upscale_at_step ({upscale_at_step}) must be smaller than total_steps ({total_steps})."
            )
        if upscale_model_name.startswith("("):
            raise RuntimeError(
                "No H3 latent-upscaler checkpoint was found. Put a checkpoint in "
                "ComfyUI/models/latent_upscale_models/ and restart ComfyUI."
            )

        # 1) Build the same low-resolution sigma schedule used by the expanded workflow.
        full_sigmas = _run_registered_node(
            "BasicScheduler", model, scheduler, total_steps, 1.0
        )[0]
        high_sigmas, _low_sigmas = _run_registered_node(
            "SplitSigmas", full_sigmas, upscale_at_step
        )[:2]

        # Pass 1 uses BasicGuider in the reference workflow. The denoised output
        # (x0 estimate) is what gets fed into the learned latent upscaler.
        guider_1 = _run_registered_node("BasicGuider", model, positive)[0]
        pass1 = _run_registered_node(
            "SamplerCustomAdvanced", noise, guider_1, sampler, high_sigmas, latent_image
        )
        if len(pass1) < 2:
            raise RuntimeError("SamplerCustomAdvanced did not return denoised_output.")
        lowres_denoised = pass1[1]

        # 2) Aspect-ratio-locked learned 3D latent upscale.
        # H3LatentUpscalerNodeResolution interprets `align` in LATENT space.
        # With MiniMax-H3's 16x spatial factor, align=2 corresponds to a 32px grid.
        source_size = _pixel_size_from_h3_latent(latent_image)
        if source_size is None:
            source_size = _pixel_size_from_h3_latent(lowres_denoised)
        if source_size is None:
            raise RuntimeError(
                "Could not determine MiniMax-H3 video latent width/height. "
                "Expected LATENT['samples'] to contain an H3 video tensor or NestedTensor."
            )

        src_width, src_height = source_size
        target_width, target_height, pixel_align = _target_resolution_for_megapixels(
            src_width, src_height, target_megapixels, align
        )

        print(
            f"[KWTR LatentUpscale] source={src_width}x{src_height} "
            f"aspect={src_width / src_height:.6f} "
            f"target={float(target_megapixels):.3f}MP -> "
            f"{target_width}x{target_height} ({_format_mp(target_width, target_height):.3f}MP) "
            f"latent_align={int(align)} pixel_align={pixel_align}"
        )

        upscaled = _run_registered_node(
            "H3LatentUpscalerNodeResolution",
            lowres_denoised,
            upscale_model_name,
            target_width,
            target_height,
            align,
            upscale_device,
            upscale_precision,
        )[0]

        actual_size = _pixel_size_from_h3_latent(upscaled)
        if actual_size is None:
            raise RuntimeError(
                "Could not determine the upscaled MiniMax-H3 video latent size after "
                "H3LatentUpscalerNodeResolution."
            )

        actual_width, actual_height = actual_size
        print(
            f"[KWTR LatentUpscale] actual_upscaled={actual_width}x{actual_height} "
            f"({_format_mp(actual_width, actual_height):.3f}MP)"
        )

        if actual_width != target_width or actual_height != target_height:
            raise RuntimeError(
                "H3 latent upscaler returned an unexpected resolution: "
                f"requested {target_width}x{target_height}, got "
                f"{actual_width}x{actual_height}. "
                "Second-pass sampling was stopped to avoid processing the wrong canvas size."
            )

        # 3) Resize MiniMax reference/keyframe conditioning metadata to the new
        # latent dimensions before the high-resolution refinement pass.
        synced = _run_registered_node(
            "H3LatentUpscalerNode3DV3", upscaled, positive, negative
        )
        upscaled_latent = synced[0]
        positive_hi = synced[1] if len(synced) > 1 else positive
        negative_hi = synced[2] if len(synced) > 2 else negative

        # Optional VRAM cleanup exactly at the low-res -> high-res boundary.
        if cleanup_between_passes:
            del pass1
            del guider_1
            del lowres_denoised
            del upscaled
            del synced
            del full_sigmas
            del high_sigmas
            del _low_sigmas
            _safe_vram_cleanup()

        # 4) Short high-resolution refinement. Default sigmas reproduce the
        # 3-step ManualSigmas node in the supplied H3 latent-upscale workflow.
        refine_sigma_tensor = _run_registered_node("ManualSigmas", refine_sigmas)[0]
        guider_2 = _run_registered_node(
            "CFGGuider", model, positive_hi, negative_hi, cfg_refine
        )[0]
        pass2 = _run_registered_node(
            "SamplerCustomAdvanced", noise, guider_2, sampler, refine_sigma_tensor, upscaled_latent
        )

        return (pass2[0], positive_hi, negative_hi)


class AmountSlider:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "amount": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
            },
        }

    RETURN_TYPES  = ("FLOAT",)
    RETURN_NAMES  = ("amount",)
    FUNCTION      = "run"
    CATEGORY      = "utils"
    DISPLAY_NAME  = "Amount Slider"

    def run(self, amount: float):
        return (amount,)



class FloatFine:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0.0,
                    "min": -1000000.0,
                    "max": 1000000.0,
                    "step": 0.1,
                    "display": "number",
                }),
            },
        }

    RETURN_TYPES  = ("FLOAT",)
    RETURN_NAMES  = ("value",)
    FUNCTION      = "run"
    CATEGORY      = "utils"
    DISPLAY_NAME  = "Float (0.1 step)"

    def run(self, value: float):
        return (value,)


class H3TileAwareResolutionSelector:
    """
    コアの Resolution Selector と同じ aspect_ratio/megapixels 指定から解像度を
    計算するが、MiniMax-H3 Video VAE のタイル分割(既定 256px タイル・64px
    オーバーラップ=192px刻み)を考慮し、幅・高さそれぞれを直近のタイル境界
    (256 + 192×(n-1))のうち近い方にスナップする。境界をわずかに跨いだだけの
    「タイル1枚分丸ごと余分に払って解像度はほぼ変わらない」ケースを回避しつつ、
    境界に近い側では素直にそこまで使い切る。単純に切り上げ続けると小さい解像度
    ではアスペクト比が大きく崩れるため(タイル1枚あたりの相対幅が大きいため)、
    その対策として「近い方に丸める」方式にしている。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": ([e.value for e in AspectRatio], {
                    "default": AspectRatio.SQUARE.value,
                }),
                "megapixels": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1,
                }),
                "multiple": ("INT", {
                    "default": 32, "min": 8, "max": 128, "step": 4,
                    "tooltip": "latentグリッド整列用。MiniMax-H3では32を推奨。",
                }),
            },
            "optional": {
                "tile_size": ("INT", {
                    "default": 256, "min": 32, "max": 1024, "step": 32,
                    "tooltip": "VAEのタイルサイズ(px)。MiniMax-H3の既定は256。",
                }),
                "tile_overlap": ("INT", {
                    "default": 64, "min": 0, "max": 512, "step": 8,
                    "tooltip": "VAEのタイルオーバーラップ(px)。MiniMax-H3の既定は64。",
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "tiles_w", "tiles_h")
    FUNCTION = "run"
    CATEGORY = "utils"
    DISPLAY_NAME = "Resolution Selector (H3 Tile-Aware)"

    @staticmethod
    def _tiles_for(size, tile_size, stride):
        if size <= tile_size:
            return 1
        return 1 + math.ceil((size - tile_size) / stride)

    @staticmethod
    def _boundary(n_tiles, tile_size, stride):
        return tile_size + stride * (n_tiles - 1)

    @classmethod
    def _snap_to_nearest_tile_boundary(cls, size, multiple, tile_size, stride):
        n = cls._tiles_for(size, tile_size, stride)
        if n <= 1:
            # 1タイル以下は境界の概念がない(0〜tile_sizeまで同じコスト)。
            # ここで無理に tile_size まで引き上げると小さい解像度ほど
            # アスペクト比が大きく崩れるため、素のサイズをそのまま使う。
            return size, n
        upper = (cls._boundary(n, tile_size, stride) // multiple) * multiple
        lower = (cls._boundary(n - 1, tile_size, stride) // multiple) * multiple
        if abs(size - lower) <= abs(upper - size):
            return lower, n - 1
        return upper, n

    def run(self, aspect_ratio, megapixels, multiple, tile_size=256, tile_overlap=64):
        w_ratio, h_ratio = ASPECT_RATIOS[AspectRatio(aspect_ratio)]
        total_pixels = megapixels * 1024 * 1024
        scale = math.sqrt(total_pixels / (w_ratio * h_ratio))
        width = round(w_ratio * scale / multiple) * multiple
        height = round(h_ratio * scale / multiple) * multiple

        stride = max(1, tile_size - tile_overlap)
        width, tiles_w = self._snap_to_nearest_tile_boundary(width, multiple, tile_size, stride)
        height, tiles_h = self._snap_to_nearest_tile_boundary(height, multiple, tile_size, stride)

        return (int(width), int(height), int(tiles_w), int(tiles_h))


class CLIPVisionLoaderDevice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name": (folder_paths.get_filename_list("clip_vision"),),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            },
        }

    RETURN_TYPES = ("CLIP_VISION",)
    RETURN_NAMES = ("CLIP_VISION",)
    FUNCTION = "load_clip"
    CATEGORY = "model/loaders"
    DESCRIPTION = "Load CLIP Vision with a device switch. device=cpu keeps the vision model in RAM instead of VRAM."

    def load_clip(self, clip_name, device="default"):
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        if clip_vision is None:
            raise RuntimeError("ERROR: clip vision file is invalid and does not contain a valid vision model.")

        if device == "cpu":
            cpu = torch.device("cpu")
            patcher = clip_vision.patcher
            # This comfy fork (aimdo/DynamicVRAM) keeps per-device pin state in
            # model.dynamic_pins; retargeting load_device without registering the
            # new device first makes partially_unload_ram() raise KeyError.
            if hasattr(patcher, "register_load_device"):
                patcher.register_load_device(cpu)
            # encode_image() uses self.load_device for image.to(...), so it must be cpu too
            clip_vision.load_device = cpu
            patcher.load_device = cpu
            patcher.offload_device = cpu
            clip_vision.model.to(cpu)

        return (clip_vision,)



class AudioDuration:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("FLOAT", "INT", "INT")
    RETURN_NAMES = ("duration", "num_samples", "sample_rate")
    FUNCTION = "get_duration"
    CATEGORY = "audio"
    DESCRIPTION = "Get duration (seconds), sample count, and sample rate from an AUDIO input."

    def get_duration(self, audio):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        num_samples = int(waveform.shape[-1])
        duration = num_samples / sample_rate
        return (float(duration), num_samples, sample_rate)


class VAEDecodeTiledProgress:
    """
    Tiled VAE Decode wrapper.

    Calls ComfyUI's built-in VAE.decode_tiled, which already emits standard
    progress events per tile. A frontend extension draws a green progress bar
    on the node frame by subscribing to those events. No custom ProgressBar
    is needed here.

    Handles both image latents (4D: b,c,h,w) and video latents (5D: b,c,t,h,w).
    Temporal tiling args are passed only when the installed VAE.decode_tiled
    supports them.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "tile_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 64}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 16}),
            },
            "optional": {
                "temporal_size": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "temporal_overlap": ("INT", {"default": 2, "min": 0, "max": 64, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "KWTR/vae"
    DESCRIPTION = (
        "Tiled VAE decode with a green progress bar on the node frame. "
        "Reduces peak VRAM by decoding in tiles. Works for image and video latents."
    )

    def decode(self, vae, samples, tile_size, overlap,
               temporal_size=8, temporal_overlap=2):
        import inspect

        latent = samples["samples"]
        if getattr(latent, "is_nested", False):
            latent = latent.unbind()[0]

        # decode_tiled expects tile sizes in LATENT space. Convert from pixel
        # space using the VAE's spatial compression factor when available.
        compression = 8
        for attr in ("spacial_compression_decode", "spatial_compression_decode"):
            fn = getattr(vae, attr, None)
            if callable(fn):
                try:
                    compression = int(fn())
                    break
                except Exception:
                    pass

        tile = max(int(tile_size) // compression, 1)
        ov = max(int(overlap) // compression, 0)

        sig_params = inspect.signature(vae.decode_tiled).parameters
        kwargs = {}

        # Spatial tile args: prefer tile_x/tile_y, fall back to tile_size.
        if "tile_x" in sig_params:
            kwargs["tile_x"] = tile
        if "tile_y" in sig_params:
            kwargs["tile_y"] = tile
        if "tile_size" in sig_params and "tile_x" not in sig_params:
            kwargs["tile_size"] = tile
        if "overlap" in sig_params:
            kwargs["overlap"] = ov

        is_video = isinstance(latent, torch.Tensor) and latent.ndim == 5
        if is_video:
            if "tile_t" in sig_params:
                kwargs["tile_t"] = int(temporal_size)
            if "overlap_t" in sig_params:
                kwargs["overlap_t"] = int(temporal_overlap)

        images = vae.decode_tiled(latent, **kwargs)

        # Flatten video output (b,t,h,w,c) -> (b*t,h,w,c) for downstream IMAGE nodes.
        if isinstance(images, torch.Tensor) and images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])

        return (images,)


_ABC_LETTERS = ['C', 'D', 'E', 'F', 'G', 'A', 'B']  # octave boundary sits at B->C, not G->A
_ABC_NATURAL_PITCH = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
_ABC_SHARP_ORDER = ['F', 'C', 'G', 'D', 'A', 'E', 'B']
_ABC_FLAT_ORDER = ['B', 'E', 'A', 'D', 'G', 'C', 'F']
_ABC_ACC_SYMBOL = {2: '^^', 1: '^', 0: '=', -1: '_', -2: '__'}
_ABC_ACC_VALUE = {'^^': 2, '^': 1, '=': 0, '_': -1, '__': -2}

_ABC_MAJOR_KEY_POS = {
    'Cb': -7, 'Gb': -6, 'Db': -5, 'Ab': -4, 'Eb': -3, 'Bb': -2, 'F': -1,
    'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
}
_ABC_MINOR_KEY_POS = {
    'Ab': -7, 'Eb': -6, 'Bb': -5, 'F': -4, 'C': -3, 'G': -2, 'D': -1,
    'A': 0, 'E': 1, 'B': 2, 'F#': 3, 'C#': 4, 'G#': 5, 'D#': 6, 'A#': 7,
}
_ABC_MAJOR_KEYS = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb']
_ABC_MINOR_KEYS = ['Am', 'Em', 'Bm', 'F#m', 'C#m', 'G#m', 'D#m', 'A#m',
                   'Dm', 'Gm', 'Cm', 'Fm', 'Bbm', 'Ebm', 'Abm']
YUE2_ABC_TARGET_KEY_CHOICES = _ABC_MAJOR_KEYS + _ABC_MINOR_KEYS

_ABC_TOKEN_RE = re.compile(r'"[^"]*"|(?:\^\^|__|\^|_|=)?[A-Ga-g][,\']*|\|+')
_ABC_NOTE_TOKEN_RE = re.compile(r"^(\^\^|__|\^|_|=)?([A-Ga-g])([,']*)$")
_ABC_CHORD_ROOT_RE = re.compile(r'^([A-G])([#b]{0,2})(.*)$')


def _abc_parse_key(key_str):
    key_str = key_str.strip()
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(m?)", key_str)
    if not m:
        raise ValueError(f"Unsupported/unparsed ABC key field: {key_str!r}")
    letter_raw, acc, minor_suffix = m.groups()
    letter = letter_raw.upper()
    mode = "minor" if (minor_suffix or letter_raw.islower()) else "major"
    table = _ABC_MINOR_KEY_POS if mode == "minor" else _ABC_MAJOR_KEY_POS
    lookup = letter + acc
    if lookup not in table:
        raise ValueError(
            f"Unsupported key for this node: {key_str!r} "
            "(only standard major/minor keys up to 7 sharps/flats)"
        )
    pos = table[lookup]
    acc_val = acc.count('#') - acc.count('b')
    return letter, acc_val, mode, pos


def _abc_key_signature(pos):
    sig = {L: 0 for L in _ABC_LETTERS}
    if pos > 0:
        for L in _ABC_SHARP_ORDER[:pos]:
            sig[L] = 1
    elif pos < 0:
        for L in _ABC_FLAT_ORDER[:-pos]:
            sig[L] = -1
    return sig


def _abc_transposition_interval(old_letter, old_acc, new_letter, new_acc):
    letter_shift_raw = (_ABC_LETTERS.index(new_letter) - _ABC_LETTERS.index(old_letter)) % 7
    natural_diff = (_ABC_NATURAL_PITCH[new_letter] - _ABC_NATURAL_PITCH[old_letter]) % 12
    s0 = (natural_diff + (new_acc - old_acc)) % 12
    if s0 > 6:
        return letter_shift_raw - 7, s0 - 12
    return letter_shift_raw, s0


def _abc_transpose_chord_symbol(text, letter_shift, semitone_shift):
    inner = text[1:-1]
    if not inner:
        return text

    def _transpose_root(part):
        m = _ABC_CHORD_ROOT_RE.match(part)
        if not m:
            return part
        letter, acc, suffix = m.groups()
        acc_val = acc.count('#') - acc.count('b')
        old_pitch = _ABC_NATURAL_PITCH[letter] + acc_val
        new_letter = _ABC_LETTERS[(_ABC_LETTERS.index(letter) + letter_shift) % 7]
        target_pitch = old_pitch + semitone_shift
        d_new = target_pitch - _ABC_NATURAL_PITCH[new_letter]
        d_new = ((d_new + 6) % 12) - 6  # fold into a chord symbol's -2..2 range
        acc_new = {2: '##', 1: '#', 0: '', -1: 'b', -2: 'bb'}.get(d_new, '')
        return new_letter + acc_new + suffix

    if '/' in inner:
        root_part, bass_part = inner.split('/', 1)
        return f'"{_transpose_root(root_part)}/{_transpose_root(bass_part)}"'
    return f'"{_transpose_root(inner)}"'


class YuE2AbcKeyTempoFix:
    """
    YuE2 Generate ABC (in-graph LLM symbolic planner) treats the requested key
    and tempo as generative hints, not hard constraints, so its output can
    drift (e.g. asked for G major/130 BPM, got K:D / Q:1/4=133). Re-prompting
    risks new notation errors in the bounded native ABC dialect (see this
    project's yue2-music skill notes). This node instead deterministically
    rewrites the already-generated score: a proper diatonic transposition
    (every note shifted by the same scale-degree + semitone interval, with
    accidentals re-spelled for the new key signature, tracked per-bar per-
    voice) plus a straight Q: tempo rewrite. No further generative pass.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"multiline": True, "default": ""}),
                "target_key": (YUE2_ABC_TARGET_KEY_CHOICES, {"default": "G"}),
                "target_bpm": ("INT", {"default": 130, "min": 20, "max": 400}),
                "transpose_chord_symbols": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("abc", "detected_source_key", "semitone_shift")
    FUNCTION = "run"
    CATEGORY = "KWTR/music"
    DESCRIPTION = (
        "Deterministically transposes a generated YuE2/SheetSage2-style two-voice "
        "(V: Vocal / V: Ins) ABC score to an exact target key and rewrites its Q: "
        "tempo, instead of re-prompting the generator. Preserves each note's "
        "scale-degree relationship and re-spells accidentals for the new key "
        "signature; chord symbols are transposed too unless disabled."
    )

    def run(self, abc, target_key, target_bpm, transpose_chord_symbols):
        out_lines = []

        old_key = None
        old_sig = None
        new_letter_full, new_acc_val, new_mode, new_pos = _abc_parse_key(target_key)
        new_sig = _abc_key_signature(new_pos)
        letter_shift = semitone_shift = 0

        voice_states_old = {}
        voice_states_new = {}
        current_voice = [None]  # mutable box so the nested closure sees live updates

        def transpose_music_line(line):
            def repl(m):
                text = m.group(0)
                if text.startswith('"'):
                    if transpose_chord_symbols:
                        return _abc_transpose_chord_symbol(text, letter_shift, semitone_shift)
                    return text
                if set(text) == {'|'}:
                    voice_states_old[current_voice[0]] = {}
                    voice_states_new[current_voice[0]] = {}
                    return text

                nm = _ABC_NOTE_TOKEN_RE.match(text)
                acc_sym, letter, marks = nm.groups()
                letter_upper = letter.upper()
                apostrophes = marks.count("'")
                commas = marks.count(',')
                base_oct = 5 if letter.islower() else 4
                oct_old = base_oct + apostrophes - commas

                state_old = voice_states_old.setdefault(current_voice[0], {})
                if acc_sym is not None:
                    d_old = _ABC_ACC_VALUE[acc_sym]
                    state_old[letter_upper] = d_old
                else:
                    d_old = state_old.get(letter_upper, old_sig.get(letter_upper, 0))

                abs_pitch_old = oct_old * 12 + _ABC_NATURAL_PITCH[letter_upper] + d_old

                old_idx = _ABC_LETTERS.index(letter_upper)
                oct_delta, new_idx = divmod(old_idx + letter_shift, 7)
                new_letter_upper = _ABC_LETTERS[new_idx]
                new_oct = oct_old + oct_delta
                required_pitch = abs_pitch_old + semitone_shift
                d_new = required_pitch - (new_oct * 12 + _ABC_NATURAL_PITCH[new_letter_upper])

                state_new = voice_states_new.setdefault(current_voice[0], {})
                base_new = new_sig.get(new_letter_upper, 0)
                active_new = state_new.get(new_letter_upper, base_new)
                acc_out = ''
                if active_new != d_new:
                    acc_out = _ABC_ACC_SYMBOL.get(d_new, '')
                    state_new[new_letter_upper] = d_new

                if new_oct >= 5:
                    out_letter = new_letter_upper.lower()
                    out_marks = "'" * (new_oct - 5)
                else:
                    out_letter = new_letter_upper
                    out_marks = ',' * max(0, 4 - new_oct)
                return acc_out + out_letter + out_marks

            return _ABC_TOKEN_RE.sub(repl, line)

        for line in abc.splitlines():
            stripped = line.strip()

            if stripped.startswith('%'):
                out_lines.append(line)
                continue

            header_match = re.match(r'^([A-Za-z]):(.*)$', stripped)
            if header_match and header_match.group(1) in "XTMLQKV":
                field = header_match.group(1)
                rest = header_match.group(2)

                if field == 'K':
                    if old_key is None:
                        old_letter, old_acc_val, old_mode, old_pos = _abc_parse_key(rest)
                        old_key = rest.strip()
                        old_sig = _abc_key_signature(old_pos)
                        letter_shift, semitone_shift = _abc_transposition_interval(
                            old_letter, old_acc_val, new_letter_full, new_acc_val,
                        )
                    out_lines.append(f"K:{target_key}")
                    continue

                if field == 'Q':
                    new_q, n_subs = re.subn(r'=\s*\d+', f'={int(target_bpm)}', rest)
                    if n_subs == 0:
                        new_q = f"1/4={int(target_bpm)}"
                    out_lines.append(f"Q:{new_q}")
                    continue

                if field == 'V':
                    voice_tokens = rest.strip().split()
                    if len(voice_tokens) == 1:
                        current_voice[0] = voice_tokens[0]
                    out_lines.append(line)
                    continue

                out_lines.append(line)  # X, T, M, L: unchanged
                continue

            if current_voice[0] is not None and old_sig is not None:
                out_lines.append(transpose_music_line(line))
            else:
                out_lines.append(line)

        if old_key is None:
            raise ValueError(
                "No K: field found in the supplied ABC; cannot determine the "
                "source key to transpose from."
            )

        return ("\n".join(out_lines), old_key, int(semitone_shift))


class YuE2AbcToMidi:
    """
    Wraps the abc2midi CLI (abcMIDI toolkit, apt package 'abcmidi', installed by
    comfyui_start.sh) to render a two-voice YuE2/SheetSage2-style ABC score
    straight to a standard multi-track MIDI file. Each V: voice lands on its
    own MIDI track (not merged into one), matching a compliant ABC->MIDI
    converter's normal behaviour.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "abc": ("STRING", {"multiline": True, "default": ""}),
                "filename_prefix": ("STRING", {"default": "yue2/abc2midi"}),
                "auto_chord_accompaniment": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Disable to inject %%MIDI gchordoff, so quoted "
                               "chord symbols don't generate an extra backing track.",
                }),
                "also_save_abc": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Also save the exact .abc text next to the .mid.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("midi_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "KWTR/music"
    DISPLAY_NAME = "Save MIDI (abc2midi)"
    DESCRIPTION = (
        "Renders ABC text to a standard multi-track .mid via the abc2midi CLI. "
        "Each V: voice becomes its own MIDI track."
    )

    def run(self, abc, filename_prefix, auto_chord_accompaniment, also_save_abc):
        if shutil.which("abc2midi") is None:
            raise RuntimeError(
                "abc2midi not found on PATH. Install the 'abcmidi' apt package "
                "(now automated in comfyui_start.sh -- restart ComfyUI via that "
                "script, or run `apt-get install -y abcmidi` manually)."
            )

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, _subfolder, _prefix = (
            folder_paths.get_save_image_path(filename_prefix, output_dir)
        )

        abc_text = abc
        if not auto_chord_accompaniment:
            lines = abc_text.splitlines()
            if lines:
                lines.insert(1, "%%MIDI gchordoff")
                abc_text = "\n".join(lines)
            else:
                abc_text = "%%MIDI gchordoff\n"

        base_name = f"{filename}_{counter:05}_"
        abc_path = os.path.join(full_output_folder, base_name + ".abc")
        midi_path = os.path.join(full_output_folder, base_name + ".mid")

        with open(abc_path, "w", encoding="utf-8") as f:
            f.write(abc_text)

        proc = subprocess.run(
            ["abc2midi", abc_path, "-o", midi_path],
            capture_output=True, text=True,
        )
        if proc.stdout:
            print(f"[KWTR YuE2AbcToMidi] abc2midi stdout:\n{proc.stdout}")
        if proc.stderr:
            print(f"[KWTR YuE2AbcToMidi] abc2midi stderr:\n{proc.stderr}")

        if not os.path.exists(midi_path):
            raise RuntimeError(
                "abc2midi did not produce a MIDI file.\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )

        if not also_save_abc:
            os.remove(abc_path)

        print(f"[KWTR YuE2AbcToMidi] wrote {midi_path}")
        return (midi_path,)


class MetaInfoFilenamePrefix:
    """
    trigger(VIDEO/IMAGE)到達時刻でタイムスタンプを確定し、
    - filename_prefix (folder + timestamp)
    - prompt をそのまま text 出力
    を返すハブ。実データとプロンプト.txtを同名・同時刻で対にする。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"forceInput": True}),
                "trigger": (any_type, {}),
                "folder": ("STRING", {"default": "MiniMax-H3"}),
                "template": ("STRING", {
                    "default": "%date:yy%_%date:MM%%date:dd%/%date:hh%%date:mm%%date:ss%",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = (any_type, "STRING", "STRING")
    RETURN_NAMES = ("trigger_passthrough", "text", "filename_prefix")
    FUNCTION = "build"
    CATEGORY = "utils"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def build(self, prompt, trigger, folder, template):
        now = datetime.datetime.now()  # ← trigger 到達時刻で確定
        tokens = {
            "%date:yy%": now.strftime("%Y"),
            "%date:MM%": now.strftime("%m"),
            "%date:dd%": now.strftime("%d"),
            "%date:hh%": now.strftime("%H"),
            "%date:mm%": now.strftime("%M"),
            "%date:ss%": now.strftime("%S"),
        }
        prefix = template
        for k, v in tokens.items():
            prefix = prefix.replace(k, v)

        parts = [p.strip("/") for p in (folder, prefix) if p]
        filename_prefix = "/".join(parts)

        return (trigger, prompt, filename_prefix)


NODE_CLASS_MAPPINGS = {
    "DualPhaseKSampler": DualPhaseKSampler,
    "LatentUpscaleKSampler": LatentUpscaleKSampler,
    "AmountSlider": AmountSlider,
    "FloatFine": FloatFine,
    "H3TileAwareResolutionSelector": H3TileAwareResolutionSelector,
    "CLIPVisionLoaderDevice": CLIPVisionLoaderDevice,
    "AudioDuration": AudioDuration,
    "VAEDecodeTiledProgress": VAEDecodeTiledProgress,
    "MetaInfoFilenamePrefix": MetaInfoFilenamePrefix,
    "YuE2AbcKeyTempoFix": YuE2AbcKeyTempoFix,
    "YuE2AbcToMidi": YuE2AbcToMidi,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DualPhaseKSampler": "KSampler (Dual Phase)",
    "LatentUpscaleKSampler": "KSampler (Latent Upscale)",
    "AmountSlider": "Amount Slider",
    "FloatFine": "Float (0.1 step)",
    "H3TileAwareResolutionSelector": "Resolution Selector (H3 Tile-Aware)",
    "CLIPVisionLoaderDevice": "Load CLIP Vision (Device)",
    "AudioDuration": "Audio Duration",
    "VAEDecodeTiledProgress": "VAE Decode Tiled (Progress) 🟢",
    "MetaInfoFilenamePrefix": "prompt + filename_prefix output",
    "YuE2AbcKeyTempoFix": "YuE2 ABC Key/Tempo Fix",
    "YuE2AbcToMidi": "Save MIDI (abc2midi)",
}
