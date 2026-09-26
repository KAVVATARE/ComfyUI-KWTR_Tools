"""
KWTR H3 Picture Reference Edit

MiniMax H3 Image Studio の「Reference Edit」(H3ReferenceEditPrepare) のラッパー。
入力ソケット名をプロンプト上の呼び名 <Picture N> と一致させ (picture_1..picture_9)、
番号のズレが起きる接続/記述を実行前にエラーで止める。

- 上流ノードは未接続ソケットを飛ばして番号を詰めるため、picture_2 を空けて picture_3 を
  つなぐとプロンプトの <Picture 3> が実際には <Picture 2> を指してしまう。これを禁止する。
- プロンプトに書かれた <Picture N> が未接続ならエラー。
- `reference_image_N` / `source_image` のようなソケット名をプロンプトに書いていたらエラー
  (モデルには意味が通じない)。
- `<picture_1>` / `<Picture_1>` / `<picture1>` / `picture_1` などの表記は正式な `<Picture 1>` に
  自動で書き換えてから上流へ渡す (書き換え内容は run_info に表示)。
- ソケットの表示名は `<Picture N>` (内部名は picture_N のまま)。
- 実処理は上流の H3ReferenceEditPrepare をそのまま呼ぶ (Image Studio 本体は無改変)。
"""

from __future__ import annotations

import re

import nodes

UPSTREAM_NODE = "H3ReferenceEditPrepare"
MAX_PICTURES = 9

_PICTURE_TAG = re.compile(r"<\s*picture\s+(\d+)\s*>", re.IGNORECASE)
# 正式表記 <Picture N> 以外の揺れ: <picture_1>, <Picture-1>, <picture1>, 括弧なしの picture_1 など
_PICTURE_VARIANT = re.compile(
    r"<\s*picture[\s_\-]*(\d+)\s*>|\bpicture_(\d+)\b", re.IGNORECASE
)
_SOCKET_NAME_IN_PROMPT = re.compile(r"\b(reference_image_\d+|source_image)\b", re.IGNORECASE)

# 上流の選択肢は INPUT_TYPES 呼び出し時に上流から取得する。上流が未ロードの場合のみ下記を使う。
_FALLBACK_QUALITY = [
    "recommended | 5 frames",
    "extended quality | 9 frames",
    "high quality | 13 frames",
    "maximum quality | 20 frames (slow)",
]


def _upstream_class():
    cls = nodes.NODE_CLASS_MAPPINGS.get(UPSTREAM_NODE)
    if cls is None:
        raise RuntimeError(
            f"{UPSTREAM_NODE} が見つかりません。ComfyUI-MiniMax-H3-Image-Studio をインストール/有効化してください。"
        )
    return cls


def _upstream_inputs():
    try:
        return _upstream_class().INPUT_TYPES()
    except Exception:
        return None


class KWTR_H3PictureReferenceEdit:
    DESCRIPTION = (
        "MiniMax H3 Image Studio の Reference Edit と同じ処理。ソケット picture_N がプロンプトの <Picture N> に"
        "そのまま対応する。飛び番号の接続や、未接続の <Picture N> への言及は実行前にエラーになる。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        upstream = _upstream_inputs()
        if upstream is not None:
            req = dict(upstream["required"])
            opt = dict(upstream.get("optional", {}))
            quality = req["quality_profile"]
            source_fit = req["source_fit"]
            reference_detail = req["reference_detail"]
            transport = opt.get("reference_transport", (["native"], {"default": "native"}))
        else:
            quality = (_FALLBACK_QUALITY, {"default": _FALLBACK_QUALITY[0]})
            source_fit = (["crop_center", "contain_pad", "stretch"], {"default": "crop_center"})
            reference_detail = (["match_generation_area", "max_identity_2048"], {"default": "match_generation_area"})
            transport = (["native", "semantic (experimental)"], {"default": "native"})

        optional = {
            f"picture_{i}": ("IMAGE", {
                "display_name": f"<Picture {i}>",
                "tooltip": f"<Picture {i}>。<Picture 2> から順に隙間なく接続すること。",
            })
            for i in range(2, MAX_PICTURES + 1)
        }
        optional["reference_transport"] = transport

        return {
            "required": {
                "clip": ("CLIP", {"tooltip": "MiniMax H3 Qwen text/vision encoder."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "picture_1": ("IMAGE", {"display_name": "<Picture 1>", "tooltip": "<Picture 1>。編集元 (source) 画像。"}),
                "edit_instruction": ("STRING", {
                    "multiline": True,
                    "dynamicPrompts": True,
                    "default": "",
                    "tooltip": "画像は <Picture 1>, <Picture 2> ... で呼ぶ。番号はソケット名 picture_N と一致する。",
                }),
                "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
                "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
                "quality_profile": quality,
                "source_fidelity": ("FLOAT", {
                    "default": 0.60, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "<Picture 1> の言及されていない特徴をどれだけ保持するか (プロンプト文言の強さ。denoise ではない)。",
                }),
                "source_fit": source_fit,
                "reference_detail": reference_detail,
                "optimize_for_still": ("BOOLEAN", {"default": True}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "IMAGE", "INT", "STRING", "STRING")
    RETURN_NAMES = ("positive", "h3_latent", "fitted_source", "requested_frames", "image_prompt", "run_info")
    FUNCTION = "prepare"
    CATEGORY = "MiniMax H3/KWTR"

    @staticmethod
    def _normalize(edit_instruction: str):
        replaced = []

        def repl(m):
            canonical = f"<Picture {m.group(1) or m.group(2)}>"
            if m.group(0) != canonical:
                replaced.append(f"{m.group(0)} -> {canonical}")
            return canonical

        text = _PICTURE_VARIANT.sub(repl, edit_instruction or "")
        return text, replaced

    @staticmethod
    def _validate(pictures: dict, edit_instruction: str) -> list:
        connected = sorted(i for i, img in pictures.items() if img is not None)

        gaps = [i for i in range(1, max(connected) + 1) if i not in connected]
        if gaps:
            raise ValueError(
                "画像ソケットが飛び番号で接続されています。未接続: "
                + ", ".join(f"<Picture {i}>" for i in gaps)
                + f" (接続済み: {', '.join(f'<Picture {i}>' for i in connected)})。"
                "上流は番号を詰めるため <Picture N> がずれます。<Picture 2> から順に詰めて接続してください。"
            )

        text = edit_instruction or ""
        socket_names = sorted({m.group(1) for m in _SOCKET_NAME_IN_PROMPT.finditer(text)})
        if socket_names:
            raise ValueError(
                f"プロンプトにソケット名 {', '.join(socket_names)} が書かれています。モデルには通じないので "
                "<Picture 1>, <Picture 2> ... の形式で書いてください。"
            )

        mentioned = sorted({int(m.group(1)) for m in _PICTURE_TAG.finditer(text)})
        missing = [n for n in mentioned if n not in connected]
        if missing:
            raise ValueError(
                "プロンプトで言及されている画像が未接続です: "
                + ", ".join(f"<Picture {n}>" for n in missing)
                + f" (接続済み: <Picture 1>〜<Picture {len(connected)}>)。"
            )

        notes = [f"KWTR: {len(connected)} picture(s) connected, socket numbers match <Picture N>."]
        unmentioned = [n for n in connected if n not in mentioned]
        if len(connected) > 1 and unmentioned:
            notes.append(
                "KWTR warning: 役割が書かれていない画像があります: "
                + ", ".join(f"<Picture {n}>" for n in unmentioned)
            )
        return notes

    def prepare(self, clip, vae, picture_1, edit_instruction, width, height, quality_profile,
                source_fidelity, source_fit, reference_detail, optimize_for_still,
                reference_transport="native", **kwargs):
        pictures = {1: picture_1}
        for i in range(2, MAX_PICTURES + 1):
            pictures[i] = kwargs.get(f"picture_{i}")

        edit_instruction, replaced = self._normalize(edit_instruction)
        notes = self._validate(pictures, edit_instruction)
        if replaced:
            notes.append("KWTR: 表記を正規化しました: " + ", ".join(dict.fromkeys(replaced)))

        result = _upstream_class()().prepare(
            clip=clip,
            vae=vae,
            source_image=picture_1,
            edit_instruction=edit_instruction,
            width=width,
            height=height,
            quality_profile=quality_profile,
            source_fidelity=source_fidelity,
            source_fit=source_fit,
            reference_detail=reference_detail,
            optimize_for_still=optimize_for_still,
            reference_transport=reference_transport,
            **{f"reference_image_{i}": pictures[i] for i in range(2, MAX_PICTURES + 1)},
        )

        result = tuple(result)
        run_info = "\n".join(notes) + "\n" + str(result[5])
        return result[:5] + (run_info,)


NODE_CLASS_MAPPINGS = {
    "KWTR_H3PictureReferenceEdit": KWTR_H3PictureReferenceEdit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KWTR_H3PictureReferenceEdit": "KWTR H3 Picture Reference Edit",
}
