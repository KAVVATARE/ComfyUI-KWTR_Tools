"""
KWTR Character Pack (Build / Load)

Reddit の OmniChar ".char" のアイデアを KWTR H3 Picture Reference Edit 向けに再現したもの。
学習はせず、「切り抜き済みの参照画像」と「固定のキャラ説明文」をキャラ単位で保存して使い回す。

- Build: face_1 (必須) / face_2 / outfit_1 / outfit_2 / body_1 を受け取り、
  顔は YuNet で検出して肩が入る程度の余白で切り抜き、SFace で face_1 との類似度を出す。
  (SFace は AI 生成の似た顔を区別しきれないため、既定は警告のみ。identity_check=error で停止)
  outfit / body に顔が写っていたら警告する (参照同士の混線防止)。
  結果を input/kwtr_characters/<name>/ に画像 + char.json として保存する。
- Load: 保存済みキャラを 顔 → 服 → 体 の順で picture_1..9 に出力し、
  <Picture N> 付きの固定説明文と scene_prompt (キャラ名は "the <subject> from <Picture 1>" に置換) を
  組み立てて edit_instruction として出力する。picture_1 は Reference Edit の編集元になるため常に正面の顔。
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re

import numpy as np
import torch
from PIL import Image

import folder_paths

PACK_DIRNAME = "kwtr_characters"
MAX_PICTURES = 9
YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
SFACE_COSINE_THRESHOLD = 0.363  # OpenCV 推奨値 (これ未満は別人判定)
DETECT_SCORE = 0.8
DETECT_MAX_SIDE = 1024
PREVIEW_HEIGHT = 512

# (ソケット名, 役割) 。Load 時の並び順もこの順 (顔 → 服 → 体)。
SLOTS = [
    ("face_1", "face"),
    ("face_2", "face"),
    ("outfit_1", "outfit"),
    ("outfit_2", "outfit"),
    ("body_1", "body"),
]


def _pack_root() -> str:
    return os.path.join(folder_paths.get_input_directory(), PACK_DIRNAME)


def _safe_dirname(name: str) -> str:
    safe = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
    if not safe:
        raise ValueError("character_name が空です。")
    return safe


def _list_characters() -> list:
    root = _pack_root()
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "char.json"))
    )


# ---------------------------------------------------------------- image utils

def _tensor_to_bgr(image: torch.Tensor) -> np.ndarray:
    arr = (image[0].clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return np.ascontiguousarray(arr[..., ::-1])


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.ascontiguousarray(bgr[..., ::-1]))


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]


# ---------------------------------------------------------------- face models

class _FaceModels:
    _cache = None

    @classmethod
    def get(cls):
        if cls._cache is None:
            import cv2

            det_dir = os.path.join(folder_paths.models_dir, "detection")
            yunet = os.path.join(det_dir, YUNET_FILE)
            sface = os.path.join(det_dir, SFACE_FILE)
            for path in (yunet, sface):
                if not os.path.isfile(path):
                    raise FileNotFoundError(
                        f"{path} がありません。opencv_zoo から models/detection/ に配置してください。"
                    )
            detector = cv2.FaceDetectorYN.create(yunet, "", (320, 320), DETECT_SCORE, 0.3, 5000)
            recognizer = cv2.FaceRecognizerSF.create(sface, "")
            cls._cache = (detector, recognizer)
        return cls._cache


def _detect_faces(bgr: np.ndarray) -> np.ndarray:
    """元画像座標の顔一覧 (N, 15) を面積の大きい順で返す。"""
    import cv2

    detector, _ = _FaceModels.get()
    h, w = bgr.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / max(h, w))
    small = cv2.resize(bgr, (round(w * scale), round(h * scale))) if scale < 1.0 else bgr
    detector.setInputSize((small.shape[1], small.shape[0]))
    _, faces = detector.detect(small)
    if faces is None:
        return np.zeros((0, 15), dtype=np.float32)
    faces = faces.copy()
    faces[:, :14] /= scale
    return faces[np.argsort(-(faces[:, 2] * faces[:, 3]))]


def _face_feature(bgr: np.ndarray, face: np.ndarray) -> np.ndarray:
    _, recognizer = _FaceModels.get()
    return recognizer.feature(recognizer.alignCrop(bgr, face))


def _face_similarity(feat_a: np.ndarray, feat_b: np.ndarray) -> float:
    import cv2

    _, recognizer = _FaceModels.get()
    return float(recognizer.match(feat_a, feat_b, cv2.FaceRecognizerSF_FR_COSINE))


def _crop_face(bgr: np.ndarray, face: np.ndarray, margin: float) -> np.ndarray:
    """顔枠を中心に margin 倍へ広げ (下方向を多めに取り首・肩を含める)、画像内に収めて切り抜く。"""
    h, w = bgr.shape[:2]
    fx, fy, fw, fh = face[:4]
    cw, ch = fw * margin, fh * margin * 1.25
    cx, cy = fx + fw / 2, fy + fh / 2 + fh * 0.35
    x0, y0 = max(0, int(cx - cw / 2)), max(0, int(cy - ch / 2))
    x1, y1 = min(w, int(cx + cw / 2)), min(h, int(cy + ch / 2))
    return bgr[y0:y1, x0:x1]


def _preview(images: list) -> torch.Tensor:
    tiles = []
    for img in images:
        ratio = PREVIEW_HEIGHT / img.height
        tiles.append(img.convert("RGB").resize((max(1, round(img.width * ratio)), PREVIEW_HEIGHT), Image.LANCZOS))
    gap = 16
    sheet = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), PREVIEW_HEIGHT), (255, 255, 255))
    x = 0
    for t in tiles:
        sheet.paste(t, (x, 0))
        x += t.width + gap
    return _pil_to_tensor(sheet)


# ---------------------------------------------------------------- Build

class KWTR_CharacterPackBuild:
    DESCRIPTION = (
        "キャラの参照画像 (顔/服/体) と固定説明文を input/kwtr_characters/<name>/ に保存する。"
        "顔は自動で切り抜き、face_2 と face_1 の類似度 (SFace) を報告。服・体に顔が写っていたら警告。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_1": ("IMAGE", {"tooltip": "正面の顔。Load 時に <Picture 1> (Reference Edit の編集元) になる。服が写っていない画像を推奨。"}),
                "character_name": ("STRING", {"default": "Mina", "tooltip": "scene_prompt 中でこの名前を書くと <Picture 1> の人物に置き換わる。"}),
                "subject": ("STRING", {"default": "woman", "tooltip": "woman / man / girl など。プロンプト内の呼び方。"}),
                "identity_description": ("STRING", {"multiline": True, "default": "", "tooltip": "顔・髪など本人の特徴 (任意)。毎回プロンプトに固定で入る。"}),
                "outfit_description": ("STRING", {"multiline": True, "default": "", "tooltip": "服の具体的な特徴。例: light-gray ribbed sleeveless turtleneck knit mini dress with the keyhole cutout"}),
                "auto_crop_face": ("BOOLEAN", {"default": True}),
                "crop_margin": ("FLOAT", {"default": 2.2, "min": 1.2, "max": 5.0, "step": 0.1, "tooltip": "顔枠に対する切り抜き倍率。"}),
                "identity_check": (["warn", "error", "off"], {"default": "warn", "tooltip": "face_2 が face_1 と別人判定のとき。SFace は AI 生成の顔では誤判定が多いので既定は warn。"}),
                "overwrite": ("BOOLEAN", {"default": False, "tooltip": "同名キャラが既にある場合に上書きするか。"}),
            },
            "optional": {
                "face_2": ("IMAGE", {"tooltip": "別角度の顔 (任意)。"}),
                "outfit_1": ("IMAGE", {"tooltip": "服だけの画像 (人物・顔なしを推奨)。"}),
                "outfit_2": ("IMAGE", {"tooltip": "服の別アングル・上下別など (任意)。"}),
                "body_1": ("IMAGE", {"tooltip": "体型参照 (顔なしを推奨)。"}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("pack_dir", "preview", "report")
    FUNCTION = "build"
    OUTPUT_NODE = True
    CATEGORY = "MiniMax H3/KWTR"

    def build(self, face_1, character_name, subject, identity_description, outfit_description,
              auto_crop_face, crop_margin, identity_check, overwrite, **kwargs):
        inputs = {"face_1": face_1, **kwargs}
        pack_dir = os.path.join(_pack_root(), _safe_dirname(character_name))
        if os.path.exists(os.path.join(pack_dir, "char.json")) and not overwrite:
            raise ValueError(f"{pack_dir} は既に存在します。上書きするには overwrite を有効にしてください。")

        report, pictures, saved = [], [], []
        ref_feature = None
        similarity = {}

        for slot, role in SLOTS:
            image = inputs.get(slot)
            if image is None:
                continue
            bgr = _tensor_to_bgr(image)
            faces = _detect_faces(bgr)

            if role == "face":
                if len(faces) == 0:
                    raise ValueError(f"{slot}: 顔が検出できませんでした。")
                if len(faces) > 1:
                    report.append(f"warning: {slot} に顔が {len(faces)} 個あります。最大の顔を使います。")
                feature = _face_feature(bgr, faces[0])
                if ref_feature is None:
                    ref_feature = feature
                elif identity_check != "off":
                    sim = _face_similarity(ref_feature, feature)
                    similarity[slot] = round(sim, 4)
                    if sim >= SFACE_COSINE_THRESHOLD:
                        report.append(f"{slot}: face_1 と同一人物判定 (SFace cosine {sim:.3f})")
                    elif identity_check == "error":
                        raise ValueError(
                            f"{slot} は face_1 と別人と判定されました (SFace cosine {sim:.3f} < {SFACE_COSINE_THRESHOLD})。"
                        )
                    else:
                        report.append(
                            f"warning: {slot} は face_1 と別人の可能性 (SFace cosine {sim:.3f} < {SFACE_COSINE_THRESHOLD})。目視で確認してください。"
                        )
                if auto_crop_face:
                    bgr = _crop_face(bgr, faces[0], crop_margin)
            elif len(faces) > 0:
                report.append(
                    f"warning: {slot} ({role}) に顔が写っています。顔の参照と混線しやすいので、顔を含まない画像を推奨します。"
                )

            pil = _bgr_to_pil(bgr)
            filename = f"{slot}.png"
            pictures.append({"file": filename, "role": role, "slot": slot})
            saved.append((filename, pil))
            report.append(f"{slot}: {role} {pil.width}x{pil.height}")

        os.makedirs(pack_dir, exist_ok=True)
        for name in os.listdir(pack_dir):
            if name.endswith(".png") and name[:-4] in dict(SLOTS):
                os.remove(os.path.join(pack_dir, name))
        for filename, pil in saved:
            pil.save(os.path.join(pack_dir, filename))

        meta = {
            "version": 1,
            "name": character_name.strip(),
            "subject": subject.strip() or "person",
            "identity_description": identity_description.strip(),
            "outfit_description": outfit_description.strip(),
            "pictures": pictures,
            "face_similarity": similarity,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(pack_dir, "char.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        report.insert(0, f"saved: {pack_dir} ({len(pictures)} pictures)")
        return (pack_dir, _preview([p for _, p in saved]), "\n".join(report))


# ---------------------------------------------------------------- Load

def _picture_list(indices: list) -> str:
    tags = [f"<Picture {i}>" for i in indices]
    return tags[0] if len(tags) == 1 else ", ".join(tags[:-1]) + " and " + tags[-1]


def _sentence(text: str) -> str:
    text = text.strip()
    return text if not text or text[-1] in ".!?" else text + "."


class KWTR_CharacterPackLoad:
    DESCRIPTION = (
        "Character Pack Build で保存したキャラを読み込み、picture_1..9 (顔 → 服 → 体) と "
        "<Picture N> 付きの edit_instruction を出力する。KWTR H3 Picture Reference Edit にそのまま接続する。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        characters = _list_characters() or ["(none)"]
        return {
            "required": {
                "character": (characters,),
                "scene_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Full-body shot of Mina, standing, smiling at the viewer.\nBright simple white background.",
                    "tooltip": "構図・ポーズ・背景など。キャラ名は <Picture 1> の人物に置き換わる。外見 (髪色・服) は書かない。",
                }),
                "use_outfit": ("BOOLEAN", {"default": True, "tooltip": "オフにすると服の参照を渡さず、服装は scene_prompt で指定する。"}),
                "use_body": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",) + ("IMAGE",) * MAX_PICTURES + ("INT", "STRING")
    RETURN_NAMES = ("edit_instruction",) + tuple(f"picture_{i}" for i in range(1, MAX_PICTURES + 1)) + ("picture_count", "run_info")
    FUNCTION = "load"
    CATEGORY = "MiniMax H3/KWTR"

    @classmethod
    def IS_CHANGED(cls, character, **kwargs):
        pack_dir = os.path.join(_pack_root(), character)
        if not os.path.isdir(pack_dir):
            return character
        h = hashlib.sha256()
        for name in sorted(os.listdir(pack_dir)):
            h.update(f"{name}:{os.path.getmtime(os.path.join(pack_dir, name))}".encode())
        return h.hexdigest()

    def load(self, character, scene_prompt, use_outfit, use_body):
        pack_dir = os.path.join(_pack_root(), character)
        meta_path = os.path.join(pack_dir, "char.json")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"{meta_path} がありません。先に KWTR Character Pack Build で作成してください。")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        subject = meta.get("subject") or "person"
        order = {"face": 0, "outfit": 1, "body": 2}
        entries = sorted(meta["pictures"], key=lambda p: order[p["role"]])
        entries = [p for p in entries
                   if not (p["role"] == "outfit" and not use_outfit)
                   and not (p["role"] == "body" and not use_body)]

        images, roles = [], {"face": [], "outfit": [], "body": []}
        for n, entry in enumerate(entries, start=1):
            images.append(_pil_to_tensor(Image.open(os.path.join(pack_dir, entry["file"]))))
            roles[entry["role"]].append(n)

        # scene_prompt 中のキャラ名: 最初は "the <subject> from <Picture 1>"、以降は "the <subject>"
        name = meta.get("name", "")
        scene = scene_prompt.strip()
        if name:
            count = {"n": 0}

            def repl(_m):
                count["n"] += 1
                return f"the {subject} from <Picture 1>" if count["n"] == 1 else f"the {subject}"

            scene = re.sub(rf"(?<!\w){re.escape(name)}(?!\w)", repl, scene, flags=re.IGNORECASE)
            # 文頭に来た "the" を大文字に
            scene = re.sub(r"(^|[.!?]\s+|\n)the ", lambda m: m.group(1) + "The ", scene)

        locked = [f"Keep the {subject}'s identity, face and hairstyle from {_picture_list(roles['face'])}."]
        if meta.get("identity_description"):
            locked.append(_sentence(meta["identity_description"]))
        if roles["outfit"]:
            outfit = meta.get("outfit_description") or "outfit"
            refs = _picture_list(roles["outfit"])
            locked.append(f"The {subject} wears the {outfit} from {refs}; the clothing must visibly match {refs}.")
        if roles["body"]:
            locked.append(f"Match the body shape and proportions from {_picture_list(roles['body'])}.")

        edit_instruction = "\n".join(filter(None, [scene, *locked]))
        outputs = images + [None] * (MAX_PICTURES - len(images))
        run_info = (
            f"{character}: " + ", ".join(f"<Picture {i}>={e['role']}({e['file']})" for i, e in enumerate(entries, start=1))
        )
        return (edit_instruction, *outputs, len(images), run_info)


NODE_CLASS_MAPPINGS = {
    "KWTR_CharacterPackBuild": KWTR_CharacterPackBuild,
    "KWTR_CharacterPackLoad": KWTR_CharacterPackLoad,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KWTR_CharacterPackBuild": "KWTR Character Pack Build",
    "KWTR_CharacterPackLoad": "KWTR Character Pack Load",
}
