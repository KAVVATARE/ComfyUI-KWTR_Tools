# ComfyUI-KWTR_Tools

ComfyUI 用のカスタムノード集です。サンプリング効率化、MiniMax-H3 動画生成の補助、細かいUI/ユーティリティ系のノードをまとめています。

## インストール

```
cd ComfyUI/custom_nodes
git clone https://github.com/KAVVATARE/ComfyUI-KWTR_Tools.git
```

ComfyUI を再起動すると、各ノードが `sampling/custom` / `utils` / `KWTR` などのカテゴリに追加されます。

## 収録ノード

### KSampler (Dual Phase) — `DualPhaseKSampler`
2つの `KSamplerAdvanced` を共有ノイズスケジュールで連結するサンプラーです。SDXLのBase+Refinerのように、指定ステップ (`shift_model_at_step`) で `model_1` から `model_2` へ切り替えて生成を継続します。モデル間でアーキテクチャやスケジュールが異なっていても、フェーズ1を完全デノイズしてからフェーズ2でノイズを乗せ直すことで、sigma不一致によるノイズ残留を避けています。

### KSampler (Latent Upscale) — `LatentUpscaleKSampler`
MiniMax-H3 向けの2段階サンプラーです。低解像度でサンプリング → 学習済みlatentアップスケーラでアスペクト比を保ったまま拡大 → conditioning(参照/キーフレーム情報)を新しい解像度に同期 → 短いステップ数で高解像度リファイン、という流れをワンノードにまとめています。
[Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) が別途必要です。`bypass_upscale` を有効にすると、アップスケーラ無しでフェーズ1のみ（フルスケジュール）を実行できます。

### Load CLIP Vision (Device) — `CLIPVisionLoaderDevice`
標準の CLIP Vision ローダーに `device` 切り替え（`default` / `cpu`）を追加したものです。`cpu` を選ぶと Vision モデルを VRAM ではなく RAM に保持したまま使えます。

### VAE Decode Tiled (Progress) 🟢 — `VAEDecodeTiledProgress`
タイル分割VAEデコードのラッパーです。ノード枠に緑色の進捗バーを表示します（`web/kwtr_vae_progress.js`）。画像/動画どちらのlatentにも対応し、`minimax_progress_patch.py` が MiniMax-H3 の VideoVAE にパッチを当てて、内部の時間方向チャンク処理から進捗イベントを発火させています。

### prompt + filename_prefix output — `MetaInfoFilenamePrefix`
VIDEO/IMAGE などの `trigger` が到達した時刻でタイムスタンプを確定し、`filename_prefix`（フォルダ＋日時）と、そのまま渡した `prompt` テキストを出力します。生成物とプロンプトログを同名・同時刻で対にして保存したい場合に使います。

### Audio Duration — `AudioDuration`
`AUDIO` 入力から再生時間（秒）・サンプル数・サンプルレートを取り出します。

### Amount Slider — `AmountSlider`
0.0〜1.0 のスライダー付き FLOAT 出力ノードです。

### Float (0.1 step) — `FloatFine`
0.1刻みで細かく調整できる FLOAT 入力ノードです。

## 依存関係

- ComfyUI 本体（`comfy.samplers` / `comfy.clip_vision` / `comfy.utils` などを利用）
- `LatentUpscaleKSampler` を使う場合は [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) の導入が必要です
- `minimax_progress_patch.py` は MiniMax-H3 系ノード（`comfy/ldm/minimax/vae.py`）が存在する環境を前提にしています。無い環境では何もせずスキップされます

## ライセンス

[LICENSE](./LICENSE) を参照してください。
