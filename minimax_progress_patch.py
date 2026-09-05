"""
Progress reporting patch for MiniMax-H3's video VAE.

comfy/ldm/minimax/vae.py's MiniMaxH3VideoVAE.decode_tiled() is a pass-through
to decode()/decode_temporal(), which streams fixed-size temporal chunks
internally (to keep VRAM low) but never reports progress through
comfy.utils.ProgressBar. That leaves VAEDecodeTiledProgress's frontend
extension with no "progress" API events to draw a bar from.

This patches two existing methods instead of duplicating decode_temporal's
chunking/blending logic:
  - decode_temporal: computes num_chunks up front (the same call the
    original makes) and opens one ProgressBar for the call.
  - _adaptive_decode: called exactly once per temporal chunk inside
    decode_temporal's loop; advances the ProgressBar by one step per call.
"""
import comfy.utils


def _patch_minimax_h3_progress():
    try:
        from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    except Exception:
        return
    if getattr(MiniMaxH3VideoVAE, "_kwtr_progress_patched", False):
        return

    orig_decode_temporal = MiniMaxH3VideoVAE.decode_temporal
    orig_adaptive_decode = MiniMaxH3VideoVAE._adaptive_decode

    def decode_temporal(self, z, output_buffer=None):
        try:
            _pad_tokens, num_chunks = self._decode_temporal_chunks(z.shape[2])
        except Exception:
            num_chunks = None

        self._kwtr_pbar = comfy.utils.ProgressBar(num_chunks) if num_chunks else None
        self._kwtr_pbar_i = 0
        try:
            return orig_decode_temporal(self, z, output_buffer=output_buffer)
        finally:
            self._kwtr_pbar = None

    def _adaptive_decode(self, z):
        pbar = getattr(self, "_kwtr_pbar", None)
        if pbar is not None:
            self._kwtr_pbar_i += 1
            pbar.update_absolute(self._kwtr_pbar_i, pbar.total)
        return orig_adaptive_decode(self, z)

    MiniMaxH3VideoVAE.decode_temporal = decode_temporal
    MiniMaxH3VideoVAE._adaptive_decode = _adaptive_decode
    MiniMaxH3VideoVAE._kwtr_progress_patched = True


_patch_minimax_h3_progress()
