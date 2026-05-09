"""Subtitld add-on entry point for Qwen3-TTS.

Wraps `qwen-tts` (PyPI) — Apache-2.0 — exposing two flavors:

  * **Custom voice**: 9 fixed timbres baked into
    `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (Vivian, Serena, Uncle_Fu, Dylan,
    Eric, Ryan, Aiden, Ono_Anna, Sohee). The voice id `qwen3-<name>` selects
    one. Native language is recommended for best quality, but each speaker
    can read any of the 10 supported languages.

  * **Voice clone**: voice id `qwen3-clone` instead loads
    `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and clones from a 3+ second reference
    audio + (optional) transcript. With transcript the quality is clearly
    higher; without it we fall back to x-vector-only mode.

Both models are 1.7B params in BF16. Loading both at once eats ~7 GB —
so we keep a single-active-model cache and swap when the request mode
changes. First load also downloads the model (3-4 GB) from HuggingFace,
which is why `startup_timeout_sec` in the manifest is 180s and we emit
a 'Loading...' progress frame upfront.

The Qwen3-TTS Python API:
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(repo_id, device_map=..., dtype=...)
    wavs, sr = model.generate_custom_voice(text, language, speaker, instruct?)
    wavs, sr = model.generate_voice_clone(text, language, ref_audio, ref_text?, x_vector_only_mode=...)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import wave
from pathlib import Path

log = logging.getLogger('qwen3-tts')
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format='[qwen3-tts] %(levelname)s %(message)s')

PROTOCOL = 1
ADDON_ID = 'qwen3-tts'
VERSION = '1.0.0'

# HF repo ids (overridable via env for offline / mirrored installs).
DEFAULT_CUSTOMVOICE_REPO = os.environ.get(
    'QWEN3_TTS_CUSTOMVOICE_REPO', 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice')
DEFAULT_BASE_REPO = os.environ.get(
    'QWEN3_TTS_BASE_REPO', 'Qwen/Qwen3-TTS-12Hz-1.7B-Base')

# Voice-id → (model variant, native language, qwen-tts speaker name).
# Speaker names mirror the table in QwenLM/Qwen3-TTS README.
VOICE_TABLE: dict[str, tuple[str, str, str]] = {
    'qwen3-vivian':   ('customvoice', 'Chinese', 'Vivian'),
    'qwen3-serena':   ('customvoice', 'Chinese', 'Serena'),
    'qwen3-uncle-fu': ('customvoice', 'Chinese', 'Uncle_Fu'),
    'qwen3-dylan':    ('customvoice', 'Chinese', 'Dylan'),
    'qwen3-eric':     ('customvoice', 'Chinese', 'Eric'),
    'qwen3-ryan':     ('customvoice', 'English', 'Ryan'),
    'qwen3-aiden':    ('customvoice', 'English', 'Aiden'),
    'qwen3-ono-anna': ('customvoice', 'Japanese', 'Ono_Anna'),
    'qwen3-sohee':    ('customvoice', 'Korean', 'Sohee'),
    'qwen3-clone':    ('base', '*', ''),
}

# BCP-47 short tag → name expected by qwen-tts. Fallback: "Auto".
LANGUAGE_TABLE: dict[str, str] = {
    'zh': 'Chinese', 'zh-cn': 'Chinese', 'zh-tw': 'Chinese',
    'en': 'English', 'en-us': 'English', 'en-gb': 'English',
    'ja': 'Japanese', 'ja-jp': 'Japanese',
    'ko': 'Korean',   'ko-kr': 'Korean',
    'de': 'German',   'de-de': 'German',
    'fr': 'French',   'fr-fr': 'French',
    'ru': 'Russian',  'ru-ru': 'Russian',
    'pt': 'Portuguese', 'pt-br': 'Portuguese', 'pt-pt': 'Portuguese',
    'es': 'Spanish',  'es-es': 'Spanish', 'es-mx': 'Spanish',
    'it': 'Italian',  'it-it': 'Italian',
}


# ---------------------------------------------------------------------------
# Wire helpers
# ---------------------------------------------------------------------------
_write_lock = threading.Lock()


def write_frame(frame: dict) -> None:
    line = json.dumps(frame, ensure_ascii=False)
    with _write_lock:
        sys.stdout.write(line + '\n')
        sys.stdout.flush()


def emit_progress(rid, value, message=''):
    write_frame({'id': rid, 'type': 'progress',
                 'data': {'value': max(0.0, min(1.0, float(value))), 'message': message}})


def emit_error(rid, code, message, retryable=False):
    write_frame({'id': rid, 'type': 'error',
                 'data': {'code': code, 'message': message, 'retryable': retryable}})


def emit_result(rid, data):
    write_frame({'id': rid, 'type': 'result', 'data': data})


# ---------------------------------------------------------------------------
# Model state — single-active-model cache to keep RAM tame
# ---------------------------------------------------------------------------
_model_lock = threading.Lock()
_model_cache: dict = {'variant': None, 'instance': None}
_pending_cancel: set[str] = set()
_pending_cancel_lock = threading.Lock()


def _resolve_dtype(dtype_pref: str, device: str):
    """Translate config string to a `torch.dtype`. CPU silently falls back to
    fp32 when fp16 is requested, since CPU fp16 is broken on most builds."""
    import torch
    pref = (dtype_pref or 'auto').lower()
    if pref == 'auto':
        return torch.bfloat16 if device.startswith('cuda') else torch.float32
    if pref == 'bfloat16':
        return torch.bfloat16
    if pref == 'float16':
        return torch.float16 if device.startswith('cuda') else torch.float32
    if pref == 'float32':
        return torch.float32
    return torch.float32


def _resolve_attn(device: str) -> str | None:
    """Use flash-attn-2 only when CUDA + the package imports cleanly. CPU and
    MPS get the eager default. We keep the call best-effort because flash-attn
    is build-time fragile."""
    if not device.startswith('cuda'):
        return None
    try:
        import flash_attn  # noqa: F401
        return 'flash_attention_2'
    except Exception:
        return None


def _load_model(variant: str, device: str, dtype_pref: str):
    """Load the CustomVoice or Base model, swapping the cached one if needed."""
    with _model_lock:
        if _model_cache['variant'] == variant and _model_cache['instance'] is not None:
            return _model_cache['instance']

        # Drop the previous model first so we don't briefly hold both in RAM.
        _model_cache['instance'] = None
        _model_cache['variant'] = None
        try:
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        except Exception:
            pass

        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f'qwen-tts python package not available: {exc}') from exc

        repo = DEFAULT_CUSTOMVOICE_REPO if variant == 'customvoice' else DEFAULT_BASE_REPO
        dtype = _resolve_dtype(dtype_pref, device)
        attn = _resolve_attn(device)

        kwargs = {'device_map': device, 'dtype': dtype}
        if attn is not None:
            kwargs['attn_implementation'] = attn

        log.info('loading %s on %s (dtype=%s, attn=%s)', repo, device, dtype, attn or 'eager')
        model = Qwen3TTSModel.from_pretrained(repo, **kwargs)
        _model_cache['variant'] = variant
        _model_cache['instance'] = model
        return model


# ---------------------------------------------------------------------------
# Audio writing
# ---------------------------------------------------------------------------
def _write_wav(path: str, wav, sample_rate: int) -> tuple[float, int, int]:
    """Write a 1-D float waveform out as PCM-16 mono WAV. Returns
    (duration_sec, sample_rate, channels)."""
    import numpy as np
    import soundfile as sf

    arr = np.asarray(wav)
    if arr.ndim > 1:
        arr = arr.squeeze()
    if arr.ndim != 1:
        raise RuntimeError(f'unexpected waveform shape: {arr.shape}')

    # qwen3-tts returns float in [-1, 1]; coerce to int16 PCM for predictable
    # consumption by the Subtitld dubbing pipeline (matches XTTS output).
    arr = np.clip(arr, -1.0, 1.0)
    sf.write(path, arr, int(sample_rate), subtype='PCM_16')

    duration = float(len(arr)) / float(sample_rate or 1)
    return duration, int(sample_rate), 1


# ---------------------------------------------------------------------------
# Request handling
# ---------------------------------------------------------------------------
def handle_tts_synthesize(rid: str, params: dict, defaults: dict) -> None:
    text = params.get('text')
    voice_id = params.get('voice')
    output_path = params.get('output_path')
    raw_lang = (params.get('language') or '').lower()
    if not text or not voice_id or not output_path:
        emit_error(rid, 'bad_params', 'text, voice, and output_path are all required')
        return

    voice_meta = VOICE_TABLE.get(voice_id)
    if voice_meta is None:
        emit_error(rid, 'unsupported_voice', f'unknown voice id: {voice_id!r}')
        return
    variant, native_lang, speaker_name = voice_meta

    with _pending_cancel_lock:
        if rid in _pending_cancel:
            _pending_cancel.discard(rid)
            emit_error(rid, 'cancelled', 'cancelled before synthesis started')
            return

    speaker_wav = params.get('voice_ref_audio') or defaults.get('voice_ref_audio') or ''
    speaker_text = params.get('voice_ref_text') or defaults.get('voice_ref_text') or ''
    if variant == 'base' and not speaker_wav:
        emit_error(rid, 'bad_params',
                   'qwen3-clone voice requires `voice_ref_audio` (path to 3+s reference clip)')
        return

    language = LANGUAGE_TABLE.get(raw_lang)
    if language is None:
        # Try the prefix (e.g. "pt-anything" → "pt") before giving up to Auto.
        language = LANGUAGE_TABLE.get(raw_lang.split('-', 1)[0]) if raw_lang else None
    if language is None:
        language = native_lang if native_lang and native_lang != '*' else 'Auto'

    emit_progress(rid, 0.05, 'Loading Qwen3-TTS model (first call may download ~3.4 GB)...')
    try:
        model = _load_model(variant, defaults['device'], defaults['dtype'])
    except Exception as exc:
        log.exception('model load failed')
        emit_error(rid, 'internal', f'failed to load Qwen3-TTS: {exc}')
        return

    emit_progress(rid, 0.4, 'Synthesizing...')
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    instruct = params.get('instruct') or ''

    try:
        if variant == 'customvoice':
            kwargs = {'text': text, 'language': language, 'speaker': speaker_name}
            if instruct:
                kwargs['instruct'] = instruct
            wavs, sr = model.generate_custom_voice(**kwargs)
        else:
            kwargs = {
                'text': text,
                'language': language,
                'ref_audio': speaker_wav,
            }
            if speaker_text:
                kwargs['ref_text'] = speaker_text
                kwargs['x_vector_only_mode'] = False
            else:
                # No transcript → x-vector-only is the supported path.
                kwargs['x_vector_only_mode'] = True
            wavs, sr = model.generate_voice_clone(**kwargs)
    except Exception as exc:
        log.exception('synth failed')
        emit_error(rid, 'internal', f'synthesize failed: {exc}')
        return

    if not wavs:
        emit_error(rid, 'internal', 'model returned no audio')
        return

    try:
        duration, sample_rate, channels = _write_wav(output_path, wavs[0], sr)
    except Exception as exc:
        log.exception('wav write failed')
        emit_error(rid, 'internal', f'failed to write {output_path}: {exc}')
        return

    emit_progress(rid, 0.99, 'Finalizing...')
    emit_result(rid, {
        'path': output_path,
        'duration_sec': duration,
        'sample_rate': sample_rate,
        'channels': channels,
    })


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> int:
    manifest_path = Path(__file__).resolve().parent / 'manifest.json'
    voices: list[dict] = []
    languages: list[str] = []
    config_defaults: dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            voices = manifest.get('voices') or []
            languages = manifest.get('languages') or []
            config_defaults = {f.get('key'): f.get('default')
                               for f in (manifest.get('config_schema') or {}).get('fields', [])
                               if f.get('default') is not None}
        except Exception:
            log.exception('manifest parse failed')

    defaults = {
        'device': os.environ.get('QWEN3_TTS_DEVICE') or config_defaults.get('device', 'cpu'),
        'dtype':  os.environ.get('QWEN3_TTS_DTYPE')  or config_defaults.get('dtype', 'auto'),
        'voice_ref_audio': os.environ.get('QWEN3_TTS_VOICE_REF_AUDIO') or '',
        'voice_ref_text':  os.environ.get('QWEN3_TTS_VOICE_REF_TEXT')  or '',
    }

    write_frame({
        'type': 'hello',
        'protocol': PROTOCOL,
        'addon': ADDON_ID,
        'version': VERSION,
        'capabilities': [
            {'task': 'tts.synthesize', 'languages': languages, 'voices': voices,
             'voice_clone': True},
        ],
    })

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            frame = json.loads(raw_line)
        except json.JSONDecodeError:
            continue

        ftype = frame.get('type')
        rid = frame.get('id', '')

        if ftype == 'shutdown':
            log.info('shutdown received; exiting')
            return 0
        if ftype == 'cancel':
            target = (frame.get('data') or {}).get('target') or frame.get('target')
            if target:
                with _pending_cancel_lock:
                    _pending_cancel.add(target)
            continue
        if ftype == 'tts.synthesize':
            threading.Thread(
                target=handle_tts_synthesize,
                args=(rid, frame.get('params') or {}, defaults),
                daemon=True,
            ).start()
            continue

        emit_error(rid, 'bad_params', f'unknown request type: {ftype!r}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
