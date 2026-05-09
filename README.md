# Qwen3-TTS add-on for Subtitld

Multilingual neural TTS based on
[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), Apache-2.0. Heavy:
the model is ~3.4 GB, peak RAM around 8 GB, CPU inference is several
seconds per line.

Two flavors selected by voice id:

- **9 fixed timbres** — `qwen3-vivian`, `qwen3-serena`, `qwen3-uncle-fu`,
  `qwen3-dylan`, `qwen3-eric` (Chinese); `qwen3-ryan`, `qwen3-aiden`
  (English); `qwen3-ono-anna` (Japanese); `qwen3-sohee` (Korean). Powered
  by `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`. Each speaker can read any
  of the 10 supported languages, but native language is recommended.
- **Voice clone** — `qwen3-clone` switches to
  `Qwen/Qwen3-TTS-12Hz-1.7B-Base` and clones from a 3+ second reference
  clip. With the matching transcript (`voice_ref_text` param) the
  quality jumps noticeably; without it we fall back to x-vector-only
  mode.

The add-on holds a single-active-model cache and swaps when the request
mode changes — loading both at once would eat ~7 GB.

## Languages

10 supported: Chinese, English, Japanese, Korean, German, French, Russian,
Portuguese, Spanish, Italian.

## Building

```bash
# CPU build (also works for AMD/Intel without CUDA)
pip install pyinstaller
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu
pip install qwen-tts
pyinstaller qwen3-tts-addon.spec --distpath dist/
cd dist/qwen3-tts-addon
zip -r ../qwen3-tts-1.0.0-linux-x86_64.zip . ../../manifest.json ../../LICENSE ../../README.md
```

For CUDA builds, install the matching `torch` wheel (`+cu121`, `+cu118`,
etc.) and ship a separate zip per CUDA version. flash-attn is detected at
runtime — when present on a CUDA box the wrapper passes
`attn_implementation="flash_attention_2"`; otherwise the model loads with
the eager attention default.

## Model storage

Weights are *not* bundled — they're downloaded on first run from
HuggingFace into the standard transformers cache (`HF_HOME`,
`~/.cache/huggingface/hub` by default). The Subtitld host shows a UI
confirmation prompt before the download begins.

`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (~3.4 GB) is fetched the first
time any of the 9 timbres is selected. `Qwen/Qwen3-TTS-12Hz-1.7B-Base`
(~3.4 GB) is fetched on the first request that picks `qwen3-clone`. Plan
for ~7 GB on disk if both modes are used.

## Voice cloning

The `qwen3-clone` voice id treats the per-request `voice_ref_audio`
parameter as the reference (any 3+ second mono WAV). Without an explicit
reference, the add-on falls back to the addon-config-level
`voice_ref_audio` (a default reference clip the user picks once).

If you also pass `voice_ref_text` (or set it in the addon config), the
model uses ICL mode for higher fidelity. Otherwise it uses
x-vector-only mode (no transcript needed, slightly lower quality).

## License

The wrapper code in this repo is Apache-2.0. The Qwen3-TTS model weights
are Apache-2.0 as well — commercial use is permitted (unlike the
Coqui XTTS-v2 weights, which are non-commercial under CPML).
