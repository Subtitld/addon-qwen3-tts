# PyInstaller spec for the qwen3-tts add-on.
# Build with: pyinstaller qwen3-tts-addon.spec --distpath dist/
# Resulting dist/qwen3-tts-addon/ + manifest.json + LICENSE + README.md is
# zipped into qwen3-tts-<version>-<platform>.zip for the catalog.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Same defensive collect pattern as the coqui-xtts addon: qwen-tts pulls a
# big, partly-optional dependency tree (transformers, accelerate, torch,
# librosa, soundfile, sox, onnxruntime, einops). We let collect_submodules
# walk each one but never fail the freeze if a sub-package is absent — the
# actual import chain pulled by the addon code is what determines runtime
# correctness.
def _safe_collect(fn, name):
    try:
        return fn(name)
    except Exception:
        return []


hiddenimports = (
    _safe_collect(collect_submodules, 'qwen_tts')
    + _safe_collect(collect_submodules, 'transformers')
    + _safe_collect(collect_submodules, 'accelerate')
    + _safe_collect(collect_submodules, 'librosa')
    + _safe_collect(collect_submodules, 'soundfile')
    + _safe_collect(collect_submodules, 'einops')
)
datas = (
    _safe_collect(collect_data_files, 'qwen_tts')
    + _safe_collect(collect_data_files, 'transformers')
    + _safe_collect(collect_data_files, 'librosa')
    + [('manifest.json', '.')]
)

block_cipher = None

a = Analysis(
    ['qwen3_tts_addon.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Excluded heavyweight optional ML libs we do not use; cuts ~500 MB
    # off the bundle when they happen to be installed in the build env.
    excludes=['tensorflow', 'jax', 'flax', 'gradio'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name='qwen3-tts-addon',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)
coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=False, upx_exclude=[],
    name='qwen3-tts-addon',
)
