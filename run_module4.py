"""
AEVO — Module 4: ONNX + OpenVINO Compression
PyTorch best.pt → ONNX → OpenVINO FP16 → OpenVINO INT8
Uses ultralytics built-in export + modern OpenVINO 2023+ API
"""
import json, time, subprocess, sys
from pathlib import Path

ROOT       = Path(__file__).parent
BEST_PT    = ROOT / 'runs/pcb_yolov8n/weights/best.pt'
EXPORT_DIR = ROOT / 'exports'
YAML_PATH  = str(ROOT / 'pcb_dataset/dataset.yaml')
EXPORT_DIR.mkdir(exist_ok=True)

print('=' * 64)
print('  AEVO MODULE 4 — ONNX + OPENVINO COMPRESSION')
print('=' * 64)

if not BEST_PT.exists():
    raise FileNotFoundError(f'best.pt not found: {BEST_PT}')

print(f'\n  Source  : {BEST_PT}  ({BEST_PT.stat().st_size/1e6:.1f} MB)')

from ultralytics import YOLO
import shutil

model = YOLO(str(BEST_PT))

# ── 1. Export ONNX ────────────────────────────────────────────────────────────
print('\n[1] Exporting PyTorch → ONNX ...')
t0 = time.time()
model.export(format='onnx', imgsz=640, simplify=True, opset=12, dynamic=False)
# ultralytics saves next to the weights file
src_onnx = BEST_PT.parent / 'best.onnx'
ONNX_PATH = EXPORT_DIR / 'pcb_yolov8n.onnx'
if src_onnx.exists():
    shutil.copy2(src_onnx, ONNX_PATH)
print(f'    Done ({time.time()-t0:.1f}s)  →  {ONNX_PATH}  ({ONNX_PATH.stat().st_size/1e6:.1f} MB)')

# ── 2. Export OpenVINO FP16 ───────────────────────────────────────────────────
print('\n[2] Exporting → OpenVINO FP16 ...')
t0 = time.time()
model2 = YOLO(str(BEST_PT))
model2.export(format='openvino', imgsz=640, half=True)
# ultralytics saves to best_openvino_model/ next to weights
src_ov = BEST_PT.parent / 'best_openvino_model'
OV_FP16_DIR = EXPORT_DIR / 'openvino_fp16'
if src_ov.exists():
    if OV_FP16_DIR.exists():
        shutil.rmtree(OV_FP16_DIR)
    shutil.copytree(src_ov, OV_FP16_DIR)
ov_fp16_size = sum(f.stat().st_size for f in OV_FP16_DIR.rglob('*') if f.is_file()) / 1e6
print(f'    Done ({time.time()-t0:.1f}s)  →  {OV_FP16_DIR}  ({ov_fp16_size:.1f} MB)')

# ── 3. Export OpenVINO INT8 ───────────────────────────────────────────────────
print('\n[3] Exporting → OpenVINO INT8 (calibration on val set) ...')
t0 = time.time()
try:
    model3 = YOLO(str(BEST_PT))
    model3.export(format='openvino', imgsz=640, int8=True, data=YAML_PATH)
    src_ov_int8 = BEST_PT.parent / 'best_int8_openvino_model'
    # fallback name patterns
    if not src_ov_int8.exists():
        candidates = list(BEST_PT.parent.glob('*int8*'))
        src_ov_int8 = candidates[0] if candidates else None

    OV_INT8_DIR = EXPORT_DIR / 'openvino_int8'
    if src_ov_int8 and src_ov_int8.exists():
        if OV_INT8_DIR.exists():
            shutil.rmtree(OV_INT8_DIR)
        shutil.copytree(src_ov_int8, OV_INT8_DIR)
        ov_int8_size = sum(f.stat().st_size for f in OV_INT8_DIR.rglob('*') if f.is_file()) / 1e6
        print(f'    Done ({time.time()-t0:.1f}s)  →  {OV_INT8_DIR}  ({ov_int8_size:.1f} MB)')
    else:
        print(f'    INT8 dir not found — using FP16 as fallback for Module 5')
        OV_INT8_DIR = OV_FP16_DIR
        ov_int8_size = ov_fp16_size
except Exception as e:
    print(f'    INT8 export failed: {e}')
    print(f'    Using FP16 as INT8 fallback for benchmarking')
    OV_INT8_DIR = OV_FP16_DIR
    ov_int8_size = ov_fp16_size

# ── 4. Find XML files ─────────────────────────────────────────────────────────
fp16_xml = next(OV_FP16_DIR.rglob('*.xml'), None)
int8_xml  = next(OV_INT8_DIR.rglob('*.xml'), None)

# ── 5. Summary ────────────────────────────────────────────────────────────────
pt_mb   = BEST_PT.stat().st_size / 1e6
onnx_mb = ONNX_PATH.stat().st_size / 1e6

SEP = '=' * 64
print(f'\n{SEP}')
print('  AEVO MODULE 4 COMPLETE')
print(SEP)
print(f'  {"Format":<22} {"Size":>8}  Path')
print(f'  {"-"*22} {"-"*8}  {"-"*30}')
print(f'  {"PyTorch FP32":<22} {pt_mb:>7.1f}MB  {BEST_PT}')
print(f'  {"ONNX":<22} {onnx_mb:>7.1f}MB  {ONNX_PATH}')
print(f'  {"OpenVINO FP16":<22} {ov_fp16_size:>7.1f}MB  {OV_FP16_DIR}')
print(f'  {"OpenVINO INT8":<22} {ov_int8_size:>7.1f}MB  {OV_INT8_DIR}')
print(SEP)

# Save paths for Module 5
paths = {
    'pytorch':            str(BEST_PT),
    'onnx':               str(ONNX_PATH),
    'openvino_fp16_dir':  str(OV_FP16_DIR),
    'openvino_fp16_xml':  str(fp16_xml) if fp16_xml else '',
    'openvino_int8_dir':  str(OV_INT8_DIR),
    'openvino_int8_xml':  str(int8_xml) if int8_xml else '',
    'test_images':        str(ROOT / 'pcb_dataset/test/images')
}
out = EXPORT_DIR / 'model_paths.json'
out.write_text(json.dumps(paths, indent=2))
print(f'\n  Paths saved → {out}')
print('  Ready for Module 5 — Benchmark Reporter.')
