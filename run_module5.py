"""
AEVO — Module 5: Benchmark Reporter
100-image inference on PyTorch / ONNX / OpenVINO — professional terminal table
"""
import time, json, sys, subprocess
import numpy as np
from pathlib import Path

try:
    from tabulate import tabulate
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'tabulate'])
    from tabulate import tabulate

ROOT         = Path(__file__).parent
PATHS_FILE   = ROOT / 'exports/model_paths.json'
TEST_IMAGES  = ROOT / 'pcb_dataset/test/images'
N_IMAGES     = 100
IMGSZ        = 640
WARMUP       = 5     # warmup runs before timing

print('=' * 70)
print('  AEVO MODULE 5 — BENCHMARK REPORTER')
print('=' * 70)

if not PATHS_FILE.exists():
    raise FileNotFoundError(f'{PATHS_FILE} not found. Run Module 4 first.')

paths = json.loads(PATHS_FILE.read_text())

# Collect test images
imgs = sorted([p for p in TEST_IMAGES.glob('*.jpg')] +
              [p for p in TEST_IMAGES.glob('*.png')])[:N_IMAGES]
print(f'\nBenchmark images : {len(imgs)} (from {TEST_IMAGES})')
print(f'Input size       : {IMGSZ}×{IMGSZ}\n')

import cv2
def load_batch(img_paths, sz=640):
    frames = []
    for p in img_paths:
        img = cv2.imread(str(p))
        img = cv2.resize(img, (sz, sz))
        frames.append(img)
    return frames

frames = load_batch(imgs)
results_table = []

# ── 1. PyTorch (YOLOv8) ───────────────────────────────────────────────────────
print('[1] Benchmarking PyTorch YOLOv8n ...')
try:
    from ultralytics import YOLO
    import torch

    model_pt = YOLO(paths['pytorch'])
    device   = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Warmup
    for img in frames[:WARMUP]:
        model_pt.predict(img, imgsz=IMGSZ, device=device, verbose=False)

    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.perf_counter()
    for img in frames:
        model_pt.predict(img, imgsz=IMGSZ, device=device, verbose=False)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed_pt = time.perf_counter() - t0

    fps_pt      = len(frames) / elapsed_pt
    ms_pt       = elapsed_pt / len(frames) * 1000
    size_mb_pt  = Path(paths['pytorch']).stat().st_size / 1e6
    results_table.append(['PyTorch FP32', f'{fps_pt:.1f}', f'{ms_pt:.1f}', f'{size_mb_pt:.1f}', device.upper()])
    print(f'    FPS: {fps_pt:.1f}  |  {ms_pt:.1f} ms/img  |  {size_mb_pt:.1f} MB')
except Exception as e:
    print(f'    SKIP: {e}')
    results_table.append(['PyTorch FP32', 'N/A', 'N/A', 'N/A', 'N/A'])

# ── 2. ONNX Runtime ───────────────────────────────────────────────────────────
print('\n[2] Benchmarking ONNX Runtime ...')
try:
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = (['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else []) + \
                ['CPUExecutionProvider']
    sess = ort.InferenceSession(paths['onnx'], sess_opts, providers=providers)
    inp_name = sess.get_inputs()[0].name

    def preprocess(img, sz=640):
        img = cv2.resize(img, (sz, sz))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, 0)

    batch = [preprocess(f) for f in frames]

    # Warmup
    for b in batch[:WARMUP]:
        sess.run(None, {inp_name: b})

    t0 = time.perf_counter()
    for b in batch:
        sess.run(None, {inp_name: b})
    elapsed_onnx = time.perf_counter() - t0

    fps_onnx     = len(frames) / elapsed_onnx
    ms_onnx      = elapsed_onnx / len(frames) * 1000
    size_mb_onnx = Path(paths['onnx']).stat().st_size / 1e6
    prov         = sess.get_providers()[0].replace('ExecutionProvider', '')
    results_table.append(['ONNX FP32', f'{fps_onnx:.1f}', f'{ms_onnx:.1f}', f'{size_mb_onnx:.1f}', prov])
    print(f'    FPS: {fps_onnx:.1f}  |  {ms_onnx:.1f} ms/img  |  {size_mb_onnx:.1f} MB  ({prov})')
except Exception as e:
    print(f'    SKIP: {e}')
    results_table.append(['ONNX FP32', 'N/A', 'N/A', 'N/A', 'N/A'])

# ── 3. OpenVINO FP16 ──────────────────────────────────────────────────────────
print('\n[3] Benchmarking OpenVINO FP16 ...')
try:
    from openvino.runtime import Core

    ie     = Core()
    fp16_xml = paths.get('openvino_fp16_xml', '')
    model_ov = ie.read_model(fp16_xml)
    compiled = ie.compile_model(model_ov, 'CPU')
    infer_req = compiled.create_infer_request()
    inp_tensor = compiled.input(0)
    out_tensor = compiled.output(0)

    def preprocess_ov(img, sz=640):
        img = cv2.resize(img, (sz, sz))
        img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        return np.expand_dims(img, 0)

    batch_ov = [preprocess_ov(f) for f in frames]

    for b in batch_ov[:WARMUP]:
        infer_req.infer({inp_tensor: b})

    t0 = time.perf_counter()
    for b in batch_ov:
        infer_req.infer({inp_tensor: b})
    elapsed_ov_fp16 = time.perf_counter() - t0

    fps_ov_fp16  = len(frames) / elapsed_ov_fp16
    ms_ov_fp16   = elapsed_ov_fp16 / len(frames) * 1000
    ov_fp16_dir  = Path(fp16_xml).parent
    size_mb_fp16 = sum(f.stat().st_size for f in ov_fp16_dir.rglob('*') if f.is_file()) / 1e6
    results_table.append(['OpenVINO FP16', f'{fps_ov_fp16:.1f}', f'{ms_ov_fp16:.1f}',
                           f'{size_mb_fp16:.1f}', 'CPU'])
    print(f'    FPS: {fps_ov_fp16:.1f}  |  {ms_ov_fp16:.1f} ms/img  |  {size_mb_fp16:.1f} MB')
except Exception as e:
    print(f'    SKIP: {e}')
    results_table.append(['OpenVINO FP16', 'N/A', 'N/A', 'N/A', 'N/A'])

# ── 4. OpenVINO INT8 ──────────────────────────────────────────────────────────
print('\n[4] Benchmarking OpenVINO INT8 ...')
try:
    from openvino.runtime import Core

    ie      = Core()
    int8_dir = Path(paths['openvino_int8_dir'])
    int8_xml = next(int8_dir.rglob('*.xml'), None)
    if int8_xml is None:
        raise FileNotFoundError('No INT8 .xml found')

    model_int8 = ie.read_model(str(int8_xml))
    compiled8  = ie.compile_model(model_int8, 'CPU')
    infer8     = compiled8.create_infer_request()
    inp8       = compiled8.input(0)

    for b in batch_ov[:WARMUP]:
        infer8.infer({inp8: b})

    t0 = time.perf_counter()
    for b in batch_ov:
        infer8.infer({inp8: b})
    elapsed_int8 = time.perf_counter() - t0

    fps_int8   = len(frames) / elapsed_int8
    ms_int8    = elapsed_int8 / len(frames) * 1000
    size_int8  = sum(f.stat().st_size for f in int8_dir.rglob('*') if f.is_file()) / 1e6
    results_table.append(['OpenVINO INT8', f'{fps_int8:.1f}', f'{ms_int8:.1f}',
                           f'{size_int8:.1f}', 'CPU'])
    print(f'    FPS: {fps_int8:.1f}  |  {ms_int8:.1f} ms/img  |  {size_int8:.1f} MB')
except Exception as e:
    print(f'    SKIP: {e}')
    results_table.append(['OpenVINO INT8', 'N/A', 'N/A', 'N/A', 'N/A'])

# ── Final professional table ──────────────────────────────────────────────────
headers = ['Model', 'FPS ↑', 'ms/img ↓', 'Size MB ↓', 'Device']

SEP = '=' * 70
print(f'\n\n{SEP}')
print('  AEVO BENCHMARK — PCB DEFECT DETECTION')
print(f'  {N_IMAGES} images @ {IMGSZ}×{IMGSZ}  |  RTX 3050 Ti 4GB')
print(SEP)
print(tabulate(results_table, headers=headers, tablefmt='rounded_outline',
               numalign='right', stralign='left'))
print(SEP)
print('  Screenshot this table — it is your #1 portfolio artifact.')
print(SEP)

# Save table to txt
table_txt = tabulate(results_table, headers=headers, tablefmt='rounded_outline',
                     numalign='right', stralign='left')
out = ROOT / 'benchmark_results.txt'
out.write_text(f'AEVO Benchmark — PCB Defect Detection\n'
               f'{N_IMAGES} images @ {IMGSZ}×{IMGSZ}\n\n{table_txt}\n')
print(f'\nSaved → {out}')
