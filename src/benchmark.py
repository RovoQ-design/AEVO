"""
AEVO — Module 5: Benchmark Reporter
100-image inference on PyTorch / ONNX / OpenVINO — professional terminal table.
"""
import time
import json
import sys
import subprocess
import numpy as np
from pathlib import Path

try:
    from tabulate import tabulate
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'tabulate'])
    from tabulate import tabulate

ROOT         = Path(__file__).parent.parent
PATHS_FILE   = ROOT / 'exports' / 'model_paths.json'
TEST_IMAGES  = ROOT / 'pcb_dataset' / 'test' / 'images'
RESULTS_DIR  = ROOT / 'results'
N_IMAGES     = 100
IMGSZ        = 640
WARMUP       = 5


def load_frames(img_paths, sz=640):
    import cv2
    frames = []
    for p in img_paths:
        img = cv2.imread(str(p))
        frames.append(cv2.resize(img, (sz, sz)))
    return frames


def preprocess(img, sz=640):
    import cv2
    img = cv2.resize(img, (sz, sz))
    return np.expand_dims(
        img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0, 0
    )


def run():
    import cv2

    print('=' * 70)
    print('  AEVO MODULE 5 — BENCHMARK REPORTER')
    print('=' * 70)

    if not PATHS_FILE.exists():
        raise FileNotFoundError(f'{PATHS_FILE} not found — run Module 4 first.')

    paths = json.loads(PATHS_FILE.read_text())
    imgs  = sorted(TEST_IMAGES.glob('*.jpg'))[:N_IMAGES]
    print(f'\nBenchmark images : {len(imgs)}  |  Input size: {IMGSZ}×{IMGSZ}\n')

    frames = load_frames(imgs)
    table  = []

    # ── 1. PyTorch ──────────────────────────────────────────────────────────────
    print('[1] PyTorch YOLOv8n ...')
    try:
        import torch
        from ultralytics import YOLO
        model_pt = YOLO(paths['pytorch'])
        device   = 'cuda' if torch.cuda.is_available() else 'cpu'
        for f in frames[:WARMUP]:
            model_pt.predict(f, imgsz=IMGSZ, device=device, verbose=False)
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for f in frames:
            model_pt.predict(f, imgsz=IMGSZ, device=device, verbose=False)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        fps  = len(frames) / elapsed
        ms   = elapsed / len(frames) * 1000
        size = Path(paths['pytorch']).stat().st_size / 1e6
        table.append(['PyTorch FP32', f'{fps:.1f}', f'{ms:.1f}', f'{size:.1f}', device.upper()])
        print(f'    FPS: {fps:.1f}  |  {ms:.1f} ms/img  |  {size:.1f} MB')
    except Exception as e:
        print(f'    SKIP: {e}')
        table.append(['PyTorch FP32', 'N/A', 'N/A', 'N/A', 'N/A'])

    # ── 2. ONNX Runtime ─────────────────────────────────────────────────────────
    print('\n[2] ONNX Runtime ...')
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        providers = (['CUDAExecutionProvider'] if ort.get_device() == 'GPU' else []) + ['CPUExecutionProvider']
        sess     = ort.InferenceSession(paths['onnx'], opts, providers=providers)
        inp_name = sess.get_inputs()[0].name
        batch    = [preprocess(f) for f in frames]
        for b in batch[:WARMUP]:
            sess.run(None, {inp_name: b})
        t0 = time.perf_counter()
        for b in batch:
            sess.run(None, {inp_name: b})
        elapsed = time.perf_counter() - t0
        fps  = len(frames) / elapsed
        ms   = elapsed / len(frames) * 1000
        size = Path(paths['onnx']).stat().st_size / 1e6
        prov = sess.get_providers()[0].replace('ExecutionProvider', '')
        table.append(['ONNX FP32', f'{fps:.1f}', f'{ms:.1f}', f'{size:.1f}', prov])
        print(f'    FPS: {fps:.1f}  |  {ms:.1f} ms/img  |  {size:.1f} MB  ({prov})')
    except Exception as e:
        print(f'    SKIP: {e}')
        table.append(['ONNX FP32', 'N/A', 'N/A', 'N/A', 'N/A'])

    # ── 3. OpenVINO FP16 ────────────────────────────────────────────────────────
    print('\n[3] OpenVINO FP16 ...')
    try:
        from openvino.runtime import Core
        ie       = Core()
        fp16_xml = paths.get('openvino_fp16_xml', '')
        model_ov = ie.read_model(fp16_xml)
        compiled = ie.compile_model(model_ov, 'CPU')
        infer    = compiled.create_infer_request()
        inp_t    = compiled.input(0)
        batch_ov = [preprocess(f) for f in frames]
        for b in batch_ov[:WARMUP]:
            infer.infer({inp_t: b})
        t0 = time.perf_counter()
        for b in batch_ov:
            infer.infer({inp_t: b})
        elapsed  = time.perf_counter() - t0
        fps      = len(frames) / elapsed
        ms       = elapsed / len(frames) * 1000
        fp16_dir = Path(fp16_xml).parent
        size     = sum(f.stat().st_size for f in fp16_dir.rglob('*') if f.is_file()) / 1e6
        table.append(['OpenVINO FP16', f'{fps:.1f}', f'{ms:.1f}', f'{size:.1f}', 'CPU'])
        print(f'    FPS: {fps:.1f}  |  {ms:.1f} ms/img  |  {size:.1f} MB')
    except Exception as e:
        print(f'    SKIP: {e}')
        table.append(['OpenVINO FP16', 'N/A', 'N/A', 'N/A', 'N/A'])
        batch_ov = [preprocess(f) for f in frames]

    # ── 4. OpenVINO INT8 ────────────────────────────────────────────────────────
    print('\n[4] OpenVINO INT8 ...')
    try:
        from openvino.runtime import Core
        ie       = Core()
        int8_dir = Path(paths['openvino_int8_dir'])
        int8_xml = next(int8_dir.rglob('*.xml'), None)
        if int8_xml is None:
            raise FileNotFoundError('No INT8 .xml found')
        model8   = ie.read_model(str(int8_xml))
        compiled8 = ie.compile_model(model8, 'CPU')
        infer8   = compiled8.create_infer_request()
        inp8     = compiled8.input(0)
        for b in batch_ov[:WARMUP]:
            infer8.infer({inp8: b})
        t0 = time.perf_counter()
        for b in batch_ov:
            infer8.infer({inp8: b})
        elapsed = time.perf_counter() - t0
        fps     = len(frames) / elapsed
        ms      = elapsed / len(frames) * 1000
        size    = sum(f.stat().st_size for f in int8_dir.rglob('*') if f.is_file()) / 1e6
        table.append(['OpenVINO INT8', f'{fps:.1f}', f'{ms:.1f}', f'{size:.1f}', 'CPU'])
        print(f'    FPS: {fps:.1f}  |  {ms:.1f} ms/img  |  {size:.1f} MB')
    except Exception as e:
        print(f'    SKIP: {e}')
        table.append(['OpenVINO INT8', 'N/A', 'N/A', 'N/A', 'N/A'])

    # ── Final table ─────────────────────────────────────────────────────────────
    headers = ['Model', 'FPS ↑', 'ms/img ↓', 'Size MB ↓', 'Device']
    SEP = '=' * 70
    print(f'\n\n{SEP}')
    print('  AEVO BENCHMARK — PCB DEFECT DETECTION')
    print(f'  {N_IMAGES} images @ {IMGSZ}×{IMGSZ}')
    print(SEP)
    print(tabulate(table, headers=headers, tablefmt='rounded_outline',
                   numalign='right', stralign='left'))
    print(SEP)
    print('  Screenshot this table — it is your #1 portfolio artifact.')
    print(SEP)

    RESULTS_DIR.mkdir(exist_ok=True)
    table_txt = tabulate(table, headers=headers, tablefmt='rounded_outline',
                         numalign='right', stralign='left')
    out = RESULTS_DIR / 'benchmark_results.txt'
    out.write_text(f'AEVO Benchmark — PCB Defect Detection\n{N_IMAGES} images @ {IMGSZ}×{IMGSZ}\n\n{table_txt}\n')
    print(f'\nSaved → {out}')


if __name__ == '__main__':
    run()
