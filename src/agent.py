"""
AEVO — Modules 2 + 3: YOLOv8 Training + Claude API Agent Loop
PCB Defect Detection | Agentic hyperparameter optimization every 5 epochs.
"""
import os
import json
import re
import time
import torch
from pathlib import Path
from ultralytics import YOLO
import anthropic

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
YAML_PATH   = str(ROOT / 'pcb_dataset' / 'dataset.yaml')
LOG_FILE    = ROOT / 'training_logs.txt'
AGENT_LOG   = ROOT / 'agent_decisions.txt'
SAVE_DIR    = str(ROOT / 'runs')
MODEL_NAME  = 'yolov8n.pt'

EPOCHS       = 100
IMGSZ        = 640
BATCH        = 8
WORKERS      = 2
PATIENCE     = 30
CLAUDE_EVERY = 5   # call Claude API every N epochs

VRAM_TOTAL = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0

_client    = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
_log_lines = []


def call_claude_agent(log_lines: list, current_lr: float, epoch: int) -> dict:
    """Send last-5-epoch metrics to Claude; receive a JSON action decision."""
    logs_text = '\n'.join(log_lines[-5:])
    prompt = f"""You are a YOLOv8 training optimizer agent for a PCB defect detection model.

Training logs (last 5 epochs):
{logs_text}

Current learning rate: {current_lr:.7f}
Current epoch: {epoch} / {EPOCHS}
GPU VRAM: {VRAM_TOTAL:.2f} GB

Analyze the trend and return EXACTLY one JSON object — no other text:
{{"action": "<action>", "value": <number_or_null>, "reason": "<one sentence>"}}

Valid actions:
- "reduce_lr"       → multiply LR by value (e.g. 0.5). Use when val loss diverges or mAP plateaus.
- "increase_lr"     → multiply LR by value (e.g. 1.5). Only if loss drops fast and is stable.
- "adjust_momentum" → set SGD momentum to value (0.85–0.98).
- "continue"        → no change, value=null.

Rules: be conservative; suggest change only if 3+ epochs show a clear bad trend;
never reduce LR below 1e-6; prefer "continue" if training is healthy."""

    try:
        msg = _client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=150,
            messages=[{'role': 'user', 'content': prompt}]
        )
        text = msg.content[0].text.strip()
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception as e:
        print(f'  [Claude] API error: {e}')
    return {'action': 'continue', 'value': None, 'reason': 'fallback'}


def apply_action(trainer, decision: dict):
    action = decision.get('action', 'continue')
    value  = decision.get('value')
    if action == 'continue' or value is None:
        return
    try:
        if action in ('reduce_lr', 'increase_lr'):
            for pg in trainer.optimizer.param_groups:
                old = pg['lr']
                pg['lr'] = max(old * float(value), 1e-7)
                if 'initial_lr' in pg:
                    pg['initial_lr'] = pg['lr']
            if hasattr(trainer, 'scheduler') and hasattr(trainer.scheduler, 'base_lrs'):
                trainer.scheduler.base_lrs = [
                    max(lr * float(value), 1e-7) for lr in trainer.scheduler.base_lrs
                ]
            new_lr = trainer.optimizer.param_groups[0]['lr']
            print(f'  [Agent] LR: {old:.7f} → {new_lr:.7f}')
        elif action == 'adjust_momentum':
            for pg in trainer.optimizer.param_groups:
                if 'momentum' in pg:
                    pg['momentum'] = float(value)
                if 'betas' in pg:
                    pg['betas'] = (float(value), pg['betas'][1])
            print(f'  [Agent] Momentum → {value}')
    except Exception as e:
        print(f'  [Agent] Apply error: {e}')


def on_fit_epoch_end(trainer):
    import csv as _csv
    epoch   = trainer.epoch + 1
    metrics = trainer.metrics
    vram    = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    box_loss = cls_loss = dfl_loss = 0.0
    try:
        csv_path = Path(trainer.csv)
        if csv_path.exists():
            with open(csv_path) as f:
                rows = list(_csv.DictReader(f))
            if rows:
                r = rows[-1]
                box_loss = float(r.get('  train/box_loss', r.get('train/box_loss', 0)))
                cls_loss = float(r.get('  train/cls_loss', r.get('train/cls_loss', 0)))
                dfl_loss = float(r.get('  train/dfl_loss', r.get('train/dfl_loss', 0)))
    except Exception:
        pass

    entry = {
        'epoch':     epoch,
        'box_loss':  round(box_loss, 5),
        'cls_loss':  round(cls_loss, 5),
        'dfl_loss':  round(dfl_loss, 5),
        'val_box':   round(float(metrics.get('val/box_loss',            0)), 5),
        'val_cls':   round(float(metrics.get('val/cls_loss',            0)), 5),
        'precision': round(float(metrics.get('metrics/precision(B)',    0)), 4),
        'recall':    round(float(metrics.get('metrics/recall(B)',       0)), 4),
        'mAP50':     round(float(metrics.get('metrics/mAP50(B)',        0)), 4),
        'mAP50_95':  round(float(metrics.get('metrics/mAP50-95(B)',     0)), 4),
        'vram_gb':   round(vram, 2),
    }

    line = ' '.join(f'{k}={v}' for k, v in entry.items())
    _log_lines.append(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

    vram_pct = (vram / VRAM_TOTAL * 100) if VRAM_TOTAL else 0
    vram_warn = ' ⚠ HIGH' if vram_pct > 85 else ''
    print(f'\n  [E{epoch:03d}] mAP50={entry["mAP50"]:.4f}  mAP50-95={entry["mAP50_95"]:.4f}'
          f'  val_box={entry["val_box"]:.4f}  VRAM={vram:.2f}GB{vram_warn}')

    if epoch % CLAUDE_EVERY == 0:
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f'\n  [Agent] Epoch {epoch} — querying Claude API ...')
        t0 = time.time()
        decision = call_claude_agent(_log_lines, current_lr, epoch)
        print(f'  [Agent] Response ({time.time()-t0:.1f}s): {json.dumps(decision)}')
        with open(AGENT_LOG, 'a') as f:
            f.write(f'Epoch {epoch:3d} | LR={current_lr:.7f} | {json.dumps(decision)}\n')
        apply_action(trainer, decision)


def train():
    LOG_FILE.write_text('')
    AGENT_LOG.write_text('AEVO Agent Decisions Log\n' + '=' * 50 + '\n')

    print('=' * 64)
    print('  AEVO MODULES 2+3 — YOLOV8 TRAINING + CLAUDE AGENT LOOP')
    print('=' * 64)
    if torch.cuda.is_available():
        print(f'  GPU : {torch.cuda.get_device_name(0)}  ({VRAM_TOTAL:.2f} GB)')

    model = YOLO(MODEL_NAME)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

    results = model.train(
        data=YAML_PATH, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        device=0 if torch.cuda.is_available() else 'cpu',
        workers=WORKERS, patience=PATIENCE, save=True,
        project=SAVE_DIR, name='pcb_yolov8n', amp=True,
        cache=False, verbose=False, exist_ok=True,
    )

    best_map = max(
        (float(l.split('mAP50=')[1].split()[0]) for l in _log_lines if 'mAP50=' in l),
        default=0
    )
    print(f'\n  Best mAP50: {best_map:.4f} ({best_map*100:.1f}%)')
    status = 'PASS — ready for Module 4' if best_map >= 0.70 else f'mAP50 < 70% — check data/training'
    print(f'  {status}')
    return results


if __name__ == '__main__':
    train()
