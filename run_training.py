"""
AEVO — Modules 2 + 3: YOLOv8 Training + Claude API Agent Loop
PCB Defect Detection | RTX 3050 Ti (4GB)
"""
import os, json, re, time, torch
from pathlib import Path
from ultralytics import YOLO
import anthropic

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
YAML_PATH   = str(ROOT / 'pcb_dataset/dataset.yaml')
LOG_FILE    = ROOT / 'training_logs.txt'
AGENT_LOG   = ROOT / 'agent_decisions.txt'
SAVE_DIR    = str(ROOT / 'runs')
MODEL_NAME  = 'yolov8n.pt'

EPOCHS      = 100
IMGSZ       = 640
BATCH       = 8          # safe for 4GB VRAM
WORKERS     = 2
PATIENCE    = 30         # early-stop if mAP doesn't improve for 30 epochs
CLAUDE_EVERY = 5         # call Claude API every N epochs

VRAM_TOTAL  = torch.cuda.get_device_properties(0).total_memory / 1e9

# ── Init logs ─────────────────────────────────────────────────────────────────
LOG_FILE.write_text('')
AGENT_LOG.write_text('AEVO Agent Decisions Log\n' + '='*50 + '\n')

print('=' * 64)
print('  AEVO MODULES 2+3 — YOLOV8 TRAINING + CLAUDE AGENT LOOP')
print('=' * 64)
print(f'  GPU    : {torch.cuda.get_device_name(0)}  ({VRAM_TOTAL:.2f} GB)')
print(f'  Model  : {MODEL_NAME}  |  imgsz={IMGSZ}  batch={BATCH}')
print(f'  Epochs : {EPOCHS}  |  patience={PATIENCE}  |  Claude every {CLAUDE_EVERY} epochs')
print(f'  Data   : {YAML_PATH}')
print('=' * 64 + '\n')

# ── Claude API agent ──────────────────────────────────────────────────────────
_client = anthropic.Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])

def call_claude_agent(log_lines, current_lr, epoch):
    logs_text = '\n'.join(log_lines[-5:])
    prompt = f"""You are a YOLOv8 training optimizer agent for a PCB defect detection model.

Training logs (last 5 epochs):
{logs_text}

Current learning rate: {current_lr:.7f}
Current epoch: {epoch} / {EPOCHS}
GPU VRAM: {VRAM_TOTAL:.2f} GB

Analyze the trend and return EXACTLY one JSON object — no other text:
{{"action": "<action>", "value": <number_or_null>, "reason": "<one sentence>"}}

Valid actions (pick one):
- "reduce_lr"      → multiply LR by value (e.g. 0.5 halves it). Use when val_box_loss diverges or mAP plateaus.
- "increase_lr"    → multiply LR by value (e.g. 1.5). Only if loss is still dropping fast and stable.
- "adjust_momentum"→ set SGD momentum to value (0.85–0.98).
- "continue"       → no change needed, value=null.

Rules:
- Be conservative. Suggest a change only if the last 3+ epochs show a clear bad trend.
- Never reduce LR below 1e-6. Never increase above initial.
- Prefer "continue" if training is healthy."""

    try:
        msg = _client.messages.create(
            model='claude-sonnet-4-6',
            max_tokens=150,
            messages=[{'role': 'user', 'content': prompt}]
        )
        text = msg.content[0].text.strip()
        m = re.search(r'\{.*?\}', text, re.DOTALL)
        if m:
            result = json.loads(m.group())
            return result
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

# ── Shared state between callbacks ───────────────────────────────────────────
_log_lines = []

# ── Callbacks ─────────────────────────────────────────────────────────────────
def on_fit_epoch_end(trainer):
    import csv as _csv
    epoch   = trainer.epoch + 1
    metrics = trainer.metrics
    vram    = torch.cuda.max_memory_allocated() / 1e9  # peak VRAM this epoch

    # Read train losses from CSV (most reliable source; trainer.metrics omits them)
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
        'epoch':       epoch,
        'box_loss':    round(box_loss, 5),
        'cls_loss':    round(cls_loss, 5),
        'dfl_loss':    round(dfl_loss, 5),
        'val_box':     round(float(metrics.get('val/box_loss',     0)), 5),
        'val_cls':     round(float(metrics.get('val/cls_loss',     0)), 5),
        'precision':   round(float(metrics.get('metrics/precision(B)', 0)), 4),
        'recall':      round(float(metrics.get('metrics/recall(B)',    0)), 4),
        'mAP50':       round(float(metrics.get('metrics/mAP50(B)',     0)), 4),
        'mAP50_95':    round(float(metrics.get('metrics/mAP50-95(B)',  0)), 4),
        'vram_gb':     round(vram, 2),
    }

    line = ' '.join(f'{k}={v}' for k, v in entry.items())
    _log_lines.append(line)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')

    # Terminal summary
    vram_pct = vram / VRAM_TOTAL * 100
    vram_warn = ' ⚠ HIGH' if vram_pct > 85 else ''
    print(f'\n  [E{epoch:03d}] mAP50={entry["mAP50"]:.4f}  mAP50-95={entry["mAP50_95"]:.4f}'
          f'  box_loss={entry["box_loss"]:.4f}  val_box={entry["val_box"]:.4f}'
          f'  VRAM={vram:.2f}/{VRAM_TOTAL:.2f}GB{vram_warn}')

    # ── Module 3: Claude agent every CLAUDE_EVERY epochs ────────────────────
    if epoch % CLAUDE_EVERY == 0:
        current_lr = trainer.optimizer.param_groups[0]['lr']
        print(f'\n  [Agent] Epoch {epoch} — querying Claude API ...')
        t0 = time.time()
        decision = call_claude_agent(_log_lines, current_lr, epoch)
        elapsed  = time.time() - t0
        print(f'  [Agent] Response ({elapsed:.1f}s): {json.dumps(decision)}')

        # Log agent decision
        with open(AGENT_LOG, 'a') as f:
            f.write(f'Epoch {epoch:3d} | LR={current_lr:.7f} | '
                    f'{json.dumps(decision)}\n')

        apply_action(trainer, decision)

    # OOM guard: if VRAM > 90%, warn
    if vram_pct > 90:
        print(f'  [VRAM WARNING] {vram:.2f}GB used ({vram_pct:.0f}%) — '
              f'consider reducing batch size if training crashes')


# ── Training ──────────────────────────────────────────────────────────────────
model = YOLO(MODEL_NAME)
model.add_callback('on_fit_epoch_end', on_fit_epoch_end)

print('Starting training...\n')
results = model.train(
    data     = YAML_PATH,
    epochs   = EPOCHS,
    imgsz    = IMGSZ,
    batch    = BATCH,
    device   = 0,
    workers  = WORKERS,
    patience = PATIENCE,
    save     = True,
    project  = SAVE_DIR,
    name     = 'pcb_yolov8n',
    amp      = True,
    cache    = False,
    verbose  = False,       # suppress ultralytics spam; our callback handles output
    exist_ok = True,
)

# ── Post-training summary ─────────────────────────────────────────────────────
best_map = max((float(l.split('mAP50=')[1].split()[0])
                for l in _log_lines if 'mAP50=' in l), default=0)

SEP = '=' * 64
print(f'\n{SEP}')
print('  AEVO MODULE 2+3 COMPLETE')
print(SEP)
print(f'  Best mAP50    : {best_map:.4f}  ({best_map*100:.1f}%)')
print(f'  Training logs : {LOG_FILE}')
print(f'  Agent log     : {AGENT_LOG}')
best_pt = Path(SAVE_DIR) / 'pcb_yolov8n' / 'weights' / 'best.pt'
last_pt = Path(SAVE_DIR) / 'pcb_yolov8n' / 'weights' / 'last.pt'
print(f'  Best weights  : {best_pt}')
print(f'  Last weights  : {last_pt}')

if best_map >= 0.70:
    print(f'\n  PASS  mAP50 {best_map*100:.1f}% >= 70% — ready for Module 4')
else:
    print(f'\n  mAP50 {best_map*100:.1f}% < 70% — continue training or check data')
print(SEP)
