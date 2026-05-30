"""AEVO Module 1 — Data Hub (offline runner)"""
import zipfile, shutil, yaml, cv2, time, random
import numpy as np
from pathlib import Path

ROOT         = Path(__file__).parent
ZIP_PATH     = ROOT / 'PCB Defect.v2i.yolov8.zip'
EXTRACT_TO   = ROOT / 'pcb_dataset'
N_AUG_COPIES = 1   # 1 copy; YOLOv8 augments further during training
IMG_EXTS     = {'.jpg', '.jpeg', '.png', '.bmp'}

# ── 1. Extract ────────────────────────────────────────────────────────────────
print('=' * 62)
print('  AEVO MODULE 1 — DATA HUB')
print('=' * 62)

if EXTRACT_TO.exists():
    shutil.rmtree(EXTRACT_TO)
EXTRACT_TO.mkdir(parents=True, exist_ok=True)

print(f'\n[1] Extracting {ZIP_PATH.name} ({ZIP_PATH.stat().st_size/1e6:.0f} MB) ...')
with zipfile.ZipFile(ZIP_PATH, 'r') as z:
    z.extractall(EXTRACT_TO)
print('    Done.')

# ── 2. Discover splits ────────────────────────────────────────────────────────
def discover(root):
    root = Path(root)
    ds   = {'root': root, 'splits': {}, 'classes': []}
    for canonical, aliases in [('train',['train']),('val',['val','valid']),('test',['test'])]:
        for alias in aliases:
            img_dir = root / alias / 'images'
            lbl_dir = root / alias / 'labels'
            if img_dir.exists():
                imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
                lbls = sorted(lbl_dir.glob('*.txt')) if lbl_dir.exists() else []
                ds['splits'][canonical] = {'img_dir': img_dir, 'lbl_dir': lbl_dir,
                                           'images': imgs, 'labels': list(lbls)}
                break
    for y in root.glob('*.yaml'):
        try:
            data = yaml.safe_load(y.read_text())
            if 'names' in data:
                ds['classes'] = data['names']; break
        except Exception:
            pass
    return ds

print('\n[2] Discovering dataset splits ...')
structure = discover(EXTRACT_TO)
print(f"    Classes ({len(structure['classes'])}): {structure['classes']}")
for split, info in structure['splits'].items():
    paired = sum(1 for i in info['images'] if (info['lbl_dir']/(i.stem+'.txt')).exists())
    print(f"    [{split:5s}]  images: {len(info['images']):5d}  labels: {len(info['labels']):5d}  paired: {paired:5d}")

# ── 3. OpenCV Augmentation ────────────────────────────────────────────────────
class AugEngine:
    def __init__(self):
        self.p = dict(flip=0.5, rotate=0.4, bright=0.5, blur=0.25, noise=0.25)
        self.max_rot = 12

    def augment(self, img_path, lbl_path):
        img = cv2.imread(str(img_path))
        if img is None: return []
        boxes = []
        if Path(lbl_path).exists():
            for line in Path(lbl_path).read_text().strip().splitlines():
                p = line.strip().split()
                if len(p) == 5:
                    boxes.append([int(p[0])] + list(map(float, p[1:])))
        results = []
        if random.random() < self.p['flip']:
            img2 = cv2.flip(img.copy(), 1)
            b2 = [[c, 1.0-cx, cy, w, h] for c,cx,cy,w,h in boxes]
            results.append((img2, b2, '_hflip'))
        if random.random() < self.p['rotate']:
            angle = random.uniform(-self.max_rot, self.max_rot)
            H, W  = img.shape[:2]
            M     = cv2.getRotationMatrix2D((W/2,H/2), angle, 1.0)
            img2  = cv2.warpAffine(img.copy(), M, (W,H), borderMode=cv2.BORDER_REFLECT_101)
            b2 = []
            for c,cx,cy,w,h in boxes:
                p2 = M @ np.array([cx*W, cy*H, 1.0])
                b2.append([c, float(np.clip(p2[0]/W,0,1)), float(np.clip(p2[1]/H,0,1)), w, h])
            if random.random() < 0.4:
                img2 = np.clip(img2.astype(np.float32)*random.uniform(0.6,1.4),0,255).astype(np.uint8)
            results.append((img2, b2, f'_rot{int(angle):+03d}'))
        if random.random() < self.p['bright']:
            img2 = np.clip(img.copy().astype(np.float32)*random.uniform(0.55,1.45),0,255).astype(np.uint8)
            results.append((img2, boxes, '_bright'))
        if random.random() < self.p['blur']:
            k = random.choice([3,5])
            results.append((cv2.GaussianBlur(img.copy(),(k,k),0), boxes, '_blur'))
        if random.random() < self.p['noise']:
            img2 = np.clip(img.copy().astype(np.float32)+np.random.randn(*img.shape)*12,0,255).astype(np.uint8)
            results.append((img2, boxes, '_noise'))
        return results

print(f'\n[3] Running OpenCV augmentation (×{N_AUG_COPIES} per image) ...')
random.seed(42); np.random.seed(42)
engine = AugEngine()
train  = structure['splits']['train']
orig_images = train['images'].copy()
train['lbl_dir'].mkdir(parents=True, exist_ok=True)
added, t0 = 0, time.time()

for idx, img_path in enumerate(orig_images):
    lbl_path = train['lbl_dir'] / (img_path.stem + '.txt')
    for _ in range(N_AUG_COPIES):
        for aug_img, aug_boxes, suffix in engine.augment(img_path, lbl_path):
            out_img = train['img_dir'] / (img_path.stem + suffix + img_path.suffix)
            out_lbl = train['lbl_dir'] / (img_path.stem + suffix + '.txt')
            cv2.imwrite(str(out_img), aug_img)
            out_lbl.write_text('\n'.join(
                f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in aug_boxes))
            added += 1
    if (idx+1) % 1000 == 0:
        print(f'    [{idx+1}/{len(orig_images)}]  +{added} aug images  ({time.time()-t0:.0f}s)')

print(f'    Done. +{added} augmented images in {time.time()-t0:.1f}s')

# ── 4. Write dataset.yaml ─────────────────────────────────────────────────────
print('\n[4] Writing dataset.yaml ...')
splits = structure['splits']
yaml_data = {
    'path':  str(EXTRACT_TO),
    'train': str(splits['train']['img_dir']),
    'val':   str(splits['val']['img_dir']),
    'nc':    len(structure['classes']),
    'names': structure['classes']
}
if 'test' in splits:
    yaml_data['test'] = str(splits['test']['img_dir'])

YAML_PATH = EXTRACT_TO / 'dataset.yaml'
with open(YAML_PATH, 'w') as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
print(f'    Written: {YAML_PATH}')

# ── 5. Final summary ──────────────────────────────────────────────────────────
structure_final = discover(EXTRACT_TO)
SEP = '=' * 62
print(f'\n{SEP}')
print('  AEVO MODULE 1 COMPLETE')
print(SEP)
print(f"  Classes : {structure_final['classes']}")
print(f"  YAML    : {YAML_PATH}")
print()
print(f"  {'Split':<8} {'Images':>8} {'Labels':>8}")
print(f"  {'-'*8} {'-'*8} {'-'*8}")
for split, info in structure_final['splits'].items():
    note = '  <- +augmented' if split == 'train' else ''
    print(f"  {split:<8} {len(info['images']):>8} {len(info['labels']):>8}{note}")
print(SEP)
print(f'\nYAML_PATH = "{YAML_PATH}"')
