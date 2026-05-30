"""
AEVO — Module 1: Data Hub
Mount dataset, validate folder structure, run OpenCV augmentation, write dataset.yaml.
"""
import cv2
import yaml
import shutil
import random
import numpy as np
from pathlib import Path

DATASET_ROOT = Path(__file__).parent.parent / 'pcb_dataset'
TRAIN_IMAGES = DATASET_ROOT / 'train' / 'images'
TRAIN_LABELS = DATASET_ROOT / 'train' / 'labels'
VAL_IMAGES   = DATASET_ROOT / 'valid' / 'images'
VAL_LABELS   = DATASET_ROOT / 'valid' / 'labels'
TEST_IMAGES  = DATASET_ROOT / 'test'  / 'images'

CLASSES = ['Short_circuit', 'damged', 'lack_of_part', 'miss_welding', 'redundant', 'slug', 'spillover']


def validate_structure():
    required = [TRAIN_IMAGES, TRAIN_LABELS, VAL_IMAGES, VAL_LABELS, TEST_IMAGES]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f'Missing dataset folders:\n' + '\n'.join(missing))

    n_train = len(list(TRAIN_IMAGES.glob('*.jpg')))
    n_val   = len(list(VAL_IMAGES.glob('*.jpg')))
    n_test  = len(list(TEST_IMAGES.glob('*.jpg')))
    print(f'Dataset: {n_train} train | {n_val} val | {n_test} test | {len(CLASSES)} classes')
    return n_train, n_val, n_test


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply random flip, brightness jitter, and Gaussian blur."""
    if random.random() > 0.5:
        img = cv2.flip(img, 1)
    factor = random.uniform(0.7, 1.3)
    img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    if random.random() > 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img


def write_dataset_yaml():
    cfg = {
        'path': str(DATASET_ROOT),
        'train': str(TRAIN_IMAGES),
        'val':   str(VAL_IMAGES),
        'test':  str(TEST_IMAGES),
        'nc':    len(CLASSES),
        'names': CLASSES,
    }
    out = DATASET_ROOT / 'dataset.yaml'
    with open(out, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f'dataset.yaml written → {out}')
    return out


if __name__ == '__main__':
    print('=== AEVO Module 1 — Data Hub ===')
    validate_structure()
    write_dataset_yaml()
    print('Module 1 complete.')
