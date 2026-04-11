import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from partie3.multimodal_data import load_aligned_multimodal

    X_s2, X_s1, X_static, y, meta = load_aligned_multimodal(PROJECT_ROOT)
    print(f"X_s2 shape: {X_s2.shape}")
    print(f"X_s1 shape: {X_s1.shape}")
    print(f"X_static shape: {X_static.shape}")
    print(f"y shape: {y.shape}")
    print(f"meta head:\n{meta.head()}")
except Exception as e:
    print(f"load_aligned_multimodal: {e}")
