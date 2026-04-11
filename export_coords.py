import numpy as np
import rasterio
from rasterio.warp import transform
import pandas as pd
from pathlib import Path

# Constantes identiques à votre Partie 1
DATASET_DIR = Path("dataset")
CDL_RASTER_PATH = DATASET_DIR / "cdl_2025_clip.tif"
S2_SAFE_ROOT = DATASET_DIR / "s2_l2a"
MAX_SAMPLES = 20_000
SEED = 42
CDL_IGNORE_VALUES = (0,)

# 1. Reproduire vos fonctions de mapping de la Partie 1
def load_cdl_labels(path: Path):
    with rasterio.open(path) as src:
        y = src.read(1).astype(np.int32)
        profile = src.profile
    return y, profile

def remap_labels(raw: np.ndarray, ignore):
    ignore_set = set(int(x) for x in ignore)
    codes = sorted(int(c) for c in np.unique(raw) if c not in ignore_set and c >= 0)
    code_to_idx = {c: i for i, c in enumerate(codes)}
    out = np.full_like(raw, -1, dtype=np.int32)
    for c, i in code_to_idx.items():
        out[raw == c] = i
    return out

def select_pixel_indices_stratified(labels_hw: np.ndarray, max_samples: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    h, w = labels_hw.shape
    flat_y = labels_hw.reshape(-1)
    valid = flat_y >= 0
    classes = np.unique(flat_y[valid])
    per_class = max(1, max_samples // int(classes.size))
    parts = []
    for cl in classes:
        pos = np.flatnonzero(valid & (flat_y == cl))
        if pos.size > 0:
            n_take = min(per_class, pos.size)
            parts.append(rng.choice(pos, size=n_take, replace=False))
    idx = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    if idx.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    rows = idx // w
    cols = idx % w
    return rows.astype(np.int32), cols.astype(np.int32)

def export_coordinates():
    print("Chargement du masque CDL...")
    y_raw, profile = load_cdl_labels(CDL_RASTER_PATH)
    y_map = remap_labels(y_raw, CDL_IGNORE_VALUES)
    
    print("Re-sélection des mêmes pixels que dans la Partie 1 (avec la seed 42)...")
    rows, cols = select_pixel_indices_stratified(y_map, max_samples=MAX_SAMPLES, seed=SEED)
    
    # Ouvrir le fichier pour récupérer le CRS (système de projection) et la matrice de transformation
    with rasterio.open(CDL_RASTER_PATH) as src:
        # Transformation pixels -> Coordonnées spatiales d'origine (Sûrement un UTM)
        xs, ys = rasterio.transform.xy(src.transform, rows, cols, offset="center")
        
        # Conversion du système d'origine vers Lat/Lon (EPSG:4326 pour Google Earth Engine)
        print(f"Conversion des coordonnées S2/CDL ({src.crs}) vers la norme lat/lon...")
        lons, lats = transform(src.crs, 'EPSG:4326', xs, ys)
    
    print(f"Sauvegarde des {len(lons)} coordonnées...")
    df = pd.DataFrame({
        "id": np.arange(len(lons), dtype=np.int64),
        "lon": lons,
        "lat": lats,
    })
    
    csv_path = DATASET_DIR / "points_agricoles.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Terminé ! Vos coordonnées sont sauvées dans : {csv_path}")

if __name__ == "__main__":
    export_coordinates()
