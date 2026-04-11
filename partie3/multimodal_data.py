"""
Chargement aligné Sentinel-2 / Sentinel-1 / covariables pour la Partie 3.

- Jointure sur la colonne `id` lorsqu'elle est présente dans les CSV S1 et covariables.
- Réordonnancement des bandes S1 : par mois (ratio, VH, VV) pour un tenseur (N, 12, 3).
- Filtrage NaN/Inf dans les entrées numériques (évite instabilités en entraînement).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def s1_monthly_feature_columns(n_months: int = 12) -> list[str]:
    """Ordre attendu par pas de temps : ratio_m, VH_m, VV_m (dB dans le CSV GEE)."""
    cols: list[str] = []
    for m in range(1, n_months + 1):
        mm = f"{m:02d}"
        cols.extend([f"ratio_{mm}", f"vh_{mm}", f"vv_{mm}"])
    return cols


def s1_dataframe_to_tensor(df: pd.DataFrame, *, n_months: int = 12) -> np.ndarray:
    """Retourne un array float32 de forme (N, n_months, 3)."""
    cols = s1_monthly_feature_columns(n_months)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colonnes S1 manquantes ({len(missing)}): {missing[:6]}{'...' if len(missing) > 6 else ''}"
        )
    arr = df[cols].to_numpy(dtype=np.float64)
    return arr.reshape(len(df), n_months, 3).astype(np.float32)


def _sanitize_array(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _map_cdl_to_paper_classes(y: np.ndarray) -> np.ndarray:
    """0: Corn (1), 1: Cotton (2), 2: Soybean (5), 3: Rice (3), 4: Others."""
    y = np.asarray(y).astype(np.int32, copy=False)
    out = np.full_like(y, 4, dtype=np.int32)
    out[y == 1] = 0
    out[y == 2] = 1
    out[y == 5] = 2
    out[y == 3] = 3
    return out


def load_aligned_multimodal(
    project_root: Path | str,
    *,
    n_months_s1: int = 12,
    id_col: str = "id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Charge X_s2, X_s1, X_static, y et retourne aussi le DataFrame fusionné (métadonnées / debug).

    Stratégie d'alignement :
    - Si `id_col` est présent dans s1_monthly_timeseries.csv et covariables_environnementales.csv :
      inner join, tri par id, sous-échantillonnage de X_s2 et y par index = id (entier).
    - Sinon : repli sur les n premières lignes (comportement historique), avec message explicite.
    """
    project_root = Path(project_root)
    s2_path = project_root / "dataset" / "X_train.npy"
    y_path = project_root / "dataset" / "y_train.npy"
    s1_path = project_root / "partie3" / "s1_monthly_timeseries.csv"
    cov_path = project_root / "dataset" / "covariables_environnementales.csv"

    X_s2 = np.load(s2_path)
    y = np.load(y_path)
    y = _map_cdl_to_paper_classes(y)

    if not s1_path.exists():
        raise FileNotFoundError(f"Fichier Sentinel-1 introuvable : {s1_path}")

    df_s1 = pd.read_csv(s1_path)
    df_cov = pd.read_csv(cov_path)

    s1_cols = s1_monthly_feature_columns(n_months_s1)
    can_merge = id_col in df_s1.columns and id_col in df_cov.columns

    if can_merge:
        df_s1[id_col] = pd.to_numeric(df_s1[id_col], errors="coerce").astype("Int64")
        df_cov[id_col] = pd.to_numeric(df_cov[id_col], errors="coerce").astype("Int64")
        df_s1 = df_s1.dropna(subset=[id_col]).astype({id_col: int})
        df_cov = df_cov.dropna(subset=[id_col]).astype({id_col: int})

        merged = df_s1.merge(df_cov, on=id_col, how="inner", suffixes=("", "_covdup"))
        merged = merged.loc[:, ~merged.columns.str.endswith("_covdup")]
        merged = merged.sort_values(id_col).reset_index(drop=True)

        ids = merged[id_col].to_numpy(dtype=np.int64)
        if ids.size == 0:
            raise RuntimeError("Jointure S1 / covariables vide : vérifiez les colonnes id.")

        if np.any(ids < 0) or np.any(ids >= len(X_s2)):
            bad = ids[(ids < 0) | (ids >= len(X_s2))]
            raise IndexError(
                f"Des id après jointure dépassent la taille de X_train.npy (n={len(X_s2)}). "
                f"Exemples invalides : {bad[:8]}"
            )

        X_s2 = X_s2[ids].astype(np.float32, copy=False)
        y = y[ids]

        X_s1 = s1_dataframe_to_tensor(merged, n_months=n_months_s1)
        static_cols = [c for c in df_cov.columns if c not in (id_col, "lon", "lat")]
        missing_static = [c for c in static_cols if c not in merged.columns]
        if missing_static:
            raise ValueError(f"Colonnes covariables manquantes après fusion : {missing_static[:10]}")
        X_static = merged[static_cols].to_numpy(dtype=np.float32)
        meta = merged[[id_col]].copy()
    else:
        print(
            "[multimodal_data] Colonne 'id' absente dans S1 ou covariables : "
            "repli sur troncature par longueur (risque de désalignement). "
            "Ajoutez 'id' (0..N-1) à points_agricoles.csv et réexportez S1 / covariables."
        )
        n = min(X_s2.shape[0], len(df_s1), len(df_cov))
        X_s2 = X_s2[:n].astype(np.float32, copy=False)
        y = y[:n]
        X_s1 = s1_dataframe_to_tensor(df_s1.iloc[:n], n_months=n_months_s1)
        static_cols = [c for c in df_cov.columns if c not in (id_col, "lon", "lat")]
        X_static = df_cov.iloc[:n][static_cols].to_numpy(dtype=np.float32)
        meta = pd.DataFrame({id_col: np.arange(n, dtype=np.int64)})

    X_s2 = _sanitize_array(X_s2)
    X_s1 = _sanitize_array(X_s1)
    X_static = _sanitize_array(X_static)

    return X_s2, X_s1, X_static, y, meta


def stratified_train_val_test_indices(
    y: np.ndarray,
    *,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Indices train / validation / test stratifiés sur y (5 classes).
    `val_size` est la fraction du jeu total pour la validation (comme sklearn train_test_split).
    """
    from sklearn.model_selection import train_test_split

    y = np.asarray(y).reshape(-1)
    idx = np.arange(len(y), dtype=np.int64)
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=random_state
    )
    y_tv = y[idx_trainval]
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=rel_val, stratify=y_tv, random_state=random_state
    )
    return idx_train.astype(np.int64), idx_val.astype(np.int64), idx_test.astype(np.int64)


def standardize_modalities_fit_transform(
    X_s2: np.ndarray,
    X_s1: np.ndarray,
    X_static: np.ndarray,
    idx_train: np.ndarray,
):
    """
    StandardScaler ajusté **uniquement** sur idx_train, puis appliqué à tout le jeu.
    Retourne (X_s2_n, X_s1_n, X_static_n), scalers (dict sklearn).
    """
    from sklearn.preprocessing import StandardScaler

    sc_s2 = StandardScaler()
    sc_s1 = StandardScaler()
    sc_st = StandardScaler()

    tr_s2 = X_s2[idx_train].reshape(len(idx_train), -1)
    tr_s1 = X_s1[idx_train].reshape(len(idx_train), -1)
    tr_st = X_static[idx_train]
    sc_s2.fit(tr_s2)
    sc_s1.fit(tr_s1)
    sc_st.fit(tr_st)

    X_s2_n = sc_s2.transform(X_s2.reshape(len(X_s2), -1)).reshape(X_s2.shape).astype(np.float32)
    X_s1_n = sc_s1.transform(X_s1.reshape(len(X_s1), -1)).reshape(X_s1.shape).astype(np.float32)
    X_st_n = sc_st.transform(X_static).astype(np.float32)
    scalers = {"s2": sc_s2, "s1": sc_s1, "static": sc_st}
    return (X_s2_n, X_s1_n, X_st_n), scalers


__all__ = [
    "load_aligned_multimodal",
    "s1_dataframe_to_tensor",
    "s1_monthly_feature_columns",
    "stratified_train_val_test_indices",
    "standardize_modalities_fit_transform",
]

