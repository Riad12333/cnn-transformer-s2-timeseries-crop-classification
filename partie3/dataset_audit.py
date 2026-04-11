"""Audit des volumes et des classes pour la Partie 3 (rapport / notebook)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def print_multimodal_audit(
    project_root: Path | str,
    X_s2: np.ndarray,
    X_s1: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    *,
    class_names: tuple[str, ...] = ("Maïs", "Coton", "Soja", "Riz", "Autres"),
) -> None:
    project_root = Path(project_root)
    raw_path = project_root / "dataset" / "X_train.npy"
    if raw_path.exists():
        n_raw = int(np.load(raw_path, mmap_mode="r").shape[0])
        n_now = int(X_s2.shape[0])
        print("--- Audit multimodal ---")
        print(f"X_train.npy (brut) : {n_raw} lignes")
        print(f"Après fusion S1 ∩ covariables : {n_now} lignes")
        print(f"Échantillons sans paire S1/stat : {n_raw - n_now}")
    else:
        print("--- Audit multimodal (X_train.npy introuvable) ---")
        print(f"Échantillons alignés : {X_s2.shape[0]}")

    print(f"\nFormes : S2 {X_s2.shape} | S1 {X_s1.shape} | statiques {X_static.shape}")
    print(f"NaN S2: {np.isnan(X_s2).sum()} | S1: {np.isnan(X_s1).sum()} | stat: {np.isnan(X_static).sum()}")

    y = np.asarray(y).reshape(-1)
    print("\nEffectifs par classe (0–4) :")
    for c in range(len(class_names)):
        n = int(np.sum(y == c))
        pct = 100.0 * n / max(len(y), 1)
        name = class_names[c] if c < len(class_names) else str(c)
        print(f"  {c} {name}: {n} ({pct:.1f} %)")


__all__ = ["print_multimodal_audit"]
