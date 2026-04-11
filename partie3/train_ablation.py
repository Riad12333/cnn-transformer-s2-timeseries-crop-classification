"""
Entraînement Partie 3 : ablations (S2 seul, S2+stat, S2+S1, S2+S1+stat), poids de classes, optional focal loss.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from partie3.mctnet_v2_model import build_mctnet_v2
from training_utils import (
    ScalarEarlyStopping,
    sparse_categorical_crossentropy_label_smoothing,
    sparse_categorical_focal_loss,
)


def keras_class_weight_dict(
    y: np.ndarray,
    *,
    minority_boost: float = 1.3,
) -> dict[int, float]:
    """
    Poids sklearn « balanced », puis léger surcroit pour les classes sous la médiane d'effectifs
    (réduit le biais « Autres » sans exploser la loss).
    """
    y = np.asarray(y).reshape(-1)
    classes = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    out = {int(c): float(wi) for c, wi in zip(classes, w)}
    if minority_boost and abs(minority_boost - 1.0) > 1e-6:
        counts = {int(c): int(np.sum(y == c)) for c in classes}
        med = float(np.median([counts[int(c)] for c in classes]))
        for c in classes:
            ci = int(c)
            if counts[ci] < med:
                out[ci] *= float(minority_boost)
    return out


def _train_sample_weights(y_labels: np.ndarray, cw: dict[int, float]) -> np.ndarray:
    y_labels = np.asarray(y_labels).reshape(-1)
    w = np.array([cw[int(t)] for t in y_labels], dtype=np.float32)
    w = w / (float(np.mean(w)) + 1e-8)
    return w


def _slice_inputs(
    X_s2: np.ndarray,
    X_s1: np.ndarray,
    X_static: np.ndarray,
    idx: np.ndarray,
    *,
    use_s1: bool,
    use_static: bool,
) -> list[np.ndarray]:
    out = [X_s2[idx]]
    if use_s1:
        out.append(X_s1[idx])
    if use_static:
        out.append(X_static[idx])
    return out


def run_ablation_suite(
    X_s2: np.ndarray,
    X_s1: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    *,
    n_classes: int = 5,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    focal_with_sample_weights: bool = True,
    label_smoothing: float = 0.0,
    minority_boost: float = 1.3,
    early_stopping_patience: int | None = None,
    random_seed: int = 42,
) -> tuple[dict[str, Any], tf.keras.Model, Any, list[np.ndarray], np.ndarray, list[np.ndarray], np.ndarray]:
    """
    Entraîne 4 variantes avec les mêmes splits et la même normalisation (déjà appliquée aux X).

    Retourne :
    - `ablation` : dict par nom de config avec history, model, test_loss, test_acc
    - `model`, `history` : modèle **complet** S2+S1+stat (pour visualisation)
    - listes/labels val et test pour le modèle complet
    """
    tf.keras.utils.set_random_seed(random_seed)
    np.random.seed(random_seed)

    y_full = np.asarray(y).reshape(-1)
    cw = keras_class_weight_dict(y_full[idx_train], minority_boost=minority_boost)
    n_static = int(X_static.shape[1])

    configs: list[tuple[str, bool, bool]] = [
        ("S2_seul", False, False),
        ("S2_stat", False, True),
        ("S2_S1", True, False),
        ("S2_S1_stat", True, True),
    ]

    ablation: dict[str, Any] = {}
    history_full = None
    model_full = None

    for name, use_s1, use_static in configs:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_seed)

        n_st = n_static if use_static else None
        model = build_mctnet_v2(
            n_timesteps_s2=X_s2.shape[1],
            n_channels_s2=X_s2.shape[2],
            n_timesteps_s1=X_s1.shape[1],
            n_channels_s1=X_s1.shape[2],
            n_classes=n_classes,
            n_static_features=n_st,
            use_s1_branch=use_s1,
            use_static_branch=use_static,
            s2_use_se=True,
            s2_post_gru=True,
            dropout=0.2,
            cross_attn_dropout=0.1,
        )

        if use_focal_loss:
            loss_fn = sparse_categorical_focal_loss(focal_gamma)
            class_weight = None
        elif label_smoothing and label_smoothing > 0:
            loss_fn = sparse_categorical_crossentropy_label_smoothing(
                n_classes, float(label_smoothing)
            )
            class_weight = cw
        else:
            loss_fn = "sparse_categorical_crossentropy"
            class_weight = cw

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            loss=loss_fn,
            metrics=[
                "accuracy",
                tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2"),
            ],
        )

        callbacks: list = [
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5,
                patience=6,
                min_lr=1e-6,
                verbose=1,
            ),
        ]
        if early_stopping_patience is not None and early_stopping_patience > 0:
            callbacks.insert(
                0,
                ScalarEarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=int(early_stopping_patience),
                    restore_best_weights=True,
                    verbose=1,
                ),
            )

        X_tr = _slice_inputs(X_s2, X_s1, X_static, idx_train, use_s1=use_s1, use_static=use_static)
        X_va = _slice_inputs(X_s2, X_s1, X_static, idx_val, use_s1=use_s1, use_static=use_static)
        X_te = _slice_inputs(X_s2, X_s1, X_static, idx_test, use_s1=use_s1, use_static=use_static)

        fit_kw: dict[str, Any] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "class_weight": class_weight,
            "callbacks": callbacks,
            "verbose": 1,
        }
        if use_focal_loss and focal_with_sample_weights:
            fit_kw["sample_weight"] = _train_sample_weights(y_full[idx_train], cw)
        hist = model.fit(
            X_tr,
            y_full[idx_train],
            validation_data=(X_va, y_full[idx_val]),
            **fit_kw,
        )
        test_loss, test_acc, test_top2 = model.evaluate(X_te, y_full[idx_test], verbose=0)
        ablation[name] = {
            "history": hist,
            "model": model,
            "test_loss": float(test_loss),
            "test_accuracy": float(test_acc),
            "test_top2": float(test_top2),
        }
        if name == "S2_S1_stat":
            history_full = hist
            model_full = model

    if model_full is None:
        raise RuntimeError("Configuration S2_S1_stat manquante.")

    X_val_f = _slice_inputs(X_s2, X_s1, X_static, idx_val, use_s1=True, use_static=True)
    X_test_f = _slice_inputs(X_s2, X_s1, X_static, idx_test, use_s1=True, use_static=True)
    y_val_f = y_full[idx_val]
    y_test_f = y_full[idx_test]

    return ablation, model_full, history_full, X_val_f, y_val_f, X_test_f, y_test_f


def quick_lr_sweep(
    X_s2: np.ndarray,
    X_s1: np.ndarray,
    X_static: np.ndarray,
    y: np.ndarray,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    *,
    learning_rates: tuple[float, ...] = (1e-3, 5e-4),
    epochs_per_lr: int = 6,
    batch_size: int = 64,
    random_seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Petite grille sur le learning rate (modèle complet uniquement). Retourne [(lr, best_val_loss), ...].
    """
    results: list[tuple[float, float]] = []
    for lr in learning_rates:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(random_seed)
        model = build_mctnet_v2(
            n_timesteps_s2=X_s2.shape[1],
            n_channels_s2=X_s2.shape[2],
            n_timesteps_s1=X_s1.shape[1],
            n_channels_s1=X_s1.shape[2],
            n_classes=len(np.unique(y)),
            n_static_features=X_static.shape[1],
            use_s1_branch=True,
            use_static_branch=True,
            dropout=0.2,
            cross_attn_dropout=0.1,
        )
        cw = keras_class_weight_dict(y[idx_train], minority_boost=1.3)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        X_tr = _slice_inputs(X_s2, X_s1, X_static, idx_train, True, True)
        X_va = _slice_inputs(X_s2, X_s1, X_static, idx_val, True, True)
        h = model.fit(
            X_tr,
            y[idx_train],
            validation_data=(X_va, y[idx_val]),
            epochs=epochs_per_lr,
            batch_size=batch_size,
            class_weight=cw,
            verbose=0,
        )
        best = float(min(h.history["val_loss"]))
        results.append((lr, best))
        print(f"lr={lr} -> min val_loss sur {epochs_per_lr} epochs: {best:.4f}")
    return results


__all__ = [
    "keras_class_weight_dict",
    "run_ablation_suite",
    "quick_lr_sweep",
]
