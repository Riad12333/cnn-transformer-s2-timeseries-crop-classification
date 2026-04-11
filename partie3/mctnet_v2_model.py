import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from mctnet_model import CTFusion, BuildSelfAttentionMask, EnsureOneValidTimeStep, MaskedGlobalMaxPooling1D


class CrossModalAttention(layers.Layer):
    """
    Cross-attention : S2 en query, S1 en key/value (complémentarité optique / SAR).
    """

    def __init__(self, d_model: int, num_heads: int = 2, *, attn_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) doit être divisible par num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = d_model // num_heads
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.key_dim,
            dropout=float(attn_dropout),
        )
        self.ln = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=None):
        s2, s1 = inputs
        q = tf.expand_dims(s2, 1)
        k = tf.expand_dims(s1, 1)
        attn_out = self.mha(query=q, value=k, key=k, training=training)
        attn_out = tf.squeeze(attn_out, axis=1)
        return self.ln(s2 + attn_out)


def _se_block_time_series(x: tf.Tensor, d_model: int, ratio: int = 8, name_prefix: str = "s2_se") -> tf.Tensor:
    """Squeeze-and-Excitation sur la dimension canal (moyenne temporelle puis ré-échelle)."""
    gap = layers.GlobalAveragePooling1D(name=f"{name_prefix}_gap")(x)
    hidden = max(d_model // ratio, 4)
    excite = layers.Dense(hidden, activation="relu", name=f"{name_prefix}_dense1")(gap)
    excite = layers.Dense(d_model, activation="sigmoid", name=f"{name_prefix}_dense2")(excite)
    excite = layers.Reshape((1, d_model), name=f"{name_prefix}_reshape")(excite)
    return layers.Multiply(name=f"{name_prefix}_scale")([x, excite])


def build_mctnet_v2(
    n_timesteps_s2: int,
    n_channels_s2: int,
    n_timesteps_s1: int,
    n_channels_s1: int,
    n_classes: int,
    n_static_features: int | None = None,
    *,
    d_model: int = 48,
    num_heads: int = 6,
    ff_dim: int = 128,
    n_stage: int = 2,
    dropout: float = 0.2,
    cross_attn_dropout: float = 0.1,
    l2: float | None = 1e-4,
    use_s1_branch: bool = True,
    use_static_branch: bool = True,
    s2_use_se: bool = True,
    s2_post_gru: bool = True,
) -> keras.Model:
    """
    MCTNet-v2 multimodal.

    - use_s1_branch=False : entrée S1 absente, fusion = embeddings S2 (+ statiques si activé).
    - use_static_branch=False : idem n_static_features=None (pas d'entrée statique).
    - s2_use_se : bloc Squeeze-and-Excitation après le stem S2.
    - s2_post_gru : GRU bidirectionnelle légère après les blocs CTFusion S2 (phénologie).
    """
    kernel_reg = keras.regularizers.l2(l2) if l2 else None

    inp_s2 = keras.Input(shape=(n_timesteps_s2, n_channels_s2), name="s2_input")
    x2 = layers.Conv1D(d_model, 1, activation="relu", kernel_regularizer=kernel_reg, name="s2_stem")(inp_s2)
    x2 = layers.BatchNormalization(name="s2_stem_bn")(x2)
    if s2_use_se:
        x2 = _se_block_time_series(x2, d_model, name_prefix="s2_se")

    mask_s2 = layers.Lambda(lambda t: tf.reduce_any(tf.not_equal(t, 0.0), axis=-1))(inp_s2)
    mask_s2 = EnsureOneValidTimeStep()(mask_s2)
    attn_mask_s2 = BuildSelfAttentionMask()(mask_s2)

    for s in range(n_stage):
        x2 = CTFusion(
            d_model=d_model,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            num_timesteps=n_timesteps_s2,
            name_prefix=f"s2_st{s}",
        )([x2, mask_s2, attn_mask_s2])

    if s2_post_gru:
        x2 = layers.GRU(d_model, return_sequences=True, dropout=dropout, name="s2_gru")(x2)
        x2 = layers.BatchNormalization(name="s2_gru_bn")(x2)

    pool_s2 = MaskedGlobalMaxPooling1D(name="s2_pool")([x2, mask_s2])

    model_inputs: list = [inp_s2]

    if use_s1_branch:
        inp_s1 = keras.Input(shape=(n_timesteps_s1, n_channels_s1), name="s1_input")
        model_inputs.append(inp_s1)

        x1 = layers.Conv1D(d_model, 1, activation="relu", kernel_regularizer=kernel_reg, name="s1_stem")(inp_s1)
        x1 = layers.BatchNormalization()(x1)
        mask_s1 = layers.Lambda(lambda t: tf.reduce_any(tf.not_equal(t, 0.0), axis=-1))(inp_s1)
        mask_s1 = EnsureOneValidTimeStep()(mask_s1)
        attn_mask_s1 = BuildSelfAttentionMask()(mask_s1)
        for s in range(n_stage):
            x1 = CTFusion(
                d_model=d_model,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                num_timesteps=n_timesteps_s1,
                name_prefix=f"s1_st{s}",
            )([x1, mask_s1, attn_mask_s1])
        pool_s1 = MaskedGlobalMaxPooling1D(name="s1_pool")([x1, mask_s1])
        fused = CrossModalAttention(
            d_model=d_model,
            num_heads=2,
            attn_dropout=cross_attn_dropout,
            name="modal_fusion",
        )([pool_s2, pool_s1])
    else:
        fused = pool_s2

    if use_static_branch and n_static_features is not None and n_static_features > 0:
        inp_static = keras.Input(shape=(n_static_features,), name="static_input")
        model_inputs.append(inp_static)
        x_static = layers.Dense(d_model, activation="relu", kernel_regularizer=kernel_reg, name="static_dense")(inp_static)
        x_static = layers.BatchNormalization()(x_static)
        fused = layers.Concatenate(name="final_fusion")([fused, x_static])
    elif use_static_branch:
        raise ValueError("use_static_branch=True mais n_static_features manquant ou 0")

    x = layers.Dense(128, activation="relu", kernel_regularizer=kernel_reg)(fused)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=kernel_reg)(x)
    x = layers.Dropout(dropout * 0.5)(x)
    out = layers.Dense(n_classes, activation="softmax", name="prediction")(x)

    return keras.Model(inputs=model_inputs, outputs=out, name="mctnet_v2_multimodal")


if __name__ == "__main__":
    m = build_mctnet_v2(3, 11, 12, 3, 5, 10, use_s1_branch=True, use_static_branch=True)
    m.summary()
    m2 = build_mctnet_v2(3, 11, 12, 3, 5, None, use_s1_branch=False, use_static_branch=False)
    m2.summary()
