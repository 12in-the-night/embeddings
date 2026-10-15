"""
Multimodal MoE Embedding Model — single-file implementation (FIXED).
Mamba-2 + Shared/Routed MoE + MRL + QAT (4/8 bit) + ViT image branch.
Streaming HF data, YAML config, training + postprocess (prune/AWQ/export).

Run:    python main.py
Resume: python main.py --resume ./ckpts_pretrain/pretrain_final.pkl
        (or set `resume:` in config.yaml)
Prune:  python main.py --action prune_experts
AWQ:    python main.py --action awq
Export: python main.py --action export
Full:   python main.py --action full
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

try:
    from kaggle_secrets import UserSecretsClient
    os.environ.setdefault("HF_TOKEN", UserSecretsClient().get_secret("hg"))
except Exception:
    pass

import sys, glob, math, time, queue, pickle, threading, functools, random, re, argparse
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
import yaml
from flax.training import train_state
from flax.jax_utils import replicate, unreplicate

# ================================================================
# CONFIG
# ================================================================
CFG_PATH = os.environ.get("MM_CONFIG", "config.yaml")
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

def _get(path, default=None):
    v = CFG
    for k in path.split("."):
        if not isinstance(v, dict) or k not in v:
            return default
        v = v[k]
    return v

PHASE       = _get("phase", "pretrain")
DATA_SOURCE = _get("data", "mixed")
CFG_RESUME  = _get("resume", None)

SEQ_LEN      = _get("training.seq_len", 128)
GLOBAL_BATCH = _get("training.batch", 128)
STEPS        = _get("training.steps", 50000)
LR           = float(_get("training.lr", 3e-4))
WARMUP       = _get("training.warmup", 1000)
MIN_LR_FRAC  = float(_get("training.min_lr_frac", 0.1))
LOG_EVERY    = _get("training.log_every", 50)
EVAL_EVERY   = _get("training.eval_every", 2000)
EVAL_BATCH   = _get("training.eval_batch", 512)
CKPT_EVERY   = _get("training.ckpt_every", 1000)
CKPT_DIR     = _get("training.ckpt_dir", f"./ckpts_{PHASE}")
GRAD_ACCUM   = _get("training.grad_accum", 1)
REMAT        = _get("training.remat", True)
PRECISION    = _get("training.precision", "fp16").lower()
WD           = float(_get("training.weight_decay", 0.01))

D_MODEL   = _get("model.d_model", 512)
N_LAYERS  = _get("model.n_layers", 32)
MRL_DIMS  = _get("model.mrl_dims", [64, 128, 256, 512])
TOK_MODEL = _get("model.vocab_tok", "mistralai/Mistral-7B-v0.1")

MAMBA_D_STATE  = _get("mamba2.d_state", 128)
MAMBA_D_CONV   = _get("mamba2.d_conv", 4)
MAMBA_CHUNK    = _get("mamba2.chunk", 64)
MAMBA_EXPAND   = _get("mamba2.expand", 2)
MAMBA_HEAD_DIM = _get("mamba2.head_dim", 64)

N_EXPERTS       = _get("moe.n_experts", 32)
TOP_K           = _get("moe.top_k", 2)
EXPERT_FFN_MULT = float(_get("moe.expert_ffn_mult", 0.5))
SHARED_FFN_MULT = float(_get("moe.shared_ffn_mult", 1.0))
MOE_EVERY       = _get("moe.every", 2)
MOE_LAYERS      = _get("moe.layers", None)
AUX_LB_COEF     = float(_get("moe.aux_lb_coef", 0.01))
AUX_Z_COEF      = float(_get("moe.aux_z_coef", 1e-3))

QAT_ENABLED    = _get("qat.enabled", True)
QAT_BITS       = _get("qat.bits", 4)
QAT_GROUP_SIZE = _get("qat.group_size", 128)

IMG_SIZE           = _get("image.size", 224)
IMG_PATCH          = _get("image.patch", 16)
IMG_DIM            = _get("image.dim", 768)
IMG_LAYERS         = _get("image.layers", 8)
IMG_HEADS          = _get("image.heads", 12)
IMG_MLP_MULT       = _get("image.mlp_mult", 4)
IMG_PROCESSOR_NAME = _get("image.processor", "google/vit-base-patch16-224")

MIX_IMAGE_TEXT = _get("data_mix.image_text", 2)
MIX_MSMARCO    = _get("data_mix.msmarco_triplet", 3)
MIX_ALLNLI     = _get("data_mix.allnli_triplet", 1)
MIX_BEKKO      = _get("data_mix.bekko_triplet", 2)

CLIP_NORM = float(_get("optim.clip_norm", 1.0))
INIT_TEMP = float(_get("optim.init_temp", 50.0))

PP            = _get("postprocess", {})
PP_INPUT      = PP.get("input_ckpt", "./ckpts_pretrain/pretrain_final.pkl")
PP_OUTPUT_DIR = PP.get("output_dir", "./ckpts_compressed")
# *_keep_ratio = fraction to KEEP (e.g. 0.75 keeps 75%).
PP_E_KEEP     = float(PP.get("expert_keep_ratio", PP.get("prune_experts_ratio", 0.75)))
PP_L_KEEP     = float(PP.get("layer_keep_ratio", PP.get("prune_layers_ratio", 0.75)))
PP_AWQ_ALPHA  = float(PP.get("awq_alpha", 0.5))
PP_AWQ_BATCHES= int(PP.get("awq_calib_batches", 32))
PP_AWQ_BS     = int(PP.get("awq_calib_batch_size", 8))
PP_EXPORT_BITS= int(PP.get("export_bits", 4))
PP_EXPORT_GRP = int(PP.get("export_group_size", 128))

assert QAT_BITS in (4, 8), f"QAT_BITS must be 4 or 8, got {QAT_BITS}"
assert PP_EXPORT_BITS in (4, 8), f"export_bits must be 4 or 8, got {PP_EXPORT_BITS}"

if PRECISION == "bf16":
    COMPUTE_DTYPE = jnp.bfloat16; LOSS_SCALE = 1.0; DYNAMIC_SCALE = False
elif PRECISION == "fp16":
    COMPUTE_DTYPE = jnp.float16; LOSS_SCALE = 2.0 ** 10; DYNAMIC_SCALE = True
else:
    COMPUTE_DTYPE = jnp.float32; LOSS_SCALE = 1.0; DYNAMIC_SCALE = False
MASTER_DTYPE = jnp.float32

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(PP_OUTPUT_DIR, exist_ok=True)


# ================================================================
# LAZY GLOBALS
# ================================================================
_tokenizer = None
_img_processor = None
_devices = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(TOK_MODEL)
    return _tokenizer

def get_img_processor():
    global _img_processor
    if _img_processor is None:
        from transformers import AutoImageProcessor
        _img_processor = AutoImageProcessor.from_pretrained(IMG_PROCESSOR_NAME)
    return _img_processor

def get_devices():
    global _devices
    if _devices is None:
        _devices = jax.devices()
    return _devices

def get_pad_id():
    tok = get_tokenizer()
    return tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def encode_text(text):
    tok = get_tokenizer()
    return tok.encode(text, add_special_tokens=True,
                      truncation=True, max_length=SEQ_LEN)

def process_image(image):
    p = get_img_processor()
    return p(image, return_tensors="np")["pixel_values"][0].astype(np.float32)


# ================================================================
# QAT
# ================================================================
def fake_quant(x, bits=None, group_size=None, eps=1e-8):
    if not QAT_ENABLED:
        return x
    bits = QAT_BITS if bits is None else bits
    group_size = QAT_GROUP_SIZE if group_size is None else group_size
    qmax = (1 << (bits - 1)) - 1
    x32 = x.astype(jnp.float32)

    if bits >= 8:
        scale = jnp.max(jnp.abs(x32), axis=-1, keepdims=True) + eps
        idx = jnp.clip(jnp.round(x32 / scale * qmax), -qmax, qmax)
        xq = (idx * (scale / qmax)).astype(x.dtype)
        return x + jax.lax.stop_gradient(xq - x)

    D = x32.shape[-1]
    if D % group_size != 0:
        scale = jnp.max(jnp.abs(x32), axis=-1, keepdims=True) + eps
        idx = jnp.clip(jnp.round(x32 / scale * qmax), -qmax, qmax)
        xq = (idx * (scale / qmax)).astype(x.dtype)
        return x + jax.lax.stop_gradient(xq - x)

    G = D // group_size
    orig = x32.shape
    xg = x32.reshape(*orig[:-1], G, group_size)
    scale = jnp.max(jnp.abs(xg), axis=-1, keepdims=True) + eps
    idx = jnp.clip(jnp.round(xg / scale * qmax), -qmax, qmax)
    xq = (idx * (scale / qmax)).reshape(orig).astype(x.dtype)
    return x + jax.lax.stop_gradient(xq - x)


# ================================================================
# LAYERS
# ================================================================
class QDense(nn.Module):
    features: int
    use_bias: bool = False
    capture: bool = False
    dtype: jnp.dtype = COMPUTE_DTYPE
    param_dtype: jnp.dtype = MASTER_DTYPE

    @nn.compact
    def __call__(self, x):
        if self.capture:
            act_mag = jnp.mean(jnp.abs(x.astype(jnp.float32)),
                               axis=tuple(range(x.ndim - 1)))
            self.sow('intermediates', 'act_mag', act_mag)
        k = self.param("kernel", nn.initializers.lecun_normal(),
                       (x.shape[-1], self.features))
        k_q = fake_quant(k).astype(self.dtype)
        y = jnp.matmul(x.astype(self.dtype), k_q,
                       preferred_element_type=jnp.float32)
        if self.use_bias:
            b = self.param("bias", nn.initializers.zeros, (self.features,))
            y = y + b
        return y.astype(self.dtype)


# ================================================================
# MAMBA-2 SSD
# ================================================================
def _pad_seq_dim(x, pad_size):
    if pad_size == 0:
        return x
    pw = [(0, 0)] * x.ndim
    pw[1] = (0, pad_size)
    return jnp.pad(x, pw, mode="constant", constant_values=0.0)


def segsum(x):
    T = x.shape[-1]
    cs = jnp.cumsum(x, axis=-1)
    ss = cs[..., :, None] - cs[..., None, :]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    return jnp.where(mask, ss, -jnp.inf)


def ssd_forward(x, dt, A, B_mat, C_mat, chunk_size, D, dt_bias,
                dt_min, dt_max):
    B_size, seq_len, num_heads, head_dim = x.shape
    dt = jax.nn.softplus(dt + dt_bias)
    dt = jnp.clip(dt, dt_min, dt_max)

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    x_p  = _pad_seq_dim(x, pad_size)
    dt_p = _pad_seq_dim(dt, pad_size)
    B_p  = _pad_seq_dim(B_mat, pad_size)
    C_p  = _pad_seq_dim(C_mat, pad_size)
    L_p  = seq_len + pad_size
    n_chunks = L_p // chunk_size

    x_c  = x_p.reshape(B_size, n_chunks, chunk_size, num_heads, head_dim)
    dt_c = dt_p.reshape(B_size, n_chunks, chunk_size, num_heads)
    B_c  = B_p.reshape(B_size, n_chunks, chunk_size, num_heads, B_mat.shape[-1])
    C_c  = C_p.reshape(B_size, n_chunks, chunk_size, num_heads, C_mat.shape[-1])

    A_disc = A.astype(jnp.float32) * dt_c.astype(jnp.float32)
    A_t    = jnp.transpose(A_disc, (0, 1, 3, 2))
    A_cumsum = jnp.cumsum(A_t, axis=-1)
    decay_end = jnp.exp(A_cumsum[..., -1:])
    L_mat = jnp.exp(segsum(A_t))
    x_disc = x_c.astype(jnp.float32) * dt_c[..., None].astype(jnp.float32)

    C32 = C_c.astype(jnp.float32)
    B32 = B_c.astype(jnp.float32)
    scores = jnp.einsum("bclhn,bcshn->bchls", C32, B32) * L_mat
    y_c = jnp.einsum("bchls,bcshp->bclhp", scores, x_disc)

    state = jnp.zeros((B_size, num_heads, head_dim, B_mat.shape[-1]),
                      dtype=jnp.float32)
    chunk_decay = decay_end.squeeze(-1)
    y_inter_list = []
    for ci in range(n_chunks):
        Bc = B32[:, ci]; Cc = C32[:, ci]; xd = x_disc[:, ci]
        Acs = A_cumsum[:, ci]; dec = chunk_decay[:, ci]
        y_i = jnp.einsum("blhn,bhpn->blhp", Cc, state)
        y_i = y_i * jnp.exp(Acs).transpose(0, 2, 1)[..., None]
        decay = jnp.exp(Acs[..., -1:] - Acs)
        cs = jnp.einsum("blhn,bhl,blhp->bhpn", Bc, decay, xd)
        state = dec[..., None, None] * state + cs
        y_inter_list.append(y_i)

    y_inter = jnp.stack(y_inter_list, axis=1)
    y_full = y_c + y_inter
    D_res = D.astype(jnp.float32).reshape(1, 1, 1, num_heads, 1) * \
            x_c.astype(jnp.float32)
    y_full = y_full + D_res
    y_full = y_full.reshape(B_size, L_p, num_heads, head_dim)
    if pad_size:
        y_full = y_full[:, :seq_len]
    return y_full.astype(x.dtype)


class Mamba2Block(nn.Module):
    d_model: int
    d_state: int = MAMBA_D_STATE
    d_conv: int = MAMBA_D_CONV
    chunk: int = MAMBA_CHUNK
    expand: int = MAMBA_EXPAND
    head_dim: int = MAMBA_HEAD_DIM
    capture: bool = False
    dtype: jnp.dtype = COMPUTE_DTYPE
    param_dtype: jnp.dtype = MASTER_DTYPE

    @nn.compact
    def __call__(self, x):
        B, T, C_ = x.shape
        d_inner = self.expand * C_
        H = d_inner // self.head_dim
        P = self.head_dim
        N = self.d_state

        h = nn.LayerNorm(dtype=jnp.float32)(
            x.astype(jnp.float32)).astype(self.dtype)

        proj_size = 2 * d_inner + 2 * N + H
        zxbcdt = QDense(proj_size, use_bias=False, capture=self.capture)(h)
        z, xBC, dt = jnp.split(zxbcdt,
                               [d_inner, 2 * d_inner + 2 * N], axis=-1)

        # FIX: flax.linen.Conv has no "CAUSAL" padding — left-pad manually.
        xBC = jnp.pad(xBC, [(0, 0), (self.d_conv - 1, 0), (0, 0)])
        xBC = nn.Conv(features=xBC.shape[-1], kernel_size=(self.d_conv,),
                      padding="VALID",
                      feature_group_count=xBC.shape[-1], use_bias=True,
                      dtype=self.dtype, param_dtype=self.param_dtype)(xBC)
        xBC = nn.silu(xBC)

        x_in, B_t, C_t = jnp.split(xBC, [d_inner, d_inner + N], axis=-1)
        x_in = x_in.reshape(B, T, H, P)
        B_t  = B_t.reshape(B, T, N)
        C_t  = C_t.reshape(B, T, N)
        B_exp = jnp.broadcast_to(B_t[:, :, None, :], (B, T, H, N))
        C_exp = jnp.broadcast_to(C_t[:, :, None, :], (B, T, H, N))

        def _a_log_init(key, shape, dtype=jnp.float32):
            return jnp.log(jax.random.uniform(key, shape, minval=1.0, maxval=16.0))
        A_log = self.param("A_log", _a_log_init, (H,))
        A = -jnp.exp(A_log.astype(jnp.float32))
        D = self.param("D", nn.initializers.ones, (H,))

        def _dt_bias_init(key, shape, dtype=jnp.float32):
            u = jax.random.uniform(key, shape)
            dt = jnp.exp(u * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
            return dt + jnp.log(-jnp.expm1(-dt))
        dt_bias = self.param("dt_bias", _dt_bias_init, (H,))

        y = ssd_forward(x=x_in, dt=dt, A=A, B_mat=B_exp, C_mat=C_exp,
                        chunk_size=self.chunk, D=D, dt_bias=dt_bias,
                        dt_min=0.0, dt_max=100.0)
        y = y.reshape(B, T, d_inner)
        y = y * nn.silu(z)
        return QDense(C_, use_bias=False)(y)


# ================================================================
# ViT
# ================================================================
class ViTBlock(nn.Module):
    dim: int
    heads: int = IMG_HEADS
    mlp_mult: int = IMG_MLP_MULT
    capture: bool = False
    dtype: jnp.dtype = COMPUTE_DTYPE
    param_dtype: jnp.dtype = MASTER_DTYPE

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm(dtype=jnp.float32)(
            x.astype(jnp.float32)).astype(self.dtype)
        attn = nn.MultiHeadDotProductAttention(
            num_heads=self.heads, qkv_features=self.dim,
            dtype=self.dtype, param_dtype=self.param_dtype)(h, h)
        x = x + attn
        h = nn.LayerNorm(dtype=jnp.float32)(
            x.astype(jnp.float32)).astype(self.dtype)
        h = QDense(self.mlp_mult * self.dim, capture=self.capture)(h)
        h = nn.gelu(h)
        h = QDense(self.dim)(h)
        x = x + h
        return x


class ViTImageEncoder(nn.Module):
    img_size: int = IMG_SIZE
    patch_size: int = IMG_PATCH
    dim: int = IMG_DIM
    depth: int = IMG_LAYERS
    heads: int = IMG_HEADS
    mlp_mult: int = IMG_MLP_MULT
    capture: bool = False
    dtype: jnp.dtype = COMPUTE_DTYPE
    param_dtype: jnp.dtype = MASTER_DTYPE

    @nn.compact
    def __call__(self, pixel_values):
        B, C_, H, W = pixel_values.shape
        ps = self.patch_size
        n_patches = (H // ps) * (W // ps)
        x = pixel_values.reshape(B, C_, H // ps, ps, W // ps, ps)
        x = x.transpose(0, 2, 4, 1, 3, 5).reshape(B, n_patches, -1)
        x = QDense(self.dim)(x)
        cls = self.param("cls", nn.initializers.normal(0.02), (1, 1, self.dim))
        pos = self.param("pos", nn.initializers.normal(0.02),
                         (1, n_patches + 1, self.dim))
        x = jnp.concatenate([jnp.broadcast_to(cls, (B, 1, self.dim)), x], axis=1)
        x = x + pos
        for _ in range(self.depth):
            x = ViTBlock(self.dim, self.heads, self.mlp_mult,
                         capture=self.capture,
                         dtype=self.dtype, param_dtype=self.param_dtype)(x)
        x = nn.LayerNorm(dtype=jnp.float32)(
            x.astype(jnp.float32)).astype(x.dtype)
        return x[:, 0]


# ================================================================
# MoE
# ================================================================
class SharedRoutedMoE(nn.Module):
    d_model: int
    d_ff_expert: int
    d_ff_shared: int
    n_experts: int = N_EXPERTS
    top_k: int = TOP_K
    capture: bool = False
    dtype: jnp.dtype = COMPUTE_DTYPE
    param_dtype: jnp.dtype = MASTER_DTYPE

    @nn.compact
    def __call__(self, x):
        B, T, C_ = x.shape
        N = B * T
        E, K = self.n_experts, self.top_k
        x_flat = x.reshape(N, C_)

        if self.capture:
            self.sow('intermediates', 'moe_in_mag',
                     jnp.mean(jnp.abs(x_flat.astype(jnp.float32)), axis=0))

        up_s   = QDense(self.d_ff_shared, capture=self.capture)(x_flat)
        h_s    = nn.gelu(up_s)
        down_s = QDense(C_)(h_s)

        router = self.param("router", nn.initializers.normal(0.02), (C_, E))
        gate_logits = x_flat.astype(jnp.float32) @ router.astype(jnp.float32)
        gates = jax.nn.softmax(gate_logits, axis=-1)
        topk_w, topk_i = jax.lax.top_k(gates, K)
        topk_w = topk_w / (topk_w.sum(-1, keepdims=True) + 1e-9)

        if self.capture:
            counts = jnp.bincount(topk_i.reshape(-1),
                                  length=E).astype(jnp.float32)
            self.sow('intermediates', 'expert_counts', counts)
            self.sow('intermediates', 'gate_mean', gates.mean(axis=0))

        expert_init = nn.initializers.lecun_normal(
            in_axis=1, out_axis=2, batch_axis=(0,))
        up   = self.param("up",   expert_init, (E, C_, self.d_ff_expert))
        down = self.param("down", expert_init, (E, self.d_ff_expert, C_))
        up_q     = fake_quant(up).astype(self.dtype)
        down_sw  = jnp.swapaxes(down, -1, -2)
        down_sw_q = fake_quant(down_sw).astype(self.dtype)
        down_q   = jnp.swapaxes(down_sw_q, -1, -2)

        idx_flat = topk_i.reshape(-1)
        x_rep    = jnp.repeat(x_flat, K, axis=0)
        order    = jnp.argsort(idx_flat)
        inv      = jnp.argsort(order)
        x_sorted = x_rep[order].astype(self.dtype)
        i_sorted = idx_flat[order]
        group_sizes = jnp.bincount(i_sorted, length=E)

        h = jax.lax.ragged_dot(x_sorted, up_q, group_sizes)
        h = nn.gelu(h)
        if self.capture:
            self.sow('intermediates', 'moe_h_mag',
                     jnp.mean(jnp.abs(h.astype(jnp.float32)), axis=0))
        y = jax.lax.ragged_dot(h, down_q, group_sizes)
        y = y[inv].reshape(N, K, C_)
        y_routed = (y * topk_w[..., None].astype(y.dtype)).sum(axis=1)

        out = (down_s + y_routed).reshape(B, T, C_)

        f_i = jnp.bincount(i_sorted, length=E).astype(jnp.float32) / (N * K)
        P_i = gates.mean(axis=0)
        aux_lb = E * jnp.sum(f_i * P_i)
        z = jax.nn.logsumexp(gate_logits, axis=-1)
        aux_z = jnp.mean(z ** 2)
        return out, AUX_LB_COEF * aux_lb + AUX_Z_COEF * aux_z


# ================================================================
# MAMBA BLOCK (with MoE)
# ================================================================
class MambaBlock(nn.Module):
    d_model: int
    layer_idx: int = 0
    remat: bool = False
    expert_ffn: int = 256
    shared_ffn: int = 512
    use_moe: bool = True
    capture: bool = False

    @nn.compact
    def __call__(self, x):
        Mamba2BlockR = nn.remat(Mamba2Block) if self.remat else Mamba2Block
        h = Mamba2BlockR(self.d_model, capture=self.capture)(x)
        x = x + h
        aux = jnp.zeros((), dtype=jnp.float32)
        if self.use_moe:
            h_ln = nn.LayerNorm(dtype=jnp.float32)(
                x.astype(jnp.float32)).astype(x.dtype)
            MoER = nn.remat(SharedRoutedMoE) if self.remat else SharedRoutedMoE
            h_moe, aux = MoER(self.d_model, self.expert_ffn,
                              self.shared_ffn, capture=self.capture)(h_ln)
            x = x + h_moe
        return x, aux


# ================================================================
# MAIN MODEL
# ================================================================
class MMEmbedModel(nn.Module):
    vocab: int
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    mrl_dims: tuple = tuple(MRL_DIMS)
    max_len: int = SEQ_LEN
    img_dim: int = IMG_DIM
    remat: bool = False
    moe_every: int = MOE_EVERY
    moe_layers: tuple | None = None
    expert_ffn_mult: float = EXPERT_FFN_MULT
    shared_ffn_mult: float = SHARED_FFN_MULT
    capture: bool = False
    dtype: jnp.dtype = COMPUTE_DTYPE
    param_dtype: jnp.dtype = MASTER_DTYPE

    def _is_moe_layer(self, li):
        if self.moe_layers is not None:
            return li in self.moe_layers
        if self.moe_every <= 0:
            return False
        return li % self.moe_every == self.moe_every - 1

    @nn.compact
    def __call__(self, tokens, mask, pixel_values=None):
        B, T = tokens.shape
        emb = nn.Embed(self.vocab, self.d_model,
                       dtype=self.dtype, param_dtype=self.param_dtype,
                       embedding_init=nn.initializers.normal(0.02))
        x = emb(tokens).astype(jnp.float32)

        expert_ffn = max(1, int(self.expert_ffn_mult * self.d_model))
        shared_ffn = max(1, int(self.shared_ffn_mult * self.d_model))
        aux_total = jnp.zeros((), dtype=jnp.float32)

        for li in range(self.n_layers):
            use_moe = self._is_moe_layer(li)
            x, aux = MambaBlock(self.d_model, layer_idx=li,
                                remat=self.remat,
                                expert_ffn=expert_ffn,
                                shared_ffn=shared_ffn,
                                use_moe=use_moe,
                                capture=self.capture)(x)
            aux_total = aux_total + aux

        x = nn.LayerNorm(dtype=jnp.float32)(
            x.astype(jnp.float32)).astype(x.dtype)
        q_pool = self.param("q_pool", nn.initializers.normal(0.02),
                            (1, 1, self.d_model))
        scores = jnp.einsum("btd,osd->bto", x, q_pool) / math.sqrt(self.d_model)
        scores = jnp.where(mask[:, :, None] > 0, scores, -1e9)
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=1)
        pooled_text = jnp.einsum("btd,bto->bod", x, attn)[:, 0]

        pooled_img = None
        if pixel_values is not None:
            vit = ViTImageEncoder(img_size=IMG_SIZE, patch_size=IMG_PATCH,
                                  dim=self.img_dim, depth=IMG_LAYERS,
                                  heads=IMG_HEADS, mlp_mult=IMG_MLP_MULT,
                                  capture=self.capture,
                                  dtype=self.dtype, param_dtype=self.param_dtype)
            pooled_img = vit(pixel_values)
            pooled_img = QDense(self.d_model, use_bias=True)(pooled_img)
            pooled_img = nn.LayerNorm(dtype=jnp.float32)(
                pooled_img.astype(jnp.float32)).astype(self.dtype)

        log_temp = self.param("log_temp",
                              lambda k, s: jnp.full(s, math.log(INIT_TEMP)),
                              ())
        temp = jnp.exp(log_temp)

        embs_text = {}
        for d in self.mrl_dims:
            h = QDense(d, use_bias=True)(pooled_text).astype(jnp.float32)
            embs_text[d] = jax.nn.l2_normalize(h, axis=-1)

        embs_img = None
        if pooled_img is not None:
            embs_img = {}
            for d in self.mrl_dims:
                h = QDense(d, use_bias=True)(pooled_img).astype(jnp.float32)
                embs_img[d] = jax.nn.l2_normalize(h, axis=-1)

        return embs_text, embs_img, temp, aux_total


# ================================================================
# LOSSES
# ================================================================
def info_nce_with_hard_neg(q, p, n, temp):
    B = q.shape[0]
    logits = q @ p.T / temp
    logits_n = jnp.sum(q * n, axis=-1, keepdims=True) / temp
    logits = jnp.concatenate([logits, logits_n], axis=1)
    targets = jnp.arange(B)
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


def mrl_contrastive(q_embs, p_embs, n_embs, temp, mrl_dims):
    total = 0.0
    for d in mrl_dims:
        total = total + info_nce_with_hard_neg(q_embs[d], p_embs[d], n_embs[d], temp)
    return total / len(mrl_dims)


def image_text_contrastive(img_embs, text_embs, temp, mrl_dims):
    total = 0.0
    for d in mrl_dims:
        logits = img_embs[d] @ text_embs[d].T / temp
        labels = jnp.arange(logits.shape[0])
        loss_i2t = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        loss_t2i = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()
        total = total + (loss_i2t + loss_t2i) / 2.0
    return total / len(mrl_dims)


# ================================================================
# DATA
# ================================================================
def _pad(ids_list, max_len):
    PAD = get_pad_id()
    out  = np.full((len(ids_list), max_len), PAD, dtype=np.int32)
    mask = np.zeros((len(ids_list), max_len), dtype=np.int32)
    for i, ids in enumerate(ids_list):
        n = min(len(ids), max_len)
        out[i, :n] = ids[:n]
        mask[i, :n] = 1
    return out, mask


def stream_msmarco():
    from datasets import load_dataset
    ds = load_dataset("sentence-transformers/msmarco", "triplets",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=20_000)
    for ex in ds:
        q, p, n = ex.get("query"), ex.get("positive"), ex.get("negative")
        if q and p and n:
            yield q, p, n


def stream_all_nli():
    from datasets import load_dataset
    ds = load_dataset("sentence-transformers/all-nli", "triplet",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10_000)
    for ex in ds:
        a = ex.get("anchor")
        e = ex.get("positive") or ex.get("entailment")
        c = ex.get("negative") or ex.get("contradiction")
        if a and e and c:
            yield a, e, c


def stream_bekko():
    from datasets import load_dataset
    subsets = [
        "hotchpotch-fineweb2-ir__eng_Latn__qd__triplet",
        "hotchpotch-fineweb2-ir__deu_Latn__qd__triplet",
        "hotchpotch-fineweb2-ir__fra_Latn__qd__triplet",
        "hotchpotch-fineweb2-ir__spa_Latn__qd__triplet",
        "huggingfacefw-finewiki__en__qd",
        "sentence-transformers-all-nli__triplet__dd__triplet",
    ]
    random.shuffle(subsets)
    for sub in subsets:
        try:
            ds = load_dataset("hotchpotch/bekko-embedding-v1-unsupervised",
                              sub, split="train", streaming=True)
            ds = ds.shuffle(seed=42, buffer_size=10_000)
        except Exception as e:
            print(f"[data] skip {sub}: {e}")
            continue
        for ex in ds:
            q = ex.get("query"); p = ex.get("pos"); n = ex.get("neg")
            if q and p:
                yield q, p, (n if n else p)


def stream_image_text():
    from datasets import load_dataset
    ds = load_dataset("laion/laion400m", split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=5_000)
    for ex in ds:
        img = ex.get("image"); cap = ex.get("caption") or ex.get("text")
        if img is not None and cap:
            yield img, cap


def _cycle(gen_fn):
    while True:
        for x in gen_fn():
            yield x


def data_worker(batch_size, max_len, steps, out_q):
    w_image = 0 if PHASE == "posttrain" else MIX_IMAGE_TEXT
    weights = {
        "image_text": w_image,
        "msmarco":    MIX_MSMARCO,
        "allnli":     MIX_ALLNLI,
        "bekko":      MIX_BEKKO,
    }
    total_w = sum(weights.values())
    assert total_w > 0, "data_mix weights sum to 0 — enable at least one source"
    probs = {k: v / total_w for k, v in weights.items()}
    srcs = list(probs.keys())
    streams = {
        "image_text": _cycle(stream_image_text),
        "msmarco":    _cycle(stream_msmarco),
        "allnli":     _cycle(stream_all_nli),
        "bekko":      _cycle(stream_bekko),
    }
    step = 0
    while step < steps:
        src = random.choices(srcs, weights=[probs[s] for s in srcs])[0]
        if src == "image_text":
            ib, cb = [], []
            while len(ib) < batch_size:
                img, cap = next(streams["image_text"])
                try:
                    px = process_image(img)
                except Exception:
                    continue
                ib.append(px); cb.append(encode_text(cap))
            ci, cm = _pad(cb, max_len)
            pi = np.stack(ib, axis=0)
            out_q.put(("imgtext", pi, ci, cm))
        else:
            qb, pb, nb = [], [], []
            while len(qb) < batch_size:
                q, p, n = next(streams[src])
                qb.append(encode_text(q))
                pb.append(encode_text(p))
                nb.append(encode_text(n))
            qi, qm = _pad(qb, max_len)
            pi, pm = _pad(pb, max_len)
            ni, nm = _pad(nb, max_len)
            out_q.put(("triplet", qi, qm, pi, pm, ni, nm))
        step += 1
    out_q.put(None)


def load_eval_batch(n_samples, max_len):
    from datasets import load_dataset
    ds = load_dataset("sentence-transformers/all-nli", "triplet",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=999, buffer_size=5_000)
    qb, pb = [], []
    for ex in ds:
        a = ex.get("anchor")
        e = ex.get("positive") or ex.get("entailment")
        if a and e:
            qb.append(encode_text(a)); pb.append(encode_text(e))
            if len(qb) >= n_samples:
                break
    qi, qm = _pad(qb, max_len)
    pi, pm = _pad(pb, max_len)
    return (jnp.asarray(qi), jnp.asarray(qm),
            jnp.asarray(pi), jnp.asarray(pm))


# ================================================================
# TRAINING
# ================================================================
def _wd_mask(params, phase=None):
    phase = phase or PHASE
    def _keep(path, leaf):
        keys = [str(getattr(k, "key", k)) for k in path]
        # In posttrain the image branch is never used (triplet batches only),
        # so ViT params receive zero grads; excluding them from decay too
        # makes "frozen" exact (zero-grad + zero-decay = no change).
        if phase == "posttrain" and any("ViT" in k for k in keys):
            return False
        if any("LayerNorm" in k for k in keys):
            return False
        return keys[-1] not in ("bias", "scale", "A_log", "D",
                                "dt_bias", "embedding", "q_pool",
                                "log_temp", "cls", "pos", "router")
    return jax.tree_util.tree_map_with_path(_keep, params)


def make_state(rng, model, seq_len, img_size):
    dt = jnp.zeros((1, seq_len), dtype=jnp.int32)
    dm = jnp.ones((1, seq_len), dtype=jnp.int32)
    di = jnp.zeros((1, 3, img_size, img_size), dtype=jnp.float32)
    params = model.init(rng, dt, dm, di)["params"]
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR,
        warmup_steps=WARMUP, decay_steps=STEPS,
        end_value=LR * MIN_LR_FRAC)
    # FIX: standard descent — positive decay, negative LR via flip_sign.
    tx = optax.chain(
        optax.clip_by_global_norm(CLIP_NORM),
        optax.adafactor(learning_rate=None,
                        multiply_by_parameter_scale=False),
        optax.add_decayed_weights(WD, _wd_mask(params, PHASE)),
        optax.scale_by_learning_rate(sched),
    )
    if GRAD_ACCUM > 1:
        tx = optax.MultiSteps(tx, every_k_schedule=GRAD_ACCUM)
    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params, tx=tx)


def init_dyn_state(scale):
    return (jnp.float32(scale), jnp.int32(0), jnp.int32(0))


def update_dyn_state(dyn_state, grads, growth_interval=2000,
                     growth=2.0, backoff=0.5,
                     min_scale=1.0, max_scale=2.0**24):
    scale, counter, valid = dyn_state
    has_inf = jnp.any(jnp.stack([
        jnp.any(~jnp.isfinite(g)) for g in jax.tree_util.tree_leaves(grads)
    ]))
    new_scale_inf = jnp.maximum(scale * backoff, min_scale)
    counter_next = jnp.where(has_inf, jnp.int32(0), counter + jnp.int32(1))
    grow_now = (counter_next >= growth_interval) & (~has_inf)
    new_scale_grow = jnp.minimum(scale * growth, max_scale)
    scale_next = jnp.where(has_inf, new_scale_inf,
                           jnp.where(grow_now, new_scale_grow, scale))
    counter_next = jnp.where(grow_now, jnp.int32(0), counter_next)
    return (scale_next, counter_next, valid + jnp.int32(~has_inf)), has_inf


@functools.partial(jax.pmap, axis_name="dev", donate_argnums=(0, 2))
def train_step_imgtext(state, batch, dyn_state):
    pv, ids, mask = batch
    scale = dyn_state[0]
    def loss_fn(params):
        t_embs, i_embs, temp, aux = state.apply_fn(
            {"params": params}, ids, mask, pv)
        loss = image_text_contrastive(i_embs, t_embs, temp, MRL_DIMS) + aux
        return loss * scale, loss
    (_, loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="dev")
    if DYNAMIC_SCALE:
        dyn_state, has_inf = update_dyn_state(dyn_state, grads)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
    else:
        has_inf = jnp.array(False, dtype=jnp.bool_)
    grads = jax.tree_util.tree_map(lambda g: g / scale, grads)
    new_state = state.apply_gradients(grads=grads)
    state = jax.tree_util.tree_map(
        lambda n, o: jnp.where(has_inf, o, n), new_state, state)
    return (state, dyn_state, jax.lax.pmean(loss, "dev"),
            has_inf.astype(jnp.float32))


@functools.partial(jax.pmap, axis_name="dev", donate_argnums=(0, 2))
def train_step_triplet(state, batch, dyn_state):
    qi, qm, pi, pm, ni, nm = batch
    scale = dyn_state[0]
    def loss_fn(params):
        q_embs, _, temp, aux_q = state.apply_fn({"params": params}, qi, qm)
        p_embs, _, _,    aux_p = state.apply_fn({"params": params}, pi, pm)
        n_embs, _, _,    aux_n = state.apply_fn({"params": params}, ni, nm)
        loss = mrl_contrastive(q_embs, p_embs, n_embs, temp, MRL_DIMS) \
             + (aux_q + aux_p + aux_n) / 3.0
        return loss * scale, loss
    (_, loss), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="dev")
    if DYNAMIC_SCALE:
        dyn_state, has_inf = update_dyn_state(dyn_state, grads)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
    else:
        has_inf = jnp.array(False, dtype=jnp.bool_)
    grads = jax.tree_util.tree_map(lambda g: g / scale, grads)
    new_state = state.apply_gradients(grads=grads)
    state = jax.tree_util.tree_map(
        lambda n, o: jnp.where(has_inf, o, n), new_state, state)
    return (state, dyn_state, jax.lax.pmean(loss, "dev"),
            has_inf.astype(jnp.float32))


@functools.partial(jax.pmap, axis_name="dev")
def eval_emb_step(state, ids, mask, dim):
    embs, _, _, _ = state.apply_fn({"params": state.params}, ids, mask)
    return embs[dim]


# ================================================================
# CHECKPOINT
# ================================================================
def save_checkpoint(state, step, path, dyn_state=None):
    host = unreplicate(state)
    to_np = lambda x: np.asarray(x)
    payload = {
        "step": step, "phase": PHASE,
        "params": jax.tree_util.tree_map(to_np, host.params),
        "opt_state": jax.tree_util.tree_map(to_np, host.opt_state),
        "config": CFG,
    }
    if dyn_state is not None:
        ds = unreplicate(dyn_state)
        payload["dyn_state"] = (float(np.asarray(ds[0])),
                                int(np.asarray(ds[1])),
                                int(np.asarray(ds[2])))
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, path)
    print(f"[ckpt] saved {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ================================================================
# EVAL
# ================================================================
def _recall_at_k(q_embs, p_embs, k=1):
    sims = q_embs @ p_embs.T
    topk = jnp.argsort(-sims, axis=-1)[:, :k]
    targets = jnp.arange(q_embs.shape[0])[:, None]
    return jnp.any(topk == targets, axis=-1).mean()


def evaluate(state, eval_batch, dim, n_dev):
    qi, qm, pi, pm = eval_batch
    qi = qi.reshape(n_dev, -1, qi.shape[-1]); qm = qm.reshape(n_dev, -1, qm.shape[-1])
    pi = pi.reshape(n_dev, -1, pi.shape[-1]); pm = pm.reshape(n_dev, -1, pm.shape[-1])
    q_e = unreplicate(eval_emb_step(state, qi, qm, dim)).reshape(-1, dim)
    p_e = unreplicate(eval_emb_step(state, pi, pm, dim)).reshape(-1, dim)
    return (float(_recall_at_k(q_e, p_e, 1)),
            float(_recall_at_k(q_e, p_e, 5)),
            float(_recall_at_k(q_e, p_e, 10)))


# ================================================================
# SANITY
# ================================================================
def sanity():
    print(f"[sanity] phase={PHASE} prec={PRECISION} E={N_EXPERTS} k={TOP_K}")
    x = jnp.array([0.1, 0.7, -1.3, 2.4, 0.0, -0.05], dtype=jnp.float32)
    g = jax.grad(lambda v: fake_quant(v).sum())(x)
    assert jnp.allclose(g, 1.0), "STE gradient wrong"
    B, D = 4, 8
    q = jax.nn.l2_normalize(jax.random.normal(jax.random.PRNGKey(0), (B, D)), -1)
    loss_perfect = info_nce_with_hard_neg(q, q, -q, temp=0.02)
    assert float(loss_perfect) < 1e-3
    m = MMEmbedModel(vocab=256, d_model=64, n_layers=4,
                     mrl_dims=(32, 64), max_len=32, img_dim=64,
                     remat=False, moe_every=2,
                     expert_ffn_mult=0.5, shared_ffn_mult=1.0)
    t = jnp.ones((2, 32), dtype=jnp.int32)
    mask = jnp.ones((2, 32), dtype=jnp.int32)
    img = jnp.zeros((2, 3, 32, 32), dtype=jnp.float32)
    p = m.init(jax.random.PRNGKey(0), t, mask, img)["params"]
    te, ie, temp, aux = m.apply({"params": p}, t, mask, img)
    assert te[32].shape == (2, 32) and ie[64].shape == (2, 64)
    # descent check: one optimizer step must DECREASE a trivial loss
    sched = optax.constant_schedule(0.1)
    tx = optax.chain(optax.add_decayed_weights(0.01),
                     optax.scale_by_learning_rate(sched))
    st = train_state.TrainState.create(
        apply_fn=None, params={"w": jnp.array(1.0)}, tx=tx)
    g = jax.grad(lambda ps: ps["w"] ** 2)(st.params)
    st2 = st.apply_gradients(grads=g)
    assert float(st2.params["w"]) < 1.0, "optimizer must descend"
    print("[sanity] OK")


# ================================================================
# TRAIN MAIN
# ================================================================
def train_main(resume_path=None):
    sanity()
    tok = get_tokenizer()
    vocab = tok.vocab_size
    assert vocab < 65536
    devices = get_devices()
    N_DEV = len(devices)
    assert GLOBAL_BATCH % N_DEV == 0
    per_dev = GLOBAL_BATCH // N_DEV
    print(f"devices: {N_DEV}  {devices}  vocab={vocab}")

    model = MMEmbedModel(
        vocab=vocab, d_model=D_MODEL, n_layers=N_LAYERS,
        mrl_dims=tuple(MRL_DIMS), max_len=SEQ_LEN,
        img_dim=IMG_DIM, remat=REMAT,
        moe_every=MOE_EVERY,
        moe_layers=tuple(MOE_LAYERS) if MOE_LAYERS else None,
        expert_ffn_mult=EXPERT_FFN_MULT,
        shared_ffn_mult=SHARED_FFN_MULT,
    )
    state = make_state(jax.random.PRNGKey(0), model, SEQ_LEN, IMG_SIZE)

    dyn_local = init_dyn_state(LOSS_SCALE)
    start_step = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = load_checkpoint(resume_path)
        start_step = int(ckpt["step"])
        state = state.replace(
            params=jax.tree_util.tree_map(jnp.asarray, ckpt["params"]),
            opt_state=jax.tree_util.tree_map(lambda _, x: jnp.asarray(x),
                                             state.opt_state, ckpt["opt_state"]),
            step=start_step)
        if "dyn_state" in ckpt:
            ds = ckpt["dyn_state"]
            dyn_local = (jnp.float32(ds[0]), jnp.int32(ds[1]), jnp.int32(ds[2]))
        print(f"[ckpt] resumed {resume_path} @ step {start_step}")

    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(state.params))
    print(f"[model] total={n_params/1e6:.1f}M  d_model={D_MODEL} "
          f"L={N_LAYERS}  E={N_EXPERTS}  k={TOP_K}  QAT={QAT_BITS}bit")

    state = replicate(state)
    dyn_state = replicate(dyn_local)

    print("[eval] loading fixed eval set ...")
    eval_batch = load_eval_batch(EVAL_BATCH, SEQ_LEN)

    q = queue.Queue(maxsize=4)
    threading.Thread(target=data_worker,
                     args=(GLOBAL_BATCH, SEQ_LEN, STEPS - start_step, q),
                     daemon=True).start()

    t0 = time.time()
    inf_acc = jnp.zeros((N_DEV,), jnp.float32)
    inf_at_last_log = 0

    for local_step in range(STEPS - start_step):
        step = start_step + local_step
        item = q.get()
        if item is None:
            break

        def _place(arr):
            return jnp.asarray(arr).reshape(N_DEV, per_dev, *arr.shape[1:])

        tag = item[0]
        if tag == "imgtext":
            _, pv, ids, mask = item
            batch = (_place(pv), _place(ids), _place(mask))
            state, dyn_state, loss, has_inf = train_step_imgtext(
                state, batch, dyn_state)
        else:
            _, qi, qm, pi, pm, ni, nm = item
            batch = (_place(qi), _place(qm), _place(pi),
                     _place(pm), _place(ni), _place(nm))
            state, dyn_state, loss, has_inf = train_step_triplet(
                state, batch, dyn_state)

        inf_acc = inf_acc + has_inf

        if step % LOG_EVERY == 0 or step == STEPS - 1:
            dt = time.time() - t0
            loss_f = float(unreplicate(loss).mean())
            sc = float(unreplicate(dyn_state[0]))
            inf_events = int(inf_acc[0])
            since = inf_events - inf_at_last_log
            inf_at_last_log = inf_events
            tok_seen = (local_step + 1) * GLOBAL_BATCH * SEQ_LEN
            print(f"step {step:6d} | {tag:8s} | loss {loss_f:.4f} "
                  f"| scale {sc:.0f} inf {since}/{inf_events} "
                  f"| {dt:6.1f}s | {tok_seen/1e6:6.1f}M tok")

        if step > 0 and step % EVAL_EVERY == 0:
            for d in MRL_DIMS:
                r1, r5, r10 = evaluate(state, eval_batch, d, N_DEV)
                print(f"[eval] step {step:6d} dim={d:4d} | "
                      f"R@1 {r1:.4f}  R@5 {r5:.4f}  R@10 {r10:.4f}")

        if step > 0 and step % CKPT_EVERY == 0:
            save_checkpoint(state, step + 1,
                            os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pkl"),
                            dyn_state=dyn_state)
            for old in sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_0*.pkl")))[:-3]:
                os.remove(old)

    final_path = os.path.join(CKPT_DIR, f"{PHASE}_final.pkl")
    save_checkpoint(state, STEPS, final_path, dyn_state=dyn_state)
    print(f"\n[final] saved to {final_path}")


# ================================================================
# POSTPROCESS — path helpers
# ================================================================
def _key_name(k):
    return str(getattr(k, "key", k))


def _path_names(path):
    return [_key_name(k) for k in path]


def _layer_idx_from_names(keys):
    for k in keys:
        m = re.match(r"MambaBlock_(\d+)", k)
        if m:
            return int(m.group(1))
    return None


def _flatten_intermediates(tree, prefix=""):
    """Flax returns nested intermediates; flatten to {'a/b/c': value}."""
    out = {}
    for k, v in tree.items():
        p = f"{prefix}/{k}" if prefix else str(k)
        if isinstance(v, dict):
            out.update(_flatten_intermediates(v, p))
        else:
            out[p] = v
    return out


def _build_model(ckpt, capture=False):
    cfg = ckpt["config"]
    m = cfg["model"]; mo = cfg["moe"]; im = cfg["image"]; tr = cfg["training"]
    return MMEmbedModel(
        vocab=get_tokenizer().vocab_size,
        d_model=m["d_model"], n_layers=m["n_layers"],
        mrl_dims=tuple(m["mrl_dims"]), max_len=tr["seq_len"],
        img_dim=im["dim"], remat=False,
        moe_every=mo["every"],
        moe_layers=tuple(mo["layers"]) if mo.get("layers") else None,
        expert_ffn_mult=mo["expert_ffn_mult"],
        shared_ffn_mult=mo["shared_ffn_mult"],
        capture=capture,
    )


def _run_capture(model, params, seq_len, img_size, n_batches, bs):
    rng = np.random.default_rng(0)
    agg = {}
    # Pull real text once, reuse across batches
    try:
        from datasets import load_dataset
        ds = load_dataset("sentence-transformers/all-nli", "triplet",
                          split="train", streaming=True).take(bs * n_batches)
        texts = [ex["anchor"] for ex in ds]
    except Exception as e:
        print(f"[capture] real data unavailable ({e}), using random")
        texts = None

    for b in range(n_batches):
        if texts:
            chunk = texts[b*bs:(b+1)*bs]
            ids_list = [encode_text(t) for t in chunk]
            toks_np, msk_np = _pad(ids_list, seq_len)
            toks = jnp.asarray(toks_np)
            msk  = jnp.asarray(msk_np)
        else:
            toks = jnp.asarray(rng.integers(0, model.vocab,
                                             size=(bs, seq_len), dtype=np.int32))
            msk  = jnp.ones_like(toks)
        img = jnp.zeros((bs, 3, img_size, img_size), dtype=jnp.float32)
        _, inter = model.apply(
            {"params": params}, toks, msk, img,
            mutable=["intermediates"])
        flat = _flatten_intermediates(inter["intermediates"])
        for key, val in flat.items():
            arr = np.asarray(val)
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            agg.setdefault(key, np.zeros_like(arr))
            agg[key] = agg[key] + arr
    for k in agg:
        agg[k] = agg[k] / n_batches
    return agg

# ---------------- expert pruning ----------------
def prune_experts(ckpt, keep_ratio, n_calib, bs):
    print(f"[prune_experts] keep_ratio={keep_ratio}")
    model = _build_model(ckpt, capture=True)
    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
    seq_len = ckpt["config"]["training"]["seq_len"]
    img_size = ckpt["config"]["image"]["size"]
    agg = _run_capture(model, params, seq_len, img_size, n_calib, bs)

    usage = {}
    for key, val in agg.items():
        if key.endswith("expert_counts"):
            li = _layer_idx_from_names(key.split("/"))
            if li is None:
                continue
            usage.setdefault(li, np.zeros_like(val))
            usage[li] = usage[li] + val

    if not usage:
        print("[prune_experts] no MoE usage found, skipping")
        return ckpt

    E_old = ckpt["config"]["moe"]["n_experts"]
    E_new = max(1, int(round(E_old * keep_ratio)))
    print(f"[prune_experts] usage captured for {len(usage)} layers")
    print(f"[prune_experts] E {E_old} -> {E_new}")

    keep_masks = {}
    for li, cnt in usage.items():
        order = np.argsort(-cnt)
        keep_idx = np.sort(order[:E_new])
        m = np.zeros(E_old, dtype=bool)
        m[keep_idx] = True
        keep_masks[li] = m

    def _transform(path, leaf):
        keys = _path_names(path)
        li = _layer_idx_from_names(keys)
        if li is None or li not in keep_masks:
            return leaf
        name = keys[-1]
        m = keep_masks[li]
        arr = jnp.asarray(leaf)
        if name in ("up", "down"):
            return arr[m]
        if name == "router":
            return arr[:, m]
        return leaf

    new_params = jax.tree_util.tree_map_with_path(_transform, ckpt["params"])

    out = dict(ckpt)
    out["params"] = new_params
    out["opt_state"] = None
    cfg = dict(ckpt["config"])
    cfg["moe"] = dict(cfg["moe"]); cfg["moe"]["n_experts"] = E_new
    out["config"] = cfg
    out["note"] = "expert-pruned; fine-tune before use"
    return out


# ---------------- layer pruning ----------------
def prune_layers(ckpt, keep_ratio):
    N_old = ckpt["config"]["model"]["n_layers"]
    N_new = max(1, int(round(N_old * keep_ratio)))
    print(f"[prune_layers] {N_old} -> {N_new}")

    new_params = {}
    for k, v in ckpt["params"].items():
        m = re.match(r"MambaBlock_(\d+)", k)
        if m:
            idx = int(m.group(1))
            if idx < N_new:
                new_params[k] = v
        else:
            new_params[k] = v

    out = dict(ckpt)
    out["params"] = new_params
    out["opt_state"] = None
    cfg = dict(ckpt["config"])
    cfg["model"] = dict(cfg["model"]); cfg["model"]["n_layers"] = N_new
    out["config"] = cfg
    out["note"] = "layer-pruned; fine-tune before using"
    return out


# ---------------- AWQ ----------------
def awq_calibrate(ckpt, n_batches, bs, alpha):
    print(f"[awq] calibrating {n_batches} batches of {bs}")
    model = _build_model(ckpt, capture=True)
    params = jax.tree_util.tree_map(jnp.asarray, ckpt["params"])
    seq_len = ckpt["config"]["training"]["seq_len"]
    img_size = ckpt["config"]["image"]["size"]
    agg = _run_capture(model, params, seq_len, img_size, n_batches, bs)

    # Map each captured activation magnitude to the weight tensor it
    # should scale (exact path matching — no substring heuristics).
    weight_scales = {}
    for key, val in agg.items():
        parts = key.split("/")
        name = parts[-1]
        base = "/".join(parts[:-1])
        mag = np.maximum(np.asarray(val, dtype=np.float32), 1e-6)
        s = mag ** alpha
        s = s / s.mean()
        if name == "act_mag":            # QDense input -> its kernel (in, out)
            weight_scales[base + "/kernel"] = s
        elif name == "moe_in_mag":       # MoE input -> expert 'up' (E, C, F)
            weight_scales[base + "/up"] = s
        elif name == "moe_h_mag":        # hidden -> expert 'down' (E, F, C)
            weight_scales[base + "/down"] = s

    print(f"[awq] computed {len(weight_scales)} scales")
    for k in list(weight_scales.keys())[:5]:
        print(f"       {k}")
    return weight_scales


def awq_apply(ckpt, weight_scales):
    def _transform(path, leaf):
        keys = _path_names(path)
        name = keys[-1]
        path_str = "/".join(keys)
        if name in ("kernel", "up", "down") and path_str in weight_scales:
            arr = jnp.asarray(leaf)
            s = jnp.asarray(weight_scales[path_str])
            if name == "kernel":            # (in, out) — scale input dim
                if arr.shape[0] == s.shape[0]:
                    return arr * s[:, None]
            else:                           # experts (E, in, out) — scale in
                if arr.shape[1] == s.shape[0]:
                    return arr * s[None, :, None]
        return leaf

    new_params = jax.tree_util.tree_map_with_path(_transform, ckpt["params"])
    n_hit = sum(1 for p in _iter_paths(ckpt["params"])
                if "/".join(_path_names(p)) in weight_scales)
    print(f"[awq] scaled {n_hit} weight tensors")

    out = dict(ckpt)
    out["params"] = new_params
    out["awq_scales"] = weight_scales
    out["note"] = "AWQ-scaled; divide activations by scale at runtime"
    return out


def _iter_paths(tree, prefix=()):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from _iter_paths(v, prefix + (k,))
    else:
        yield prefix


# ---------------- int4/int8 export ----------------
def _quantize_group(w32, group_size, bits):
    qmax = (1 << (bits - 1)) - 1
    orig = w32.shape
    D = orig[-1]
    assert D % group_size == 0, f"D={D} not divisible by {group_size}"
    G = D // group_size
    wg = w32.reshape(*orig[:-1], G, group_size)
    scale = np.max(np.abs(wg), axis=-1, keepdims=True) / qmax
    scale = np.maximum(scale, 1e-8)
    idx = np.clip(np.round(wg / scale), -qmax, qmax).astype(np.int8)
    idx = idx.reshape(orig)
    return idx, scale.astype(np.float32)


def _pack_int4(idx):
    idx = np.asarray(idx, dtype=np.int8)
    D = idx.shape[-1]
    assert D % 2 == 0
    u = idx.astype(np.uint8) & 0x0F
    hi = u[..., 0::2]; lo = u[..., 1::2]
    return ((hi << 4) | lo).astype(np.uint8)


def export_intN(ckpt, group_size, bits):
    params = ckpt["params"]
    bundle = {"quantized": {}, "unquantized": {}, "config": ckpt["config"],
              "bits": bits, "group_size": group_size}

    def _walk(path, leaf):
        keys = _path_names(path)
        name = keys[-1]
        path_str = "/".join(keys)
        if name in ("kernel", "up", "down"):
            arr = np.asarray(leaf, dtype=np.float32)
            D = arr.shape[-1]
            if D % group_size == 0:
                idx, scale = _quantize_group(arr, group_size, bits)
                entry = {"scale": scale, "shape": arr.shape,
                         "bits": bits, "group_size": group_size}
                if bits == 4:
                    entry["packed"] = _pack_int4(idx)
                else:
                    entry["idx"] = idx
                bundle["quantized"][path_str] = entry
                return
        bundle["unquantized"][path_str] = np.asarray(leaf)

    jax.tree_util.tree_map_with_path(_walk, params)
    if "awq_scales" in ckpt:
        bundle["awq_scales"] = ckpt["awq_scales"]
    return bundle


# ---------------- CLI ----------------
def _load_input(path=None):
    p = path or PP_INPUT
    if not os.path.exists(p):
        print(f"[error] checkpoint not found: {p}")
        sys.exit(1)
    print(f"[load] {p}")
    return load_checkpoint(p)


def _save(payload, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, path)
    print(f"[ckpt] saved {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


def postprocess_main(action, input_path, output_path):
    if action == "prune_experts":
        ckpt = _load_input(input_path)
        ckpt = prune_experts(ckpt, PP_E_KEEP, PP_AWQ_BATCHES, PP_AWQ_BS)
        _save(ckpt, output_path or os.path.join(PP_OUTPUT_DIR, "pruned_experts.pkl"))
    elif action == "prune_layers":
        ckpt = _load_input(input_path)
        ckpt = prune_layers(ckpt, PP_L_KEEP)
        _save(ckpt, output_path or os.path.join(PP_OUTPUT_DIR, "pruned_layers.pkl"))
    elif action == "awq":
        ckpt = _load_input(input_path)
        scales = awq_calibrate(ckpt, PP_AWQ_BATCHES, PP_AWQ_BS, PP_AWQ_ALPHA)
        ckpt = awq_apply(ckpt, scales)
        _save(ckpt, output_path or os.path.join(PP_OUTPUT_DIR, "awq.pkl"))
    elif action == "export":
        ckpt = _load_input(input_path)
        bundle = export_intN(ckpt, PP_EXPORT_GRP, PP_EXPORT_BITS)
        _save(bundle, output_path or os.path.join(PP_OUTPUT_DIR, "int_bundle.pkl"))
    elif action == "full":
        ckpt = _load_input(input_path)
        ckpt = prune_experts(ckpt, PP_E_KEEP, PP_AWQ_BATCHES, PP_AWQ_BS)
        _save(ckpt, os.path.join(PP_OUTPUT_DIR, "pruned_experts.pkl"))
        ckpt = prune_layers(ckpt, PP_L_KEEP)
        _save(ckpt, os.path.join(PP_OUTPUT_DIR, "pruned_layers.pkl"))
        scales = awq_calibrate(ckpt, PP_AWQ_BATCHES, PP_AWQ_BS, PP_AWQ_ALPHA)
        ckpt = awq_apply(ckpt, scales)
        _save(ckpt, os.path.join(PP_OUTPUT_DIR, "awq.pkl"))
        bundle = export_intN(ckpt, PP_EXPORT_GRP, PP_EXPORT_BITS)
        _save(bundle, os.path.join(PP_OUTPUT_DIR, "int_bundle.pkl"))
        print("[full] done")


# ================================================================
# ENTRY POINT
# ================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume", default=None, help="resume from checkpoint")
    ap.add_argument("--action", default="train",
                    choices=["train", "prune_experts", "prune_layers",
                             "awq", "export", "full"])
    ap.add_argument("--input", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    # config.yaml `resume:` is used when --resume is not given
    resume = args.resume or CFG_RESUME

    if args.action == "train":
        train_main(resume_path=resume)
    else:
        postprocess_main(args.action, args.input, args.output)


if __name__ == "__main__":
    main()
