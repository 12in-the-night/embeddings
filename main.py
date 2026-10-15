"""
GPU-only. Pure Mamba-2 + MoE (E=8, top-2) + INT8 QAT.
d_model=768, 24 layers, S=1024, Mistral 32k vocab, tied embeddings.
Target: 2x T4 (16 GB each), fp16, dynamic loss scale, Adafactor, remat.

Run:    python train.py
Resume: python train.py --resume ./ckpts/ckpt_final.pkl
Smoke:  GPT_STEPS=20 GPT_LOG_EVERY=1 GPT_D_MODEL=128 GPT_LAYERS=4 \
        GPT_SEQ_LEN=256 GPT_BATCH=2 GPT_PREC=fp16 python train.py
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

import sys, glob, math, time, queue, pickle, threading, functools
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax.jax_utils import replicate, unreplicate
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.95")

# ---------------------------------------------------------------- HF auth
try:
    from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("hg")
except Exception:
    pass

import sys, glob, math, time, queue, pickle, threading, functools
import numpy as np
import jax, jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state
from flax.jax_utils import replicate, unreplicate

# ---------------------------------------------------------------- config
_e = lambda k, d: type(d)(os.environ.get(k, d))
SEQ_LEN       = _e("GPT_SEQ_LEN", 1024)
GLOBAL_BATCH  = _e("GPT_BATCH", 4)
STEPS         = _e("GPT_STEPS", 40000)
LR            = _e("GPT_LR", 3e-4)
WARMUP        = min(1000, max(STEPS // 20, 1))
MIN_LR_FRAC   = 0.1
CACHE_DIR     = os.environ.get("GPT_CACHE", "./gpt_cache")
CKPT_DIR      = os.environ.get("GPT_CKPT", "./ckpts")
TOKENS_GOAL   = _e("GPT_TOKENS_GOAL", 200_000_000)
PREFETCH      = 4
CKPT_EVERY    = _e("GPT_CKPT_EVERY", 1000)
LOG_EVERY     = _e("GPT_LOG_EVERY", 50)

D_MODEL    = _e("GPT_D_MODEL", 768)
N_LAYERS   = _e("GPT_LAYERS", 24)
N_EXPERTS  = 8
TOP_K      = 2
EXPERT_FFN_MULT = 4

MAMBA_D_STATE  = _e("GPT_MAMBA_D_STATE", 128)
MAMBA_D_CONV   = _e("GPT_MAMBA_D_CONV", 4)
MAMBA_CHUNK    = _e("GPT_MAMBA_CHUNK", 256)
MAMBA_EXPAND   = 2
MAMBA_HEAD_DIM = _e("GPT_MAMBA_HEAD_DIM", 64)

REMAT = _e("GPT_REMAT", 1) == 1

AUX_LB_COEF = 0.01
AUX_Z_COEF  = 1e-3
CLIP_NORM   = 1.0

# ---------------------------------------------------------------- precision
PRECISION = os.environ.get("GPT_PREC", "fp16").lower()
if PRECISION == "bf16":
    COMPUTE_DTYPE = jnp.bfloat16
    LOSS_SCALE = 1.0
    DYNAMIC_SCALE = False
elif PRECISION == "fp16":
    COMPUTE_DTYPE = jnp.float16
    LOSS_SCALE = 2.0 ** 10
    DYNAMIC_SCALE = True
else:
    COMPUTE_DTYPE = jnp.float32
    LOSS_SCALE = 1.0
    DYNAMIC_SCALE = False
MASTER_DTYPE = jnp.float32

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ---------------------------------------------------------------- tokenizer
from transformers import AutoTokenizer
TOK_MODEL = os.environ.get("GPT_TOK", "mistralai/Mistral-7B-v0.1")
tok = AutoTokenizer.from_pretrained(TOK_MODEL)
VOCAB = tok.vocab_size
EOT = tok.eos_token_id

def _encode(text):
    return tok.encode(text, add_special_tokens=False)

def _decode(ids):
    return tok.decode(ids, skip_special_tokens=True)

# ---------------------------------------------------------------- devices
devices = jax.devices()
N_DEV   = len(devices)
print(f"devices: {N_DEV}  {devices}")
assert GLOBAL_BATCH % N_DEV == 0, \
    f"GLOBAL_BATCH={GLOBAL_BATCH} not divisible by N_DEV={N_DEV}"
PER_DEV_BATCH = GLOBAL_BATCH // N_DEV
print(f"vocab={VOCAB}  ctx={SEQ_LEN}  layers={N_LAYERS}  "
      f"mamba d_state={MAMBA_D_STATE} chunk={MAMBA_CHUNK}  "
      f"B/device={PER_DEV_BATCH}  prec={PRECISION}  remat={REMAT}  "
      f"loss_scale={LOSS_SCALE}  dynamic={DYNAMIC_SCALE}")

# ---------------------------------------------------------------- data
def _stream_text():
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-350BT",
                      split="train", streaming=True)
    ds = ds.shuffle(seed=42, buffer_size=10_000)
    for ex in ds:
        t = ex["text"]
        if t:
            yield t
def build_cache(cache_dir, tokens_goal):
    done = os.path.join(cache_dir, "DONE")
    if os.path.exists(done):
        return
    print(f"[data] tokenizing ~{tokens_goal/1e6:.0f}M tokens -> {cache_dir}")
    buf = np.empty(1 << 20, dtype=np.uint16)
    fill = shard_id = total = 0
    t0 = time.time()
    for text in _stream_text():
        ids = np.asarray(_encode(text) + [EOT], dtype=np.uint16)
        i = 0
        while i < len(ids):
            n = min(len(buf) - fill, len(ids) - i)
            buf[fill:fill + n] = ids[i:i + n]
            fill += n; i += n; total += n
            if fill == len(buf):
                buf.tofile(os.path.join(cache_dir, f"shard_{shard_id:05d}.bin"))
                shard_id += 1; fill = 0
                if shard_id % 16 == 0:
                    rate = total / max(time.time() - t0, 1e-6)
                    print(f"[data] {total/1e6:7.1f}M tok ({rate/1e3:6.1f}k tok/s)")
        if total >= tokens_goal:
            break
    if fill:
        buf[:fill].tofile(os.path.join(cache_dir, f"shard_{shard_id:05d}.bin"))
        shard_id += 1
    with open(done, "w") as f:
        f.write(f"{total}\n{shard_id}\n")
    print(f"[data] done: {total/1e6:.1f}M tokens, {shard_id} shards")

def load_cache(cache_dir):
    with open(os.path.join(cache_dir, "DONE")) as f:
        total, n_shards = f.read().split("\n")[:2]
    total, n_shards = int(total), int(n_shards)
    shards = [np.memmap(os.path.join(cache_dir, f"shard_{i:05d}.bin"),
                        dtype=np.uint16, mode="r") for i in range(n_shards)]
    return shards, total

def batch_worker(shards, total_tokens, gbs, seq_len, steps, out_q, seed, start=0):
    rng = np.random.default_rng(seed + start)
    starts = rng.integers(0, max(total_tokens - seq_len - 1, 1),
                          size=(steps, gbs))
    bounds = np.cumsum([0] + [s.shape[0] for s in shards])
    for step in range(steps):
        out = np.empty((gbs, seq_len + 1), dtype=np.int32)
        for b in range(gbs):
            p = starts[step, b]
            si = np.searchsorted(bounds, p, side="right") - 1
            off = p - bounds[si]
            if off + seq_len + 1 <= shards[si].shape[0]:
                out[b] = shards[si][off:off + seq_len + 1]
            else:
                tail = shards[si][off:]
                need = seq_len + 1 - len(tail)
                si2 = (si + 1) % len(shards)
                out[b, :len(tail)] = tail
                out[b, len(tail):] = shards[si2][:need]
        out_q.put(out)
    out_q.put(None)

# ---------------------------------------------------------------- int8 QAT
QMAX = 127
def fake_quant_int8(x, axis=-1, eps=1e-8):
    x32 = x.astype(jnp.float32)
    scale = jnp.max(jnp.abs(x32), axis=axis, keepdims=True) + eps
    idx = jnp.clip(jnp.round(x32 / scale * QMAX), -QMAX, QMAX)
    xq = (idx * (scale / QMAX)).astype(x.dtype)
    return x + jax.lax.stop_gradient(xq - x)

# ---------------------------------------------------------------- layers
class QDense(nn.Module):
    features: int
    use_bias: bool = False
    dtype: jnp.dtype = COMPUTE_DTYPE
    param_dtype: jnp.dtype = MASTER_DTYPE

    @nn.compact
    def __call__(self, x):
        k = self.param("kernel", nn.initializers.lecun_normal(),
                       (x.shape[-1], self.features))
        k_q = fake_quant_int8(k, axis=0).astype(self.dtype)
        y = jnp.matmul(x.astype(self.dtype), k_q,
                       preferred_element_type=jnp.float32)
        if self.use_bias:
            b = self.param("bias", nn.initializers.zeros, (self.features,))
            y = y + b
        return y.astype(self.dtype)

class MoEMLP(nn.Module):
    d_model: int
    d_ff: int
    n_experts: int = N_EXPERTS
    top_k: int = TOP_K
    dtype: jnp.dtype = COMPUTE_DTYPE
    param_dtype: jnp.dtype = MASTER_DTYPE

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        N = B * T
        E, K = self.n_experts, self.top_k
        x_flat = x.reshape(N, C)

        router = self.param("router", nn.initializers.normal(0.02), (C, E))
        gate_logits = x_flat.astype(jnp.float32) @ router.astype(jnp.float32)
        gates = jax.nn.softmax(gate_logits, axis=-1)
        topk_w, topk_i = jax.lax.top_k(gates, K)
        topk_w = topk_w / (topk_w.sum(-1, keepdims=True) + 1e-9)

        expert_init = nn.initializers.lecun_normal(
            in_axis=1, out_axis=2, batch_axis=(0,))
        up   = self.param("up",   expert_init, (E, C, self.d_ff))
        down = self.param("down", expert_init, (E, self.d_ff, C))
        up_q   = fake_quant_int8(up,   axis=1).astype(self.dtype)
        down_q = fake_quant_int8(down, axis=1).astype(self.dtype)

        idx_flat = topk_i.reshape(-1)
        x_rep    = jnp.repeat(x_flat, K, axis=0)
        order    = jnp.argsort(idx_flat)
        inv      = jnp.argsort(order)
        x_sorted = x_rep[order].astype(self.dtype)
        i_sorted = idx_flat[order]
        group_sizes = jnp.bincount(i_sorted, length=E).astype(jnp.int32)

        h = jax.lax.ragged_dot(x_sorted, up_q, group_sizes)
        h = nn.gelu(h)
        y = jax.lax.ragged_dot(h, down_q, group_sizes)
        y = y[inv].reshape(N, K, C)
        y = (y * topk_w[..., None].astype(y.dtype)).sum(axis=1).reshape(B, T, C)

        f_i = jnp.bincount(i_sorted, length=E).astype(jnp.float32) / (N * K)
        P_i = gates.mean(axis=0)
        aux_lb = E * jnp.sum(f_i * P_i)
        z = jax.nn.logsumexp(gate_logits, axis=-1)
        aux_z = jnp.mean(z ** 2)
        return y, AUX_LB_COEF * aux_lb + AUX_Z_COEF * aux_z

# ---------------------------------------------------------------- Mamba-2 SSD
def _pad_seq_dim(x, pad_size):
    if pad_size == 0:
        return x
    pw = [(0, 0)] * x.ndim
    pw[1] = (0, pad_size)
    return jnp.pad(x, pw, mode="constant", constant_values=0.0)

def segsum(x):
    T = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((T, T), dtype=bool), k=0)
    return jnp.where(mask, x_segsum, -jnp.inf)

def ssd_forward(x, dt, A, B_mat, C_mat, chunk_size, D, dt_bias,
                dt_min, dt_max):
    """x (B,L,H,P), dt (B,L,H), A (H,), B_mat (B,L,H,N), C_mat (B,L,H,N), D (H,)."""
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
        Bc = B32[:, ci]
        Cc = C32[:, ci]
        xd = x_disc[:, ci]
        Acs = A_cumsum[:, ci]
        dec = chunk_decay[:, ci]

        y_inter = jnp.einsum("blhn,bhpn->blhp", Cc, state)
        y_inter = y_inter * jnp.exp(Acs).transpose(0, 2, 1)[..., None]

        decay = jnp.exp(Acs[..., -1:] - Acs)
        chunk_state = jnp.einsum("blhn,bhl,blhp->bhpn", Bc, decay, xd)
        state = dec[..., None, None] * state + chunk_state
        y_inter_list.append(y_inter)

    y_inter = jnp.stack(y_inter_list, axis=1)
    y_full = y_c + y_inter

    D_res = D.astype(jnp.float32).reshape(1, 1, 1, num_heads, 1) * \
            x_c.astype(jnp.float32)
    y_full = y_full + D_res

    y_full = y_full.reshape(B_size, L_p, num_heads, head_dim)
    if pad_size:
        y_full = y_full[:, :seq_len]
    return y_full.astype(x.dtype)

def ssd_naive(x, dt, A, B, C, D, dt_bias, dt_min, dt_max):
    dt = jnp.clip(jax.nn.softplus(dt + dt_bias), dt_min, dt_max)
    Bsz, L, H, P = x.shape
    N = B.shape[-1]
    def step(s, inp):
        x_t, dt_t, B_t, C_t = inp
        decay = jnp.exp(A[None, :, None, None] * dt_t[..., None, None])
        s = decay * s + (dt_t[..., None] * x_t)[..., None] * B_t[:, :, None, :]
        y = jnp.einsum("bhpn,bhn->bhp", s, C_t) + D[None, :, None] * x_t
        return s, y
    s0 = jnp.zeros((Bsz, H, P, N), jnp.float32)
    xs = tuple(jnp.moveaxis(a, 1, 0) for a in
               (x.astype(jnp.float32), dt.astype(jnp.float32),
                B.astype(jnp.float32), C.astype(jnp.float32)))
    _, ys = jax.lax.scan(step, s0, xs)
    return jnp.moveaxis(ys, 0, 1)

# ---------------------------------------------------------------- Mamba-2 block
class Mamba2Block(nn.Module):
    d_model: int
    d_state: int = MAMBA_D_STATE
    d_conv: int = MAMBA_D_CONV
    chunk: int = MAMBA_CHUNK
    expand: int = MAMBA_EXPAND
    head_dim: int = MAMBA_HEAD_DIM
    dtype: jnp.dtype = COMPUTE_DTYPE
    param_dtype: jnp.dtype = MASTER_DTYPE

    @nn.compact
    def __call__(self, x):
        B, T, C = x.shape
        d_inner = self.expand * C
        H = d_inner // self.head_dim
        P = self.head_dim
        N = self.d_state

        h = nn.LayerNorm(dtype=jnp.float32)(
            x.astype(jnp.float32)).astype(self.dtype)

        proj_size = 2 * d_inner + 2 * N + H
        zxbcdt = QDense(proj_size, use_bias=False)(h)

        z, xBC, dt = jnp.split(zxbcdt,
                               [d_inner, 2 * d_inner + 2 * N], axis=-1)

        xBC = nn.Conv(features=xBC.shape[-1],
                      kernel_size=(self.d_conv,),
                      padding="CAUSAL",
                      feature_group_count=xBC.shape[-1],
                      use_bias=True,
                      dtype=self.dtype,
                      param_dtype=self.param_dtype)(xBC)
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
        return QDense(C, use_bias=False)(y)

# ---------------------------------------------------------------- Block
class MambaBlock(nn.Module):
    d_model: int
    layer_idx: int = 0
    remat: bool = False

    @nn.compact
    def __call__(self, x):
        Mamba2BlockR = nn.remat(Mamba2Block) if self.remat else Mamba2Block
        h = Mamba2BlockR(self.d_model)(x)
        x = x + h

        aux = jnp.zeros((), dtype=jnp.float32)
        if self.layer_idx % 2 == 1:
            h_ln = nn.LayerNorm(dtype=jnp.float32)(
                x.astype(jnp.float32)).astype(x.dtype)
            MoEMLPR = nn.remat(MoEMLP) if self.remat else MoEMLP
            h_moe, aux = MoEMLPR(self.d_model,
                                 EXPERT_FFN_MULT * self.d_model)(h_ln)
            x = x + h_moe
        return x, aux

class GPT(nn.Module):
    vocab: int
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    max_len: int = SEQ_LEN
    tie_embeddings: bool = True
    remat: bool = False
    dtype: jnp.dtype = COMPUTE_DTYPE
    param_dtype: jnp.dtype = MASTER_DTYPE

    @nn.compact
    def __call__(self, tokens):
        B, T = tokens.shape
        emb = nn.Embed(self.vocab, self.d_model,
                       dtype=self.dtype, param_dtype=self.param_dtype,
                       embedding_init=nn.initializers.normal(0.02))
        x = emb(tokens).astype(jnp.float32)

        aux_total = jnp.zeros((), dtype=jnp.float32)
        for li in range(self.n_layers):
            x, aux = MambaBlock(self.d_model, layer_idx=li,
                                remat=self.remat)(x)
            aux_total = aux_total + aux

        x = nn.LayerNorm(dtype=jnp.float32)(
            x.astype(jnp.float32)).astype(x.dtype)

        if self.tie_embeddings:
            w = emb.variables["params"]["embedding"]
            w_q = fake_quant_int8(w, axis=1).astype(self.dtype)
            logits = jnp.matmul(x.astype(self.dtype), w_q.T,
                                preferred_element_type=jnp.float32)
        else:
            logits = QDense(self.vocab, use_bias=False)(x).astype(jnp.float32)
        return logits.astype(jnp.float32), aux_total

# ---------------------------------------------------------------- weight decay mask
def _wd_mask(params):
    def _keep(path, leaf):
        keys = [str(getattr(k, "key", k)) for k in path]
        if any("LayerNorm" in k for k in keys):
            return False
        return keys[-1] not in ("bias", "scale", "A_log", "D", "dt_bias", "embedding")
    return jax.tree_util.tree_map_with_path(_keep, params)

# ---------------------------------------------------------------- train
def make_state(rng, model, seq_len):
    dummy = jnp.zeros((1, seq_len), dtype=jnp.int32)
    params = model.init(rng, dummy)["params"]
    sched = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=LR,
        warmup_steps=WARMUP, decay_steps=STEPS, end_value=LR * MIN_LR_FRAC)
    tx = optax.chain(
        optax.clip_by_global_norm(CLIP_NORM),
        optax.adafactor(learning_rate=None,
                        multiply_by_parameter_scale=False),
        optax.add_decayed_weights(-0.01, _wd_mask(params)),
        optax.scale_by_learning_rate(sched, flip_sign=False),
    )
    tx = optax.MultiSteps(tx, every_k_schedule=8)
    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params, tx=tx)
def _init_dyn_state(scale):
    return (jnp.float32(scale), jnp.int32(0), jnp.int32(0))

def _update_dyn_state(dyn_state, grads, growth_interval=2000,
                      growth=2.0, backoff=0.5,
                      min_scale=1.0, max_scale=2.0**24):
    scale, counter, valid = dyn_state
    has_inf = jnp.any(jnp.stack([
        jnp.any(~jnp.isfinite(g))
        for g in jax.tree_util.tree_leaves(grads)
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
def train_step(state, batch, dyn_state):
    scale = dyn_state[0]

    def loss_fn(params):
        inp, tgt = batch[:, :-1], batch[:, 1:]
        logits, aux = state.apply_fn({"params": params}, inp)
        nll = optax.softmax_cross_entropy_with_integer_labels(logits, tgt).mean()
        return (nll + aux) * scale, (nll, aux)

    (_, (nll, aux)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grads = jax.lax.pmean(grads, axis_name="dev")

    if DYNAMIC_SCALE:
        dyn_state, has_inf = _update_dyn_state(dyn_state, grads)
        grads = jax.tree_util.tree_map(
            lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads)
    else:
        has_inf = jnp.array(False, dtype=jnp.bool_)

    grads = jax.tree_util.tree_map(lambda g: g / scale, grads)
    new_state = state.apply_gradients(grads=grads)

    state = jax.tree_util.tree_map(
        lambda n, o: jnp.where(has_inf, o, n), new_state, state)

    return (state, dyn_state,
            jax.lax.pmean(nll, "dev"),
            jax.lax.pmean(aux, "dev"),
            has_inf.astype(jnp.float32))

# ---------------------------------------------------------------- checkpoint
def save_checkpoint(state, step, path, dyn_state=None):
    host = unreplicate(state)
    to_np = lambda x: np.asarray(x)
    payload = {
        "step": step,
        "params": jax.tree_util.tree_map(to_np, host.params),
        "opt_state": jax.tree_util.tree_map(to_np, host.opt_state),
        "config": {
            "arch": "mamba2", "d_model": D_MODEL, "n_layers": N_LAYERS,
            "n_experts": N_EXPERTS, "top_k": TOP_K,
            "expert_ffn_mult": EXPERT_FFN_MULT, "vocab": VOCAB,
            "seq_len": SEQ_LEN, "mamba_d_state": MAMBA_D_STATE,
            "mamba_chunk": MAMBA_CHUNK, "mamba_expand": MAMBA_EXPAND,
            "mamba_head_dim": MAMBA_HEAD_DIM, "n_dev": N_DEV,
            "remat": REMAT,
        },
    }
    if dyn_state is not None:
        ds_host = unreplicate(dyn_state)
        payload["dyn_state"] = (
            float(np.asarray(ds_host[0])),
            int(np.asarray(ds_host[1])),
            int(np.asarray(ds_host[2])),
        )
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    os.replace(tmp, path)
    print(f"[ckpt] saved {path}  ({os.path.getsize(path)/1e6:.1f} MB)")
    return path

def load_checkpoint(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# ---------------------------------------------------------------- sanity
def sanity():
    print(f"[sanity] arch=mamba2 prec={PRECISION} E={N_EXPERTS} k={TOP_K} "
          f"chunk={MAMBA_CHUNK} loss_scale={LOSS_SCALE} dev={N_DEV} remat={REMAT}")

    x = jnp.array([0.1, 0.7, -1.3, 2.4, 0.0, -0.05], dtype=jnp.float32)
    g = jax.grad(lambda v: fake_quant_int8(v).sum())(x)
    assert jnp.allclose(g, 1.0), ("STE gradient wrong", g)
    q1 = fake_quant_int8(x); q2 = fake_quant_int8(q1)
    assert jnp.allclose(q1, q2, atol=1e-6), ("not idempotent",)

    Bsz, L, H, P, N = 2, 200, 4, 8, 16
    chunk = 64
    ks = jax.random.split(jax.random.PRNGKey(0), 7)
    x_r  = jax.random.normal(ks[0], (Bsz, L, H, P))
    dt_r = jax.random.uniform(ks[1], (Bsz, L, H), minval=-2, maxval=2)
    A_r  = -jnp.exp(jax.random.normal(ks[2], (H,)))
    B_r  = jax.random.normal(ks[3], (Bsz, L, H, N))
    C_r  = jax.random.normal(ks[4], (Bsz, L, H, N))
    D_r  = jax.random.normal(ks[5], (H,))
    b_r  = jax.random.normal(ks[6], (H,))
    y_fast = ssd_forward(x_r, dt_r, A_r, B_r, C_r, chunk, D_r, b_r, 0.0, 100.0)
    y_ref  = ssd_naive  (x_r, dt_r, A_r, B_r, C_r, D_r, b_r, 0.0, 100.0)
    err = float(jnp.max(jnp.abs(y_fast - y_ref)))
    print(f"[sanity] ssd wide-dt max|Δ| = {err:.3e}")
    assert err < 1e-4, f"SSD mismatch: {err:.3e}"

    m = Mamba2Block(d_model=64, d_state=16, chunk=32, head_dim=16)
    dummy = jnp.zeros((2, 128, 64), dtype=jnp.float32)
    p = m.init(jax.random.PRNGKey(0), dummy)["params"]
    y = m.apply({"params": p}, dummy)
    assert y.shape == dummy.shape, (y.shape, dummy.shape)
    print("[sanity] OK")

# ---------------------------------------------------------------- sample
def sample(params, prompt="The future of computing is ", n_new=200, temp=0.8):
    ids = list(_encode(prompt))
    model_eval = GPT(vocab=VOCAB, remat=REMAT)
    fwd = jax.jit(lambda p, t: model_eval.apply({"params": p}, t))
    for _ in range(n_new):
        window = ids[-SEQ_LEN:]
        arr = np.zeros((1, SEQ_LEN), dtype=np.int32)
        arr[0, :len(window)] = window
        logits, _ = fwd(params, jnp.asarray(arr))
        l = np.asarray(logits[0, len(window) - 1], dtype=np.float64) / temp
        p = np.exp(l - l.max()); p /= p.sum()
        ids.append(int(np.random.choice(len(p), p=p)))
    return _decode(ids)

# ---------------------------------------------------------------- main
def main():
    resume_path = None
    if "--resume" in sys.argv:
        resume_path = sys.argv[sys.argv.index("--resume") + 1]

    sanity()
    build_cache(CACHE_DIR, TOKENS_GOAL)
    shards, total_tokens = load_cache(CACHE_DIR)
    print(f"[data] {total_tokens/1e6:.1f}M tokens in {len(shards)} shards")

    model = GPT(vocab=VOCAB, d_model=D_MODEL, n_layers=N_LAYERS,
                max_len=SEQ_LEN, remat=REMAT)
    state = make_state(jax.random.PRNGKey(0), model, SEQ_LEN)

    dyn_state_local = _init_dyn_state(LOSS_SCALE)
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
            dyn_state_local = (jnp.float32(ds[0]), jnp.int32(ds[1]), jnp.int32(ds[2]))
            print(f"[ckpt] restored dyn_state: scale={ds[0]:.0f} counter={ds[1]} valid={ds[2]}")
        print(f"[ckpt] resumed {resume_path} @ step {start_step}")

    n_params = sum(int(x.size) for x in jax.tree_util.tree_leaves(state.params))
    print(f"[model] arch=mamba2 total={n_params/1e6:.1f}M | d_model={D_MODEL} "
          f"L={N_LAYERS} E={N_EXPERTS} k={TOP_K} "
          f"S={SEQ_LEN} B={GLOBAL_BATCH} B/dev={PER_DEV_BATCH} prec={PRECISION} "
          f"tied=True vocab={VOCAB} dev={N_DEV} remat={REMAT}")

    state = replicate(state)
    dyn_state = replicate(dyn_state_local)

    q = queue.Queue(maxsize=PREFETCH)
    threading.Thread(target=batch_worker,
                     args=(shards, total_tokens, GLOBAL_BATCH, SEQ_LEN,
                           STEPS - start_step, q, 0, start_step),
                     daemon=True).start()

    t0 = time.time()
    inf_acc = jnp.zeros((N_DEV,), jnp.float32)
    inf_at_last_log = 0

    for local_step in range(STEPS - start_step):
        step = start_step + local_step
        np_batch = q.get()
        if np_batch is None:
            break
        batch = jnp.asarray(np_batch).reshape(
            N_DEV, PER_DEV_BATCH, SEQ_LEN + 1)
        state, dyn_state, nll, aux, has_inf = train_step(state, batch, dyn_state)

        inf_acc = inf_acc + has_inf

        if step % LOG_EVERY == 0 or step == STEPS - 1:
            dt = time.time() - t0
            tok_seen = (local_step + 1) * GLOBAL_BATCH * SEQ_LEN
            nll_f = float(unreplicate(nll).mean())
            aux_f = float(unreplicate(aux).mean())
            sc = float(unreplicate(dyn_state[0]))
            inf_events = int(inf_acc[0])
            since = inf_events - inf_at_last_log
            inf_at_last_log = inf_events
            print(f"step {step:6d} | nll {nll_f:.4f} "
                  f"| aux {aux_f:.4f} "
                  f"| ppl {math.exp(min(nll_f, 20)):7.2f} "
                  f"| scale {sc:.0f} inf {since}/{inf_events} "
                  f"| {dt:6.1f}s | {tok_seen/1e6:6.1f}M tok")

        if step > 0 and step % CKPT_EVERY == 0:
            save_checkpoint(state, step + 1,
                            os.path.join(CKPT_DIR, f"ckpt_{step:06d}.pkl"),
                            dyn_state=dyn_state)
            for old in sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_0*.pkl")))[:-3]:
                os.remove(old)

    save_checkpoint(state, STEPS, os.path.join(CKPT_DIR, "ckpt_final.pkl"),
                    dyn_state=dyn_state)
    host = unreplicate(state)
    print("\n--- sample ---\n" + sample(host.params))

if __name__ == "__main__":
    main()
