import jax  # Bring in JAX, a super fast math and GPU calculation engine.
import jax.numpy as jnp  # Bring in JAX NumPy for handling arrays and matrices.
import flax.linen as nn  # Bring in Flax, a tool to build neural networks.
import optax  # Bring in Optax, a library for optimizer tools to update model weights.
from flax.training import train_state  # Bring in train_state to store weights and optimizer.

# --- FUNCTION TO SHRINK NUMBERS TO 8-BITS (INT8) ---
def quantize_int8(x):
    # Find the absolute maximum number in the array, divide by 127 to create a scaling ruler.
    scale = jnp.max(jnp.abs(x), axis=-1, keepdims=True) / 127.0
    # Prevent division by zero errors by ensuring scale is at least a tiny number.
    scale = jnp.maximum(scale, 1e-8)
    # Scale the original numbers down.
    x_scaled = x / scale
    # Round the scaled numbers and squeeze them into the integer range between -127 and 127.
    x_int8 = jnp.clip(jnp.round(x_scaled), -127, 127).astype(jnp.int8)
    # Un-shrink (dequantize) back to float numbers so the neural network can still calculate gradients.
    x_dequant = x_int8.astype(jnp.float32) * scale
    # Return the dequantized tensor (Straight-Through Estimator trick).
    return x_dequant

# --- CUSTOM INT8 DENSE LAYER ---
class INT8Dense(nn.Module):
    features: int  # Define output size dimension for this layer.

    @nn.compact
    def __call__(self, x):
        # Create a learnable weight matrix (kernel) using random initialization.
        kernel = self.param('kernel', nn.initializers.lecun_normal(), (x.shape[-1], self.features))
        # Create a learnable bias vector initialized to zeros.
        bias = self.param('bias', nn.initializers.zeros_init(), (self.features,))

        # Shrink the weight matrix and the input data down to INT8 on-the-fly.
        q_kernel = quantize_int8(kernel)
        q_x = quantize_int8(x)

        # Multiply the shrunken inputs by the shrunken weights and add the bias.
        return jnp.dot(q_x, q_kernel) + bias

# --- FULL MoE EMBEDDER MODEL WITH INT8 LAYERS ---
class FullMoEEmbedderINT8(nn.Module):
    vocab_size: int = 32000  # Total number of unique words in dictionary.
    hidden_dim: int = 2048   # Size/width of the vector for each token.
    max_len: int = 512       # Maximum length of text sequence allowed.
    num_experts: int = 16    # Total number of expert neural network pathways.
    top_k: int = 2           # Number of active experts chosen for each token.

    def setup(self):
        # Create a lookup table to convert token IDs into semantic meaning vectors.
        self.token_embed = nn.Embed(num_embeddings=self.vocab_size, features=self.hidden_dim)
        # Create a lookup table to give position order coordinates to tokens.
        self.pos_embed = nn.Embed(num_embeddings=self.max_len, features=self.hidden_dim)

        # Create a router gate layer using INT8 dense logic to score tokens across experts.
        self.gate = INT8Dense(self.num_experts)
        # Create a list of 16 parallel INT8 expert FFN layers that expand dimension 4x.
        self.experts = [INT8Dense(self.hidden_dim * 4) for _ in range(self.num_experts)]

    def __call__(self, input_ids):
        # Extract batch size and sequence length dimensions from incoming token IDs.
        batch_size, seq_len = input_ids.shape
        # Create position index numbers [0, 1, 2, ..., seq_len-1] for the sequence.
        positions = jnp.arange(seq_len)[None, :]

        # Combine the word meaning vectors and position coordinate vectors together.
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        # Pass combined tensor through router gate to get routing score logits.
        logits = self.gate(x)
        # Apply softmax and find the top-2 highest scoring expert indices and weights.
        weights, indices = jax.lax.top_k(nn.softmax(logits), self.top_k)
        # Create an empty zero tensor to accumulate outputs from active experts.
        out = jnp.zeros_like(x)

        # Loop through all 16 expert networks.
        for i, expert in enumerate(self.experts):
            # Make a boolean mask checking if expert 'i' is selected for any token.
            mask = jnp.any(indices == i, axis=-1, keepdims=True)
            # Run token through expert, apply SiLU activation, multiply by mask, and add to output.
            out = out + mask * nn.silu(expert(x))

        # Average pool all token vectors across the sequence length into one final text embedding vector.
        emb = jnp.mean(out, axis=1)
        # Return the final dense embedding vector.
        return emb

# --- KAGGLE EXECUTION SETUP ---
model = FullMoEEmbedderINT8()  # Instantiate the model architecture object.
key = jax.random.PRNGKey(42)   # Generate a base random seed key for reproducibility.
dummy_ids = jax.randint(key, (8, 64), 0, 32000)  # Create fake batch of token IDs for testing.
params = model.init(key, dummy_ids)  # Initialize all model weights parameters.

tx = optax.adamw(learning_rate=1e-4)  # Configure AdamW optimizer with a learning rate of 0.0001.
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)  # Package state container.

# --- LOSS FUNCTION ---
def contrastive_loss(params, q_ids, p_ids):
    # Run query token IDs through model to get query embedding vectors.
    q_emb = model.apply(params, q_ids)
    # Run passage token IDs through model to get passage embedding vectors.
    p_emb = model.apply(params, p_ids)

    # L2 normalize query embedding vector so it sits on the surface of a unit hypersphere.
    q_emb = q_emb / jnp.linalg.norm(q_emb, axis=-1, keepdims=True)
    # L2 normalize passage embedding vector so it sits on the surface of a unit hypersphere.
    p_emb = p_emb / jnp.linalg.norm(p_emb, axis=-1, keepdims=True)

    # Compute dot product cosine similarity between queries and passages, scaled by temperature 0.02.
    logits = jnp.sum(q_emb * p_emb, axis=-1) / 0.02
    # Calculate and return InfoNCE cross-entropy loss error penalty.
    return -jnp.mean(jax.nn.log_softmax(logits))

# --- JIT GPU COMPILATION ---
@jax.jit  # Compile function into fast GPU machine code binary.
def train_step(state, q_ids, p_ids):
    # Simultaneously compute loss value and parameter gradients (slopes/derivatives).
    loss, grads = jax.value_and_grad(contrastive_loss)(state.params, q_ids, p_ids)
    # Apply gradient updates to weights, return new training state and current loss value.
    return state.apply_gradients(grads=grads), loss

# --- TRAINING LOOP ---
for epoch in range(5):  # Run training loop for 5 epochs.
    k1, k2 = jax.random.split(jax.random.PRNGKey(epoch))  # Split unique random keys per epoch.
    q = jax.randint(k1, (16, 64), 0, 32000)  # Generate batch of mock query token IDs.
    p = jax.randint(k2, (16, 64), 0, 32000)  # Generate batch of mock passage token IDs.
    state, loss = train_step(state, q, p)  # Execute compiled training step on GPU.
    print(f"Epoch {epoch} | INT8 Loss: {loss:.4f}")  # Print current training loss value.
