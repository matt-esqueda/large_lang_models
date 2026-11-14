"""
GPT Language Model Architecture
Shared model components for training and inference.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    One linear layer projects queries, keys and values for every head at
    once, and F.scaled_dot_product_attention applies the causal mask,
    softmax and dropout in a single fused kernel.
    """

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head = n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_dropout if self.training else 0.0, is_causal=True
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble the heads
        return self.resid_dropout(self.proj(y))


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(ln1(x)), then x + ffwd(ln2(x))."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.sa = CausalSelfAttention(n_embd, n_head, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    """GPT Language Model"""

    def __init__(self, vocab_size, n_embd=384, n_head=6, n_layer=6, block_size=64, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.config = {
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "n_head": n_head,
            "n_layer": n_layer,
            "block_size": block_size,
            "dropout": dropout,
        }

        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Tied embeddings: the output layer reuses the input embedding matrix,
        # so a 50K-token vocabulary is stored once rather than twice.
        self.lm_head.weight = self.token_embedding_table.weight

        self.apply(self._init_weights)
        # GPT-2: projections that write into the residual stream start smaller,
        # by 1/sqrt(2 * n_layer), so the stream's variance doesn't grow with depth.
        for name, param in self.named_parameters():
            if name.endswith(("sa.proj.weight", "ffwd.net.2.weight")):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, index, targets=None):
        B, T = index.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(index)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=index.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate_stream(
        self, index, max_new_tokens, temperature=1.0, top_k=None, top_p=None, repetition_penalty=1.0
    ):
        """Yield each new token as a (B, 1) tensor.

        Order: repetition penalty, then either greedy argmax (temperature 0)
        or temperature scaling, top-k, top-p and sampling. generate() is
        built on this method, so streamed and collected output are identical
        for the same random state.

        Args:
            index: (B, T) tensor of token ids for the context
            max_new_tokens: number of tokens to generate
            temperature: 0.0 = greedy, 1.0 = unchanged, >1.0 = more random
            top_k: if set, sample only from the k most likely tokens
            top_p: if set, sample from the smallest set with cumulative probability >= p
            repetition_penalty: >1.0 discourages tokens already in the sequence
        """
        for _ in range(max_new_tokens):
            logits, _ = self(index[:, -self.block_size :])
            logits = logits[:, -1, :]  # (B, C)
            if repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(logits, index, repetition_penalty)
            if temperature == 0.0:
                index_next = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    logits = self._top_k_filtering(logits, top_k)
                if top_p is not None:
                    logits = self._top_p_filtering(logits, top_p)
                index_next = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            index = torch.cat((index, index_next), dim=1)
            yield index_next

    def generate(self, index, max_new_tokens, **sampling):
        """Return index extended by max_new_tokens tokens: (B, T + max_new_tokens).

        Takes the same sampling arguments as generate_stream.
        """
        for index_next in self.generate_stream(index, max_new_tokens, **sampling):
            index = torch.cat((index, index_next), dim=1)
        return index

    def _apply_repetition_penalty(self, logits, previous_tokens, penalty):
        """
        Apply repetition penalty to logits

        Args:
            logits: (B, C) unnormalized log probabilities
            previous_tokens: (B, T) previously generated tokens
            penalty: repitition penalty factor (>1.0 to discourage repetition)

        Returns:
            Modified logits with repetition penalty applied
        """
        batch_size, vocab_size = logits.shape

        for i in range(batch_size):
            # Get unique tokens in the sequence
            unique_tokens = torch.unique(previous_tokens[i])

            # Apply penalty to previously seen tokens
            for token in unique_tokens:
                # If logit is positive, divide by penalty; if negative, multiply by penalty
                if logits[i, token] > 0:
                    logits[i, token] /= penalty
                else:
                    logits[i, token] *= penalty

        return logits

    def _top_k_filtering(self, logits, top_k):
        """
        Filter logits to only keep top k tokens

        Args:
            logits: (B, C) unormalized log probabilities
            top_k: number of top tokens to keep

        Returns:
            Filitered logits with only top k values, rest set to -inf
        """
        top_k = min(top_k, logits.size(-1))  # Safety check

        # Get top k values and indices
        top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)

        # Create a mask for values below the k-th largest
        indices_to_remove = logits < top_k_values[:, -1, None]

        # Set filtered values to -inf (will have ~0 probability after softamax)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits

    def _top_p_filtering(self, logits, top_p):
        """
        Nucleus sampling: filter logits to keep tokens with cumalative probability >= top_p

        Args:
            logits: (B, C) unnormalized log probabilities
            top_p: cumulative probability threshold (e.g., 0.9)

        Returns:
            Filtered logits with only nuclues tokens, rest set to -inf
        """
        # Sort logits in descending order
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

        # Compute cumulative probabilities
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p

        # Shift the indices to the right to keep the first token above threshold
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        # Create mask in original order
        indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
        for i in range(logits.size(0)):
            indices_to_remove[i, sorted_indices[i]] = sorted_indices_to_remove[i]

        # Set filtered values to -inf
        logits = logits.masked_fill(indices_to_remove, float('-inf'))

        return logits
