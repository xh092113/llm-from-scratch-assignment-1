import torch
from torch import Tensor, nn
import torch.nn.functional as F
from typing import Dict, Optional, Literal, Tuple
from jaxtyping import Float, Int
from dataclasses import dataclass
import json


@dataclass
class ModelArgs:
    vocab_size: int
    embed_dim: int
    inter_dim: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    n_layers: int
    attention_type: Literal["MHA", "MLA"]
    rope_theta: float


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.weight = nn.Parameter(torch.empty(self.vocab_size, self.dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]
    

def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    """
    Given the weights of an Embedding layer, get the embeddings for a batch of token ids.

    Args:
        vocab_size (int): The number of embeddings in the vocabulary
        d_model (int): The size of the embedding dimension
        weights (Float[Tensor, "vocab_size d_model"]): The embedding vectors to fetch from
        token_ids (Int[Tensor, "..."]): The set of token ids to fetch from the Embedding layer
    
    Returns:
        Float[Tensor, "... d_model"]: Batch of embeddings returned by your Embedding layer.
    """

    embed = Embedding(vocab_size, d_model, weights.device, weights.dtype)
    embed.load_state_dict({"weight": weights})
    return embed(token_ids)


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(dim))
    
    def forward(self, x: torch.Tensor):
        input_type = x.dtype
        x = x.to(torch.float32)
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(input_type) * self.weight


class MLP(nn.Module):
    def __init__(self, embed_dim: int, inter_dim: int):
        super().__init__()
        self.gate_proj = Linear(embed_dim, inter_dim)
        self.up_proj = Linear(embed_dim, inter_dim)
        self.down_proj = Linear(inter_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        self.dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.dim, 2, dtype=torch.float64) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq_ = self.inv_freq[None, None, :].to(torch.float32)
        position_ids_ = position_ids[:, :, None].to(torch.float32)

        emb = inv_freq_ * position_ids_
        cos = emb.cos()
        sin = emb.sin()
        return cos, sin
    

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d].type_as(cos)
    x2 = x[..., d:].type_as(cos)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], -1).type_as(x)


class KVCache:
    def __init__(self, n_layers: int):
        self.cache = [None] * n_layers
        self.seq_length = 0
    
    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int):
        if self.cache[layer_idx] is None:
            self.cache[layer_idx] = (k, v)
        else:
            k_, v_ = self.cache[layer_idx]
            k_ = torch.cat([k_, k], dim=-2)
            v_ = torch.cat([v_, v], dim=-2)
            self.cache[layer_idx] = (k_, v_)
        return self.cache[layer_idx]


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    attention_mask: torch.Tensor) -> torch.Tensor:
    
    assert q.shape[-3] % k.shape[-3] == 0
    n_groups = q.shape[-3] // k.shape[-3]
    if n_groups > 1:
        k = k.repeat_interleave(n_groups, dim=-3)
        v = v.repeat_interleave(n_groups, dim=-3)
    scaling = float(q.shape[-1] ** -0.5)
    attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
    attn_weights = F.softmax(attn_weights.masked_fill(~attention_mask, -torch.inf), dim=-1, dtype=torch.float32)
    attn_output = torch.matmul(attn_weights, v)
    attn_output = attn_output.transpose(-3, -2).contiguous()
    return attn_output


class MHA(nn.Module):
    def __init__(self, embed_dim: int, head_dim: int, num_attention_heads: int, num_key_value_heads: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = head_dim
        self.scaling = self.head_dim ** -0.5
        
        self.q_proj = Linear(embed_dim, num_attention_heads * head_dim)
        self.k_proj = Linear(embed_dim, num_key_value_heads * head_dim)
        self.v_proj = Linear(embed_dim, num_key_value_heads * head_dim)
        self.o_proj = Linear(num_attention_heads * head_dim, embed_dim)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)

    def forward(self,
                x: torch.Tensor,
                attention_mask: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                past_key_values: Optional[KVCache] = None):
        input_shape = x.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        q = self.q_norm(self.q_proj(x).view(hidden_shape)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        if past_key_values is not None:
            k, v = past_key_values.update(k, v, self.layer_idx)
        
        attn_output = scaled_dot_product_attention(q, k, v, attention_mask)
        attn_output = self.o_proj(attn_output.view(*input_shape, -1))
        return attn_output
        

class Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attn = MHA(embed_dim=args.embed_dim, head_dim=args.head_dim, num_attention_heads=args.num_attention_heads, num_key_value_heads=args.num_key_value_heads, layer_idx=layer_idx)
        self.mlp = MLP(args.embed_dim, args.inter_dim)
        self.attn_norm = RMSNorm(args.embed_dim)
        self.mlp_norm = RMSNorm(args.embed_dim)
    
    def forward(self, x: torch.Tensor, **attention_kwargs):
        x = x + self.attn(self.attn_norm(x), **attention_kwargs)
        x = x + self.mlp(self.mlp_norm(x))
        return x


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed = Embedding(args.vocab_size, args.embed_dim)
        self.layers = torch.nn.ModuleList([Block(args, i) for i in range(args.n_layers)])
        self.norm = RMSNorm(args.embed_dim)
        self.lm_head = Linear(args.embed_dim, args.vocab_size)
        self.rotary_emb = RotaryEmbedding(args.head_dim, args.rope_theta)
    
    def forward(self, *, input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor]=None,
                position_ids: Optional[torch.Tensor]=None,
                past_key_values: Optional[KVCache]=None):
        past_len = past_key_values.seq_length if past_key_values is not None else 0
        new_len = input_ids.shape[-1]

        if attention_mask is None:
            mask = torch.full((new_len, past_len + new_len), True, device=input_ids.device)
        elif attention_mask.dim() == 2:
            mask = (attention_mask[:, -new_len:, None] * attention_mask[:, None, :]).to(torch.bool)
        else:
            mask = attention_mask.to(torch.bool)
        mask = torch.tril(mask, diagonal=past_len)

        x = self.embed(input_ids)

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + new_len)[None, :].reshape_as(input_ids).to(input_ids.device)
        pos_embed = self.rotary_emb(position_ids)

        for block in self.layers:
            x = block(x,
                      attention_mask=mask,
                      position_embeddings=pos_embed,
                      past_key_values=past_key_values)
        
        x = self.norm(x)
        logits = self.lm_head(x)

        if past_key_values is not None:
            past_key_values.seq_length += input_ids.shape[1]
            
        return logits
    
    def print_info(self):
        n_embed_params = sum(p.numel() for p in self.embed.parameters())
        n_params = sum(p.numel() for p in self.parameters())
        
        print(f"Number of Embedding Parameters: {n_embed_params}")
        print(f"Number of Non-Embedding Parameters: {n_params - n_embed_params}")
        print(f"Number of Parameters: {n_params}")
    
    @torch.inference_mode()
    def generate(self, *,
                 input_ids: torch.Tensor,
                 attention_mask: torch.Tensor,
                 max_new_tokens: int=-1,
                 temperature: float=1.0,
                 top_k: int=0,
                 eos_token_id: Optional[int]=None):
        cache = KVCache(self.args.n_layers)
        ended = torch.full((input_ids.shape[0],), False, dtype=torch.bool, device=input_ids.device)

        k = 0
        while k < max_new_tokens or max_new_tokens == -1:
            if k == 0:
                logits = self(input_ids=input_ids, attention_mask=attention_mask, past_key_values=cache)
            else:
                logits = self(input_ids=input_ids[:, -1:], attention_mask=attention_mask, past_key_values=cache)
            logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -torch.inf
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            if eos_token_id is not None:
                idx_next[ended] = eos_token_id
                ended = idx_next == eos_token_id
                if ended.all():
                    break
            input_ids = torch.cat((input_ids, idx_next), dim=1)
            attention_mask = torch.cat((attention_mask, torch.ones_like(idx_next, dtype=attention_mask.dtype)), dim=1)

            k += 1
        
        return input_ids


if __name__ == "__main__":
    with open("architectures/qwen3-0.2B.json", 'r') as f:
        model_args = ModelArgs(**json.load(f))
    model = Transformer(model_args)
    model.print_info()
