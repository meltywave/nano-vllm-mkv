import torch
from torch import nn
import triton
import triton.language as tl

# ===================== FlashAttention 自适应导入 =====================
# 优先使用官方flash-attn，未安装时自动降级为原生PyTorch SDPA兼容实现
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

from nanovllm.utils.context import get_context


# ===================== FlashAttention 兼容实现（无flash时自动启用） =====================
def _repeat_kv(k: torch.Tensor, v: torch.Tensor, num_heads: int):
    """处理GQA分组查询，重复KV头到与Q头数一致"""
    num_kv_heads = k.shape[-2]
    if num_heads == num_kv_heads:
        return k, v
    repeat_times = num_heads // num_kv_heads
    k = k.repeat_interleave(repeat_times, dim=-2)
    v = v.repeat_interleave(repeat_times, dim=-2)
    return k, v


def _varlen_attention_padded(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal):
    """普通连续varlen注意力实现（非paged cache场景）"""
    batch = cu_seqlens_q.shape[0] - 1
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    num_kv_heads = k.shape[1]

    # 拆分每个序列并padding到统一长度
    q_list, k_list, v_list = [], [], []
    for i in range(batch):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i+1].item()
        q_seq = q[q_start:q_end]
        q_pad = torch.zeros(max_seqlen_q, num_heads, head_dim, device=q.device, dtype=q.dtype)
        q_pad[:q_end - q_start] = q_seq
        q_list.append(q_pad)

        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i+1].item()
        k_seq = k[k_start:k_end]
        k_pad = torch.zeros(max_seqlen_k, num_kv_heads, head_dim, device=k.device, dtype=k.dtype)
        k_pad[:k_end - k_start] = k_seq
        k_list.append(k_pad)

        v_seq = v[k_start:k_end]
        v_pad = torch.zeros(max_seqlen_k, num_kv_heads, head_dim, device=v.device, dtype=v.dtype)
        v_pad[:k_end - k_start] = v_seq
        v_list.append(v_pad)

    # 转置为SDPA要求的 (batch, num_heads, seq_len, head_dim)
    q_batch = torch.stack(q_list, dim=0).transpose(1, 2)
    k_batch = torch.stack(k_list, dim=0).transpose(1, 2)
    v_batch = torch.stack(v_list, dim=0).transpose(1, 2)

    # GQA头数对齐
    k_batch, v_batch = _repeat_kv(k_batch, v_batch, num_heads)

    # 原生缩放点积注意力
    out = torch.nn.functional.scaled_dot_product_attention(
        q_batch, k_batch, v_batch,
        is_causal=causal,
        scale=softmax_scale
    )

    # 转回varlen格式
    out = out.transpose(1, 2)
    out_list = []
    for i in range(batch):
        q_len = cu_seqlens_q[i+1].item() - cu_seqlens_q[i].item()
        out_list.append(out[i, :q_len])
    return torch.cat(out_list, dim=0)


def _varlen_attention_paged(q, k_cache, v_cache, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, block_table):
    """Paged KV Cache的varlen注意力实现（prefix cache场景）"""
    batch = cu_seqlens_q.shape[0] - 1
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]

    # 从paged cache中取出每个序列的KV，拼接成连续格式
    k_contig_list, v_contig_list = [], []
    for i in range(batch):
        seq_len_k = cu_seqlens_k[i+1].item() - cu_seqlens_k[i].item()
        blocks = block_table[i]
        # 取出该序列所有块的KV并展平为token维度
        seq_k = k_cache[blocks].view(-1, num_kv_heads, head_dim)[:seq_len_k]
        seq_v = v_cache[blocks].view(-1, num_kv_heads, head_dim)[:seq_len_k]
        k_contig_list.append(seq_k)
        v_contig_list.append(seq_v)

    k_contig = torch.cat(k_contig_list, dim=0)
    v_contig = torch.cat(v_contig_list, dim=0)

    # 复用连续varlen实现
    return _varlen_attention_padded(q, k_contig, v_contig, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal)


def _flash_attn_varlen_func_compat(q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k, softmax_scale, causal=True, block_table=None):
    """flash_attn_varlen_func 接口兼容实现"""
    if block_table is not None:
        return _varlen_attention_paged(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal, block_table)
    else:
        return _varlen_attention_padded(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal)


def _flash_attn_with_kvcache_compat(q, k_cache, v_cache, cache_seqlens, block_table, softmax_scale, causal=True):
    """flash_attn_with_kvcache 接口兼容实现（decode阶段paged kv cache）"""
    batch = q.shape[0]
    num_heads = q.shape[2]
    head_dim = q.shape[3]
    num_kv_heads = k_cache.shape[2]

    # 每个序列从paged cache取出KV并padding
    k_list, v_list = [], []
    max_seq_len = cache_seqlens.max().item()
    for i in range(batch):
        seq_len = cache_seqlens[i].item()
        blocks = block_table[i]
        seq_k = k_cache[blocks].view(-1, num_kv_heads, head_dim)[:seq_len]
        pad_k = torch.zeros(max_seq_len, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)
        pad_k[:seq_len] = seq_k
        k_list.append(pad_k)

        seq_v = v_cache[blocks].view(-1, num_kv_heads, head_dim)[:seq_len]
        pad_v = torch.zeros(max_seq_len, num_kv_heads, head_dim, device=q.device, dtype=q.dtype)
        pad_v[:seq_len] = seq_v
        v_list.append(pad_v)

    k_batch = torch.stack(k_list, dim=0).transpose(1, 2)
    v_batch = torch.stack(v_list, dim=0).transpose(1, 2)
    k_batch, v_batch = _repeat_kv(k_batch, v_batch, num_heads)

    q_t = q.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_t, k_batch, v_batch,
        is_causal=causal,
        scale=softmax_scale
    )
    return out.transpose(1, 2)


# 未安装flash-attn时自动替换为兼容实现
if flash_attn_varlen_func is None:
    flash_attn_varlen_func = _flash_attn_varlen_func_compat
if flash_attn_with_kvcache is None:
    flash_attn_with_kvcache = _flash_attn_with_kvcache_compat


# ===================== 原有Triton KV存储内核（无需修改） =====================
@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


# ===================== Attention层 =====================
class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)

        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
        return o