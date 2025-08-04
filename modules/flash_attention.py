"""
FlashAttention Implementation

A memory-efficient implementation of attention that reduces memory usage 
from O(N²) to O(N) through block-wise computation and IO-aware design.

Key features:
1. Block-wise computation to reduce memory usage
2. Online softmax computation
3. Causal masking support
4. Numerically stable implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange


class FlashCausalSelfAttention(nn.Module):
    """
    Causal Self-Attention with FlashAttention implementation
    """
    def __init__(self, config):
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Block size for FlashAttention
        self.block_size = getattr(config, 'block_size', 64)
        
        # Linear transformations for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transform(self, x, linear_layer):
        # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
        proj = linear_layer(x)
        # Next, we need to produce multiple heads for the proj. This is done by spliting the
        # hidden state to self.num_attention_heads, each of size self.attention_head_size.
        proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
        # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
        proj = rearrange(proj, 'b t h d -> b h t d')
        return proj
    
    def attention(self, key, query, value, attention_mask):
        """
        Block-wise FlashAttention implementation
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = 1.0 / math.sqrt(head_dim)
        
        # Initialize output
        output = torch.zeros_like(query)
        
        # Number of blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        for i in range(num_blocks):
            # Query block indices (对应公式中的第i个tile)
            q_start = i * self.block_size
            q_end = min((i + 1) * self.block_size, seq_len)
            q_block = query[:, :, q_start:q_end, :]  # [B, H, block_size, D]
            
            # Initialize block statistics (对应公式中的o'_i, m_i, d'_i)
            block_output = torch.zeros_like(q_block)
            block_max = torch.full((batch_size, num_heads, q_end - q_start), -float('inf'),
                                  device=query.device, dtype=query.dtype)
            block_sum = torch.zeros((batch_size, num_heads, q_end - q_start),
                                   device=query.device, dtype=query.dtype)
            
            # 对于causal attention，只需要处理到当前query位置的key/value
            kv_end = q_end
            
            # Process key/value blocks (对应公式中的内层循环，处理每个KV block)
            for j in range((kv_end + self.block_size - 1) // self.block_size):
                kv_start = j * self.block_size
                kv_end_block = min((j + 1) * self.block_size, kv_end)
                
                k_block = key[:, :, kv_start:kv_end_block, :]
                v_block = value[:, :, kv_start:kv_end_block, :]
                
                # Compute attention scores for this block (对应公式中的x_i)
                scores = torch.matmul(q_block, k_block.transpose(-2, -1)) * scale
                
                # Apply causal mask within block
                q_indices = torch.arange(q_start, q_end, device=query.device)[:, None]
                kv_indices = torch.arange(kv_start, kv_end_block, device=query.device)[None, :]
                causal_mask = (q_indices < kv_indices).float() * -1e4
                scores = scores + causal_mask
                
                # Apply additional attention mask if provided
                if attention_mask is not None:
                    mask_block = attention_mask[:, :, q_start:q_end, kv_start:kv_end_block]
                    scores = scores + mask_block
            
                # Online softmax computation (对应公式中的m_i^{local})
                scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
                scores_exp = torch.exp(scores - scores_max)
                scores_sum = torch.sum(scores_exp, dim=-1, keepdim=True)
                
                # Update global statistics (对应公式中的m_i, d'_i更新)
                new_max = torch.max(block_max.unsqueeze(-1), scores_max).squeeze(-1)
                old_scale = torch.exp(block_max - new_max)
                new_scale = torch.exp(scores_max.squeeze(-1) - new_max)
                
                # Update output and statistics (对应公式中的o'_i, d'_i更新)
                block_output = block_output * old_scale.unsqueeze(-1) + \
                              torch.matmul(scores_exp, v_block) * new_scale.unsqueeze(-1)
                block_sum = block_sum * old_scale + scores_sum.squeeze(-1) * new_scale
                block_max = new_max
            
            # Normalize block output (对应公式最后的O[k,:] ← o'_{N/b})
            if torch.any(block_sum > 0):  # 避免除以0
                block_output = block_output / block_sum.unsqueeze(-1)

            # Store block output
            output[:, :, q_start:q_end, :] = block_output
        
        return output
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional FlashAttention
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_len]
            
        Returns:
            output: [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute Q, K, V
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        
        # Calculate the multi-head attention.
        # 使用 FlashAttention
        attn_value_1 = F.scaled_dot_product_attention(query_layer, key_layer, value_layer, attention_mask, is_causal=True)
        attn_value_2 = self.attention(key_layer, query_layer, value_layer, attention_mask)
        
        # 比较两种实现的输出
        assert torch.allclose(attn_value_1, attn_value_2, atol=1e-6), "Outputs do not match!"
        y = rearrange(attn_value_1, 'b h t d -> b t (h d)')
        return y