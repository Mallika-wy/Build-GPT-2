from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention


class LoRALayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config

    # LoRA parameters
    self.lora_alpha = config.lora_alpha
    self.lora_r = config.lora_r

    # Linear layers for LoRA
    self.lora_A = nn.Linear(config.hidden_size, self.lora_r, bias=False)
    self.lora_B = nn.Linear(self.lora_r, config.hidden_size, bias=False)

  def forward(self, x):
    lora_output = self.lora_B(F.relu(self.lora_A(x))) * self.lora_alpha
    return lora_output


class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)
    # LoRA layer.
    if config.use_lora:
      self.lora_layer = LoRALayer(config)
    else:
      self.lora_layer = None

  def add(self, input, output, dense_layer, dropout, lora_layer=None):
    out = dense_layer(output)
    out = dropout(out)
    out = input + out
    if lora_layer is not None:
      out += lora_layer(input)
    return out


  def forward(self, hidden_states, attention_mask):
    # LayerNorm before attention
    attn_input = self.attention_layer_norm(hidden_states)
    attn_output = self.self_attention(attn_input, attention_mask)
    attn_residual = self.add(hidden_states, attn_output, self.attention_dense, self.attention_dropout, self.lora_layer)
    # LayerNorm before feed-forward
    ffn_input = self.out_layer_norm(attn_residual)
    ffn_output = self.interm_af(self.interm_dense(ffn_input))
    ffn_residual = self.add(attn_residual, ffn_output, self.out_dense, self.out_dropout, self.lora_layer)
    return ffn_residual
