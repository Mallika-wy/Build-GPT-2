import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRALayer(nn.Module):
    """
    LoRA: Low-Rank Adaptation for Efficient Model Fine-tuning
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lora_alpha = config.lora_alpha
        self.lora_r = config.lora_r
        # Linear layers for LoRA
        self.lora_A = nn.Linear(config.hidden_size, self.lora_r, bias=False)
        self.lora_B = nn.Linear(self.lora_r, config.hidden_size, bias=False)
        # Optional dropout
        self.lora_dropout = getattr(config, 'lora_dropout', 0.0)
        self.dropout = nn.Dropout(self.lora_dropout)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.lora_A.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        lora_output = self.lora_B(F.relu(self.lora_A(x))) * self.lora_alpha
        lora_output = self.dropout(lora_output)
        return lora_output
