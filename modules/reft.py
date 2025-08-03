"""
ReFT (Representation Fine-Tuning) Implementation

ReFT is a parameter-efficient fine-tuning method that intervenes on 
representations at specific layers of a pre-trained model, rather than 
modifying the model's weights directly.

Key concepts:
1. Intervention on hidden representations
2. Low-rank transformations for efficiency
3. Position-specific interventions
4. Minimal parameter overhead
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


class ReFTIntervention(nn.Module):
    """
    ReFT Intervention Module
    
    This module applies a low-rank transformation to hidden representations
    at specific positions in the sequence.
    """
    
    def __init__(self, 
                 hidden_size: int,
                 low_rank_dimension: int = 4,
                 dropout: float = 0.0,
                 intervention_type: str = "distributed"):
        """
        Args:
            hidden_size: Dimension of hidden representations
            low_rank_dimension: Rank of the intervention transformation
            dropout: Dropout probability
            intervention_type: Type of intervention ('distributed' or 'position_specific')
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.low_rank_dimension = low_rank_dimension
        self.intervention_type = intervention_type
        
        # Low-rank intervention matrices
        # R = W_down @ W_up, where W_down: hidden_size -> low_rank, W_up: low_rank -> hidden_size
        self.W_down = nn.Linear(hidden_size, low_rank_dimension, bias=False)
        self.W_up = nn.Linear(low_rank_dimension, hidden_size, bias=False)
        
        # Optional learnable scaling factor
        self.scale = nn.Parameter(torch.ones(1))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Position embedding for position-specific interventions
        if intervention_type == "position_specific":
            self.position_embedding = nn.Embedding(1024, low_rank_dimension)  # Max seq length
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following ReFT paper recommendations"""
        # Initialize W_down with small random values
        nn.init.normal_(self.W_down.weight, mean=0.0, std=0.02)
        # Initialize W_up with zeros to start with identity-like behavior
        nn.init.zeros_(self.W_up.weight)
        
        if hasattr(self, 'position_embedding'):
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, 
                hidden_states: torch.Tensor,
                intervention_positions: Optional[torch.Tensor] = None,
                intervention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply ReFT intervention to hidden states
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            intervention_positions: Positions to apply intervention [batch_size, num_positions]
            intervention_mask: Mask for which tokens to intervene on [batch_size, seq_len]
            
        Returns:
            Modified hidden states with intervention applied
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        if self.intervention_type == "distributed":
            # Apply intervention to all positions
            intervention = self._compute_intervention(hidden_states)
            if intervention_mask is not None:
                # Apply mask to selectively intervene
                intervention = intervention * intervention_mask.unsqueeze(-1)
            return hidden_states + self.scale * intervention
            
        elif self.intervention_type == "position_specific":
            # Apply intervention only at specific positions
            if intervention_positions is None:
                # Default to last position if not specified
                intervention_positions = torch.full((batch_size, 1), seq_len - 1, 
                                                  dtype=torch.long, device=hidden_states.device)
            
            # Create intervention for specific positions
            intervention = torch.zeros_like(hidden_states)
            for batch_idx in range(batch_size):
                for pos in intervention_positions[batch_idx]:
                    if 0 <= pos < seq_len:
                        pos_hidden = hidden_states[batch_idx:batch_idx+1, pos:pos+1]
                        pos_intervention = self._compute_intervention(pos_hidden, pos)
                        intervention[batch_idx, pos] = pos_intervention.squeeze()
            
            return hidden_states + self.scale * intervention
        
        else:
            raise ValueError(f"Unknown intervention type: {self.intervention_type}")
    
    def _compute_intervention(self, hidden_states: torch.Tensor, position: Optional[int] = None) -> torch.Tensor:
        """Compute the low-rank intervention"""
        # Low-rank transformation: hidden -> low_rank -> hidden
        low_rank_repr = self.W_down(hidden_states)
        low_rank_repr = F.relu(low_rank_repr)  # Non-linearity
        low_rank_repr = self.dropout(low_rank_repr)
        
        # Add position information if position-specific
        if self.intervention_type == "position_specific" and position is not None:
            pos_embedding = self.position_embedding(torch.tensor([position], device=hidden_states.device))
            low_rank_repr = low_rank_repr + pos_embedding
        
        intervention = self.W_up(low_rank_repr)
        return intervention


class ReFTLayer(nn.Module):
    """
    ReFT Layer that wraps around existing transformer layers
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # ReFT interventions
        self.reft_attention_intervention = None
        self.reft_mlp_intervention = None
        
        if config.use_reft:
            # Intervention after attention
            if config.reft_layers is None or layer_idx in config.reft_layers:
                self.reft_attention_intervention = ReFTIntervention(
                    hidden_size=config.hidden_size,
                    low_rank_dimension=config.reft_rank,
                    dropout=config.reft_dropout,
                    intervention_type=config.reft_intervention_type
                )
            
            # Intervention after MLP (optional)
            if config.reft_intervene_mlp and (config.reft_layers is None or layer_idx in config.reft_layers):
                self.reft_mlp_intervention = ReFTIntervention(
                    hidden_size=config.hidden_size,
                    low_rank_dimension=config.reft_rank,
                    dropout=config.reft_dropout,
                    intervention_type=config.reft_intervention_type
                )
    
    def apply_reft_intervention(self, 
                               hidden_states: torch.Tensor,
                               intervention_type: str = "attention",
                               **kwargs) -> torch.Tensor:
        """Apply ReFT intervention to hidden states"""
        
        if intervention_type == "attention" and self.reft_attention_intervention is not None:
            return self.reft_attention_intervention(hidden_states, **kwargs)
        elif intervention_type == "mlp" and self.reft_mlp_intervention is not None:
            return self.reft_mlp_intervention(hidden_states, **kwargs)
        else:
            return hidden_states


class ReFTConfig:
    """Configuration for ReFT parameters"""
    
    def __init__(self,
                 use_reft: bool = False,
                 reft_rank: int = 4,
                 reft_dropout: float = 0.0,
                 reft_intervention_type: str = "distributed",
                 reft_layers: Optional[List[int]] = None,
                 reft_intervene_mlp: bool = False,
                 reft_intervention_positions: Optional[List[int]] = None):
        """
        Args:
            use_reft: Whether to use ReFT
            reft_rank: Rank of the low-rank intervention
            reft_dropout: Dropout rate for ReFT layers
            reft_intervention_type: Type of intervention ('distributed' or 'position_specific')
            reft_layers: List of layer indices to apply ReFT to (None means all layers)
            reft_intervene_mlp: Whether to also intervene after MLP layers
            reft_intervention_positions: Specific positions to intervene (for position_specific type)
        """
        self.use_reft = use_reft
        self.reft_rank = reft_rank
        self.reft_dropout = reft_dropout
        self.reft_intervention_type = reft_intervention_type
        self.reft_layers = reft_layers
        self.reft_intervene_mlp = reft_intervene_mlp
        self.reft_intervention_positions = reft_intervention_positions


def count_reft_parameters(model) -> int:
    """Count the number of trainable ReFT parameters"""
    reft_params = 0
    for name, param in model.named_parameters():
        if any(reft_keyword in name for reft_keyword in ['reft', 'W_down', 'W_up', 'scale', 'position_embedding']):
            if param.requires_grad:
                reft_params += param.numel()
    return reft_params


def print_reft_info(model, config):
    """Print information about ReFT configuration and parameters"""
    if not config.use_reft:
        print("ReFT is not enabled")
        return
    
    total_params = sum(p.numel() for p in model.parameters())
    reft_params = count_reft_parameters(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n=== ReFT Configuration ===")
    print(f"ReFT Rank: {config.reft_rank}")
    print(f"ReFT Intervention Type: {config.reft_intervention_type}")
    print(f"ReFT Layers: {config.reft_layers if config.reft_layers else 'All layers'}")
    print(f"ReFT Intervene MLP: {config.reft_intervene_mlp}")
    print(f"ReFT Dropout: {config.reft_dropout}")
    
    print(f"\n=== Parameter Statistics ===")
    print(f"Total parameters: {total_params:,}")
    print(f"ReFT parameters: {reft_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"ReFT parameter ratio: {reft_params/total_params*100:.4f}%")
    print(f"Trainable parameter ratio: {trainable_params/total_params*100:.4f}%")
