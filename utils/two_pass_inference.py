import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TwoPassConfig:
    input_dim: int
    hidden_dim: int
    num_experts: int
    num_tasks: int
    dropout: float = 0.1
    temperature: float = 1.0

class ExpertGating(nn.Module):
    """Expert gating network for first pass"""
    
    def __init__(self, config: TwoPassConfig):
        super().__init__()
        self.config = config
        
        # Task identification network
        self.task_network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_tasks)
        )
        
        # Expert selection network
        self.expert_network = nn.Sequential(
            nn.Linear(config.input_dim + config.num_tasks, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_experts)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Task identification
        task_logits = self.task_network(x)
        task_probs = F.softmax(task_logits / self.config.temperature, dim=-1)
        
        # Expert selection
        expert_input = torch.cat([x, task_probs], dim=-1)
        expert_logits = self.expert_network(expert_input)
        expert_probs = F.softmax(expert_logits / self.config.temperature, dim=-1)
        
        return task_probs, expert_probs

class WeightAdjustment(nn.Module):
    """Weight adjustment network for second pass"""
    
    def __init__(self, config: TwoPassConfig):
        super().__init__()
        self.config = config
        
        # Weight adjustment network
        self.adjustment_network = nn.Sequential(
            nn.Linear(config.input_dim + config.num_experts, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Scaling factors for weight adjustment
        self.scaling_factors = nn.Parameter(torch.ones(config.num_experts))
        
    def forward(self, x: torch.Tensor, expert_probs: torch.Tensor) -> torch.Tensor:
        # Combine input with expert probabilities
        adjustment_input = torch.cat([x, expert_probs], dim=-1)
        
        # Generate weight adjustments
        base_adjustment = self.adjustment_network(adjustment_input)
        
        # Apply expert-specific scaling
        scaled_adjustment = torch.einsum('be,e->be', base_adjustment, 
                                       F.softplus(self.scaling_factors))
        
        return scaled_adjustment

class TwoPassInference:
    """Two-pass inference mechanism"""
    
    def __init__(self, config: TwoPassConfig):
        self.config = config
        self.gating = ExpertGating(config)
        self.adjustment = WeightAdjustment(config)
        self.expert_memories = {}
        
    def first_pass(self, input_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """First pass: Task identification and expert selection"""
        # Get task and expert probabilities
        task_probs, expert_probs = self.gating(input_embedding)
        
        # Store intermediate results
        self.expert_memories['task_probs'] = task_probs
        self.expert_memories['expert_probs'] = expert_probs
        
        return {
            'task_probs': task_probs,
            'expert_probs': expert_probs
        }
        
    def second_pass(self, input_embedding: torch.Tensor,
                   base_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Second pass: Dynamic weight adjustment"""
        # Get expert probabilities from first pass
        expert_probs = self.expert_memories['expert_probs']
        
        # Generate weight adjustments
        weight_adjustment = self.adjustment(input_embedding, expert_probs)
        
        # Apply adjustments to base weights
        adjusted_weights = base_weights + weight_adjustment
        
        return {
            'adjusted_weights': adjusted_weights,
            'adjustment_magnitude': torch.norm(weight_adjustment)
        }
        
    def get_expert_contribution(self, expert_idx: int) -> Dict[str, float]:
        """Get contribution statistics for specific expert"""
        expert_probs = self.expert_memories['expert_probs']
        return {
            'average_probability': expert_probs[:, expert_idx].mean().item(),
            'max_probability': expert_probs[:, expert_idx].max().item(),
            'activation_frequency': (expert_probs[:, expert_idx] > 0.5).float().mean().item()
        }
        
    def analyze_task_distribution(self) -> Dict[str, List[float]]:
        """Analyze task type distribution"""
        task_probs = self.expert_memories['task_probs']
        return {
            'task_distribution': task_probs.mean(0).tolist(),
            'task_entropy': -(task_probs * torch.log(task_probs + 1e-10)).sum(1).mean().item()
        }
        
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        return {
            'expert_utilization': [
                self.get_expert_contribution(i)
                for i in range(self.config.num_experts)
            ],
            'task_analysis': self.analyze_task_distribution(),
            'gating_parameters': {
                'temperature': self.gating.config.temperature,
                'dropout': self.gating.config.dropout
            }
        }
