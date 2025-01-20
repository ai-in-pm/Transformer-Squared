import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

@dataclass
class AdaptationConfig:
    input_dim: int
    hidden_dim: int
    num_experts: int
    num_templates: int
    learning_rate: float = 0.001
    temperature: float = 1.0

class PromptEngineering:
    """Dynamic prompt engineering and template management"""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.templates = {}
        self.template_usage = {}
        self.performance_history = []
        
    def add_template(self, name: str, template: str, metadata: Dict[str, Any]) -> None:
        """Add new prompt template"""
        self.templates[name] = {
            'template': template,
            'metadata': metadata,
            'performance': [],
            'usage_count': 0
        }
        
    def select_template(self, task: Dict[str, Any]) -> Tuple[str, str]:
        """Select best template for task"""
        scores = []
        for name, info in self.templates.items():
            # Calculate template score based on:
            # 1. Historical performance
            # 2. Task similarity
            # 3. Usage frequency
            score = self._calculate_template_score(task, info)
            scores.append((score, name))
            
        # Select best template
        best_score, best_name = max(scores)
        self.template_usage[best_name] = self.template_usage.get(best_name, 0) + 1
        
        return best_name, self.templates[best_name]['template']
        
    def update_performance(self, template_name: str, 
                          performance: float) -> None:
        """Update template performance history"""
        if template_name in self.templates:
            self.templates[template_name]['performance'].append(performance)
            self.performance_history.append({
                'template': template_name,
                'performance': performance
            })
            
    def _calculate_template_score(self, task: Dict[str, Any], 
                                template_info: Dict[str, Any]) -> float:
        """Calculate template score for task"""
        # Performance score
        perf_score = np.mean(template_info['performance']) if template_info['performance'] else 0.5
        
        # Task similarity score
        sim_score = self._calculate_task_similarity(task, template_info['metadata'])
        
        # Usage penalty (encourage exploration)
        usage_penalty = 1.0 / (1.0 + template_info['usage_count'])
        
        return 0.4 * perf_score + 0.4 * sim_score + 0.2 * usage_penalty
        
    def _calculate_task_similarity(self, task: Dict[str, Any], 
                                 metadata: Dict[str, Any]) -> float:
        """Calculate similarity between task and template metadata"""
        # Implement similarity calculation
        return 0.5  # Placeholder

class ClassificationExperts(nn.Module):
    """Ensemble of specialized classification experts"""
    
    def __init__(self, config: AdaptationConfig):
        super().__init__()
        self.config = config
        
        # Expert networks
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(config.num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_experts)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get expert outputs
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Get gating weights
        gate_logits = self.gate(x)
        gate_weights = F.softmax(gate_logits / self.config.temperature, dim=-1)
        
        # Combine expert outputs
        combined_output = torch.sum(expert_outputs * gate_weights.unsqueeze(-1), dim=1)
        
        return combined_output, gate_weights
        
    def _create_expert(self) -> nn.Module:
        """Create single expert network"""
        return nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )

class MixtureOfExperts(nn.Module):
    """Adaptive Mixture of Experts system"""
    
    def __init__(self, config: AdaptationConfig):
        super().__init__()
        self.config = config
        
        # Expert networks
        self.experts = nn.ModuleList([
            self._create_expert() for _ in range(config.num_experts)
        ])
        
        # Gating network with attention
        self.gate = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            dropout=0.1
        )
        
        # Expert combination layer
        self.combine = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor, 
                task_embedding: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Get expert outputs
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        # Use task embedding for attention if provided
        if task_embedding is None:
            task_embedding = x
            
        # Get attention weights
        attn_output, attn_weights = self.gate(
            query=task_embedding.unsqueeze(0),
            key=expert_outputs.transpose(0, 1),
            value=expert_outputs.transpose(0, 1)
        )
        
        # Combine expert outputs
        combined = torch.sum(expert_outputs * attn_weights.transpose(0, 1), dim=1)
        final_output = self.combine(combined)
        
        return {
            'output': final_output,
            'expert_weights': attn_weights,
            'expert_outputs': expert_outputs
        }
        
    def _create_expert(self) -> nn.Module:
        """Create single expert network"""
        return nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.LayerNorm(self.config.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim)
        )
        
    def get_expert_stats(self) -> Dict[str, List[float]]:
        """Get expert utilization statistics"""
        return {
            'expert_norms': [
                torch.norm(expert[0].weight).item()
                for expert in self.experts
            ],
            'combination_weights': self.combine[0].weight.mean(dim=0).tolist()
        }
