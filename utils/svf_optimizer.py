import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

@dataclass
class SVFConfig:
    input_dim: int
    hidden_dim: int
    num_singular_values: int
    learning_rate: float
    regularization: float

class SVFOptimizer:
    """Singular Value Fine-tuning Optimizer"""
    
    def __init__(self, config: SVFConfig):
        self.config = config
        self.U = np.random.randn(config.input_dim, config.num_singular_values)
        self.S = np.ones(config.num_singular_values)
        self.V = np.random.randn(config.num_singular_values, config.hidden_dim)
        self.gradients = {"U": [], "S": [], "V": []}
        
    def decompose(self, weight_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform SVD decomposition of weight matrix"""
        U, S, V = np.linalg.svd(weight_matrix, full_matrices=False)
        return U[:, :self.config.num_singular_values], \
               S[:self.config.num_singular_values], \
               V[:self.config.num_singular_values, :]
               
    def reconstruct(self) -> np.ndarray:
        """Reconstruct weight matrix from U, S, V components"""
        return np.dot(self.U * self.S, self.V)
        
    def update_singular_values(self, task_embedding: np.ndarray,
                             performance_metric: float) -> None:
        """Update singular values based on task performance"""
        # Compute gradients for singular values
        output = self.forward(task_embedding)
        gradient = self._compute_gradient(output, performance_metric)
        
        # Update S with regularization
        S_gradient = np.dot(self.U.T, gradient)
        S_gradient = S_gradient * self.V
        S_gradient = np.mean(S_gradient, axis=1)
        
        # Apply regularization
        S_gradient += self.config.regularization * self.S
        
        # Store gradient history
        self.gradients["S"].append(S_gradient)
        
        # Update singular values
        self.S -= self.config.learning_rate * S_gradient
        
        # Ensure non-negative singular values
        self.S = np.maximum(self.S, 0)
        
    def forward(self, input_embedding: np.ndarray) -> np.ndarray:
        """Forward pass through SVF layer"""
        return np.dot(input_embedding, self.reconstruct())
        
    def adapt_to_task(self, task_embedding: np.ndarray,
                      task_type: str) -> np.ndarray:
        """Adapt singular values for specific task type"""
        # Get task-specific scaling factors
        scale_factors = self._get_task_scale_factors(task_type)
        
        # Scale singular values
        scaled_S = self.S * scale_factors
        
        # Compute adapted output
        return np.dot(np.dot(task_embedding, self.U * scaled_S), self.V)
        
    def optimize_batch(self, embeddings: List[np.ndarray],
                      metrics: List[float]) -> Dict[str, float]:
        """Optimize SVF parameters for a batch of tasks"""
        total_loss = 0
        
        for embedding, metric in zip(embeddings, metrics):
            # Forward pass
            output = self.forward(embedding)
            
            # Compute loss
            loss = self._compute_loss(output, metric)
            total_loss += loss
            
            # Update parameters
            self.update_singular_values(embedding, metric)
            
        return {
            "average_loss": total_loss / len(embeddings),
            "singular_value_norm": np.linalg.norm(self.S),
            "max_singular_value": np.max(self.S)
        }
        
    def _compute_gradient(self, output: np.ndarray,
                         target_metric: float) -> np.ndarray:
        """Compute gradient for backpropagation"""
        # Simple MSE gradient
        return 2 * (output - target_metric)
        
    def _compute_loss(self, output: np.ndarray,
                     target_metric: float) -> float:
        """Compute loss with regularization"""
        mse_loss = np.mean((output - target_metric) ** 2)
        reg_loss = self.config.regularization * np.sum(self.S ** 2)
        return mse_loss + reg_loss
        
    def _get_task_scale_factors(self, task_type: str) -> np.ndarray:
        """Get task-specific scaling factors for singular values"""
        # Define scaling factors for different task types
        scale_factors = {
            "classification": np.linspace(1.0, 0.1, self.config.num_singular_values),
            "generation": np.linspace(0.8, 0.2, self.config.num_singular_values),
            "analysis": np.linspace(0.9, 0.3, self.config.num_singular_values)
        }
        
        return scale_factors.get(task_type,
                               np.ones(self.config.num_singular_values))
                               
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            "singular_values": self.S.tolist(),
            "condition_number": np.max(self.S) / np.min(self.S),
            "gradient_norm_history": [np.linalg.norm(g) for g in self.gradients["S"]],
            "parameter_norm": {
                "U": np.linalg.norm(self.U),
                "S": np.linalg.norm(self.S),
                "V": np.linalg.norm(self.V)
            }
        }
