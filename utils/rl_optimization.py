import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import deque
import random

@dataclass
class RLConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int
    buffer_size: int = 10000
    batch_size: int = 32
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 0.001
    critic_lr: float = 0.001

class ReplayBuffer:
    """Experience replay buffer for RL"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, action: np.ndarray,
             reward: float, next_state: np.ndarray, done: bool) -> None:
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        return (np.array(state), np.array(action),
                np.array(reward), np.array(next_state),
                np.array(done))
                
    def __len__(self) -> int:
        return len(self.buffer)

class Actor(nn.Module):
    """Actor network for determining actions"""
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        
        self.network = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
            nn.Tanh()
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class Critic(nn.Module):
    """Critic network for value estimation"""
    
    def __init__(self, config: RLConfig):
        super().__init__()
        self.config = config
        
        self.network = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class RLOptimizer:
    """RL-based optimization for expert vectors"""
    
    def __init__(self, config: RLConfig):
        self.config = config
        
        # Networks
        self.actor = Actor(config)
        self.actor_target = Actor(config)
        self.critic = Critic(config)
        self.critic_target = Critic(config)
        
        # Copy weights
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training info
        self.training_step = 0
        self.episode_reward = 0
        self.best_reward = float('-inf')
        
    def select_action(self, state: np.ndarray, 
                     add_noise: bool = True) -> np.ndarray:
        """Select action based on current policy"""
        with torch.no_grad():
            state = torch.FloatTensor(state)
            action = self.actor(state).numpy()
            
        if add_noise:
            noise = np.random.normal(0, 0.1, size=self.config.action_dim)
            action = np.clip(action + noise, -1, 1)
            
        return action
        
    def train(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """Train the RL optimizer"""
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if len(self.replay_buffer) < batch_size:
            return {}
            
        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done).unsqueeze(1)
        
        # Update critic
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * self.config.gamma * target_Q
            
        current_Q = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q, target_Q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        actor_loss = -self.critic(state, self.actor(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        self.training_step += 1
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'average_Q': current_Q.mean().item()
        }
        
    def optimize_expert_vector(self, expert_vector: np.ndarray,
                             task_embedding: np.ndarray,
                             performance: float) -> np.ndarray:
        """Optimize expert vector using RL"""
        # Convert expert vector to state representation
        state = self._get_state_representation(expert_vector, task_embedding)
        
        # Select action (vector adjustment)
        action = self.select_action(state)
        
        # Apply action to expert vector
        new_vector = expert_vector + action
        
        # Store experience
        next_state = self._get_state_representation(new_vector, task_embedding)
        self.replay_buffer.push(state, action, performance, next_state, False)
        
        # Train if enough samples
        if len(self.replay_buffer) > self.config.batch_size:
            self.train()
            
        return new_vector
        
    def _soft_update(self, local_model: nn.Module, 
                    target_model: nn.Module) -> None:
        """Soft update target network parameters"""
        for target_param, local_param in zip(target_model.parameters(),
                                           local_model.parameters()):
            target_param.data.copy_(
                self.config.tau * local_param.data +
                (1.0 - self.config.tau) * target_param.data
            )
            
    def _get_state_representation(self, expert_vector: np.ndarray,
                                task_embedding: np.ndarray) -> np.ndarray:
        """Convert expert vector and task embedding to state representation"""
        return np.concatenate([expert_vector, task_embedding])
        
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        return {
            'training_step': self.training_step,
            'buffer_size': len(self.replay_buffer),
            'episode_reward': self.episode_reward,
            'best_reward': self.best_reward,
            'actor_grad_norm': self._get_grad_norm(self.actor),
            'critic_grad_norm': self._get_grad_norm(self.critic)
        }
        
    def _get_grad_norm(self, model: nn.Module) -> float:
        """Calculate gradient norm for model"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return np.sqrt(total_norm)
