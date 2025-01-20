from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

class BaseAgent(ABC):
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.logger = logging.getLogger(name)
        
    @abstractmethod
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a given task and return results"""
        pass
    
    @abstractmethod
    async def collaborate(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborate with other agents"""
        pass
    
    @abstractmethod
    async def adapt(self, feedback: Dict[str, Any]) -> None:
        """Adapt based on feedback"""
        pass
    
    def log_info(self, message: str) -> None:
        """Log information about agent's activities"""
        self.logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str) -> None:
        """Log errors encountered by the agent"""
        self.logger.error(f"[{self.name}] {message}")
        
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize agent-specific components"""
        pass
