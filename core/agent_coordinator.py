from typing import Dict, List, Any
import asyncio
import logging
from core.base_agent import BaseAgent

class AgentCoordinator:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.logger = logging.getLogger("AgentCoordinator")
        
    def register_agent(self, agent: BaseAgent) -> None:
        """Register a new agent with the coordinator"""
        self.agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name} ({agent.role})")
        
    async def initialize_agents(self) -> None:
        """Initialize all registered agents"""
        init_tasks = [agent.initialize() for agent in self.agents.values()]
        await asyncio.gather(*init_tasks)
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate task processing across all agents"""
        # First pass: Task identification and expert vector selection
        first_pass_results = {}
        for agent in self.agents.values():
            result = await agent.process_task(task)
            first_pass_results[agent.name] = result
            
        # Second pass: Collaborative refinement
        second_pass_results = {}
        for agent in self.agents.values():
            result = await agent.collaborate(first_pass_results)
            second_pass_results[agent.name] = result
            
        # Adaptation phase
        adaptation_tasks = []
        for agent in self.agents.values():
            adaptation_tasks.append(agent.adapt(second_pass_results))
        await asyncio.gather(*adaptation_tasks)
        
        return {
            "first_pass": first_pass_results,
            "second_pass": second_pass_results
        }
        
    def get_agent(self, name: str) -> BaseAgent:
        """Get an agent by name"""
        return self.agents.get(name)
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents"""
        return list(self.agents.values())
