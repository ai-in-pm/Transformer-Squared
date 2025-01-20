import asyncio
from typing import Dict, Any
from core.agent_coordinator import AgentCoordinator
from agents.openai_agent import OpenAIAgent
from agents.anthropic_agent import AnthropicAgent
from agents.mistral_agent import MistralAgent
from agents.groq_agent import GroqAgent
from agents.gemini_agent import GeminiAgent
from agents.cohere_agent import CohereAgent
from agents.emergence_agent import EmergenceAgent
import logging
from PIL import Image
import io

async def run_multimodal_demo():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Initialize coordinator
    coordinator = AgentCoordinator()
    
    # Register all agents
    coordinator.register_agent(OpenAIAgent())
    coordinator.register_agent(AnthropicAgent())
    coordinator.register_agent(MistralAgent())
    coordinator.register_agent(GroqAgent())
    coordinator.register_agent(GeminiAgent())
    coordinator.register_agent(CohereAgent())
    coordinator.register_agent(EmergenceAgent())
    
    # Initialize all agents
    await coordinator.initialize_agents()
    
    # Example multi-modal task
    task = {
        "type": "multimodal_analysis",
        "text": "Analyze this image and provide a detailed description of the architectural style, historical period, and cultural significance.",
        "image": Image.new('RGB', (100, 100), color='red'),  # Placeholder image
        "requirements": {
            "analysis_depth": "detailed",
            "focus_areas": [
                "architectural_style",
                "historical_context",
                "cultural_impact"
            ],
            "output_format": "structured"
        }
    }
    
    try:
        # Process task through agent network
        logger.info("Starting multi-modal task processing")
        results = await coordinator.process_task(task)
        
        # Log results from each agent
        logger.info("Task processing complete")
        logger.info("First pass results:")
        for agent_name, result in results["first_pass"].items():
            logger.info(f"{agent_name}: {result}")
            
        logger.info("\nSecond pass results:")
        for agent_name, result in results["second_pass"].items():
            logger.info(f"{agent_name}: {result}")
            
    except Exception as e:
        logger.error(f"Error in task processing: {str(e)}")
        
if __name__ == "__main__":
    asyncio.run(run_multimodal_demo())
