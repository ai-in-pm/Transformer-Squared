import asyncio
import logging
from dotenv import load_dotenv
from core.agent_coordinator import AgentCoordinator
from agents.openai_agent import OpenAIAgent
from agents.anthropic_agent import AnthropicAgent
from agents.mistral_agent import MistralAgent
from agents.groq_agent import GroqAgent
from agents.gemini_agent import GeminiAgent
from agents.cohere_agent import CohereAgent
from agents.emergence_agent import EmergenceAgent
import os

async def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load environment variables
    load_dotenv()
    
    # Verify API keys
    required_keys = [
        'OPENAI_API_KEY',
        'ANTHROPIC_API_KEY',
        'MISTRAL_API_KEY',
        'GROQ_API_KEY',
        'GOOGLE_API_KEY',
        'COHERE_API_KEY'
    ]
    
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        logger.error(f"Missing required API keys: {', '.join(missing_keys)}")
        return
    
    # Initialize coordinator
    coordinator = AgentCoordinator()
    
    # Register all agents
    coordinator.register_agent(OpenAIAgent())  # Task Coordinator
    coordinator.register_agent(AnthropicAgent())  # Architecture Specialist
    coordinator.register_agent(MistralAgent())  # Adaptation Strategist
    coordinator.register_agent(GroqAgent())  # RL Optimizer
    coordinator.register_agent(GeminiAgent())  # Multi-Modal Expert
    coordinator.register_agent(CohereAgent())  # Task Adaptation Specialist
    coordinator.register_agent(EmergenceAgent())  # System Monitor
    
    # Initialize all agents
    logger.info("Initializing agents...")
    await coordinator.initialize_agents()
    
    # Example complex task
    task = {
        "type": "multi_modal_analysis",
        "content": {
            "text": """
            Analyze the architectural design patterns in modern cloud-native applications,
            focusing on microservices architecture, event-driven systems, and serverless
            computing. Consider both technical implementation details and business impact.
            """,
            "requirements": {
                "depth": "comprehensive",
                "focus_areas": [
                    "architectural_patterns",
                    "scalability",
                    "maintainability",
                    "cost_efficiency"
                ],
                "output_format": "structured",
                "adaptation_required": True
            }
        },
        "constraints": {
            "response_time": "real-time",
            "resource_efficiency": "high",
            "accuracy": "high"
        }
    }
    
    try:
        # Process task through agent network
        logger.info("Starting task processing...")
        results = await coordinator.process_task(task)
        
        # Log results from each agent
        logger.info("\nTask processing complete. Results from each agent:")
        
        # First pass results
        logger.info("\nFirst Pass Results:")
        for agent_name, result in results["first_pass"].items():
            logger.info(f"\n{agent_name}:")
            logger.info("-" * 40)
            logger.info(result)
            
        # Second pass results
        logger.info("\nSecond Pass Results:")
        for agent_name, result in results["second_pass"].items():
            logger.info(f"\n{agent_name}:")
            logger.info("-" * 40)
            logger.info(result)
            
    except Exception as e:
        logger.error(f"Error in task processing: {str(e)}")
        logger.exception("Detailed error information:")

if __name__ == "__main__":
    asyncio.run(main())
