# Transformer-Squared

A powerful multi-agent system for collaborative problem-solving and task execution, leveraging state-of-the-art language models and AI technologies.

The development of this repository was inspired by the "Transformer squared" paper, which focuses on the design and implementation of a multi-agent system for solving complex tasks using state-of-adaptive LLMs. To read the full paper, visit https://arxiv.org/pdf/2501.06252

## Overview

Transformer-Squared is a sophisticated multi-agent system that combines multiple AI models and technologies to create a collaborative environment for solving complex tasks. The system integrates various AI agents, each specialized in different aspects of task processing and problem-solving.

## Features

- **Multi-Agent Collaboration**: Integrates multiple AI agents including OpenAI, Anthropic, Mistral, Groq, Gemini, and Cohere for comprehensive task processing
- **CrewAI Integration**: Enhanced web automation and system monitoring capabilities through CrewAI tools
- **Adaptive Task Processing**: Dynamic task distribution and processing based on agent specializations
- **Real-time System Monitoring**: Comprehensive system performance tracking and analysis
- **Scalable Architecture**: Designed to handle complex tasks through distributed processing

## System Requirements

- Python 3.10 or higher
- Required dependencies (see `requirements.txt`)
- API keys for various AI services:
  - OpenAI API key
  - Anthropic API key
  - Mistral API key
  - Groq API key
  - Google API key (for Gemini)
  - Cohere API key
  - Emergence API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Transformer-Squared.git
cd Transformer-Squared
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory and add your API keys:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
MISTRAL_API_KEY=your_mistral_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
COHERE_API_KEY=your_cohere_key
EMERGENCE_API_KEY=your_emergence_key
```

## Usage

To run the system:

```bash
python main.py
```

## Architecture

The system consists of several key components:

1. **Agent Coordinator**: Manages the interaction between different AI agents
2. **Specialized Agents**:
   - OpenAI (Task Coordinator)
   - Anthropic (Architecture Specialist)
   - Mistral (Adaptation Strategist)
   - Groq (RL Optimizer)
   - Gemini (Multi-Modal Expert)
   - Cohere (Task Adaptation Specialist)
   - Emergence (System Monitor)

3. **CrewAI Integration**:
   - Web automation capabilities
   - System monitoring tools
   - Performance metrics collection

## Performance Monitoring

The system includes comprehensive monitoring capabilities:

- CPU usage tracking
- Memory utilization
- Disk I/O performance
- Network traffic analysis
- Application performance metrics
- System load monitoring
- Error rate tracking

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Last Updated

2025-01-20 14:07
