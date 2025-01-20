import os
from typing import Dict, Any, List
from core.base_agent import BaseAgent
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import requests
import json
import asyncio
from aiohttp import ClientSession
import time
from crewai import Crew, Process, Task, Agent as CrewAgent
from crewai import tools

import sqlite3

@dataclass
class SystemMetrics:
    latency: float
    throughput: float
    error_rate: float
    memory_usage: float
    adaptation_score: float

class EmergenceAgent(BaseAgent):
    def __init__(self):
        super().__init__("Emergence", "System Monitor")
        self.api_key = os.getenv("EMERGENCE_API_KEY")
        self.api_url = "https://api.emergence.ai/v0/orchestrators/em-web-automation/workflows"
        self.performance_history = []
        self.error_patterns = {}
        self.optimization_strategies = {}
        self.crew = self._initialize_crew()
        
    def _initialize_crew(self) -> Crew:
        """Initialize CrewAI components"""
        # Create web automation agent
        web_automation_agent = CrewAgent(
            role="Web Automation Specialist",
            goal="Monitor system performance through web automation",
            backstory="Expert in web automation and system monitoring",
            verbose=True,
            allow_delegation=False,
            share_crew=True,
            tools=[self.web_automation_tool]
        )
        
        # Create web automation task
        websearch_task = Task(
            description="Monitor system performance and collect metrics",
            expected_output="Detailed system performance metrics and analysis",
            tools=[self.web_automation_tool],
            agent=web_automation_agent
        )
        
        # Create and return crew
        return Crew(
            agents=[web_automation_agent],
            tasks=[websearch_task],
            process=Process.sequential,
            max_rpm=100,
            share_crew=True
        )
        
    @tools.tool("Web automation tool for system monitoring")
    def web_automation_tool(self, prompt: str) -> str:
        """Web automation tool for system monitoring"""
        try:
            # Create request payload
            payload = {"prompt": prompt}
            
            # Set headers
            headers = {
                "Content-Type": "application/json",
                "apikey": self.api_key
            }
            
            # Send initial request
            response = self._get_api_response(
                base_url=self.api_url,
                method="POST",
                headers=headers,
                payload=payload
            )
            
            workflow_id = response["workflowId"]
            status_url = f"{self.api_url}/{workflow_id}"
            
            # Poll for results
            while True:
                status_response = self._get_api_response(
                    base_url=status_url,
                    method="GET",
                    headers={"apikey": self.api_key}
                )
                
                if status_response["data"]["status"] in ["SUCCESS"]:
                    return status_response["data"]["output"]["result"]
                elif status_response["data"]["status"] in ["FAILED", "CANCELLED"]:
                    raise Exception(f"Workflow failed: {status_response['data']['error']}")
                    
                time.sleep(5)
                
        except Exception as e:
            self.log_error(f"Web automation error: {str(e)}")
            return str(e)
            
    def _get_api_response(self, base_url: str, method: str, 
                         headers: dict, payload: dict = {}) -> dict:
        """Send HTTP request to API"""
        response = requests.request(method, base_url, headers=headers, json=payload)
        return json.loads(response.text)
        
    async def initialize(self) -> None:
        """Initialize monitoring components"""
        # Initialize performance tracking
        self.performance_history = []
        
        # Initialize error pattern recognition
        self.error_patterns = self._initialize_error_patterns()
        
        # Initialize optimization strategies
        self.optimization_strategies = self._initialize_optimization_strategies()
        
        self.log_info("Initialized system monitoring components")
        
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor system performance during task processing"""
        try:
            # Collect system metrics
            metrics = self._collect_system_metrics()
            
            # Analyze performance bottlenecks
            bottlenecks = self._analyze_bottlenecks(metrics)
            
            # Get optimization recommendations using Emergence API
            crew_result = self.crew.kickoff(inputs={
                "prompt": f"Analyze system performance metrics: {metrics.__dict__}"
            })
            
            recommendations = await self._get_emergence_recommendations(
                task, metrics, bottlenecks, crew_result
            )
            
            # Update performance history
            self._update_performance_history(metrics)
            
            return {
                "metrics": metrics.__dict__,
                "bottlenecks": bottlenecks,
                "recommendations": recommendations,
                "crew_analysis": crew_result,
                "status": "healthy"
            }
            
        except Exception as e:
            self.log_error(f"Error in monitoring: {str(e)}")
            return {"error": str(e)}
            
    async def collaborate(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system-wide performance patterns"""
        try:
            # Analyze agent interactions
            interaction_patterns = self._analyze_interactions(agent_outputs)
            
            # Get optimization insights from Emergence API
            crew_result = self.crew.kickoff(inputs={
                "prompt": f"Analyze agent interactions: {agent_outputs}"
            })
            
            optimization_insights = await self._get_emergence_insights(
                agent_outputs, crew_result
            )
            
            # Update optimization strategies
            self._update_optimization_strategies(optimization_insights)
            
            return {
                "interaction_analysis": interaction_patterns,
                "optimization_insights": optimization_insights,
                "crew_analysis": crew_result,
                "system_health": self._get_system_health()
            }
            
        except Exception as e:
            self.log_error(f"Error in collaboration: {str(e)}")
            return {"error": str(e)}
            
    async def adapt(self, feedback: Dict[str, Any]) -> None:
        """Adapt monitoring and optimization strategies"""
        try:
            if "performance_metrics" in feedback:
                # Update monitoring thresholds
                self._update_monitoring_thresholds(feedback["performance_metrics"])
                
                # Adapt error pattern recognition
                self._adapt_error_patterns(feedback["performance_metrics"])
                
                # Get adaptation recommendations
                crew_result = self.crew.kickoff(inputs={
                    "prompt": f"Generate adaptation plan for metrics: {feedback}"
                })
                
                adaptation_plan = await self._get_emergence_adaptation_plan(
                    feedback, crew_result
                )
                
                # Apply adaptation plan
                self._apply_adaptation_plan(adaptation_plan)
                
                self.log_info("Monitoring strategies adapted based on feedback")
                
        except Exception as e:
            self.log_error(f"Error in adaptation: {str(e)}")
            
    async def _get_emergence_recommendations(self, task: Dict[str, Any],
                                          metrics: SystemMetrics,
                                          bottlenecks: Dict[str, Any],
                                          crew_result: str) -> Dict[str, Any]:
        """Get optimization recommendations from Emergence API"""
        payload = {
            "prompt": json.dumps({
                "task": task,
                "metrics": metrics.__dict__,
                "bottlenecks": bottlenecks,
                "crew_analysis": crew_result,
                "request_type": "optimization_recommendations"
            })
        }
        
        async with ClientSession() as session:
            async with session.post(
                self.api_url,
                json=payload,
                headers={
                    "apikey": self.api_key,
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Emergence API error: {await response.text()}")
                    
    async def _get_emergence_insights(self, agent_outputs: Dict[str, Any],
                                    crew_result: str) -> Dict[str, Any]:
        """Get system optimization insights from Emergence API"""
        payload = {
            "prompt": json.dumps({
                "agent_outputs": agent_outputs,
                "crew_analysis": crew_result,
                "request_type": "optimization_insights"
            })
        }
        
        async with ClientSession() as session:
            async with session.post(
                self.api_url,
                json=payload,
                headers={
                    "apikey": self.api_key,
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Emergence API error: {await response.text()}")
                    
    async def _get_emergence_adaptation_plan(self, feedback: Dict[str, Any],
                                           crew_result: str) -> Dict[str, Any]:
        """Get adaptation recommendations from Emergence API"""
        payload = {
            "prompt": json.dumps({
                "feedback": feedback,
                "crew_analysis": crew_result,
                "request_type": "adaptation_plan"
            })
        }
        
        async with ClientSession() as session:
            async with session.post(
                self.api_url,
                json=payload,
                headers={
                    "apikey": self.api_key,
                    "Content-Type": "application/json"
                }
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    raise Exception(f"Emergence API error: {await response.text()}")
                    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        return SystemMetrics(
            latency=self._measure_latency(),
            throughput=self._measure_throughput(),
            error_rate=self._calculate_error_rate(),
            memory_usage=self._measure_memory_usage(),
            adaptation_score=self._calculate_adaptation_score()
        )
        
    def _analyze_bottlenecks(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Analyze system bottlenecks"""
        bottlenecks = {}
        
        if metrics.latency > 1.0:  # Example threshold
            bottlenecks["latency"] = {
                "severity": "high",
                "impact": "response_time"
            }
            
        if metrics.error_rate > 0.05:  # Example threshold
            bottlenecks["error_rate"] = {
                "severity": "medium",
                "impact": "reliability"
            }
            
        return bottlenecks
        
    def _initialize_error_patterns(self) -> Dict[str, Any]:
        """Initialize error pattern recognition"""
        return {
            "timeout": {"pattern": "timeout", "count": 0},
            "memory": {"pattern": "memory", "count": 0},
            "api": {"pattern": "api", "count": 0}
        }
        
    def _initialize_optimization_strategies(self) -> Dict[str, Any]:
        """Initialize optimization strategies"""
        return {
            "caching": {"enabled": True, "ttl": 300},
            "batching": {"enabled": True, "size": 10},
            "retry": {"enabled": True, "max_attempts": 3}
        }
        
    def _update_performance_history(self, metrics: SystemMetrics) -> None:
        """Update performance history"""
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.__dict__
        })
        
        # Keep only last 1000 entries
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
            
    def _analyze_interactions(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent interaction patterns"""
        return {
            "communication_efficiency": self._calculate_communication_efficiency(outputs),
            "collaboration_patterns": self._identify_collaboration_patterns(outputs),
            "bottlenecks": self._identify_interaction_bottlenecks(outputs)
        }
        
    def _update_optimization_strategies(self, insights: Dict[str, Any]) -> None:
        """Update optimization strategies based on insights"""
        if "recommendations" in insights:
            for strategy, config in insights["recommendations"].items():
                if strategy in self.optimization_strategies:
                    self.optimization_strategies[strategy].update(config)
                    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status"""
        metrics = self._collect_system_metrics()
        return {
            "status": "healthy" if metrics.error_rate < 0.05 else "degraded",
            "metrics": metrics.__dict__,
            "bottlenecks": self._analyze_bottlenecks(metrics)
        }
        
    def _update_monitoring_thresholds(self, metrics: Dict[str, float]) -> None:
        """Update monitoring thresholds based on performance metrics"""
        pass
        
    def _adapt_error_patterns(self, metrics: Dict[str, float]) -> None:
        """Adapt error pattern recognition based on metrics"""
        pass
        
    def _apply_adaptation_plan(self, plan: Dict[str, Any]) -> None:
        """Apply adaptation plan to monitoring system"""
        pass
        
    def _measure_latency(self) -> float:
        """Measure current system latency"""
        return 0.5  # Example value
        
    def _measure_throughput(self) -> float:
        """Measure current system throughput"""
        return 100.0  # Example value
        
    def _calculate_error_rate(self) -> float:
        """Calculate current error rate"""
        return 0.02  # Example value
        
    def _measure_memory_usage(self) -> float:
        """Measure current memory usage"""
        return 0.6  # Example value
        
    def _calculate_adaptation_score(self) -> float:
        """Calculate system adaptation score"""
        return 0.85  # Example value
        
    def _calculate_communication_efficiency(self, outputs: Dict[str, Any]) -> float:
        """Calculate communication efficiency between agents"""
        return 0.9  # Example value
        
    def _identify_collaboration_patterns(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Identify patterns in agent collaboration"""
        return {
            "sequential": 0.7,
            "parallel": 0.3,
            "hybrid": 0.0
        }
        
    def _identify_interaction_bottlenecks(self, outputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify bottlenecks in agent interactions"""
        return [
            {
                "type": "communication",
                "severity": "low",
                "impact": "minimal"
            }
        ]
