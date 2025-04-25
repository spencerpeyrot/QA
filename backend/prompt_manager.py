import json
import os
from typing import Dict, List, Optional
from jinja2 import Template
from fastapi import HTTPException

class PromptManager:
    def __init__(self):
        self.registry = self._load_registry()
        
    def _load_registry(self) -> Dict:
        """Load the prompt registry from JSON file."""
        registry_path = os.path.join(os.path.dirname(__file__), "prompts", "registry.json")
        try:
            with open(registry_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail="Prompt registry not found. Please ensure prompts/registry.json exists."
            )
    
    def validate_variables(self, agent: str, sub_component: Optional[str], variables: Dict) -> List[str]:
        """Validate that all required variables are present."""
        required_vars = self._get_required_variables(agent, sub_component)
        missing_vars = [var for var in required_vars if var not in variables]
        return missing_vars
    
    def _get_required_variables(self, agent: str, sub_component: Optional[str]) -> List[str]:
        """Get required variables for a given agent and sub-component."""
        if agent not in self.registry:
            raise HTTPException(status_code=400, detail=f"Unknown agent: {agent}")
            
        if not sub_component:
            # Agent without sub-component
            if "variables" in self.registry[agent]:
                return self.registry[agent]["variables"]
            raise HTTPException(
                status_code=400,
                detail=f"Agent {agent} requires a sub-component"
            )
            
        if sub_component not in self.registry[agent]:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown sub-component {sub_component} for agent {agent}"
            )
            
        return self.registry[agent][sub_component]["variables"]
    
    def load_prompt(self, agent: str, sub_component: Optional[str]) -> str:
        """Load and return the prompt template for the given agent/sub-component."""
        # Construct prompt file path
        base_path = os.path.join(os.path.dirname(__file__), "prompts")
        if sub_component:
            prompt_path = os.path.join(base_path, agent, f"{sub_component}.jinja")
        else:
            prompt_path = os.path.join(base_path, f"{agent}.jinja")
            
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except FileNotFoundError:
            raise HTTPException(
                status_code=500,
                detail=f"Prompt template not found at {prompt_path}"
            )
    
    def format_prompt(self, template: str, variables: Dict) -> str:
        """Format the prompt template with the provided variables."""
        try:
            return Template(template).render(**variables)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error formatting prompt template: {str(e)}"
            ) 