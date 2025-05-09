import json
import os
import logging
from typing import Dict, List, Optional
from jinja2 import Template
from fastapi import HTTPException

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
        
        # Log the attempt to load the prompt
        logger.info(f"Attempting to load prompt for agent '{agent}' with sub-component '{sub_component}'")
        
        if sub_component:
            # Try multiple variations of the filename
            variations = [
                (f"{sub_component}.jinja", "original with spaces"),
                (f"{sub_component.replace(' ', '')}.jinja", "without spaces"),
                (f"{sub_component.replace(' ', '_')}.jinja", "with underscores")
            ]
            
            for filename, variant_type in variations:
                prompt_path = os.path.join(base_path, agent, filename)
                logger.info(f"Trying {variant_type} variant at path: {prompt_path}")
                
                if os.path.exists(prompt_path):
                    logger.info(f"Found prompt file at: {prompt_path}")
                    try:
                        with open(prompt_path, 'r') as f:
                            content = f.read()
                            # Verify the content is not empty
                            if not content.strip():
                                logger.warning(f"Prompt file exists but is empty: {prompt_path}")
                                continue
                            logger.info(f"Successfully loaded prompt content from: {prompt_path}")
                            # Log first few lines of the template for verification
                            preview_lines = content.split('\n')[:5]
                            logger.info(f"Template preview (first 5 lines):\n{''.join(preview_lines)}")
                            return content
                    except Exception as e:
                        logger.error(f"Error reading prompt file {prompt_path}: {str(e)}")
                        continue
            
            # If we get here, we tried all variations but found nothing
            error_msg = f"Could not find valid prompt template for agent '{agent}' and sub-component '{sub_component}'. Tried paths: {[os.path.join(base_path, agent, v[0]) for v in variations]}"
            logger.error(error_msg)
            raise HTTPException(status_code=500, detail=error_msg)
        else:
            # Handle agent-level prompts
            prompt_path = os.path.join(base_path, f"{agent}.jinja")
            logger.info(f"Attempting to load agent-level prompt: {prompt_path}")
            
            if not os.path.exists(prompt_path):
                error_msg = f"Agent-level prompt template not found at {prompt_path}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
            
            try:
                with open(prompt_path, 'r') as f:
                    content = f.read()
                    if not content.strip():
                        error_msg = f"Prompt file exists but is empty: {prompt_path}"
                        logger.error(error_msg)
                        raise HTTPException(status_code=500, detail=error_msg)
                    logger.info(f"Successfully loaded agent-level prompt from: {prompt_path}")
                    return content
            except Exception as e:
                error_msg = f"Error reading prompt file {prompt_path}: {str(e)}"
                logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)
    
    def format_prompt(self, template: str, variables: Dict) -> str:
        """Format the prompt template with the provided variables."""
        try:
            # Log template and variables before rendering
            logger.info("Template before rendering:")
            logger.info("-" * 80)
            logger.info(f"Template size: {len(template)} characters")
            logger.info("Full template:")
            logger.info(template)  # Log the entire template
            logger.info("-" * 80)
            logger.info("Variables for rendering:")
            logger.info(json.dumps(variables, indent=2))
            
            # Render the template
            rendered = Template(template).render(**variables)
            
            # Log the rendered result
            logger.info("Rendered template:")
            logger.info("-" * 80)
            logger.info(f"Rendered size: {len(rendered)} characters")
            logger.info("Full rendered template:")
            logger.info(rendered)  # Log the entire rendered template
            logger.info("-" * 80)
            
            # Check for potential truncation
            if len(rendered) > 100000:  # OpenAI's typical token limit is around 100k characters
                logger.warning(f"Rendered template is very large ({len(rendered)} chars). This may exceed token limits.")
            
            return rendered
        except Exception as e:
            error_msg = f"Error formatting prompt template: {str(e)}"
            logger.error(error_msg)
            # Log the actual error traceback
            logger.exception("Template rendering failed with traceback:")
            raise HTTPException(
                status_code=500,
                detail=error_msg
            ) 