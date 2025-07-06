import ollama
import time
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger(__name__)

class TopicExplainerNode:
    def __init__(self, max_retries: int = 3, timeout: int = 120):
        self.max_retries = max_retries
        self.timeout = timeout
        self.ollama_client = ollama.Client(
            host="http://localhost:11434",
            timeout=timeout
        )
    
    def _get_explanation(self, topic: str) -> Optional[str]:
        """Get explanation from Ollama with retries and error handling."""
        for attempt in range(self.max_retries):
            try:
                response = self.ollama_client.chat(
                    model="mistral",
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are an expert research tutor. Provide a clear, concise explanation 
                            of the given topic suitable for a beginner. Structure your response with an introduction, 
                            key concepts, and a brief conclusion. Keep it under 300 words."""
                        },
                        {
                            "role": "user", 
                            "content": f"Explain the topic '{topic}' in a way that's easy to understand."
                        }
                    ],
                    options={
                        "temperature": 0.3,
                        "max_tokens": 500
                    }
                )
                return response['message']['content'].strip()
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request to Ollama timed out (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    return "Explanation could not be generated at this time (timeout)."
                time.sleep(1)  # Wait before retrying
                
            except Exception as e:
                logger.error(f"Error generating explanation: {str(e)}")
                if attempt == self.max_retries - 1:
                    return f"Error: Could not generate explanation. {str(e)}"
                time.sleep(1)  # Wait before retrying
        
        return "Error: Failed to generate explanation after multiple attempts."

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an explanation for the given topic."""
        if not state.get("topic"):
            logger.error("No topic provided in state")
            state["explanation"] = "Error: No topic provided for explanation."
            return state
            
        logger.info(f"Generating explanation for topic: {state['topic']}")
        
        try:
            explanation = self._get_explanation(state["topic"])
            state["explanation"] = explanation or "No explanation was generated."
            
        except Exception as e:
            logger.error(f"Unexpected error in TopicExplainerNode: {str(e)}", exc_info=True)
            state["explanation"] = "An error occurred while generating the explanation."
        
        return state
