import ollama
import logging
import json
from typing import Dict, Any, List, Optional
import requests

logger = logging.getLogger(__name__)

class RelatedTopicsNode:
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        self.max_retries = max_retries
        self.timeout = timeout
        self.ollama_client = ollama.Client(
            host="http://localhost:11434",
            timeout=timeout
        )
    
    def _get_related_topics(self, topic: str) -> List[str]:
        """Get related topics using Ollama with retries and error handling."""
        system_prompt = """You are an expert research assistant. 
        Provide a list of 5-8 key related topics or subfields for the given research topic. 
        Return ONLY a JSON array of topic strings, no additional text or explanation.
        Example: ["topic1", "topic2", "topic3"]"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.ollama_client.chat(
                    model="mistral",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Provide related topics for: {topic}"}
                    ],
                    options={
                        "temperature": 0.3,
                        "max_tokens": 500
                    },
                    format="json"
                )
                
                # Try to parse the response as JSON
                try:
                    content = response['message']['content'].strip()
                    # Clean up the response to ensure it's valid JSON
                    if content.startswith('```json'):
                        content = content[content.find('['):content.rfind(']')+1]
                    elif '```' in content:
                        content = content[content.find('['):content.rfind(']')+1]
                    
                    topics = json.loads(content)
                    if isinstance(topics, list) and all(isinstance(t, str) for t in topics):
                        return topics[:8]  # Return max 8 topics
                    else:
                        logger.warning(f"Unexpected response format: {content}")
                        
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Failed to parse response as JSON: {e}")
                    # Fallback: Extract topics from text response
                    content = response['message']['content'].strip()
                    lines = [line.strip('-*# ') for line in content.split('\n') if line.strip()]
                    return lines[:8] if lines else ["Machine Learning in Security", "Network Security", "Threat Intelligence", "Anomaly Detection", "Cybersecurity Frameworks"]
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request to Ollama timed out (attempt {attempt + 1}/{self.max_retries})")
                if attempt == self.max_retries - 1:
                    return ["Machine Learning in Security", "Network Security", "Threat Intelligence"]
                
            except Exception as e:
                logger.error(f"Error getting related topics: {str(e)}")
                if attempt == self.max_retries - 1:
                    return ["Machine Learning in Security", "Network Security"]
        
        return ["Cybersecurity Fundamentals", "AI Security"]

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a list of related topics for the given topic."""
        if not state.get("topic"):
            logger.error("No topic provided in state")
            state["related_topics"] = ["Cybersecurity Fundamentals", "AI Security"]
            return state
            
        logger.info(f"Generating related topics for: {state['topic']}")
        
        try:
            related_topics = self._get_related_topics(state["topic"])
            state["related_topics"] = related_topics or ["Cybersecurity Fundamentals"]
            logger.info(f"Generated {len(related_topics)} related topics")
            
        except Exception as e:
            logger.error(f"Unexpected error in RelatedTopicsNode: {str(e)}", exc_info=True)
            state["related_topics"] = ["Cybersecurity Fundamentals", "AI Security"]
        
        return state
