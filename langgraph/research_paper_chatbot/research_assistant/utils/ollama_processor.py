import requests
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def process_with_ollama(content: str, model: str = "llama3") -> str:
    """
    Send content to Ollama for better formatting and summarization.
    
    Args:
        content: The text content to process
        model: The Ollama model to use (default: "llama3")
        
    Returns:
        Processed text from Ollama, or original content if processing fails
    """
    if not content or len(content.strip()) < 10:
        return content
        
    url = "http://localhost:11434/api/generate"
    
    # Create a prompt that asks for better formatting and summarization
    prompt = f"""Please process the following research content and provide a well-structured summary. 
    - Extract and organize key information into clear sections
    - Ensure the output is coherent and properly formatted
    - Remove any truncated text or incomplete sentences
    - Use markdown formatting with appropriate headers
    
    Content to process:
    {content}
    """
    
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "max_tokens": 4000
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get("response", content)  # Return original if no response
    except Exception as e:
        logger.warning(f"Ollama processing failed: {str(e)}")
        return content  # Return original if processing fails
