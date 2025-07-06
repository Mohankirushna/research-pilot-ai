import requests
import json
from typing import Dict, Any, Optional

def process_with_ollama(content: str, model: str = "llama3") -> str:
    """Send content to Ollama for processing and get improved output."""
    url = "http://localhost:11434/api/generate"
    
    prompt = f"""Please process the following research paper content and provide a well-structured summary. 
    Focus on extracting and organizing the key information into clear sections. 
    Make sure the output is coherent, properly formatted, and free of any truncated text.
    
    Content to process:
    {content}
    
    Please provide the output in markdown format with appropriate headers and sections."""
    
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
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "No response from Ollama")
    except Exception as e:
        return f"Error processing with Ollama: {str(e)}"

def process_content(content: str, model: str = "llama3") -> str:
    """Process content directly with Ollama and return the improved output."""
    print("Processing content with Ollama...")
    improved_content = process_with_ollama(content, model)
    print("-" * 80)
    print("PROCESSED OUTPUT:")
    print("-" * 80)
    return improved_content

if __name__ == "__main__":
    # Example content - replace this with your actual content
    example_content = """
    **Problem Statement & Objective**
    Artificial intelligence (AI) has the potential to revolutionize the way we learn and teach...
    """
    
    print("Processing content with Ollama...")
    result = process_content(example_content)
    print(result)
