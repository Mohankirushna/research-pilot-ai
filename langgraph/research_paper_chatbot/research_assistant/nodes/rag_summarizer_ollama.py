"""
Enhanced RAG Summarizer with Ollama Integration

This module provides an enhanced version of the RAG summarizer that uses Ollama
for better content formatting and summarization.
"""

from typing import Dict, Any, List, Optional
import os
import logging
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests

logger = logging.getLogger(__name__)

class OllamaEnhancedResearchSummarizer:
    """Enhanced research paper summarizer with Ollama integration for better formatting."""
    
    def __init__(self, model_name: str = "llama2"):
        self.model_name = model_name
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            num_ctx=1024,
            num_predict=200,
            repeat_penalty=1.1,
            top_k=20,
            top_p=0.9
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " "]
        )
        
        # Section patterns for extraction
        self.section_patterns = {
            'problem_statement': [
                r'problem\s+statement', r'objective', r'aim', r'purpose', r'goal',
                r'research\s+question', r'motivation', r'challenge'
            ],
            'research_gap': [
                r'gap', r'limitation', r'shortcoming', r'lack\s+of', r'missing',
                r'insufficient', r'inadequate', r'need\s+for', r'however', r'but'
            ],
            'methodology': [
                r'method', r'approach', r'technique', r'algorithm', r'framework',
                r'architecture', r'model', r'procedure', r'implementation'
            ],
            'datasets': [
                r'dataset', r'data', r'corpus', r'benchmark', r'collection',
                r'samples', r'instances', r'records', r'training\s+set'
            ],
            'results': [
                r'result', r'performance', r'accuracy', r'precision', r'recall',
                r'f1', r'score', r'metric', r'evaluation', r'experiment'
            ],
            'contributions': [
                r'contribution', r'novel', r'new', r'innovative', r'propose',
                r'introduce', r'first', r'advance', r'breakthrough', r'significant'
            ]
        }

    def process_with_ollama(self, content: str) -> str:
        """Process content with Ollama for better formatting and summarization."""
        if not content or len(content.strip()) < 10:
            return content
            
        url = "http://localhost:11434/api/generate"
        prompt = f"""Please process the following research content and provide a well-structured summary. 
        - Extract and organize key information into clear sections
        - Ensure the output is coherent and properly formatted
        - Remove any truncated text or incomplete sentences
        - Use markdown formatting with appropriate headers
        
        Content to process:
        {content}"""
        
        payload = {
            "model": self.model_name,
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
            return result.get("response", content)
        except Exception as e:
            logger.warning(f"Ollama processing failed: {str(e)}")
            return content

    def extract_structured_info(self, text: str) -> Dict[str, str]:
        """Extract structured information from text with Ollama enhancement."""
        # First extract basic structure
        structured_info = {
            'problem_statement': "The paper does not explicitly state a clear problem statement.",
            'research_gap': "The research gap is not clearly identified in the available text.",
            'methodology': "The methodology section is not clearly defined in the extracted content.",
            'datasets': "No specific dataset information is mentioned in the available text.",
            'results': "The results are not explicitly stated in the extracted content.",
            'contributions': "The key contributions are not clearly outlined in the available text."
        }
        
        # Process with Ollama for better extraction
        try:
            processed_content = self.process_with_ollama(text)
            # Here you can add more sophisticated extraction logic
            # For now, we'll just return the processed content in a structured format
            return {
                'problem_statement': processed_content,
                'research_gap': "See full content above.",
                'methodology': "See full content above.",
                'datasets': "See full content above.",
                'results': "See full content above.",
                'contributions': "See full content above."
            }
        except Exception as e:
            logger.error(f"Error processing content with Ollama: {str(e)}")
            return structured_info

    def process_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single paper with Ollama enhancement."""
        title = paper.get("title", "Untitled Paper")
        content = paper.get("content", "")
        source = paper.get("source", "")
        
        if not content.strip():
            return {
                "title": title,
                "summary": "No content available for summarization.",
                "source": source
            }
        
        # Extract structured information with Ollama enhancement
        structured_info = self.extract_structured_info(content)
        
        # Format the output
        formatted_summary = "\n\n".join(
            f"### {section.replace('_', ' ').title()}\n\n{content}"
            for section, content in structured_info.items()
        )
        
        return {
            "title": title,
            "summary": formatted_summary,
            "source": source
        }
