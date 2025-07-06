import ollama
import logging
from typing import Dict, Any, List, Optional
import time

logger = logging.getLogger(__name__)

class ResearchDraftNode:
    def __init__(self, max_retries: int = 3, timeout: int = 120):
        self.max_retries = max_retries
        self.timeout = timeout
        self.ollama_client = ollama.Client(
            host="http://localhost:11434",
            timeout=timeout
        )
    
    def _format_summaries(self, summaries: List[Dict[str, Any]]) -> str:
        """Format the list of summary dictionaries into a readable string."""
        if not summaries or not isinstance(summaries, list):
            return "No summaries available."
            
        formatted = []
        for i, summary in enumerate(summaries, 1):
            if not isinstance(summary, dict):
                logger.warning(f"Skipping invalid summary format at index {i-1}")
                continue
                
            title = summary.get('title', f'Paper {i}')
            content = summary.get('summary', 'No summary available')
            source = summary.get('source', 'Source not available')
            
            formatted.append(f"""
            --- PAPER {i}: {title} ---
            Source: {source}
            
            {content}
            
            -----------------------------
            """)
        
        return "\n".join(formatted) if formatted else "No summaries available."
    
    def _generate_research_draft(self, topic: str, summaries: List[Dict[str, Any]]) -> str:
        """Generate a research draft using the Ollama model."""
        if not summaries or not isinstance(summaries, list):
            logger.error("No valid summaries provided for research draft")
            return "Error: No valid research summaries available to generate a draft."
            
        # Format the summaries for the prompt
        formatted_summaries = []
        for i, summary in enumerate(summaries, 1):
            if not isinstance(summary, dict):
                logger.warning(f"Skipping invalid summary format at index {i-1}")
                continue
                
            title = summary.get('title', f'Research Paper {i}')
            content = summary.get('summary', 'No summary content available')
            source = summary.get('source', 'Source not available')
            
            formatted = f"""
            --- PAPER {i}: {title} ---
            Source: {source}
            
            {content}
            
            -----------------------------
            """
            formatted_summaries.append(formatted)
        
        if not formatted_summaries:
            return "Error: No valid summaries could be processed for the research draft."
            
        summaries_text = "\n".join(formatted_summaries)
        
        system_prompt = """You are an expert academic research writer. Your task is to write a comprehensive 
        research paper draft based on the provided research summaries. The paper should be well-structured, 
        coherent, and properly cited where appropriate."""
        
        user_prompt = f"""
        Research Topic: {topic}
        
        Please write a comprehensive research paper draft that includes the following sections:
        1. Abstract: A brief summary of the research
        2. Introduction: Background, problem statement, and objectives
        3. Literature Review: Synthesis of existing research
        4. Methodology: Research approach and methods
        5. Results: Key findings from the research
        6. Discussion: Interpretation of results and implications
        7. Conclusion: Summary and future research directions
        
        Here are the research summaries to base your draft on:
        {summaries}
        
        Please ensure that:
        - The paper is well-organized with clear section headings
        - You maintain an academic tone throughout
        - You properly synthesize information from multiple sources
        - You highlight key findings and their significance
        - You include in-text citations where appropriate (e.g., [1], [2], etc.)
        - The paper should be comprehensive but concise, approximately 1500-2000 words
        - Include relevant examples and evidence from the provided research
        - Conclude with practical implications and suggestions for future research
        """.format(summaries=summaries_text)
        
        for attempt in range(self.max_retries):
            try:
                response = self.ollama_client.chat(
                    model="mistral",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    options={
                        "temperature": 0.3,
                        "max_tokens": 4000
                    }
                )
                return response['message']['content'].strip()
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2)  # Wait before retrying
        
        return "Error: Failed to generate research draft after multiple attempts."

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a research paper draft based on the provided summaries."""
        if "summaries" not in state or not state["summaries"]:
            logger.error("No summaries found in state")
            state["research_draft"] = "Error: No research summaries available to generate a draft."
            return state
            
        if not state.get("topic"):
            logger.error("No topic found in state")
            state["research_draft"] = "Error: No research topic specified."
            return state
            
        logger.info("Generating research draft...")
        
        try:
            draft = self._generate_research_draft(state["topic"], state["summaries"])
            state["research_draft"] = draft
            logger.info("Successfully generated research draft")
            
        except Exception as e:
            logger.error(f"Error generating research draft: {str(e)}", exc_info=True)
            state["research_draft"] = f"Error: Failed to generate research draft. {str(e)}"
        
        return state
