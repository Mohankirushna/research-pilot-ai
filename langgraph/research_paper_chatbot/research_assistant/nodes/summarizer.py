import ollama
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
import re
from pathlib import Path

logger = logging.getLogger(__name__)

class SummarizerNode:
    def __init__(self, model: str = "mistral", max_chars: int = 32000):
        self.model = model
        self.max_chars = max_chars  # Increased for handling full papers
        self.ollama_client = ollama.Client(host="http://localhost:11434")
        self.max_retries = 3
        self.retry_delay = 2

    def _chunk_text(self, text: str, max_chars: int) -> List[Dict[str, Any]]:
        """
        Split text into chunks with metadata about their position and context.
        Returns a list of dictionaries with 'text' and 'metadata' keys.
        """
        if not text.strip():
            return []
            
        # First, try to split by sections if they exist
        if isinstance(text, dict) and 'sections' in text:
            chunks = []
            for section_name, section_text in text['sections'].items():
                if not section_text.strip():
                    continue
                    
                # Split section into smaller chunks if needed
                if len(section_text) > max_chars:
                    section_chunks = self._split_text_by_paragraphs(section_text, max_chars)
                    for i, chunk in enumerate(section_chunks, 1):
                        chunks.append({
                            'text': chunk,
                            'metadata': {
                                'section': section_name,
                                'chunk': f"{i}/{len(section_chunks)}",
                                'total_chunks': len(section_chunks)
                            }
                        })
                else:
                    chunks.append({
                        'text': section_text,
                        'metadata': {
                            'section': section_name,
                            'chunk': '1/1',
                            'total_chunks': 1
                        }
                    })
            return chunks
            
        # Fallback: split by paragraphs if no sections
        return [
            {'text': chunk, 'metadata': {}}
            for chunk in self._split_text_by_paragraphs(text, max_chars)
        ]
    
    def _split_text_by_paragraphs(self, text: str, max_chars: int) -> List[str]:
        """Split text into chunks based on paragraphs, respecting max_chars."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para) + 2  # +2 for newlines
            
            if current_length + para_length > max_chars and current_chunk:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = []
                current_length = 0
                
            current_chunk.append(para)
            current_length += para_length
            
            # If a single paragraph is too long, split it by sentences
            if current_length > max_chars * 1.5 and len(current_chunk) == 1:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                current_chunk = []
                current_length = 0
                for sent in sentences:
                    sent_length = len(sent) + 1
                    if current_length + sent_length > max_chars and current_chunk:
                        chunks.append(' '.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    current_chunk.append(sent)
                    current_length += sent_length
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

    def _generate_section_summary(self, section_name: str, section_text: str, paper_title: str = "", context: str = "") -> str:
        """Generate a summary for a specific section of a paper."""
        try:
            # Prepare the prompt based on section type
            if section_name.lower() == 'abstract':
                prompt = f"""
                You are summarizing the ABSTRACT of a research paper titled: {paper_title}
                
                Please provide a concise summary that captures:
                1. The main research question or objective
                2. The approach or methodology used
                3. Key findings or results
                4. The significance or implications
                
                Abstract:
                {section_text}
                """
            elif section_name.lower() in ['introduction', 'background']:
                prompt = f"""
                You are summarizing the INTRODUCTION/BACKGROUND section of a research paper titled: {paper_title}
                
                Focus on:
                1. The research problem and its context
                2. Gaps in current knowledge
                3. Research objectives or hypotheses
                
                Introduction/Background:
                {section_text}
                """
            elif section_name.lower() in ['method', 'methodology', 'methods']:
                prompt = f"""
                You are summarizing the METHODOLOGY section of a research paper titled: {paper_title}
                
                Focus on:
                1. Research design and approach
                2. Data collection methods
                3. Analysis techniques
                4. Any tools or frameworks used
                
                Methodology:
                {section_text}
                """
            elif section_name.lower() in ['results', 'findings']:
                prompt = f"""
                You are summarizing the RESULTS/FINDINGS section of a research paper titled: {paper_title}
                
                Focus on:
                1. Key results and findings
                2. Important data points or statistics
                3. Any patterns or trends observed
                
                Results/Findings:
                {section_text}
                """
            elif section_name.lower() in ['discussion', 'conclusion']:
                prompt = f"""
                You are summarizing the DISCUSSION/CONCLUSION section of a research paper titled: {paper_title}
                
                Focus on:
                1. Interpretation of results
                2. Implications for the field
                3. Limitations of the study
                4. Suggestions for future research
                
                Discussion/Conclusion:
                {section_text}
                """
            else:
                prompt = f"""
                You are summarizing the '{section_name.upper()}' section of a research paper titled: {paper_title}
                
                Please provide a concise summary of the key points from this section.
                
                {section_name}:
                {section_text}
                """
            
            # Add context if available
            if context:
                prompt = f"Context from other sections:\n{context}\n\n" + prompt
            
            response = self.ollama_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful research assistant that provides accurate and concise summaries of academic papers."},
                    {"role": "user", "content": prompt}
                ],
                options={
                    "temperature": 0.2,  # Low temperature for factual accuracy
                    "top_p": 0.9,
                    "max_tokens": 1000,
                    "num_ctx": 8000  # Larger context window for full papers
                }
            )
            
            return response['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Error generating summary for section '{section_name}': {str(e)}")
            return f"[Summary for {section_name} could not be generated due to an error]"
    
    def _generate_comprehensive_summary(self, paper_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of a full research paper."""
        try:
            title = paper_data.get('title', 'Untitled Paper')
            sections = paper_data.get('sections', {})
            
            # Initialize summary sections
            summary = {
                'title': title,
                'source': paper_data.get('url', ''),
                'sections': {},
                'key_points': [],
                'full_summary': ''
            }
            
            # Process each section
            section_order = [
                'abstract', 'introduction', 'related work', 'methodology',
                'results', 'discussion', 'conclusion', 'references'
            ]
            
            # Build context as we go through sections
            context = ""
            
            for section_name in section_order:
                if section_name in sections and sections[section_name]:
                    section_summary = self._generate_section_summary(
                        section_name, 
                        sections[section_name], 
                        title,
                        context
                    )
                    summary['sections'][section_name] = section_summary
                    context += f"\n\n{section_name.upper()}:\n{section_summary}"
            
            # Generate overall key points
            key_points_prompt = f"""
            Based on the following paper summary, extract 3-5 key points:
            
            {context}
            
            Format as a bulleted list of the most important findings or contributions.
            """
            
            response = self.ollama_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that extracts key points from academic papers."},
                    {"role": "user", "content": key_points_prompt}
                ],
                options={
                    "temperature": 0.1,
                    "max_tokens": 500
                }
            )
            
            summary['key_points'] = response['message']['content'].strip().split('\n')
            
            # Generate a concise overall summary
            overall_summary_prompt = f"""
            Write a comprehensive but concise summary of this research paper based on the following sections:
            
            {context}
            
            Your summary should be structured as follows:
            1. Introduction to the research problem
            2. Methodology overview
            3. Key findings
            4. Implications and conclusions
            
            Keep it under 500 words.
            """
            
            response = self.ollama_client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a research assistant that writes clear, concise summaries of academic papers."},
                    {"role": "user", "content": overall_summary_prompt}
                ],
                options={
                    "temperature": 0.2,
                    "max_tokens": 1000
                }
            )
            
            summary['full_summary'] = response['message']['content'].strip()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating comprehensive summary: {str(e)}")
            return {'error': str(e)}

    def __call__(self, state: Dict) -> Dict:
        """Process parsed content and generate comprehensive summaries."""
        if 'processed_papers' not in state or not state['processed_papers']:
            logger.warning("No processed papers found in state")
            # Fallback to using search results if no processed papers
            if 'search_results' in state and state['search_results']:
                logger.info("Using search results as fallback for summarization")
                state['processed_papers'] = [
                    {
                        'title': paper.get('title', 'Untitled'),
                        'sections': {
                            'abstract': paper.get('snippet') or paper.get('abstract', 'No content available')
                        },
                        'url': paper.get('link', '')
                    }
                    for paper in state['search_results']
                    if isinstance(paper, dict)
                ]
            else:
                logger.error("No content available for summarization")
                state['summaries'] = []
                return state
        
        summaries = []
        
        for paper in state['processed_papers']:
            if not isinstance(paper, dict):
                continue
                
            title = paper.get('title', 'Untitled')
            source = paper.get('url', '')
            
            logger.info(f"Generating summary for paper: {title}")
            
            try:
                # Check if we have processed content with sections
                if 'sections' in paper and paper['sections']:
                    # Generate comprehensive summary for full paper
                    summary = self._generate_comprehensive_summary(paper)
                    
                    if 'error' in summary:
                        raise Exception(summary['error'])
                    
                    # Format the summary for display
                    formatted_summary = f"# {title}\n\n"
                    
                    # Add abstract if available
                    if 'abstract' in summary['sections']:
                        formatted_summary += f"## Abstract\n{summary['sections']['abstract']}\n\n"
                    
                    # Add key points
                    if 'key_points' in summary and summary['key_points']:
                        formatted_summary += "## Key Points\n"
                        if isinstance(summary['key_points'], list):
                            formatted_summary += '\n'.join(f"- {point}" for point in summary['key_points'] if point.strip())
                        else:
                            formatted_summary += summary['key_points']
                        formatted_summary += "\n\n"
                    
                    # Add full summary
                    if 'full_summary' in summary and summary['full_summary']:
                        formatted_summary += f"## Summary\n{summary['full_summary']}\n\n"
                    
                    # Add section summaries
                    if 'sections' in summary and summary['sections']:
                        formatted_summary += "## Detailed Section Summaries\n\n"
                        for section_name, section_summary in summary['sections'].items():
                            if section_name != 'abstract' and section_summary.strip():
                                formatted_summary += f"### {section_name.title()}\n{section_summary}\n\n"
                    
                    # Add source information to the summary
                    content_source = 'full text' if len(paper.get('sections', {}).get('full_text', '')) > 100 else 'abstract only'
                    formatted_summary += f"\n\n*[Summary based on {content_source}]*"
                    
                    summaries.append({
                        'title': title,
                        'summary': formatted_summary,
                        'source': source,
                        'metadata': paper.get('metadata', {}),
                        'content_source': content_source
                    })
                else:
                    # Fallback to simple summarization if no sections
                    text = paper.get('text', '')
                    if not text.strip():
                        text = str(paper)  # Last resort
                        
                    summary = self._generate_section_summary('paper', text, title)
                    
                    # Add source information to the summary
                    content_source = 'abstract only'  # Fallback case is always abstract only
                    summary_with_source = f"# {title}\n\n## Summary\n{summary}\n\n*[Summary based on {content_source}]*"
                    
                    summaries.append({
                        'title': title,
                        'summary': summary_with_source,
                        'source': source,
                        'metadata': paper.get('metadata', {}),
                        'content_source': content_source
                    })
                
                # Add a small delay between API calls to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing paper {title}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                
                summaries.append({
                    'title': title,
                    'summary': f"# {title}\n\n## Error\nFailed to generate summary: {str(e)}",
                    'source': source,
                    'metadata': paper.get('metadata', {})
                })
        
        logger.info(f"Generated {len(summaries)} summaries from {len(state.get('processed_papers', []))} papers")
        state["summaries"] = summaries
        return state
