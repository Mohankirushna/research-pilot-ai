import time
import concurrent.futures
import logging
from typing import Dict, Any, List
import requests
import re
from langchain_community.chat_models import ChatOllama

logger = logging.getLogger(__name__)

class EnhancedResearchSummarizerNode:
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.model_name = model_name
        
        # FIXED: More conservative Ollama configuration
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.1,
            num_ctx=2048,        # Increased context window
            num_predict=400,     # Increased output length
            repeat_penalty=1.1,
            top_k=20,
            top_p=0.9,
            # CRITICAL: Add timeout configurations
            timeout=180,         # 3 minutes total timeout
            request_timeout=180, # HTTP request timeout
            # Performance optimizations
            num_thread=4,        # Reduce CPU threads to prevent overload
            num_batch=256,       # Smaller batch size
        )
        
        # Connection pool settings for better stability
        self.max_retries = 3
        self.retry_delay = 2  # seconds
        
        # Other initialization code remains the same...
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

    def _check_ollama_health(self) -> bool:
        """Check if Ollama service is responsive."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False

    def _wait_for_ollama(self, max_wait: int = 30) -> bool:
        """Wait for Ollama to become available."""
        for i in range(max_wait):
            if self._check_ollama_health():
                return True
            logger.info(f"Waiting for Ollama... ({i+1}/{max_wait})")
            time.sleep(1)
        return False

    def _process_with_ollama_retry(self, content: str, max_retries: int = 3) -> str:
        """Process content with Ollama with retry logic."""
        for attempt in range(max_retries):
            try:
                # Check Ollama health before processing
                if not self._check_ollama_health():
                    logger.warning(f"Ollama not responsive, attempt {attempt + 1}")
                    if not self._wait_for_ollama():
                        raise Exception("Ollama service unavailable")
                
                # Process with timeout
                logger.info(f"Processing with Ollama (attempt {attempt + 1})")
                
                # Use a simple prompt for faster processing
                prompt = f"""Summarize this research paper content in a structured format:

{content[:2000]}  # Limit content to prevent huge context

Please provide:
1. Main objective
2. Key methodology
3. Important results
4. Contributions

Keep it concise and informative."""
                
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
                
            except Exception as e:
                logger.warning(f"Ollama processing attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"All Ollama processing attempts failed: {e}")
                    return f"Processing failed after {max_retries} attempts: {str(e)}"

    def _extract_structured_info(self, text: str) -> Dict[str, str]:
        """Extract structured information with complete sentences and better formatting."""
        if not text or not text.strip():
            return {}
        
        # Convert to lowercase for pattern matching
        text_lower = text.lower()
        
        # Initialize result dictionary
        extracted_info = {}
        
        try:
            # Extract information for each section
            for section, patterns in self.section_patterns.items():
                extracted_content = []
                
                for pattern in patterns:
                    # Find sentences containing the pattern
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if re.search(pattern, sentence.lower()) and len(sentence.strip()) > 20:
                            # Clean and add the sentence
                            clean_sentence = sentence.strip()
                            if clean_sentence and clean_sentence not in extracted_content:
                                extracted_content.append(clean_sentence)
                
                # Combine and format the extracted content
                if extracted_content:
                    # Take the best 2-3 sentences for each section
                    best_content = extracted_content[:3]
                    extracted_info[section] = ' '.join(best_content)
                else:
                    extracted_info[section] = "Not clearly identified in the text"
            
            return extracted_info
            
        except Exception as e:
            logger.error(f"Error in _extract_structured_info: {e}")
            # Return empty dict with section keys to prevent None issues
            return {section: "Extraction failed" for section in self.section_patterns.keys()}

    def _format_basic_output(self, structured_info: Dict[str, str]) -> str:
        """Format structured info without Ollama processing."""
        # FIXED: Add None check and ensure we have a dictionary
        if not structured_info or not isinstance(structured_info, dict):
            return "No structured information could be extracted from the text."
        
        sections = []
        for section, content in structured_info.items():
            if content and content != "Can't access" and content != "Extraction failed":
                section_title = ' '.join(word.capitalize() for word in section.split('_'))
                sections.append(f"### {section_title}\n\n{content}\n")
        
        return '\n'.join(sections) if sections else "No structured information extracted."

    def _process_single_paper_safe(self, paper: Dict[str, Any], paper_num: int) -> Dict[str, Any]:
        """Process a single paper with enhanced error handling."""
        title = paper.get("title", f"Paper {paper_num}")
        content = paper.get("content", "")
        source = paper.get("source", "")
        
        logger.info(f"Processing paper {paper_num}: {title[:50]}...")
        
        try:
            if not content.strip():
                return {
                    "title": title,
                    "summary": "No content available for processing.",
                    "source": source,
                    "method": "skipped",
                    "status": "skipped"
                }
            
            # Extract basic structured info first (fast, no Ollama)
            structured_info = self._extract_structured_info(content)
            
            # FIXED: Ensure structured_info is always a dictionary
            if not isinstance(structured_info, dict):
                structured_info = {}
            
            # Format basic output
            basic_summary = self._format_basic_output(structured_info)
            
            # Try Ollama enhancement (with fallback)
            try:
                enhanced_summary = self._process_with_ollama_retry(content)
                final_summary = f"{enhanced_summary}\n\n---\n\n{basic_summary}"
                processing_method = "ollama_enhanced"
            except Exception as e:
                logger.warning(f"Ollama enhancement failed for {title}: {e}")
                final_summary = basic_summary
                processing_method = "pattern_extraction_only"
            
            # Add source information
            source_note = self._format_source(source)
            full_summary = f"{final_summary}\n\n{source_note}"
            
            return {
                "title": title,
                "summary": full_summary,
                "structured_info": structured_info,
                "source": source,
                "method": processing_method,
                "status": "completed"
            }
            
        except Exception as e:
            logger.error(f"Error processing paper {paper_num}: {e}")
            return {
                "title": title,
                "summary": f"[Processing error: {str(e)}]",
                "source": source,
                "method": "error",
                "status": "error"
            }

    def _sequential_process_with_delay(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process papers sequentially with delays to prevent Ollama overload."""
        summaries = []
        
        for idx, paper in enumerate(papers):
            logger.info(f"Processing paper {idx + 1}/{len(papers)}")
            
            # Process single paper
            summary = self._process_single_paper_safe(paper, idx + 1)
            summaries.append(summary)
            
            # Add delay between papers to prevent Ollama overload
            if idx < len(papers) - 1:  # Don't delay after last paper
                time.sleep(1)  # 1 second delay between papers
        
        return summaries

    def _parallel_process_conservative(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process papers in parallel with conservative settings."""
        summaries = []
        
        # FIXED: Reduce max_workers to prevent Ollama overload
        max_workers = min(2, len(papers))  # Max 2 concurrent processes
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_paper = {}
            for idx, paper in enumerate(papers):
                future = executor.submit(self._process_single_paper_safe, paper, idx + 1)
                future_to_paper[future] = (paper, idx + 1)
                
                # Add small delay between submissions
                time.sleep(0.5)
            
            # Collect results with increased timeout
            for future in concurrent.futures.as_completed(future_to_paper, timeout=300):  # 5 minutes total
                try:
                    summary = future.result(timeout=180)  # 3 minutes per paper
                    summaries.append(summary)
                except Exception as e:
                    paper, paper_num = future_to_paper[future]
                    logger.error(f"Error processing paper {paper_num}: {e}")
                    summaries.append({
                        "title": paper.get("title", f"Paper {paper_num}"),
                        "summary": f"[Processing timeout/error: {str(e)}]",
                        "source": paper.get("source", ""),
                        "method": "timeout_error",
                        "status": "error"
                    })
        
        return summaries

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Process the state with enhanced error handling and timeouts."""
        start_time = time.time()
        
        try:
            state["current_stage"] = "structured_extraction"
            
            if not state.get("parsed_content"):
                logger.warning("No parsed content found in state")
                state["error"] = "No parsed content available for summarization"
                return state

            papers = state["parsed_content"]
            total_papers = len(papers)
            
            logger.info(f"Processing {total_papers} papers with structured extraction...")
            
            # Check Ollama health before starting
            if not self._check_ollama_health():
                logger.warning("Ollama service not responsive, waiting...")
                if not self._wait_for_ollama():
                    logger.error("Ollama service unavailable, using pattern extraction only")
                    # Fall back to pattern extraction only
                    summaries = []
                    for idx, paper in enumerate(papers):
                        summary = self._process_single_paper_safe(paper, idx + 1)
                        summaries.append(summary)
                else:
                    logger.info("Ollama service restored, proceeding with processing")
            
            # Choose processing method based on paper count
            if total_papers <= 3:
                # Use parallel processing for small batches
                summaries = self._parallel_process_conservative(papers)
            else:
                # Use sequential processing for larger batches
                logger.info("Using sequential processing to prevent Ollama overload")
                summaries = self._sequential_process_with_delay(papers)
            
            # Update state
            state["summaries"] = summaries
            state["current_stage"] = "structured_extraction_complete"
            
            # Log performance
            elapsed_time = time.time() - start_time
            completed = sum(1 for s in summaries if s.get("status") == "completed")
            errors = sum(1 for s in summaries if s.get("status") == "error")
            
            logger.info(f"Processing complete: {completed} succeeded, {errors} failed, {elapsed_time:.2f}s total")
            
            return state
            
        except Exception as e:
            logger.error(f"Critical error in structured extractor: {str(e)}")
            state["error"] = f"Structured extraction failed: {str(e)}"
            state["current_stage"] = "extraction_failed"
            return state
    
    def _format_source(self, source: str) -> str:
        """Format source information."""
        if not source:
            return "**Source:** Not specified"
        elif "arxiv" in source.lower():
            return "**Source:** ArXiv"
        elif "doi.org" in source.lower():
            return f"**Source:** DOI"
        else:
            return f"**Source:** {source[:50]}..." if len(source) > 50 else f"**Source:** {source}"