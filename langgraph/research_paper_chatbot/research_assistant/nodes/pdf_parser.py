import logging
import os
import PyPDF2
from typing import Dict, List, Any, Optional
import io
import requests

logger = logging.getLogger(__name__)

class PDFParserNode:
    def __init__(self, max_pages: int = 10):
        self.max_pages = max_pages  # Limit pages to process per PDF

    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from a local PDF file."""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                # Limit the number of pages to process
                for page_num in range(min(len(reader.pages), self.max_pages)):
                    text += reader.pages[page_num].extract_text() + "\n\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {str(e)}")
            return None

    def _extract_text_from_url(self, url: str) -> Optional[str]:
        """Extract text directly from a PDF URL."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            with io.BytesIO(response.content) as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                # Limit the number of pages to process
                for page_num in range(min(len(reader.pages), self.max_pages)):
                    text += reader.pages[page_num].extract_text() + "\n\n"
                return text.strip()
        except Exception as e:
            logger.warning(f"Error extracting text from URL {url}: {str(e)}")
            return None

    def __call__(self, state: Dict) -> Dict:
        """Process papers and extract text from their PDFs or use available metadata."""
        papers = []
        # Check for downloaded files first
        if "downloaded_files" in state and state["downloaded_files"]:
            papers = [{"file_path": path} for path in state["downloaded_files"]]
        # Fall back to selected papers
        elif "selected_papers" in state and state["selected_papers"]:
            papers = state["selected_papers"]
        # Fall back to search results
        elif "search_results" in state and state["search_results"]:
            papers = state["search_results"]
            logger.info(f"Using search results as fallback: {len(papers)} papers")
        
        parsed_content = []
        
        for paper in papers:
            try:
                if not isinstance(paper, dict):
                    logger.warning(f"Skipping invalid paper format: {paper}")
                    continue
                
                content = None
                source = ""
                title = paper.get("title", "Untitled")
                
                # Try to get content from local file first
                if "file_path" in paper and paper["file_path"] and os.path.exists(paper["file_path"]):
                    content = self._extract_text_from_pdf(paper["file_path"])
                    source = paper["file_path"]
                # Fall back to URL if local file doesn't exist or failed
                elif "pdf_url" in paper and paper["pdf_url"]:
                    content = self._extract_text_from_url(paper["pdf_url"])
                    source = paper["pdf_url"]
                elif "link" in paper and paper["link"] and paper["link"].endswith('.pdf'):
                    content = self._extract_text_from_url(paper["link"])
                    source = paper["link"]
                # Fall back to abstract if available
                elif "abstract" in paper and paper["abstract"]:
                    content = paper["abstract"]
                    source = paper.get("source", "")
                elif "snippet" in paper and paper["snippet"]:
                    content = paper["snippet"]
                    source = paper.get("link", "")
                
                if content:
                    logger.info(f"Successfully parsed content for: {title}")
                    parsed_content.append({
                        "title": title,
                        "content": content,
                        "source": source
                    })
                else:
                    logger.warning(f"Could not extract content for paper: {title}")
                    
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('title', 'Unknown')}: {str(e)}")
                continue
        
        if not parsed_content and papers:
            logger.warning("No content could be parsed from any papers. Please check the logs for details.")
        else:
            logger.info(f"Successfully parsed {len(parsed_content)} out of {len(papers)} papers")
        
        state["parsed_content"] = parsed_content
        return state
