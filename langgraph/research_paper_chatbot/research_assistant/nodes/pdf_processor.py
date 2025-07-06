import os
import re
import logging
import requests
import PyPDF2
import io
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, download_dir: str = "downloaded_pdfs"):
        """Initialize the PDF processor with download directory."""
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/pdf,text/html,application/xhtml+xml',
        }

    def download_pdf(self, url: str, paper_title: str) -> Optional[str]:
        """Download a PDF from a URL and return the local file path."""
        try:
            # Clean and create a safe filename
            safe_title = re.sub(r'[^\w\s-]', '', paper_title).strip().replace(' ', '_')
            filename = f"{safe_title[:100]}.pdf"
            filepath = self.download_dir / filename
            
            # Skip if already downloaded
            if filepath.exists():
                return str(filepath)

            # Download the PDF
            response = requests.get(url, headers=self.headers, stream=True, timeout=30)
            response.raise_for_status()
            
            # Verify content type
            content_type = response.headers.get('content-type', '').lower()
            if 'application/pdf' not in content_type and not url.lower().endswith('.pdf'):
                logger.warning(f"URL does not appear to be a PDF: {url}")
                return None
            
            # Save the PDF
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Downloaded PDF to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {str(e)}")
            return None

    def extract_text_from_pdf(self, filepath: str) -> Dict[str, str]:
        """Extract text from a PDF file with section-wise splitting."""
        try:
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                
                # Extract text from each page
                for page in reader.pages:
                    text += page.extract_text() + "\n\n"
                
                # Basic section detection (can be enhanced with more sophisticated parsing)
                sections = self._split_into_sections(text)
                
                return {
                    'full_text': text,
                    'sections': sections,
                    'page_count': len(reader.pages)
                }
                
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filepath}: {str(e)}")
            return {'error': str(e)}

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """Split text into sections based on common section headers."""
        # Common section headers in research papers
        section_headers = [
            'abstract', 'introduction', 'related work', 'methodology',
            'experiments', 'results', 'discussion', 'conclusion',
            'references', 'acknowledgments'
        ]
        
        sections = {}
        current_section = 'preamble'
        current_text = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line matches a section header
            is_header = False
            for header in section_headers:
                if line.lower().startswith(header):
                    # Save previous section
                    if current_text:
                        sections[current_section] = '\n'.join(current_text).strip()
                    # Start new section
                    current_section = header.lower()
                    current_text = []
                    is_header = True
                    break
            
            if not is_header:
                current_text.append(line)
        
        # Add the last section
        if current_text:
            sections[current_section] = '\n'.join(current_text).strip()
        
        return sections

    def process_paper(self, url: str, title: str) -> Dict:
        """Process a paper by downloading and extracting its content."""
        # Download the PDF
        pdf_path = self.download_pdf(url, title)
        if not pdf_path:
            return {'error': 'Failed to download PDF'}
        
        # Extract text and sections
        content = self.extract_text_from_pdf(pdf_path)
        if 'error' in content:
            return content
        
        return {
            'title': title,
            'url': url,
            'local_path': pdf_path,
            **content
        }

class PDFProcessorNode:
    """Node for processing PDFs in the research pipeline."""
    
    def __init__(self, download_dir: str = "downloaded_pdfs"):
        self.processor = PDFProcessor(download_dir)
    
    def __call__(self, state: Dict) -> Dict:
        """Process PDFs from the state."""
        if 'search_results' not in state or not state['search_results']:
            logger.warning("No search results found in state")
            return state
        
        processed_papers = []
        
        for paper in state['search_results']:
            try:
                if not isinstance(paper, dict):
                    continue
                    
                url = paper.get('pdf_url') or paper.get('link')
                title = paper.get('title', 'Untitled')
                
                if not url:
                    logger.warning(f"No URL found for paper: {title}")
                    continue
                
                logger.info(f"Processing paper: {title}")
                processed = self.processor.process_paper(url, title)
                
                if 'error' not in processed:
                    processed_papers.append({
                        **paper,
                        'processed_content': processed
                    })
                
            except Exception as e:
                logger.error(f"Error processing paper {paper.get('title', 'unknown')}: {str(e)}")
                continue
        
        state['processed_papers'] = processed_papers
        logger.info(f"Processed {len(processed_papers)} papers")
        return state
