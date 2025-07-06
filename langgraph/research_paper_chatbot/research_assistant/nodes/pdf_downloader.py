import logging
import requests
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse
import os

logger = logging.getLogger(__name__)

class PDFDownloaderNode:
    def __init__(self, download_dir: str = "downloaded_pdfs"):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def _is_pdf_link(self, url: str) -> bool:
        """Check if the URL points to a PDF file."""
        parsed = urlparse(url)
        return parsed.path.lower().endswith('.pdf')

    def _download_pdf(self, url: str, paper_title: str) -> Optional[Dict[str, Any]]:
        """Download a PDF and return its local path and metadata."""
        try:
            if not self._is_pdf_link(url):
                logger.info(f"Skipping non-PDF URL: {url}")
                return None
                
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Create a safe filename from the paper title
            safe_title = "".join(c if c.isalnum() else "_" for c in paper_title[:50]).strip('_')
            pdf_path = os.path.join(self.download_dir, f"{safe_title}.pdf")
            
            with open(pdf_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
            logger.info(f"Downloaded PDF: {pdf_path}")
            return {
                "path": pdf_path,
                "url": url,
                "status": "downloaded"
            }
            
        except Exception as e:
            logger.warning(f"Failed to download PDF from {url}: {str(e)}")
            return None

    def __call__(self, state: Dict) -> Dict:
        """Process papers and attempt to download their PDFs."""
        papers = state.get("search_results") or state.get("selected_papers") or []
        processed_papers = []
        
        for paper in papers:
            paper_info = {
                "title": paper.get("title", "Untitled"),
                "link": paper.get("link"),
                "snippet": paper.get("snippet", ""),
                "publication_info": paper.get("publication_info"),
                "year": paper.get("year"),
                "download_status": "not_attempted"
            }
            
            # Try to download PDF if we have a link
            if paper_info["link"]:
                download_result = self._download_pdf(paper_info["link"], paper_info["title"])
                if download_result:
                    paper_info.update({
                        "pdf_path": download_result["path"],
                        "download_status": "success"
                    })
                else:
                    paper_info["download_status"] = "failed"
            
            processed_papers.append(paper_info)
        
        logger.info(f"Processed {len(processed_papers)} papers. "
                   f"Successfully downloaded {len([p for p in processed_papers if p.get('download_status') == 'success'])} PDFs.")
        
        # Update the state with processed papers
        state["processed_papers"] = processed_papers
        state["pdf_links"] = [p["link"] for p in processed_papers if p.get("link")]
        return state
