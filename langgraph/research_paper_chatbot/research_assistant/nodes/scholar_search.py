import os
import time
import logging
from serpapi.google_search import GoogleSearch
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class ScholarSearchNode:
    def __init__(self, max_results: int = 10, year_min: Optional[int] = None):
        self.max_results = max_results
        self.year_min = year_min
        self.api_key = os.getenv("SERPAPI_API_KEY")
        if not self.api_key:
            logger.error("SERPAPI_API_KEY not found in environment variables")

    def _build_search_query(self, topic: str) -> str:
        """Create a focused search query for the given topic."""
        # Use the exact topic as the main query
        query = f'"{topic}"'
        
        # Add year filter if specified
        if self.year_min:
            query += f" after:{self.year_min}"
            
        logger.info(f"Search query: {query}")
        return query

    def _search_scholar(self, query: str) -> Optional[Dict]:
        """Execute the search with error handling and retries."""
        if not self.api_key:
            logger.error("API key not available")
            return None
            
        params = {
            "q": query,
            "engine": "google_scholar",
            "api_key": self.api_key,
            "num": min(self.max_results, 20),  # Max 20 results per page
            "as_ylo": self.year_min,
            "hl": "en",
            "as_vis": "1"  # Include both citations and versions
        }

        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            
            if "error" in results:
                logger.error(f"API Error: {results.get('error')}")
                return None
                
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return None

    def __call__(self, state: Dict) -> Dict:
        if not self.api_key:
            logger.error("SERPAPI_API_KEY not configured")
            state["search_error"] = "API key not configured"
            return state
            
        topic = state.get("topic", "")
        if not topic:
            logger.error("No topic provided")
            state["search_error"] = "No topic provided"
            return state

        logger.info(f"Searching for papers on: {topic}")
        
        # Build enhanced query
        query = self._build_search_query(topic)
        logger.debug(f"Search query: {query}")

        # Execute search
        results = self._search_scholar(query)
        if not results:
            state["search_error"] = "No results found or search failed"
            return state

        # Process results
        papers = []
        if "organic_results" in results:
            for i, result in enumerate(results["organic_results"][:self.max_results], 1):
                paper = {
                    "title": result.get("title", "No title"),
                    "link": result.get("link"),
                    "snippet": result.get("snippet", ""),
                    "publication_info": result.get("publication_info", {}).get("summary"),
                    "year": next((y for y in range(2025, 1900, -1) if str(y) in result.get("snippet", "")), None),
                    "result_id": i  # Add a sequential ID for reference
                }
                logger.debug(f"Found paper {i}: {paper.get('title', 'Untitled')}")
                if paper["link"]:
                    logger.debug(f"  - Has direct link: {paper['link']}")
                else:
                    logger.debug("  - No direct link available")
                papers.append(paper)

        logger.info(f"Found {len(papers)} papers")
        state["search_results"] = papers
        state["papers"] = papers  # Keep for backward compatibility
        return state
