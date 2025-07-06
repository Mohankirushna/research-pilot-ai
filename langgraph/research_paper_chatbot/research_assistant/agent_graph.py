import logging
import json
import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_state(state: Dict, node_name: str):
    """Log the current state in a readable format."""
    logger.info(f"\n{'='*50}")
    logger.info(f"STATE AFTER {node_name.upper()}")
    logger.info("-"*50)
    
    # Log basic state info
    for key in ['topic', 'explanation', 'related_topics', 'search_error']:
        if key in state:
            logger.info(f"{key.upper()}: {state[key]}")
    
    # Log search results count if available
    if 'search_results' in state and state['search_results']:
        logger.info(f"SEARCH RESULTS: {len(state['search_results'])} papers found")
        for i, paper in enumerate(state['search_results'][:3], 1):
            logger.info(f"  {i}. {paper.get('title', 'No title')}")
            if i == 3 and len(state['search_results']) > 3:
                logger.info(f"  ... and {len(state['search_results']) - 3} more")
                break
    
    # Log PDF download status
    if 'processed_papers' in state and state['processed_papers']:
        success = len([p for p in state['processed_papers'] if p.get('download_status') == 'success'])
        total = len(state['processed_papers'])
        logger.info(f"PDF DOWNLOADS: {success}/{total} successful")
    
    # Log parsed content status
    if 'parsed_content' in state and state['parsed_content'] is not None:
        logger.info(f"PARSED CONTENT: {len(state['parsed_content'])} items")
    else:
        logger.info("PARSED CONTENT: No content parsed yet")
    
    # Log summaries status
    if 'summaries' in state and state['summaries'] is not None:
        logger.info(f"SUMMARIES: {len(state['summaries'])} generated")
    elif 'summaries' in state:
        logger.info("SUMMARIES: No summaries generated (None)")
    else:
        logger.info("SUMMARIES: Not available in state")
    
    logger.info("="*50 + "\n")

class State(TypedDict):
    topic: str
    explanation: Optional[str]
    related_topics: Optional[List[str]]
    search_results: Optional[List[Dict[str, Any]]]
    selected_papers: Optional[List[Dict[str, Any]]]
    pdf_links: Optional[List[str]]
    downloaded_files: Optional[List[str]]
    parsed_content: Optional[List[Dict[str, Any]]]
    summaries: Optional[List[Dict[str, Any]]]
    notes: Optional[List[Dict[str, Any]]]
    draft: Optional[str]
    # Add error tracking
    error: Optional[str]
    current_stage: Optional[str]

from research_assistant.nodes.topic_explainer import TopicExplainerNode
from research_assistant.nodes.related_topics import RelatedTopicsNode
from research_assistant.nodes.scholar_search import ScholarSearchNode
from research_assistant.nodes.pdf_downloader import PDFDownloaderNode
from research_assistant.nodes.pdf_processor import PDFProcessorNode
from research_assistant.nodes.pdf_parser import PDFParserNode
from research_assistant.nodes.rag_summarizer import EnhancedResearchSummarizerNode
from research_assistant.nodes.note_saver import NoteSaverNode
from research_assistant.nodes.research_draft import ResearchDraftNode
from research_assistant.nodes.report_generator import ReportGeneratorNode

# Create the graph with state schema
graph = StateGraph(State)

# Add nodes with logging wrappers
def wrap_with_logging(node_func, node_name):
    def wrapper(state):
        logger.info(f"\n{'*'*20} ENTERING {node_name.upper()} {'*'*20}")
        result = node_func(state)
        log_state(result, node_name)
        return result
    return wrapper

# Add nodes with logging
graph.add_node("topic_explainer", wrap_with_logging(TopicExplainerNode(), "topic_explainer"))
graph.add_node("related_topics", wrap_with_logging(RelatedTopicsNode(), "related_topics"))
graph.add_node("scholar_search", wrap_with_logging(ScholarSearchNode(max_results=10, year_min=2020), "scholar_search"))
graph.add_node("pdf_downloader", wrap_with_logging(PDFDownloaderNode(), "pdf_downloader"))
graph.add_node("pdf_processor", wrap_with_logging(PDFProcessorNode(), "pdf_processor"))
graph.add_node("pdf_parser", wrap_with_logging(PDFParserNode(), "pdf_parser"))
# Initialize the enhanced research summarizer
summarizer = EnhancedResearchSummarizerNode(model_name="llama2")
graph.add_node("summarizer", wrap_with_logging(summarizer, "summarizer"))
graph.add_node("note_saver", NoteSaverNode(output_dir=os.path.join(os.path.dirname(__file__), "..", "notes")))
graph.add_node("research_draft", ResearchDraftNode())
graph.add_node("report_generator", ReportGeneratorNode(output_dir=os.path.join(os.path.dirname(__file__), "..", "notes")))

# Create a notes directory if it doesn't exist
notes_dir = os.path.join(os.path.dirname(__file__), "..", "notes")
os.makedirs(notes_dir, exist_ok=True)

# Define the edges with conditional routing
graph.add_edge("topic_explainer", "related_topics")
graph.add_edge("related_topics", "scholar_search")
graph.add_edge("scholar_search", "pdf_downloader")
graph.add_edge("pdf_downloader", "pdf_processor")
graph.add_edge("pdf_processor", "pdf_parser")

def should_summarize(state: State) -> str:
    """Decide whether to summarize or skip to draft generation."""
    if state.get("parsed_content") and len(state["parsed_content"]) > 0:
        return "summarizer"
    logger.warning("No parsed content available, skipping summarization")
    return "research_draft"

# Add conditional edge for summarization
graph.add_conditional_edges(
    "pdf_parser",
    should_summarize,
    {
        "summarizer": "summarizer",
        "research_draft": "research_draft"
    }
)

graph.add_edge("summarizer", "research_draft")
graph.add_edge("research_draft", "note_saver")
graph.add_edge("note_saver", "report_generator")

# Set the entry point
graph.set_entry_point("topic_explainer")

# Compile the graph with error handling
try:
    app = graph.compile()
    logger.info("Graph compiled successfully")
except Exception as e:
    logger.error(f"Error compiling graph: {str(e)}")
    raise

if __name__ == "__main__":
    try:
        logger.info("Starting research assistant...")
        # Example usage with proper state dictionary
        initial_state = {
            "topic": "AI in education",
            "explanation": None,
            "related_topics": None,
            "search_results": None,
            "selected_papers": None,
            "pdf_links": None,
            "downloaded_files": None,
            "parsed_content": None,
            "summaries": None,
            "notes": None,
            "draft": None
        }
        
        logger.info(f"Processing topic: {initial_state['topic']}")
        result = app.invoke(initial_state)
        
        # Log final state
        logger.info("\n" + "="*50)
        logger.info("PIPELINE EXECUTION COMPLETE")
        logger.info("="*50)
        logger.info(f"Final state keys: {list(result.keys())}")
        
        if 'summaries' in result and result['summaries']:
            logger.info(f"Generated {len(result['summaries'])} summaries")
            for i, summary in enumerate(result['summaries'][:3], 1):
                title = summary.get('title', 'Untitled')
                logger.info(f"\nSummary {i} ({title}):")
                logger.info(summary.get('summary', 'No summary')[:200] + "...")
        else:
            logger.warning("No summaries were generated")
            
        if 'notes_file' in result:
            logger.info(f"\nNotes saved to: {result['notes_file']}")
        
    except Exception as e:
        logger.error(f"Error in research pipeline: {str(e)}", exc_info=True)
        raise
