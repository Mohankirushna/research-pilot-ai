import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class NoteSaverNode:
    def __init__(self, output_dir: str = "."):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _sanitize_filename(self, filename: str) -> str:
        """Remove or replace invalid filename characters."""
        return re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata dictionary as markdown."""
        if not metadata:
            return ""
            
        lines = ["### Metadata\n"]
        for key, value in metadata.items():
            if value and value not in ["", "N/A"]:
                if isinstance(value, (list, tuple)):
                    value = ", ".join(str(v) for v in value if v)
                lines.append(f"- **{key.replace('_', ' ').title()}:** {value}")
        return "\n".join(lines) + "\n"

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Save research notes to a markdown file."""
        topic = state.get("topic", "research_notes")
        sanitized_topic = self._sanitize_filename(topic)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{timestamp}_{sanitized_topic[:50]}.md"
        
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                # Header
                f.write(f"# Research Notes: {topic}\n\n")
                f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
                
                # Topic Explanation
                if "explanation" in state and state["explanation"]:
                    f.write("## Topic Explanation\n\n")
                    f.write(f"{state['explanation']}\n\n")
                
                # Related Topics
                if "related_topics" in state and state["related_topics"]:
                    f.write("## Related Topics\n\n")
                    if isinstance(state["related_topics"], str):
                        f.write(f"{state['related_topics']}\n\n")
                    elif isinstance(state["related_topics"], (list, tuple)):
                        f.write("\n".join(f"- {t}" for t in state["related_topics"]))
                        f.write("\n\n")
                
                # Search Results
                if "search_results" in state and state["search_results"]:
                    f.write("## Search Results\n\n")
                    f.write(f"Found {len(state['search_results'])} papers.\n\n")
                    
                    for i, paper in enumerate(state["search_results"], 1):
                        f.write(f"### {i}. {paper.get('title', 'Untitled')}\n")
                        
                        if 'snippet' in paper and paper['snippet']:
                            f.write(f"**Abstract:** {paper['snippet']}\n\n")
                        
                        if 'link' in paper and paper['link']:
                            f.write(f"**🔗 [View Paper]({paper['link']})**  ")
                        
                        if 'publication_info' in paper and paper['publication_info']:
                            f.write(f"**Published in:** {paper['publication_info']}  ")
                        
                        if 'year' in paper and paper['year']:
                            f.write(f"**Year:** {paper['year']}")
                        
                        f.write("\n\n---\n\n")
                
                # Research Summaries
                if "summaries" in state and state["summaries"]:
                    f.write("## Research Summaries\n\n")
                    
                    for i, summary_info in enumerate(state["summaries"], 1):
                        if not isinstance(summary_info, dict):
                            continue
                            
                        title = summary_info.get("title", f"Summary {i}")
                        summary = summary_info.get("summary", "")
                        source = summary_info.get("source")
                        metadata = summary_info.get("metadata", {})
                        
                        f.write(f"### {i}. {title}\n\n")
                        
                        if source:
                            f.write(f"**Source:** [{source}]({source})\n\n")
                        
                        if metadata:
                            f.write(self._format_metadata(metadata))
                        
                        f.write(f"{summary}\n\n---\n\n")
                else:
                    f.write("## Research Summaries\n\nNo paper summaries were generated.\n")
                
                # Add footer
                f.write("\n---\n")
                f.write("*This document was automatically generated by the Research Assistant.*\n")
            
            logger.info(f"Notes successfully saved to: {output_file.absolute()}")
            state["notes_file"] = str(output_file.absolute())
            
        except Exception as e:
            logger.error(f"Error saving notes to {output_file}: {str(e)}")
            state["error"] = f"Failed to save notes: {str(e)}"
        
        return state
