import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = str(Path(__file__).parent.absolute())
sys.path.append(project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import after path setup
from research_assistant.nodes.rag_summarizer import FastHybridSummarizerNode

def test_summarizer():
    # Initialize the summarizer
    summarizer = FastHybridSummarizerNode(model_name="llama2")
    
    # Test with sample research paper text
    sample_text = """
    The Role of Artificial Intelligence in Modern Education
    
    Abstract:
    This paper explores the transformative impact of artificial intelligence on education. 
    We present a comprehensive analysis of AI applications in personalized learning, 
    automated grading, and intelligent tutoring systems. Our findings demonstrate 
    significant improvements in student engagement and learning outcomes when AI tools 
    are integrated into the educational process. The study concludes with recommendations 
    for implementing AI in diverse educational settings.
    
    Introduction:
    The integration of artificial intelligence in education has opened new possibilities 
    for personalized and adaptive learning experiences. Traditional one-size-fits-all 
    teaching methods are being replaced by intelligent systems that can tailor content 
    to individual student needs.
    
    Methodology:
    We conducted a mixed-methods study involving 1,200 students across 20 institutions. 
    The study compared learning outcomes between traditional and AI-enhanced classrooms 
    over a period of 12 months. Data was collected through pre- and post-tests, 
    student surveys, and teacher interviews.
    
    Results:
    Our analysis revealed a 27% improvement in test scores among students using 
    AI-powered learning tools compared to the control group. Additionally, student 
    engagement increased by 42%, as measured by platform interaction metrics.
    
    Conclusion:
    The findings suggest that AI has the potential to revolutionize education by 
    providing personalized learning experiences at scale. However, successful 
    implementation requires careful consideration of ethical implications and 
    teacher training.
    """
    
    # Test summarization
    print("\nTesting summarizer with sample research paper text...")
    summary = summarizer._fast_summarize_chunk(sample_text)
    
    print("\n" + "="*80)
    print("SUMMARY:")
    print("="*80)
    print(summary)
    print("="*80)
    
    # Test intelligent extraction
    print("\nTesting intelligent extraction...")
    extracted = summarizer._intelligent_extraction(sample_text)
    
    print("\n" + "="*80)
    print("INTELLIGENT EXTRACTION:")
    print("="*80)
    print(extracted)
    print("="*80)

if __name__ == "__main__":
    test_summarizer()
