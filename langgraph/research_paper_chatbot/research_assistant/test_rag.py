import os
import sys
import time
import PyPDF2
import re
from typing import List, Dict, Tuple
from collections import Counter

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import the original summarizer, fallback to simple implementation
try:
    from research_assistant.nodes.rag_summarizer import HybridSummarizerNode
    HAVE_ORIGINAL_SUMMARIZER = True
except ImportError:
    HAVE_ORIGINAL_SUMMARIZER = False
    print("Original summarizer not found, using fast fallback implementation")

class AdvancedFastSummarizer:
    """Fast summarizer that processes all pages and extracts key information"""
    
    def __init__(self):
        # Pre-compile regex patterns for speed
        self.sentence_pattern = re.compile(r'[.!?]+\s+')
        self.word_pattern = re.compile(r'\b\w+\b')
        self.number_pattern = re.compile(r'\d+(?:\.\d+)?%?')
        
        # Enhanced keyword categories
        self.keyword_categories = {
            'research_methods': {
                'methodology', 'method', 'approach', 'technique', 'procedure',
                'experiment', 'study', 'analysis', 'evaluation', 'assessment',
                'survey', 'interview', 'questionnaire', 'data collection'
            },
            'ai_education': {
                'artificial intelligence', 'machine learning', 'deep learning',
                'neural network', 'algorithm', 'automated', 'intelligent',
                'personalized learning', 'adaptive learning', 'educational technology'
            },
            'education_concepts': {
                'students', 'learning', 'teaching', 'education', 'academic',
                'curriculum', 'pedagogy', 'classroom', 'instructor', 'teacher',
                'performance', 'achievement', 'assessment', 'engagement'
            },
            'results_findings': {
                'results', 'findings', 'conclusion', 'discovered', 'found',
                'showed', 'demonstrated', 'revealed', 'indicated', 'suggests',
                'evidence', 'significant', 'improvement', 'effective'
            },
            'key_metrics': {
                'accuracy', 'precision', 'recall', 'f1-score', 'performance',
                'effectiveness', 'efficiency', 'score', 'rating', 'percentage'
            }
        }
        
        # Section headers to identify
        self.section_headers = {
            'abstract', 'introduction', 'literature review', 'methodology',
            'results', 'discussion', 'conclusion', 'future work', 'references'
        }
    
    def _extract_page_content(self, page_text: str, page_num: int) -> Dict:
        """Extract structured content from a page"""
        content = {
            'page_num': page_num,
            'text': page_text,
            'sentences': [],
            'keywords': Counter(),
            'numbers': [],
            'sections': []
        }
        
        # Extract sentences
        sentences = [s.strip() for s in self.sentence_pattern.split(page_text) if len(s.strip()) > 20]
        content['sentences'] = sentences
        
        # Extract and categorize keywords
        text_lower = page_text.lower()
        for category, keywords in self.keyword_categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    content['keywords'][keyword] += text_lower.count(keyword)
        
        # Extract numbers and percentages
        content['numbers'] = self.number_pattern.findall(page_text)
        
        # Identify sections
        for section in self.section_headers:
            if section in text_lower:
                content['sections'].append(section)
        
        return content
    
    def _score_sentence_importance(self, sentence: str, global_keywords: Counter) -> float:
        """Score sentence importance based on multiple factors"""
        sentence_lower = sentence.lower()
        score = 0
        
        # Length factor (prefer medium-length sentences)
        word_count = len(self.word_pattern.findall(sentence))
        if 15 <= word_count <= 40:
            score += 2
        elif 10 <= word_count <= 50:
            score += 1
        
        # Keyword importance
        for keyword, count in global_keywords.most_common(20):
            if keyword in sentence_lower:
                score += count * 0.5
        
        # Numbers and metrics
        if self.number_pattern.search(sentence):
            score += 1.5
        
        # Question or statement strength
        if sentence.strip().endswith('?'):
            score += 0.5
        if any(word in sentence_lower for word in ['significant', 'important', 'crucial', 'key']):
            score += 1
        
        # Avoid very short or very long sentences
        if word_count < 8 or word_count > 60:
            score -= 1
        
        return score
    
    def _extract_key_insights(self, all_content: List[Dict]) -> Dict:
        """Extract key insights from all pages"""
        insights = {
            'document_structure': [],
            'main_topics': [],
            'key_findings': [],
            'methodology': [],
            'results_metrics': [],
            'important_sentences': []
        }
        
        # Combine all keywords
        global_keywords = Counter()
        all_sentences = []
        
        for page_content in all_content:
            global_keywords.update(page_content['keywords'])
            all_sentences.extend([
                (sent, page_content['page_num']) 
                for sent in page_content['sentences']
            ])
            
            # Track document structure
            if page_content['sections']:
                insights['document_structure'].extend([
                    f"Page {page_content['page_num']}: {section.title()}"
                    for section in page_content['sections']
                ])
        
        # Get main topics
        insights['main_topics'] = [
            f"{keyword} (mentioned {count} times)"
            for keyword, count in global_keywords.most_common(10)
            if count > 1
        ]
        
        # Score and select important sentences
        scored_sentences = []
        for sentence, page_num in all_sentences:
            score = self._score_sentence_importance(sentence, global_keywords)
            scored_sentences.append((score, sentence, page_num))
        
        # Sort by score and select top sentences
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Categorize sentences
        for score, sentence, page_num in scored_sentences[:15]:
            sentence_lower = sentence.lower()
            
            # Categorize based on content
            if any(word in sentence_lower for word in ['method', 'approach', 'technique']):
                if len(insights['methodology']) < 3:
                    insights['methodology'].append(f"[Page {page_num}] {sentence}")
            elif any(word in sentence_lower for word in ['result', 'finding', 'conclusion']):
                if len(insights['key_findings']) < 4:
                    insights['key_findings'].append(f"[Page {page_num}] {sentence}")
            elif self.number_pattern.search(sentence):
                if len(insights['results_metrics']) < 3:
                    insights['results_metrics'].append(f"[Page {page_num}] {sentence}")
            else:
                if len(insights['important_sentences']) < 5:
                    insights['important_sentences'].append(f"[Page {page_num}] {sentence}")
        
        return insights
    
    def summarize_full_document(self, all_pages_text: List[Tuple[str, int]]) -> str:
        """Create comprehensive summary from all pages"""
        # Process all pages
        all_content = []
        for page_text, page_num in all_pages_text:
            if page_text.strip():
                content = self._extract_page_content(page_text, page_num)
                all_content.append(content)
        
        # Extract insights
        insights = self._extract_key_insights(all_content)
        
        # Build comprehensive summary
        summary_parts = []
        
        # Document overview
        summary_parts.append("📄 **COMPREHENSIVE DOCUMENT ANALYSIS**")
        summary_parts.append(f"📊 **Pages Processed**: {len(all_content)}")
        
        # Document structure
        if insights['document_structure']:
            summary_parts.append("\n📋 **Document Structure**:")
            for structure in insights['document_structure'][:8]:
                summary_parts.append(f"  • {structure}")
        
        # Main topics
        if insights['main_topics']:
            summary_parts.append("\n🔍 **Main Topics & Keywords**:")
            for topic in insights['main_topics'][:8]:
                summary_parts.append(f"  • {topic}")
        
        # Key findings
        if insights['key_findings']:
            summary_parts.append("\n🎯 **Key Findings & Conclusions**:")
            for i, finding in enumerate(insights['key_findings'], 1):
                summary_parts.append(f"  {i}. {finding}")
        
        # Methodology
        if insights['methodology']:
            summary_parts.append("\n⚙️ **Methodology & Approach**:")
            for method in insights['methodology']:
                summary_parts.append(f"  • {method}")
        
        # Results and metrics
        if insights['results_metrics']:
            summary_parts.append("\n📈 **Results & Metrics**:")
            for metric in insights['results_metrics']:
                summary_parts.append(f"  • {metric}")
        
        # Important sentences
        if insights['important_sentences']:
            summary_parts.append("\n💡 **Additional Important Information**:")
            for sentence in insights['important_sentences']:
                summary_parts.append(f"  • {sentence}")
        
        return "\n".join(summary_parts)

def extract_all_pages_optimized(pdf_path: str) -> List[Tuple[str, int]]:
    """Extract text from all pages with optimization"""
    pages_text = []
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            total_pages = len(reader.pages)
            
            print(f"📖 Processing {total_pages} pages...")
            
            for page_num in range(total_pages):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text.strip():
                        pages_text.append((page_text, page_num + 1))
                        
                        # Progress indicator for large documents
                        if (page_num + 1) % 5 == 0:
                            print(f"  ✓ Processed {page_num + 1}/{total_pages} pages")
                            
                except Exception as e:
                    print(f"⚠️ Error processing page {page_num + 1}: {str(e)}")
                    continue
        
        print(f"✅ Successfully extracted text from {len(pages_text)} pages")
        return pages_text
        
    except Exception as e:
        print(f"❌ Error reading PDF: {str(e)}")
        return []

def get_pdf_metadata_detailed(pdf_path: str) -> Dict[str, str]:
    """Extract detailed metadata"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            metadata = reader.metadata or {}
            
            return {
                'title': metadata.get('/Title', 'Unknown'),
                'author': metadata.get('/Author', 'Unknown'),
                'creator': metadata.get('/Creator', 'Unknown'),
                'producer': metadata.get('/Producer', 'Unknown'),
                'pages': str(len(reader.pages)),
                'size': f"{os.path.getsize(pdf_path) / 1024:.1f} KB",
                'created': str(metadata.get('/CreationDate', 'Unknown')),
                'modified': str(metadata.get('/ModDate', 'Unknown'))
            }
    except Exception as e:
        return {'error': str(e)}

def main():
    print("🚀 Starting Comprehensive PDF Analyzer...")
    print("📌 This version processes ALL pages for complete analysis")
    overall_start = time.time()
    
    # Path to the PDF
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_path = os.path.join(base_dir, "downloaded_pdfs", "Role_of_AI_in_Education.pdf")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please ensure the PDF file exists in the specified location.")
        return
    
    # Get detailed metadata
    print("📊 Extracting detailed PDF metadata...")
    metadata = get_pdf_metadata_detailed(pdf_path)
    
    # Extract ALL pages
    print("📄 Extracting text from ALL pages...")
    extraction_start = time.time()
    all_pages_text = extract_all_pages_optimized(pdf_path)
    extraction_time = time.time() - extraction_start
    
    if not all_pages_text:
        print("❌ Failed to extract text from PDF.")
        return
    
    total_chars = sum(len(page_text) for page_text, _ in all_pages_text)
    print(f"✅ Extracted {total_chars:,} characters from {len(all_pages_text)} pages in {extraction_time:.2f}s")
    
    # Analyze with advanced summarizer
    analysis_start = time.time()
    print("🔍 Performing comprehensive analysis...")
    
    summarizer = AdvancedFastSummarizer()
    summary = summarizer.summarize_full_document(all_pages_text)
    
    analysis_time = time.time() - analysis_start
    total_time = time.time() - overall_start
    
    # Print results
    print("\n" + "="*80)
    print("📋 DOCUMENT METADATA")
    print("="*80)
    for key, value in metadata.items():
        if key != 'error':
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    print("\n" + "="*80)
    print("📝 COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(summary)
    
    print("\n" + "="*80)
    print("⏱️ PERFORMANCE METRICS")
    print("="*80)
    print(f"Text Extraction: {extraction_time:.2f}s")
    print(f"Analysis & Summarization: {analysis_time:.2f}s")
    print(f"Total Processing Time: {total_time:.2f}s")
    print(f"Processing Speed: {total_chars/total_time:.0f} chars/second")
    
    if total_time < 10:
        print("🎯 EXCELLENT: Comprehensive analysis completed quickly!")
    elif total_time < 20:
        print("✅ GOOD: Reasonable processing time for full document")
    else:
        print("⚠️ SLOW: Consider optimizations for very large documents")
    
    print(f"\n📌 Complete analysis of all {len(all_pages_text)} pages with {total_chars:,} characters")

if __name__ == "__main__":
    main()