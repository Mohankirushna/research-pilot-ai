# 📚 AI Research Assistant

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/built%20with-LangGraph-ff69b4.svg)](https://langchain.com/)

An intelligent research assistant that automates the process of academic research, from topic exploration to comprehensive report generation using cutting-edge AI technologies.

## 🚀 Features

- **Smart Topic Analysis** - Get detailed explanations and related concepts for any research topic
- **Automated Paper Discovery** - Find relevant academic papers with smart filtering
- **AI-Powered Summarization** - Extract key insights from multiple research papers
- **Comprehensive Reporting** - Generate well-structured research reports with proper citations
- **Local Processing** - Run everything on your machine with Ollama's local LLMs

## 🏗️ Tech Stack

### Core Technologies

| Technology | Purpose |
|------------|---------|
| **Python 3.8+** | Core programming language |
| **LangGraph** | Workflow orchestration and state management |
| **Ollama** | Local LLM inference (supports llama2, mistral, etc.) |
| **scholarly** | Academic paper search and metadata extraction |
| **PyPDF2** | PDF text extraction and processing |

### Key Dependencies

- **langchain**: Framework for building LLM applications
- **pydantic**: Data validation and settings management
- **requests**: HTTP requests for API calls
- **tqdm**: Progress bars for long-running operations
- **python-dotenv**: Environment variable management

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/research-assistant-ai.git
   cd research-assistant-ai
   ```

2. **Set up a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   cd research_assistant
   pip install -r requirements.txt
   ```

4. **Set up Ollama**
   - Download and install from [ollama.ai](https://ollama.ai/)
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
   - In a new terminal, download a model:
     ```bash
     ollama pull llama2  # or mistral, etc.
     ```

## 🚀 Quick Start

```bash
python -m research_assistant.agent_graph --topic "Your Research Topic"
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--topic` | Research topic (required) | - |
| `--max_papers` | Maximum papers to process | 10 |
| `--min_year` | Minimum publication year | 2020 |
| `--output_dir` | Directory to save outputs | ./notes |

## 🏗️ Project Structure

```
research-assistant-ai/
├── research_assistant/
│   ├── __init__.py
│   ├── agent_graph.py         # Main workflow definition
│   ├── nodes/                 # Individual processing nodes
│   │   ├── __init__.py
│   │   ├── topic_explainer.py
│   │   ├── scholar_search.py
│   │   ├── pdf_downloader.py
│   │   └── ...
│   └── utils/                 # Helper functions
│       ├── __init__.py
│       ├── pdf_processor.py
│       └── ollama_processor.py
├── notes/                     # Generated research notes
├── downloaded_pdfs/           # Downloaded research papers
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔄 Workflow

1. **Topic Analysis**
   - The system first analyzes the provided research topic
   - Generates a detailed explanation and identifies related concepts

2. **Paper Discovery**
   - Searches academic databases for relevant papers
   - Filters results based on relevance and publication year

3. **Content Processing**
   - Downloads available PDFs
   - Extracts and cleans text content
   - Processes multiple papers in parallel

4. **Analysis & Summarization**
   - Uses local LLMs to generate summaries
   - Extracts key insights and methodologies
   - Identifies research gaps and trends

5. **Report Generation**
   - Compiles findings into structured markdown
   - Includes proper citations and references
   - Generates an executive summary

## 🤖 Customization

### Changing the Research Topic
You can modify the research topic in two ways:

1. **Command Line Argument** (Recommended):
   ```bash
   python -m research_assistant.agent_graph --topic "Your Research Topic"
   ```

2. **Direct Code Modification**:
   Edit the `initial_state` dictionary in `agent_graph.py`:
   ```python
   initial_state = {
       "topic": "Your Research Topic Here",  # Change this to your desired topic
       "explanation": None,
       "related_topics": None,
       "search_results": None,
       # ... other fields remain the same
   }
   ```

### Using Different Models
Edit the `.env` file to specify your preferred Ollama model:
```
OLLAMA_MODEL=llama2  # Change to mistral, codellama, etc.
```

### Adding New Features
1. Create a new node in `research_assistant/nodes/`
2. Import and add it to the workflow in `agent_graph.py`
3. Define the edges to connect it with existing nodes

## 📝 Example Output

```markdown
# Research Report: AI in Education

## Executive Summary
[Generated summary of findings...]

## Key Papers
1. **Title of Paper 1**  
   Authors, Year  
   Summary: [Generated summary...]

2. **Title of Paper 2**  
   Authors, Year  
   Summary: [Generated summary...]

## Research Gaps
- [Identified gap 1]
- [Identified gap 2]

## Future Directions
- [Suggested research direction 1]
- [Suggested research direction 2]
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://python.langchain.com/)
- Powered by [Ollama](https://ollama.ai/)
- Inspired by the latest research in AI and NLP
