# 📚 Research Assistant Agent (LangGraph + Ollama)

A powerful, modular AI Research Assistant built with LangGraph and Ollama that automates the research process from topic exploration to comprehensive report generation.

## 🚀 Features

🔍 **Topic Analysis**
- Detailed topic explanations
- Related concepts and subtopics
- Contextual understanding of research areas

📄 **Paper Discovery**
- Automated Google Scholar search
- Smart paper filtering and selection
- Metadata extraction (title, authors, year, etc.)

🤖 **AI-Powered Processing**
- Automatic PDF downloading (when available)
- Content extraction and cleaning
- Multi-paper summarization
- Key insights extraction

📝 **Report Generation**
- Comprehensive markdown reports
- Structured research summaries
- Proper citations and references
- Executive summaries and key findings

## 🛠️ Prerequisites

- Python 3.8+
- Ollama (with at least one model installed, e.g., llama2, mistral)
- Git
- pip (Python package manager)

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/research-paper-chatbot.git
   cd research-paper-chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start Ollama server (in a new terminal):
   ```bash
   ollama serve
   ```

   In another terminal, pull a model (if not already done):
   ```bash
   ollama pull llama2  # or mistral, etc.
   ```

## 🚀 Quick Start

Run the research assistant with your topic:
```bash
python -m research_assistant.agent_graph --topic "Your Research Topic Here"
```

### Optional Arguments
- `--max_papers`: Maximum number of papers to process (default: 10)
- `--min_year`: Minimum publication year (default: 2020)
- `--output_dir`: Directory to save outputs (default: ./notes)

Example:
```bash
python -m research_assistant.agent_graph --topic "Blockchain in Healthcare" --max_papers 15 --min_year 2019
```

## 📂 Output

The assistant will create markdown files in the `notes` directory:
- `{Topic}_{Timestamp}.md`: Raw research notes
- `{Topic}_Comprehensive_Report_{Timestamp}.md`: Formatted research report

## 🏗️ Architecture

The system is built using:
- **LangGraph** for workflow orchestration
- **Ollama** for local LLM inference
- **scholarly** for paper discovery
- **PyPDF2** for PDF processing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain for the amazing framework
- Ollama for local LLM support
- Google Scholar for research paper data
