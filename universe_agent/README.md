# Research Agent

AI-powered research agent for character and content analysis. Built with LangChain and designed to research characters from web novels, manga, and other entertainment sources.

## Features

- **Multi-Source Research**: Google Search, Twitter, Namu Wiki, and community sites
- **Intelligent Analysis**: LLM-powered synthesis and summarization
- **Vector Knowledge Base**: Store and retrieve character information efficiently
- **Character Personas**: Create interactive character agents based on research
- **Markdown Reports**: Export research findings as structured markdown documents

## Architecture

```
Research Agent (Data Collection)
    ↓
Vector Knowledge Base (Storage)
    ↓
Persona Agents (Character Embodiment)
    ↓
Content Creation (Your Use Case)
```

## Installation

### Prerequisites

- Python 3.10 or higher
- API keys for LLM provider (Claude or OpenAI)

### Setup

1. **Clone the repository** (or navigate to project directory)

```bash
cd gottawork
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Install Playwright browsers** (for web scraping)

```bash
playwright install
```

4. **Initialize configuration**

```bash
python main.py init
```

This creates a `.env` file from the template.

5. **Add your API keys**

Edit `.env` and add your API keys:

```bash
# Required: Choose one
ANTHROPIC_API_KEY=sk-ant-...
# OR
OPENAI_API_KEY=sk-...

# Optional: For official APIs (otherwise uses web scraping)
TWITTER_BEARER_TOKEN=...
GOOGLE_API_KEY=...
GOOGLE_CSE_ID=...
```

## Usage

### Research a Character

```bash
python main.py research "캐릭터 이름"
```

With options:

```bash
python main.py research "김독자" --type character --llm anthropic
```

This will:
1. Search multiple sources (Google, Twitter, Namu Wiki, etc.)
2. Analyze and synthesize the findings
3. Save to the knowledge base
4. Export a markdown report to `outputs/`

### Chat with a Character Persona

After researching a character, you can chat with them:

```bash
python main.py chat "캐릭터 이름"
```

The persona agent will:
- Load all research data about the character
- Embody the character's personality and traits
- Respond in character during conversations

### Search the Knowledge Base

```bash
# General search
python main.py kb-search "검색어"

# Filter by character
python main.py kb-search "성격" --character "김독자" --limit 10
```

### View Character Knowledge

```bash
python main.py kb-info "캐릭터 이름"
```

## Project Structure

```
gottawork/
├── agents/
│   ├── researcher.py      # Main research agent
│   └── persona.py          # Character persona agents
├── tools/
│   ├── google_search.py    # Google search tool
│   ├── twitter_search.py   # Twitter search tool
│   ├── community_scraper.py # Community site scraper
│   └── web_scraper.py      # Generic web scraper
├── storage/
│   ├── vector_store.py     # Vector database for knowledge
│   └── markdown_exporter.py # Report generation
├── config.py               # Configuration management
├── main.py                 # CLI entry point
├── requirements.txt        # Dependencies
└── outputs/                # Generated reports
```

## Configuration

All configuration is managed through environment variables in `.env`:

### LLM Settings

```bash
DEFAULT_MODEL=claude-3-5-sonnet-20241022
TEMPERATURE=0.7
MAX_TOKENS=4096
```

### Search Settings

```bash
MAX_SEARCH_RESULTS=10
SCRAPING_DELAY=1.0  # Delay between requests (seconds)
```

### Vector Store Settings

```bash
CHROMA_PERSIST_DIR=./data/chroma
COLLECTION_NAME=character_knowledge
```

## Advanced Usage

### Custom Persona Configuration

```python
from agents.persona import create_persona_agent

agent = create_persona_agent(
    character_name="김독자",
    personality_traits=["intelligent", "strategic", "compassionate"],
    speech_style="Calm and analytical, occasionally self-deprecating",
    background="Reader who became a character in his favorite novel"
)

response = agent.chat("What's your goal?")
```

### Programmatic Research

```python
from agents.researcher import ResearchAgent

agent = ResearchAgent(llm_provider="anthropic")
report = agent.research(
    subject="전지적 독자 시점",
    subject_type="work",
    save_to_kb=True
)

print(report.summary)
```

### Direct Knowledge Base Access

```python
from storage.vector_store import VectorKnowledgeBase, CharacterKnowledge

kb = VectorKnowledgeBase()

# Add custom knowledge
kb.add_knowledge(CharacterKnowledge(
    character_name="김독자",
    source="custom",
    content="Custom information about the character...",
    metadata={"type": "personality"}
))

# Search
results = kb.search("성격", character_name="김독자", k=5)
```

## Development

### Install development dependencies

```bash
pip install -e ".[dev]"
```

### Code formatting

```bash
black .
ruff check .
```

## Limitations

- **API Costs**: LLM API calls incur costs (Claude/OpenAI)
- **Rate Limits**: Web scraping and APIs have rate limits
- **Language**: Optimized for Korean content (can be adapted for other languages)
- **Data Accuracy**: Research quality depends on available online sources

## Roadmap

- [ ] Additional search sources (Reddit, YouTube, etc.)
- [ ] Multi-character interaction support
- [ ] Content generation templates
- [ ] Web UI interface
- [ ] Export to different formats (PDF, JSON, etc.)
- [ ] Custom scraping rules for specific sites

## Troubleshooting

### "Module not found" errors

```bash
pip install -r requirements.txt
```

### Twitter API not working

Twitter search requires a Bearer Token. If you don't have one, the tool will skip Twitter search and use other sources.

### Web scraping fails

Some sites may block scrapers. The tool will continue with other available sources.

### Vector store errors

Delete the chroma database and restart:

```bash
rm -rf data/chroma
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

---

Built with [LangChain](https://langchain.com/) and [Claude](https://claude.ai/)
