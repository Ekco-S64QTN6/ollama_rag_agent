# Kaia: **K**aia **A**rtificial **I**ntelligence **A**ssistant

![Kaia CLI Screenshot](images/kaia_cli_screenshot.png)

## Overview

Kaia is an advanced local AI assistant combining:
- **Mistral-Instruct:** For natural language understanding and conversational AI.
- **ChromaDB + LlamaIndex:** For powerful contextual Retrieval-Augmented Generation (RAG).
- **Intelligent Command Execution:** Understands natural language requests to propose and execute Linux shell commands.
- **Text-to-Speech (Optional):** Provides spoken responses via Piper/speech-dispatcher.

## Key Features

### Core Capabilities
- Natural language conversations with persona consistency
- Context-aware responses using personal/general knowledge
- Safe Linux command generation & execution
- Real-time system monitoring (`/status`)
- Streaming responses with first-token latency metrics

### Technical Highlights
- Persistent ChromaDB vector storage
- Ollama API with strict JSON response enforcement
- Configurable TTS backend (Piper recommended)
- GPU awareness (NVIDIA monitoring)
- Pre-warmed LLM for reduced cold-start latency

## Installation

### Prerequisites
- Ollama server running (`mistral:instruct` and `nomic-embed-text` models)
- Python 3.10+
- speech-dispatcher + Piper TTS (optional)

```bash
# On Arch Linux:
yay -S piper-tts-bin speech-dispatcher piper-voices-en-us
sudo systemctl enable --now speech-dispatcher.service
```
## Setup

```bash
git clone [https://github.com/Ekco-S64QTN6/ollama_rag_agent.git](https://github.com/Ekco-S64QTN6/ollama_rag_agent.git)
cd kaia-assistant

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Prepare knowledge bases
```bash
mkdir -p data personal_context
echo "Place your Markdown knowledge files here" > data/README.md
```

## Configuration

Edit llamaindex_ollama_rag.py to customize:

# Core settings (lines 20-30)
```python
LLM_MODEL = "mistral:instruct"          # Primary chat model
TTS_ENABLED = True                      # Auto-detect if unset
CHROMA_DB_PATH = "./storage/chroma_db"  # Vector store location
```
# Usage
```python
python llamaindex_ollama_rag.py
```

# First run will build vector indexes
# Subsequent runs load persisted indexes

# Interaction Examples
```bash
Query: check disk space
Kaia (Command): df -h

Query: explain ChromaDB
Kaia: ChromaDB is an open-source vector database...

Query: /status
[Displays real-time system metrics]
```

# Symptoms
- TTS not working	Verify spd-say "test" works first
- Ollama timeouts	Increase request_timeout in config
- Missing persona	Ensure Kaia_Desktop_Persona.md exists in ./data

## Contributing

This project welcomes:

    Documentation improvements

    Additional TTS backend integrations

    Enhanced safety checks for command execution
    
    Other

License

MIT License - See LICENSE.md
