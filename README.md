# Kaia: **K**aia **A**rtificial **I**ntelligence **A**ssistant

![Kaia CLI Screenshot](images/kaia_cli_screenshot.png)

## Overview

Kaia is an advanced local AI assistant combining:
- **Mistral-Instruct:** For natural language understanding and conversational AI
- **ChromaDB + LlamaIndex:** For powerful contextual Retrieval-Augmented Generation (RAG)
- **Intelligent Command Execution:** Understands natural language requests to propose and execute Linux shell commands
- **Text-to-Speech (Optional):** Provides spoken responses via Piper/speech-dispatcher
- **Long-Term Memory:** Stores user preferences, facts, and interaction history in PostgreSQL

## Key Features

### Core Capabilities
- Natural language conversations with persona consistency
- Context-aware responses using personal/general knowledge
- Safe Linux command generation & execution with user confirmation
- Real-time system monitoring (`/status` command)
- Streaming responses with first-token latency metrics
- Persistent memory for user preferences and facts
- Unified data retrieval for persona details, facts, and preferences

### Technical Highlights
- Persistent ChromaDB vector storage
- PostgreSQL database for long-term memory
- Ollama API with strict JSON response enforcement
- Configurable TTS backend (Piper recommended)
- GPU awareness (NVIDIA monitoring)
- Pre-warmed LLM for reduced cold-start latency

## Installation

### Prerequisites
- Ollama server running (`mistral:instruct` and `nomic-embed-text` models)
- Python 3.10+
- PostgreSQL database
- speech-dispatcher + Piper TTS (optional)
```bash
# On Arch Linux:
sudo pacman -S python python-pip postgresql
yay -S piper-tts-bin speech-dispatcher piper-voices-en-us
sudo systemctl enable --now postgresql
sudo systemctl enable --now speech-dispatcher.service
```
### Setup
```bash
git clone https://github.com/Ekco-S64QTN6/ollama_rag_agent.git
cd kaia-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
### Prepare knowledge bases
```bash
mkdir -p data personal_context storage
echo "Place your Markdown knowledge files here" > data/README.md
```

## Configuration

Edit llamaindex_ollama_rag.py to customize:

```python
LLM_MODEL = "mistral:instruct"          # Primary chat model
TTS_ENABLED = True                      # Auto-detect if unset
CHROMA_DB_PATH = "./storage/chroma_db"  # Vector store location
```

## Secure Database Setup

### 1. Create PostgreSQL User and Database**

```bash
sudo -u postgres createuser --pwprompt kaiauser
sudo -u postgres createdb -O kaiauser kaiadb
```

### 2. Set Environment Variables:

    Add to your shell profile (~/.bashrc, ~/.zshrc, or ~/.profile):

```bash
echo 'export KAIA_DB_USER="kaiauser"' >> ~/.bashrc
echo 'export KAIA_DB_PASS="your_secure_password"' >> ~/.bashrc
echo 'export KAIA_DB_HOST="localhost"' >> ~/.bashrc
echo 'export KAIA_DB_NAME="kaiadb"' >> ~/.bashrc
```
Then reload:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

### 3. Initialize Database Schema:
The database schema will be automatically initialized when you start Kaia.

Usage

Start Kaia:
```python
python llamaindex_ollama_rag.py
```
First Run

    Will build vector indexes from your knowledge files

    Subsequent runs will load persisted indexes
    
## Interaction Examples
```bash
# Execute system commands
Query: check disk space
Kaia (Command): df -h

# Get explanations
Query: explain ChromaDB
Kaia: ChromaDB is an open-source vector database...

# Check system status
Query: /status
[Displays real-time system metrics]

# Store preferences
Query: remember I prefer dark mode
Kaia: Preference stored: 'dark_mode'

# Recall preferences
Query: what are my accessibility preferences?
Kaia: Your preferences: dark_mode

# Store personal facts
Query: remember my partner's birthday is June 15
Kaia: Fact stored: "My partner's birthday is June 15"

# Exit session
Query: exit
Kaia: Session ended. Until next time!
```
Troubleshooting

    TTS not working: Verify spd-say "test" works first

    Ollama timeouts: Increase request_timeout in config

    Missing persona: Ensure Kaia_Desktop_Persona.md exists in ./data

    Database connection issues: Verify PostgreSQL is running and environment variables are set

Contributing

This project welcomes contributions for:

    Documentation improvements

    Additional TTS backend integrations

    Enhanced safety checks for command execution

    New features and bug fixes

License

MIT License - See LICENSE.md
