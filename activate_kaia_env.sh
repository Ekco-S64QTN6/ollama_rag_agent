#!/bin/bash

COLOR_GREEN="\033[92m"
COLOR_BLUE="\033[94m"
COLOR_YELLOW="\033[93m"
COLOR_RED="\033[91m"
COLOR_RESET="\033[0m"

KAIA_PROJECT_DIR="/home/ekco/projects/personal/ollama_dev"

cd "$KAIA_PROJECT_DIR" || { echo -e "${COLOR_RED}Error: Could not navigate to project directory: $KAIA_PROJECT_DIR. Exiting.${COLOR_RESET}"; exit 1; }

source .venv/bin/activate || { echo -e "${COLOR_RED}Error: Could not activate virtual environment at $KAIA_PROJECT_DIR/.venv. Ensure it exists and is valid. Exiting.${COLOR_RESET}"; exit 1; }

echo -e "${COLOR_BLUE}Kaia's virtual environment activated and you are in $(pwd)${COLOR_RESET}"

echo -e "${COLOR_BLUE}Checking PostgreSQL status...${COLOR_RESET}"
if ! pg_isready -h localhost -p 5432 -U kaiauser > /dev/null 2>&1; then
    echo -e "${COLOR_YELLOW}PostgreSQL is not running. Attempting to start...${COLOR_RESET}"
    sudo systemctl start postgresql
    sleep 3
    if pg_isready -h localhost -p 5432 -U kaiauser > /dev/null 2>&1; then
        echo -e "${COLOR_GREEN}PostgreSQL started successfully.${COLOR_RESET}"
    else
        echo -e "${COLOR_YELLOW}Warning: Failed to start PostgreSQL. Kaia might not function correctly without it.${COLOR_RESET}"
    fi
else
    echo -e "${COLOR_GREEN}PostgreSQL is already running.${COLOR_RESET}"
fi

echo -e "${COLOR_BLUE}Checking ChromaDB server status...${COLOR_RESET}"
CHROMA_NC_OUTPUT=$(nc -z localhost 8000 2>&1)
CHROMA_NC_STATUS=$?
if [ $CHROMA_NC_STATUS -ne 0 ]; then
    echo -e "${COLOR_YELLOW}ChromaDB server not running on port 8000. Attempting to start...${COLOR_RESET}"
    nohup chroma run --host localhost --port 8000 --path "$KAIA_PROJECT_DIR/storage/chroma_db" > "$KAIA_PROJECT_DIR/chroma.log" 2>&1 &
    CHROMA_PID=$!
    echo -e "${COLOR_BLUE}ChromaDB server started with PID $CHROMA_PID. Log: $KAIA_PROJECT_DIR/chroma.log${COLOR_RESET}"
    sleep 5
    CHROMA_NC_OUTPUT=$(nc -z localhost 8000 2>&1)
    CHROMA_NC_STATUS=$?
    if [ $CHROMA_NC_STATUS -eq 0 ]; then
        echo -e "${COLOR_GREEN}ChromaDB server is now running.${COLOR_RESET}"
    else
        echo -e "${COLOR_RED}Error: ChromaDB server failed to start. Check $KAIA_PROJECT_DIR/chroma.log for details.${COLOR_RESET}"
    fi
else
    echo -e "${COLOR_GREEN}ChromaDB server is already running.${COLOR_RESET}"
fi

echo -e "${COLOR_BLUE}Checking Ollama status...${COLOR_RESET}"
OLLAMA_NC_OUTPUT=$(nc -z localhost 11434 2>&1)
OLLAMA_NC_STATUS=$?
if [ $OLLAMA_NC_STATUS -ne 0 ]; then
    echo -e "${COLOR_YELLOW}Ollama server not running on port 11434. Attempting to start...${COLOR_RESET}"
    nohup ollama serve > "$KAIA_PROJECT_DIR/ollama.log" 2>&1 &
    OLLAMA_PID=$!
    echo -e "${COLOR_BLUE}Ollama server started with PID $OLLAMA_PID. Log: $KAIA_PROJECT_DIR/ollama.log${COLOR_RESET}"
    sleep 10
    OLLAMA_NC_OUTPUT=$(nc -z localhost 11434 2>&1)
    OLLAMA_NC_STATUS=$?
    if [ $OLLAMA_NC_STATUS -eq 0 ]; then
        echo -e "${COLOR_GREEN}Ollama server is now running.${COLOR_RESET}"
    else
        echo -e "${COLOR_RED}Error: Ollama server failed to start. Check $KAIA_PROJECT_DIR/ollama.log for details.${COLOR_RESET}"
        exit 1
    fi
else
    echo -e "${COLOR_GREEN}Ollama server is already running.${COLOR_RESET}"
fi

echo -e "${COLOR_BLUE}All services checked. Starting Kaia CLI application...${COLOR_RESET}"

python llamaindex_ollama_rag.py

echo -e "${COLOR_BLUE}Kaia CLI application session ended.${COLOR_RESET}"
