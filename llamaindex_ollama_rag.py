import json
import logging
import os
import subprocess
import sys
import time
import requests
import database_utils
import chromadb
import chromadb.config
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    SQLDatabase,
)
from llama_index.core.chat_engine.simple import SimpleChatEngine
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from sqlalchemy import create_engine
from kaia_cli import KaiaCLI, DEFAULT_COMMAND_MODEL
from contextlib import redirect_stdout, contextmanager
from typing import Optional, List, Dict, Union, Any
from datetime import datetime # Added datetime import

# --- Config ---
TIMEOUT_SECONDS = 300  # 5 minutes timeout for all operations
LLM_MODEL = "mistral:instruct"
EMBEDDING_MODEL = "nomic-embed-text"

GENERAL_KNOWLEDGE_DIR = "./data"
PERSONAL_CONTEXT_DIR = "./personal_context"
PERSONA_DIR = "./data"
PERSIST_DIR = "./storage"
CHROMA_DB_PATH = "./storage/chroma_db"
CHROMA_SERVER_HOST = "localhost"
CHROMA_SERVER_PORT = 8000
LLAMA_INDEX_METADATA_PATH = "./storage/llama_index_metadata"
TTS_ENABLED = False
SQL_RAG_ENABLED = True

# --- ANSI Color Codes ---
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"

# --- Kaia's Core System Prompt ---
KAIA_SYSTEM_PROMPT = """
You are Kaia, a helpful and knowledgeable AI assistant.
"""

# --- Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
for noisy in ["httpx", "httpcore", "fsspec", "urllib3", "llama_index.core.storage.kvstore", "chromadb", "llama_index"]:
    logging.getLogger(noisy).setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

# --- Context Manager to suppress stdout ---
@contextmanager
def suppress_stdout():
    """Temporarily redirects stdout to devnull to suppress noisy output."""
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def main():
    global SQL_RAG_ENABLED
    cli = KaiaCLI()

    # --- Initialize Models ---
    print(f"{COLOR_BLUE}Initializing LLM and Embedding Model...{COLOR_RESET}")
    start_time_init_models = time.time()
    try:
        with suppress_stdout():
            Settings.llm = Ollama(model=LLM_MODEL, request_timeout=360.0, stream=True)
            Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
            _ = Settings.llm.complete("Hello")
        logger.info("LLM and embedding models initialized successfully")
        print(f"{COLOR_GREEN}LLM and embedding models initialized successfully in {time.time() - start_time_init_models:.2f}s.{COLOR_RESET}")
    except Exception as e:
        logger.error(f"Failed to initialize Ollama: {e}", exc_info=True)
        print(f"{COLOR_RED}Failed to initialize Ollama: {e}{COLOR_RESET}")
        sys.exit(1)

    # --- Load Persona ---
    print(f"{COLOR_BLUE}Loading Kaia persona...{COLOR_RESET}")
    persona_doc_path = os.path.join(PERSONA_DIR, "Kaia_Desktop_Persona.md")
    kaia_persona_content = ""
    try:
        if os.path.exists(persona_doc_path):
            kaia_persona_content = SimpleDirectoryReader(input_files=[persona_doc_path]).load_data()[0].text
            kaia_persona_content = kaia_persona_content.replace('\x00', '').replace('\u0000', '')
            logger.info("Kaia persona loaded successfully")
            print(f"{COLOR_GREEN}Kaia persona loaded successfully.{COLOR_RESET}")
        else:
            logger.warning(f"Persona file not found: {persona_doc_path}")
            print(f"{COLOR_YELLOW}Warning: Persona file not found. Responses may be generic.{COLOR_RESET}")
    except Exception as e:
        logger.warning(f"Could not load persona: {e}", exc_info=True)
        print(f"{COLOR_YELLOW}Warning: Could not load persona: {e}. Responses may be generic.{COLOR_RESET}")

    # --- Initialize ChromaDB ---
    print(f"{COLOR_BLUE}Initializing ChromaDB...{COLOR_RESET}")
    try:
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        if not os.access(CHROMA_DB_PATH, os.W_OK):
            logger.error(f"Insufficient permissions for ChromaDB path: {CHROMA_DB_PATH}")
            print(f"{COLOR_RED}Error: Insufficient permissions for ChromaDB path: {CHROMA_DB_PATH}{COLOR_RESET}")
            sys.exit(1)

        with suppress_stdout():
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            chroma_collection = chroma_client.get_or_create_collection("kaia_documents")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        logger.info(f"ChromaDB persistent client initialized at '{CHROMA_DB_PATH}'.")
        print(f"{COLOR_GREEN}ChromaDB persistent client initialized.{COLOR_RESET}")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB persistent client: {e}", exc_info=True)
        print(f"{COLOR_RED}Failed to initialize ChromaDB persistent client: {e}{COLOR_RESET}")
        print(f"{COLOR_YELLOW}Please ensure write permissions for ChromaDB path: {CHROMA_DB_PATH}{COLOR_RESET}")
        sys.exit(1)

    # --- Configure LlamaIndex ---
    print(f"{COLOR_BLUE}Loading/Building LlamaIndex...{COLOR_RESET}")
    index = None
    try:
        if not os.path.exists(LLAMA_INDEX_METADATA_PATH):
            print(f"{COLOR_BLUE}Building index...{COLOR_RESET}")
            general_docs = [doc for doc in SimpleDirectoryReader(GENERAL_KNOWLEDGE_DIR).load_data()
                            if "Kaia_Desktop_Persona.md" not in doc.id_]
            personal_docs = SimpleDirectoryReader(PERSONAL_CONTEXT_DIR).load_data()
            all_docs = general_docs + personal_docs

            if not all_docs:
                print(f"{COLOR_YELLOW}No documents found to build index.{COLOR_RESET}")
            else:
                with suppress_stdout():
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_documents(all_docs, storage_context=storage_context)
                    index.storage_context.persist(persist_dir=LLAMA_INDEX_METADATA_PATH)
                logger.info("Index built and saved.")
                print(f"{COLOR_GREEN}Index built and saved.{COLOR_RESET}")
        else:
            print(f"{COLOR_BLUE}Loading index...{COLOR_RESET}")
            with suppress_stdout():
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=LLAMA_INDEX_METADATA_PATH
                )
                index = load_index_from_storage(storage_context=storage_context)
            logger.info("Index loaded.")
            print(f"{COLOR_GREEN}Index loaded.{COLOR_RESET}")
    except Exception as e:
        logger.error(f"Failed to initialize LlamaIndex: {e}", exc_info=True)
        print(f"{COLOR_RED}Failed to initialize LlamaIndex: {e}{COLOR_RESET}")
        index = None

    # --- Chat Engines ---
    def get_rag_chat_engine(index_param=None):
        if not hasattr(get_rag_chat_engine, '_cached_engine') or index_param != get_rag_chat_engine._cached_index:
            if index_param:
                combined_system_prompt = KAIA_SYSTEM_PROMPT + "\n\n" + kaia_persona_content
                get_rag_chat_engine._cached_engine = index_param.as_chat_engine(
                    chat_mode="condense_plus_context",
                    memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
                    system_prompt=combined_system_prompt,
                    similarity_top_k=2
                )
                get_rag_chat_engine._cached_index = index_param
            else:
                print(f"{COLOR_YELLOW}Warning: RAG index not available. Using basic chat engine for RAG queries.{COLOR_RESET}")
                get_rag_chat_engine._cached_engine = SimpleChatEngine.from_defaults(
                    llm=Settings.llm,
                    memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
                    system_prompt=KAIA_SYSTEM_PROMPT + "\n\n" + kaia_persona_content,
                )
                get_rag_chat_engine._cached_index = None
        return get_rag_chat_engine._cached_engine

    def get_pure_chat_engine():
        if not hasattr(get_pure_chat_engine, '_cached_engine'):
            get_pure_chat_engine._cached_engine = SimpleChatEngine.from_defaults(
                llm=Settings.llm,
                chat_mode="best",
                memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
                system_prompt=KAIA_SYSTEM_PROMPT + "\n\n" + kaia_persona_content,
            )
        return get_pure_chat_engine._cached_engine

    rag_chat_engine = get_rag_chat_engine(index)
    pure_chat_engine = get_pure_chat_engine()

    # --- SQL Database ---
    sql_query_engine = None
    if SQL_RAG_ENABLED:
        print(f"{COLOR_BLUE}Initializing PostgreSQL database...{COLOR_RESET}")
        try:
            with suppress_stdout():
                database_utils.initialize_db()
                sql_database = SQLDatabase(database_utils.engine)
                sql_query_engine = NLSQLTableQueryEngine(
                    sql_database=sql_database,
                    llm=Settings.llm,
                    tables=["facts", "interaction_history", "user_preferences"]
                )
            logger.info("PostgreSQL initialized successfully")
            print(f"{COLOR_GREEN}PostgreSQL initialized successfully.{COLOR_RESET}")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL Database: {e}", exc_info=True)
            print(f"{COLOR_RED}Failed to initialize PostgreSQL Database: {e}{COLOR_RESET}")
            SQL_RAG_ENABLED = False

    # --- Helper Functions ---
    def speak_text_async(text):
        if not TTS_ENABLED:
            return
        try:
            subprocess.run(["spd-say", "--wait", text], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        except Exception as e:
            logger.error(f"TTS failed: {str(e)}")

    def get_dynamic_system_status_llama_index_rag():
        """This function is no longer used for /status, but kept for historical context if needed."""
        system_query = "What is the current system status, including CPU, memory, disk, and GPU information? Also, tell me if Ollama is running."
        try:
            response = rag_chat_engine.query(system_query)
            return str(response)
        except Exception as e:
            return f"An error occurred during the RAG query for system status: {e}"

    def confirm_and_execute_command(command):
        if any(blocked in command for blocked in [';', '&&', '||', '`', '$(', '>', '<', '|']):
            print(f"{COLOR_RED}Dangerous command detected - aborting.{COLOR_RESET}")
            return False, "Dangerous command detected - aborting."

        print(f"\n{COLOR_YELLOW}Execute command? (y/n)\n{command}{COLOR_RESET}")
        if input("Your choice: ").lower().strip() == 'y':
            try:
                result = subprocess.run(command, shell=True, text=True, capture_output=True)
                print(f"{COLOR_GREEN}Output:\n{result.stdout}{COLOR_RESET}")
                if result.stderr:
                    print(f"{COLOR_RED}Error:\n{result.stderr}{COLOR_RESET}")
                return True, result.stdout.strip()
            except Exception as e:
                print(f"{COLOR_RED}Command failed: {e}{COLOR_RESET}")
                return False, f"Command failed: {e}"
        else:
            print(f"{COLOR_BLUE}Command execution cancelled.{COLOR_RESET}")
            return False, "Command execution cancelled."

    def generate_action_plan(user_input):
        system_prompt = """You are an AI assistant that classifies user intents. Respond ONLY with valid JSON:
        {
            "action": "command"|"chat"|"sql"|"retrieve_data"|"store_data"|"system_status"|"get_persona_content",
            "content": "appropriate content for the action"
        }

        Categories:
        - "command": Terminal commands, file operations, system info requests that require a command (e.g., "list files", "check running processes").
        - "chat": General conversation, greetings, small talk, or questions that do not require specific knowledge retrieval from documents or databases.
        - "sql": Database queries requiring SQL joins/aggregations on `facts`, `interaction_history`, `user_preferences` tables.
        - "retrieve_data": Recall personal facts/preferences stored in the database ("What's my favorite color?", "Tell me what you know about X", "What do you know about me?", "What preferences do you know?", "What facts do you know?", "What is our interaction history?", "List interactions?", "List history?", "List all memories?").
        - "store_data": Remember personal facts/preferences ("My favorite food is pizza", "Remember that X is Y").
        - "system_status": Direct requests for system status (e.g., "show system status", "how is my computer doing", "/status" - though /status is handled directly, the intent for similar natural language queries should be classified here).
        - "get_persona_content": Directly retrieve Kaia's persona content ("List your persona", "What is your persona?", "Tell me about yourself Kaia").
        """

        payload = {
            "model": DEFAULT_COMMAND_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a command to list all files in the current directory."},
                {"role": "assistant", "content": json.dumps({"action": "command", "content": "ls -a"})},
                {"role": "user", "content": f"What is the capital of France?"},
                {"role": "assistant", "content": json.dumps({"action": "chat", "content": "What is the capital of France?"})},
                {"role": "user", "content": f"Hello, how are you?"},
                {"role": "assistant", "content": json.dumps({"action": "chat", "content": "Hello, how are you?"})},
                {"role": "user", "content": f"Tell me a joke."},
                {"role": "assistant", "content": json.dumps({"action": "chat", "content": "Tell me a joke."})},
                {"role": "user", "content": f"What's up?"},
                {"role": "assistant", "content": json.dumps({"action": "chat", "content": "What's up."})},
                {"role": "user", "content": f"Show me all the facts you know."},
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "all facts"})},
                {"role": "user", "content": f"What do you know about me?"},
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "what do you know about me"})},
                {"role": "user", "content": f"What preferences do you know?"},
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "what preferences do you know"})},
                {"role": "user", "content": f"What facts do you know?"},
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "what facts do you know"})},
                {"role": "user", "content": f"What is our interaction history?"},
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "what is our interaction history"})},
                {"role": "user", "content": f"List interactions?"},
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "list interactions"})},
                {"role": "user", "content": f"List history?"},
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "list history"})},
                {"role": "user", "content": f"List all memories?"},
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "list all memories"})},
                {"role": "user", "content": f"List your persona?"},
                {"role": "assistant", "content": json.dumps({"action": "get_persona_content", "content": "list your persona"})},
                {"role": "user", "content": f"What is your persona?"},
                {"role": "assistant", "content": json.dumps({"action": "get_persona_content", "content": "what is your persona"})},
                {"role": "user", "content": f"Tell me about yourself Kaia."},
                {"role": "assistant", "content": json.dumps({"action": "get_persona_content", "content": "tell me about yourself kaia"})},
                {"role": "user", "content": f"Remember that my favorite food is pizza."},
                {"role": "assistant", "content": json.dumps({"action": "store_data", "content": "my favorite food is pizza"})},
                {"role": "user", "content": f"remember that the sky is blue"},
                {"role": "assistant", "content": json.dumps({"action": "store_data", "content": "the sky is blue"})},
                {"role": "user", "content": f"remember I prefer dark mode"},
                {"role": "assistant", "content": json.dumps({"action": "store_data", "content": "I prefer dark mode"})},
                {"role": "user", "content": f"remember my preferred editor is Neovim"},
                {"role": "assistant", "content": json.dumps({"action": "store_data", "content": "my preferred editor is Neovim"})},
                {"role": "user", "content": f"remember that"},
                {"role": "assistant", "content": json.dumps({"action": "chat", "content": "Please tell me what fact you want me to remember."})},
                {"role": "user", "content": f"remember"},
                {"role": "assistant", "content": json.dumps({"action": "chat", "content": "Please tell me what you want me to remember, either a fact or a preference."})},
                {"role": "user", "content": f"What is the current system status?"},
                {"role": "assistant", "content": json.dumps({"action": "system_status", "content": "current system status"})},
                {"role": "user", "content": str(user_input)}
            ],
            "stream": False,
            "format": "json"
        }

        try:
            response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            result = response.json()["message"]["content"].strip()
            return json.loads(result)
        except Exception as e:
            logger.error(f"Action plan error: {e}", exc_info=True)
            # Removed the system_status fallback from here as it's now handled directly in the main loop
            if any(keyword in str(user_input).lower() for keyword in ['list', 'find', 'search', 'show files', 'check processes', 'disk usage', 'memory usage', 'cpu usage', 'gpu usage']):
                return {"action": "command", "content": user_input}
            elif any(keyword in str(user_input).lower() for keyword in ['remember', 'my name is', 'i like']):
                return {"action": "store_data", "content": user_input}
            elif any(keyword in str(user_input).lower() for keyword in ['what is', 'who is', 'tell me about']):
                return {"action": "chat", "content": user_input}
            return {"action": "chat", "content": user_input}

    # --- Helper function for color coding percentages ---
    def get_color_for_percentage(percent: Union[int, float]) -> str:
        if percent <= 70:
            return COLOR_GREEN
        elif 70 < percent <= 80:
            return COLOR_YELLOW
        else:
            return COLOR_RED

    # --- Main Loop ---
    print(f"""{COLOR_BLUE}
██╗  ██╗ █████╗ ██╗ █████╗
██║ ██╔╝██╔══██╗██║██╔══██╗
█████╔╝ ███████║██║███████║
██╔═██╗ ██╔══██║██║██╔══██║
██║  ██╗██║  ██║██║██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
{COLOR_RESET}
Kaia (Personal AI Assistant) - Ready

{COLOR_GREEN}Welcome to Kaia, your AI assistant.{COLOR_RESET}
Type 'exit' or 'quit' to end the session.
You can also ask about system status (e.g., 'show disk usage').
To store memories, use phrases like 'remember I prefer...' or 'remember that...'
To retrieve memories, use 'list my preferences' or 'show facts'.
To run a command, type 'run command <your command>' or '!<your command>'.
""")

    while True:
        try:
            user_id = database_utils.get_current_user()
            query = input("\nYou: ").strip()

            if query.lower() in ['exit', 'quit']:
                print(f"{COLOR_BLUE}Kaia: Session ended. Until next time!{COLOR_RESET}")
                break
            if not query:
                continue

            start_time = time.time()
            response = ""
            response_type = "unclassified_query"

            # Handle slash commands and direct system status queries (direct handling)
            system_status_keywords = ['status', 'how is my computer doing', 'system info', 'show system status', 'display system status', 'system status']
            if query.startswith('/') or any(keyword in query.lower() for keyword in system_status_keywords):
                cmd = query[1:].strip().lower() if query.startswith('/') else query.lower() # Handle both /cmd and cmd
                if cmd == 'help':
                    response = """Available commands:
/help - Show this help message
/status - Show system status
/exit - End the session"""
                    response_type = "help"
                elif any(keyword in cmd for keyword in system_status_keywords): # Consolidated status handling
                    status_info = cli.get_system_status()
                    msg_parts = [
                        f"• {COLOR_BLUE}Date & Time:{COLOR_RESET} {datetime.fromisoformat(status_info.get('timestamp', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')}",
                        f"• {COLOR_BLUE}Uptime:{COLOR_RESET} {status_info.get('uptime', 'N/A')}",
                        f"• {COLOR_BLUE}Board:{COLOR_RESET} {status_info.get('board_info', 'N/A')}",
                        f"• {COLOR_BLUE}OS:{COLOR_RESET} {status_info.get('os_info', 'N/A')}",
                        f"• {COLOR_BLUE}Kernel:{COLOR_RESET} {status_info.get('kernel_info', 'N/A')}",
                        f"• {COLOR_BLUE}Python Version:{COLOR_RESET} {status_info.get('python_version', 'N/A')}",
                    ]

                    # CPU Info
                    cpu_info = status_info.get('cpu_info', {})
                    if cpu_info:
                        cpu_name = cpu_info.get('name', 'N/A')
                        cpu_speed = cpu_info.get('speed', 'N/A')
                        cpu_cores = cpu_info.get('logical_cores', 'N/A')
                        msg_parts.append(f"• {COLOR_BLUE}CPU:{COLOR_RESET} {cpu_name} ({cpu_cores}) @ {cpu_speed}")

                    # Memory Info
                    mem_info = status_info.get('memory_info', {})
                    if mem_info:
                        total_gb = round(mem_info.get('total', 0) / (1024**3), 2)
                        available_gb = round(mem_info.get('available', 0) / (1024**3), 2)
                        percent_used = mem_info.get('percent', 'N/A')
                        percent_color = get_color_for_percentage(percent_used) # Get color for memory percentage
                        msg_parts.append(f"• {COLOR_BLUE}Memory:{COLOR_RESET} {total_gb} GB total, {available_gb} GB available ({percent_color}{percent_used}% used{COLOR_RESET})")

                    # Disk Usage Info
                    all_disk_usage = status_info.get('all_disk_usage', [])

                    # Create a temporary dictionary to hold formatted disk strings for easy lookup
                    formatted_disks = {}
                    for disk in all_disk_usage:
                        path = disk.get('path', disk.get('mount_point', 'N/A'))
                        label = disk.get('label', disk.get('mount_point', 'N/A'))

                        if disk.get('status') == 'Error':
                            formatted_disks[path] = f"• {COLOR_RED}Disk Usage ('{label}'):{COLOR_RESET} Error - {disk.get('error_message', 'N/A')}"
                        else:
                            total_gb = round(disk.get('total', 0) / (1024**3), 2)
                            used_gb = round(disk.get('used', 0) / (1024**3), 2)
                            percent_used = disk.get('percent', 'N/A')
                            percent_color = get_color_for_percentage(percent_used) # Get color for disk percentage
                            formatted_disks[path] = {
                                'string': f"• {COLOR_BLUE}Disk Usage ('{label}'):{COLOR_RESET} {total_gb} GB total, {used_gb} GB used ({percent_color}{percent_used}% used{COLOR_RESET})",
                                'data': disk # Keep original data for reconstruction
                            }

                    # Define the desired order and new labels
                    desired_order = [
                        {'path': '/boot', 'new_label': 'Boot'},
                        {'path': '/', 'new_label': 'Root'},
                        {'path': '/home', 'new_label': 'Home'}
                    ]

                    # Add disks in desired order with new labels
                    for item in desired_order:
                        path = item['path']
                        new_label = item['new_label']
                        if path in formatted_disks:
                            disk_data = formatted_disks[path]['data']
                            if disk_data.get('status') == 'Error':
                                msg_parts.append(f"• {COLOR_RED}Disk Usage ('{new_label}'):{COLOR_RESET} Error - {disk_data.get('error_message', 'N/A')}")
                            else:
                                total_gb = round(disk_data.get('total', 0) / (1024**3), 2)
                                used_gb = round(disk_data.get('used', 0) / (1024**3), 2)
                                percent_used = disk_data.get('percent', 'N/A')
                                percent_color = get_color_for_percentage(percent_used) # Get color for disk percentage
                                msg_parts.append(f"• {COLOR_BLUE}Disk Usage ('{new_label}'):{COLOR_RESET} {total_gb} GB total, {used_gb} GB used ({percent_color}{percent_used}% used{COLOR_RESET})")

                            # Remove from formatted_disks so it's not added again
                            del formatted_disks[path]

                    # Add any remaining disks (e.g., KingSpec1, KingSpec2, Removable)
                    for path in formatted_disks:
                        msg_parts.append(formatted_disks[path]['string'])

                    if not all_disk_usage: # If no disks were found at all
                        msg_parts.append(f"• {COLOR_BLUE}Disk Usage:{COLOR_RESET} N/A")

                    # GPU Info
                    gpu_info = status_info.get('gpu_info', [])
                    if gpu_info:
                        for i, gpu in enumerate(gpu_info):
                            gpu_name = gpu.get('name', 'N/A')
                            gpu_type = gpu.get('type', 'N/A')
                            msg_parts.append(f"• {COLOR_BLUE}GPU {i+1}:{COLOR_RESET} {gpu_name} [{gpu_type}]")
                    else:
                        msg_parts.append(f"• {COLOR_BLUE}GPU:{COLOR_RESET} N/A")

                    # Vulkan and OpenCL Info
                    vulkan_info = status_info.get('vulkan_info', 'N/A')
                    opencl_info = status_info.get('opencl_info', 'N/A')
                    msg_parts.append(f"• {COLOR_BLUE}Vulkan:{COLOR_RESET} {vulkan_info}")
                    msg_parts.append(f"• {COLOR_BLUE}OpenCL:{COLOR_RESET} {opencl_info}")

                    ollama_status = cli._check_ollama_status() if hasattr(cli, '_check_ollama_status') else "N/A"
                    msg_parts.append(f"• {COLOR_BLUE}Ollama Server:{COLOR_RESET} {ollama_status}")

                    db_status = database_utils.get_database_status()
                    if db_status['connected']:
                        msg_parts.append(f"• {COLOR_BLUE}Database:{COLOR_RESET} Connected (Tables: {', '.join(db_status['tables'])})")
                    else:
                        msg_parts.append(f"• {COLOR_BLUE}Database:{COLOR_RESET} Not Connected ({db_status.get('error', 'Unknown error')})")

                    response = "\n".join(msg_parts)
                    response_type = "system_status"
                    print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}")
                    print(f"{COLOR_GREEN}┌── System Status ──┐{COLOR_RESET}")
                    print(response)
                    print(f"{COLOR_GREEN}└───────────────────┘{COLOR_RESET}")
                if TTS_ENABLED:
                    speak_text_async(response)
                database_utils.log_interaction(user_id=user_id, user_query=query, kaia_response=response, response_type=response_type)
                if cmd in ['exit', 'quit']:
                    break
                continue

            # Process user query through action planner
            plan = generate_action_plan(query)
            action = plan.get("action", "chat")
            content = plan.get("content", query)

            if action == "store_data":
                if isinstance(content, list):
                    content = ' '.join(content)
                storage_handled, storage_response = database_utils.handle_memory_storage(user_id, content)
                response = storage_response
                response_type = "store_data"
                print(f"\n{COLOR_BLUE}Kaia: {response}{COLOR_RESET}")

            elif action == "command":
                print(f"\n{COLOR_BLUE}Kaia (Command Mode):{COLOR_RESET}")
                command, error = cli.generate_command(str(content))

                if error:
                    response = f"Command generation failed: {error}"
                    response_type = "command_error"
                    print(f"{COLOR_RED}{response}{COLOR_RESET}")
                else:
                    print(f"\n{COLOR_YELLOW}┌── Proposed Command ──┐{COLOR_RESET}")
                    print(f"{COLOR_BLUE}{command}{COLOR_RESET}")
                    print(f"{COLOR_YELLOW}└──────────────────────┘{COLOR_RESET}")

                    confirm = input(f"{COLOR_YELLOW}Execute? (y/N): {COLOR_RESET}").lower().strip()
                    if confirm == 'y':
                        print(f"\n{COLOR_GREEN}Executing...{COLOR_RESET}")
                        success, stdout, stderr = cli.execute_command(command)
                        if success:
                            response = f"Command executed successfully. Output:\n{stdout}"
                            response_type = "command_executed"
                            print(f"{COLOR_GREEN}{response}{COLOR_RESET}")
                            if stderr:
                                print(f"{COLOR_YELLOW}Stderr:\n{stderr}{COLOR_RESET}")
                        else:
                            response = f"Command failed. Stderr:\n{stderr}\nStdout:\n{stdout}"
                            response_type = "command_error"
                            print(f"{COLOR_RED}{response}{COLOR_RESET}")
                    else:
                        response = f"Command cancelled: {command}"
                        response_type = "command_cancelled"
                        print(f"{COLOR_BLUE}{response}{COLOR_RESET}")

            elif action == "system_status":
                # This block is now redundant for 'status' and similar queries due to direct handling above
                # It remains here as a fallback for cases where LLM might still classify something as 'system_status'
                # but it's not one of the directly handled keywords.
                status_info = cli.get_system_status()
                msg_parts = [
                        f"• {COLOR_BLUE}Date & Time:{COLOR_RESET} {datetime.fromisoformat(status_info.get('timestamp', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')}",
                        f"• {COLOR_BLUE}Uptime:{COLOR_RESET} {status_info.get('uptime', 'N/A')}",
                        f"• {COLOR_BLUE}Board:{COLOR_RESET} {status_info.get('board_info', 'N/A')}",
                        f"• {COLOR_BLUE}OS:{COLOR_RESET} {status_info.get('os_info', 'N/A')}",
                        f"• {COLOR_BLUE}Kernel:{COLOR_RESET} {status_info.get('kernel_info', 'N/A')}",
                        f"• {COLOR_BLUE}Python Version:{COLOR_RESET} {status_info.get('python_version', 'N/A')}",
                    ]

                cpu_info = status_info.get('cpu_info', {})
                if cpu_info:
                    cpu_name = cpu_info.get('name', 'N/A')
                    cpu_speed = cpu_info.get('speed', 'N/A')
                    cpu_cores = cpu_info.get('logical_cores', 'N/A')
                    msg_parts.append(f"• {COLOR_BLUE}CPU:{COLOR_RESET} {cpu_name} ({cpu_cores}) @ {cpu_speed}")

                mem_info = status_info.get('memory_info', {})
                if mem_info:
                    total_gb = round(mem_info.get('total', 0) / (1024**3), 2)
                    available_gb = round(mem_info.get('available', 0) / (1024**3), 2)
                    percent_used = mem_info.get('percent', 'N/A')
                    percent_color = get_color_for_percentage(percent_used) # Get color for memory percentage
                    msg_parts.append(f"• {COLOR_BLUE}Memory:{COLOR_RESET} {total_gb} GB total, {available_gb} GB available ({percent_color}{percent_used}% used{COLOR_RESET})")

                all_disk_usage = status_info.get('all_disk_usage', [])

                # Create a temporary dictionary to hold formatted disk strings for easy lookup
                formatted_disks = {}
                for disk in all_disk_usage:
                    path = disk.get('path', disk.get('mount_point', 'N/A'))
                    label = disk.get('label', disk.get('mount_point', 'N/A'))

                    if disk.get('status') == 'Error':
                        formatted_disks[path] = f"• {COLOR_RED}Disk Usage ('{label}'):{COLOR_RESET} Error - {disk.get('error_message', 'N/A')}"
                    else:
                        total_gb = round(disk.get('total', 0) / (1024**3), 2)
                        used_gb = round(disk.get('used', 0) / (1024**3), 2)
                        percent_used = disk.get('percent', 'N/A')
                        percent_color = get_color_for_percentage(percent_used) # Get color for disk percentage
                        formatted_disks[path] = {
                            'string': f"• {COLOR_BLUE}Disk Usage ('{label}'):{COLOR_RESET} {total_gb} GB total, {used_gb} GB used ({percent_color}{percent_used}% used{COLOR_RESET})",
                            'data': disk # Keep original data for reconstruction
                        }

                # Define the desired order and new labels
                desired_order = [
                    {'path': '/boot', 'new_label': 'Boot'},
                    {'path': '/', 'new_label': 'Root'},
                    {'path': '/home', 'new_label': 'Home'}
                ]

                # Add disks in desired order with new labels
                for item in desired_order:
                    path = item['path']
                    new_label = item['new_label']
                    if path in formatted_disks:
                        disk_data = formatted_disks[path]['data']
                        if disk_data.get('status') == 'Error':
                            msg_parts.append(f"• {COLOR_RED}Disk Usage ('{new_label}'):{COLOR_RESET} Error - {disk_data.get('error_message', 'N/A')}")
                        else:
                            total_gb = round(disk_data.get('total', 0) / (1024**3), 2)
                            used_gb = round(disk_data.get('used', 0) / (1024**3), 2)
                            percent_used = disk_data.get('percent', 'N/A')
                            percent_color = get_color_for_percentage(percent_used) # Get color for disk percentage
                            msg_parts.append(f"• {COLOR_BLUE}Disk Usage ('{new_label}'):{COLOR_RESET} {total_gb} GB total, {used_gb} GB used ({percent_color}{percent_used}% used{COLOR_RESET})")

                        # Remove from formatted_disks so it's not added again
                        del formatted_disks[path]

                # Add any remaining disks (e.g., KingSpec1, KingSpec2, Removable)
                for path in formatted_disks:
                    msg_parts.append(formatted_disks[path]['string'])

                if not all_disk_usage: # If no disks were found at all
                    msg_parts.append(f"• {COLOR_BLUE}Disk Usage:{COLOR_RESET} N/A")

                gpu_info = status_info.get('gpu_info', [])
                if gpu_info:
                    for i, gpu in enumerate(gpu_info):
                        gpu_name = gpu.get('name', 'N/A')
                        gpu_type = gpu.get('type', 'N/A')
                        msg_parts.append(f"• {COLOR_BLUE}GPU {i+1}:{COLOR_RESET} {gpu_name} [{gpu_type}]")
                else:
                    msg_parts.append(f"• {COLOR_BLUE}GPU:{COLOR_RESET} N/A")

                vulkan_info = status_info.get('vulkan_info', 'N/A')
                opencl_info = status_info.get('opencl_info', 'N/A')
                msg_parts.append(f"• {COLOR_BLUE}Vulkan:{COLOR_RESET} {vulkan_info}")
                msg_parts.append(f"• {COLOR_BLUE}OpenCL:{COLOR_RESET} {opencl_info}")

                ollama_status = cli._check_ollama_status() if hasattr(cli, '_check_ollama_status') else "N/A"
                msg_parts.append(f"• {COLOR_BLUE}Ollama Server:{COLOR_RESET} {ollama_status}")

                db_status = database_utils.get_database_status()
                if db_status['connected']:
                    msg_parts.append(f"• {COLOR_BLUE}Database:{COLOR_RESET} Connected (Tables: {', '.join(db_status['tables'])})")
                else:
                    msg_parts.append(f"• {COLOR_BLUE}Database:{COLOR_RESET} Not Connected ({db_status.get('error', 'Unknown error')})")

                response = "\n".join(msg_parts)
                response_type = "system_status"
                print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}")
                print(f"{COLOR_GREEN}┌── System Status ──┐{COLOR_RESET}")
                print(response)
                print(f"{COLOR_GREEN}└───────────────────┘{COLOR_RESET}")

            elif action == "sql" and SQL_RAG_ENABLED and sql_query_engine:
                try:
                    print(f"\n{COLOR_BLUE}Kaia (Querying Database):{COLOR_RESET}")
                    sql_response = sql_query_engine.query(content)
                    response = str(sql_response)
                    response_type = "sql_query"
                    print(f"{COLOR_GREEN}┌── Query Results ──┐{COLOR_RESET}")
                    print(f"{response}")
                    print(f"{COLOR_GREEN}└───────────────────┘{COLOR_RESET}")
                except Exception as e:
                    response = f"Database Error: {e}"
                    response_type = "sql_error"
                    print(f"{COLOR_RED}{response}{COLOR_RESET}")

            elif action == "retrieve_data":
                if isinstance(content, list):
                    content = ' '.join(content)
                result = database_utils.handle_data_retrieval(user_id, content)
                response = result['message']
                response_type = result['response_type']

                print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}")
                if isinstance(result['data'], list) and result['data']:
                    print(f"{COLOR_GREEN}┌── {result['message']} ──┐{COLOR_RESET}")
                    for item in result['data']:
                        if isinstance(item, dict):
                            formatted_item = ", ".join([f"{k}: {v}" for k, v in item.items()])
                            print(f"• {formatted_item}")
                        else:
                            print(f"• {str(item)}")
                    print(f"{COLOR_GREEN}└──────────────────────┘{COLOR_RESET}")
                else:
                    print(response)

            elif action == "get_persona_content":
                print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}")
                print(f"{COLOR_GREEN}┌── Kaia's Persona ──┐{COLOR_RESET}")
                print(kaia_persona_content)
                print(f"{COLOR_GREEN}└────────────────────┘{COLOR_RESET}")
                response = kaia_persona_content
                response_type = "persona_retrieved"


            else: # Default to chat if no specific action or if other actions failed/not enabled
                print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}", end=" ", flush=True)
                full_response = ""
                line_length = 0

                is_casual_greeting = any(phrase in query.lower() for phrase in ["hello", "hi", "hey", "how are you", "what's up", "are you operational", "are you there", "you there"])

                if not any(char in query for char in ['?', '!', '.']):
                    print("...", end="", flush=True)

                if action == "chat" and is_casual_greeting:
                    current_chat_engine = pure_chat_engine
                else:
                    current_chat_engine = rag_chat_engine

                response_stream = current_chat_engine.stream_chat(content)
                first_token_received = False
                first_token_time = None

                for token in response_stream.response_gen:
                    if not first_token_received:
                        first_token_received = True
                        first_token_time = time.time()
                        print(f"\n{COLOR_YELLOW}⚡ First token in {first_token_time - start_time:.2f}s{COLOR_RESET}\n{COLOR_BLUE}", end="", flush=True)
                        sys.stdout.write("\033[K")

                    line_length += len(token)
                    if line_length > 80:
                        print()
                        line_length = 0

                    sys.stdout.write(token)
                    sys.stdout.flush()
                    full_response += token
                response = full_response
                response_type = "chat"
                print(f"\n\n{COLOR_YELLOW}⏱ Total time: {time.time() - start_time:.2f}s", end=" ")
                if first_token_time:
                    print(f"(First token: {first_token_time - start_time:.2f}s)", end="")
                print(f"{COLOR_RESET}")

            if TTS_ENABLED:
                speak_text_async(response)

            try:
                database_utils.log_interaction(
                    user_id=user_id,
                    user_query=query,
                    kaia_response=response,
                    response_type=response_type
                )
            except Exception as log_e:
                logger.error(f"Failed to log interaction: {log_e}", exc_info=True)

        except KeyboardInterrupt:
            print(f"\n{COLOR_BLUE}Kaia: Exiting gracefully...{COLOR_RESET}")
            break
        except Exception as e:
            logger.exception("Unexpected error in main loop")
            response = f"System error: {e}"
            response_type = "system_error"
            print(f"{COLOR_RED}{response}{COLOR_RESET}")

            try:
                database_utils.log_interaction(
                    user_id=user_id,
                    user_query=query,
                    kaia_response=f"System error: {str(e)[:200]}",
                    response_type="system_error"
                )
            except Exception as log_e:
                logger.error(f"Failed to log error: {log_e}")

if __name__ == "__main__":
    main()
