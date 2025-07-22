import json
import logging
import os
import platform
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
from kaia_cli import KaiaCLI
from contextlib import redirect_stdout, contextmanager
from typing import Optional, List, Dict, Union, Any
from datetime import datetime

# --- Config ---
TIMEOUT_SECONDS = 300  # 5 minutes timeout for all operations
LLM_MODEL = "llama2:7b-chat" # Changed to llama2:7b-chat for RAG testing
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_COMMAND_MODEL = "mistral:instruct" # Retained mistral:instruct for action planning
MAX_LINE_WIDTH = 80 # Define maximum line width for word wrapping

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
    logging.getLogger(noisy).setLevel(logging.CRITICAL) # Corrected: logging.CRITICAL
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
            # FIX: Added recursive=True to SimpleDirectoryReader to load documents from subdirectories
            general_docs = [doc for doc in SimpleDirectoryReader(GENERAL_KNOWLEDGE_DIR, recursive=True).load_data()
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
                # Updated system prompt to explicitly instruct the LLM to use provided context
                combined_system_prompt = (
                    "You are Kaia, a helpful and knowledgeable AI assistant. "
                    "Your responses should be concise and directly answer the user's question. "
                    "You MUST ONLY use the provided context to answer the question. "
                    "If the answer is NOT explicitly present in the provided context, you MUST state clearly that you do not have enough information to answer based on the available data. "
                    "Do NOT invent information or use your general knowledge if it contradicts the context. "
                    "NEVER refer to yourself as a language model or AI assistant in the context of answering a factual question from the provided data, unless the question is explicitly about your own nature or capabilities. "
                    "Maintain your core persona: strategic precision, intellectual curiosity, dry, often sarcastic wit. "
                    "Persona details: " + kaia_persona_content
                )
                get_rag_chat_engine._cached_engine = index_param.as_chat_engine(
                    chat_mode="condense_plus_context",
                    memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
                    system_prompt=combined_system_prompt,
                    similarity_top_k=2 # Reduced similarity_top_k to limit context size
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

    def generate_action_plan(user_input):
        # FIX: Overhauled system prompt and added a 'knowledge_query' category for better intent classification.
        system_prompt = """You are an AI assistant that classifies user intents. Respond ONLY with valid JSON.

        Categories:
        - "command": For requests to run terminal commands (e.g., "list files", "check running processes").
        - "knowledge_query": For questions that require information retrieval from a knowledge base (e.g., "What is...", "Explain...", "According to...", "Summarize...").
        - "sql": For complex database queries requiring SQL joins/aggregations on `facts`, `interaction_history`, `user_preferences` tables.
        - "retrieve_data": For simple retrieval of stored personal data ("What are my preferences?", "List my facts", "What do you know about me?", "List history?").
        - "store_data": For requests to remember information ("Remember that...", "My favorite food is...").
        - "system_status": For requests about system health ("show system status", "how is my computer doing", "status kaia").
        - "get_persona_content": For questions about Kaia's identity ("Tell me about yourself", "What is your persona?").
        - "chat": For general conversation, greetings, and queries that don't fit other categories.

        Respond with: {"action": "action_name", "content": "query_content"}
        """

        payload = {
            "model": DEFAULT_COMMAND_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                # FIX: Added examples for 'knowledge_query' to improve model accuracy.
                {"role": "user", "content": "What is the DRY principle in programming?"},
                {"role": "assistant", "content": json.dumps({"action": "knowledge_query", "content": "What is the DRY principle in programming?"})},
                {"role": "user", "content": "According to Russell and Norvig, what are the four approaches to AI?"},
                {"role": "assistant", "content": json.dumps({"action": "knowledge_query", "content": "What are the four approaches to AI as described by Russell and Norvig?"})},
                {"role": "user", "content": "Summarize H+ Magazine - Issue 2"}, # Added example for summarize
                {"role": "assistant", "content": json.dumps({"action": "knowledge_query", "content": "Summarize H+ Magazine - Issue 2"})},
                {"role": "user", "content": f"Generate a command to list all files in the current directory."},
                {"role": "assistant", "content": json.dumps({"action": "command", "content": "ls -a"})},
                {"role": "user", "content": f"Hello, how are you?"},
                {"role": "assistant", "content": json.dumps({"action": "chat", "content": "Hello, how are you?"})},
                {"role": "user", "content": f"Show me all the facts you know."},
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "all facts"})},
                {"role": "user", "content": f"List my facts."}, # Added example for retrieve_data
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "list my facts"})},
                {"role": "user", "content": f"List history."}, # Added example for retrieve_data
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "list history"})},
                {"role": "user", "content": f"Show interaction history."}, # Added example for retrieve_data
                {"role": "assistant", "content": json.dumps({"action": "retrieve_data", "content": "show interaction history"})},
                {"role": "user", "content": f"Remember that my favorite food is pizza."},
                {"role": "assistant", "content": json.dumps({"action": "store_data", "content": "my favorite food is pizza"})},
                {"role": "user", "content": f"What is the current system status?"},
                {"role": "assistant", "content": json.dumps({"action": "system_status", "content": "current system status"})},
                {"role": "user", "content": "status kaia"}, # Added example for system_status
                {"role": "assistant", "content": json.dumps({"action": "system_status", "content": "status kaia"})},
                {"role": "user", "content": "kaia status"}, # Added example for system_status
                {"role": "assistant", "content": json.dumps({"action": "system_status", "content": "kaia status"})},
                {"role": "user", "content": str(user_input)}
            ],
            "stream": False,
            "format": "json"
        }

        try:
            # Check if DEFAULT_COMMAND_MODEL is available before making the request
            ollama_models_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            ollama_models_response.raise_for_status()
            available_models = [m['name'] for m in ollama_models_response.json().get('models', [])]

            # If the configured DEFAULT_COMMAND_MODEL is not available, try to find a suitable alternative
            model_to_use = DEFAULT_COMMAND_MODEL
            if DEFAULT_COMMAND_MODEL not in available_models:
                logger.warning(f"Configured Ollama model '{DEFAULT_COMMAND_MODEL}' not found. Attempting to use '{LLM_MODEL}' for action planning.")
                if LLM_MODEL in available_models:
                    model_to_use = LLM_MODEL
                else:
                    # Fallback to a generic chat model if neither is found
                    if 'llama2:7b-chat' in available_models:
                        model_to_use = 'llama2:7b-chat'
                    elif 'mistral:instruct' in available_models:
                        model_to_use = 'mistral:instruct'
                    else:
                        logger.error(f"No suitable Ollama model found for action planning. Available models: {available_models}")
                        # Fallback to chat action if no model can be used for classification
                        return {"action": "chat", "content": user_input}

            payload["model"] = model_to_use # Update the model in the payload

            response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            result = response.json()["message"]["content"].strip()
            return json.loads(result)
        except Exception as e:
            logger.error(f"Action plan error: {e}", exc_info=True)
            # Fallback to knowledge_query for question-like inputs, otherwise chat.
            user_input_lower = user_input.lower()
            if any(keyword in user_input_lower for keyword in ['what is', 'who is', 'explain', 'tell me about', 'according to', 'summarize']):
                return {"action": "knowledge_query", "content": user_input}
            # FIX: Added specific fallbacks for retrieve_data and system_status
            if any(keyword in user_input_lower for keyword in ['list my facts', 'list history', 'show interaction history', 'what do you know about me', 'my preferences']):
                return {"action": "retrieve_data", "content": user_input}
            if any(keyword in user_input_lower for keyword in ['status', 'how is my computer doing', 'system info', 'show system status', 'display system status', 'status kaia', 'kaia status']):
                return {"action": "system_status", "content": user_input}
            return {"action": "chat", "content": user_input}

    def get_color_for_percentage(percent: Union[int, float]) -> str:
        if percent <= 70:
            return COLOR_GREEN
        elif 70 < percent <= 80:
            return COLOR_YELLOW
        else:
            return COLOR_RED

    def stream_and_print_response(response_stream, start_time):
        """Helper function to stream response, handle word wrap, and print timing info."""
        full_response = ""
        first_token_received = False
        first_token_time = None
        current_line_length = 0
        word_buffer = []

        for token in response_stream.response_gen:
            if not first_token_received:
                first_token_received = True
                first_token_time = time.time()
                print(f"\n{COLOR_YELLOW}⚡ First token in {first_token_time - start_time:.2f}s{COLOR_RESET}\n{COLOR_BLUE}", end="", flush=True)
                sys.stdout.write("\033[K")

            for char in token:
                word_buffer.append(char)
                if char.isspace() or char in ['.', ',', '!', '?', ';', ':'] or len(word_buffer) > MAX_LINE_WIDTH:
                    word_to_print = "".join(word_buffer)
                    word_buffer = []

                    if current_line_length + len(word_to_print) > MAX_LINE_WIDTH and current_line_length > 0:
                        sys.stdout.write("\n")
                        current_line_length = 0

                    sys.stdout.write(word_to_print)
                    current_line_length += len(word_to_print)
                    full_response += word_to_print
                    sys.stdout.flush()

        # Print any remaining word in the buffer
        if word_buffer:
            word_to_print = "".join(word_buffer)
            if current_line_length + len(word_to_print) > MAX_LINE_WIDTH and current_line_length > 0:
                sys.stdout.write("\n")
            sys.stdout.write(word_to_print)
            full_response += word_to_print
            sys.stdout.flush()


        response = full_response
        response_type = "chat"
        print(f"\n\n{COLOR_YELLOW}⏱ Total time: {time.time() - start_time:.2f}s", end=" ")
        if first_token_time:
            print(f"(First token: {first_token_time - start_time:.2f}s)", end="")
        print(f"{COLOR_RESET}")
        return full_response

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

            # Direct command handling
            if query.startswith('/') or query.startswith('!'):
                cmd_query = query[1:].strip()
                # FIX: Moved /status handling to the top of direct command block
                if cmd_query.lower() == 'status': # Explicitly handle /status
                    status_info = cli.get_system_status()
                    msg_parts = [
                        f"• {COLOR_BLUE}Date & Time:{COLOR_RESET} {datetime.fromisoformat(status_info.get('timestamp', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')}",
                        f"• {COLOR_BLUE}Uptime:{COLOR_RESET} {status_info.get('uptime', 'N/A')}",
                        f"• {COLOR_BLUE}Board:{COLOR_RESET} {status_info.get('board_info', 'N/A')}",
                        f"• {COLOR_BLUE}OS:{COLOR_RESET} {status_info.get('os_info', 'N/A')}",
                        f"• {COLOR_BLUE}Kernel:{COLOR_RESET} {status_info.get('kernel_info', 'N/A')}",
                        f"• {COLOR_BLUE}Python Version:{COLOR_RESET} {platform.python_version()}",
                    ]

                    cpu_info = status_info.get('cpu_info', {})
                    if cpu_info:
                        cpu_name = cpu_info.get('name', 'N/A')
                        cpu_speed = cpu_info.get('speed', 'N/A')
                        cpu_cores = cpu_info.get('logical_cores', 'N/A')
                        msg_parts.append(f"• {COLOR_BLUE}CPU:{COLOR_RESET} {cpu_name} ({cpu_cores}) @ {cpu_speed}")

                    mem_info = status_info.get('memory_info', {})
                    if mem_info:
                        total_gb = round(mem_info.get('total_gb', 0), 2)
                        available_gb = round(mem_info.get('available_gb', 0), 2)
                        percent_used = mem_info.get('percent_used', 'N/A')
                        percent_color = get_color_for_percentage(percent_used)
                        msg_parts.append(f"• {COLOR_BLUE}Memory:{COLOR_RESET} {total_gb} GB total, {available_gb} GB available ({percent_color}{percent_used}% used{COLOR_RESET})")

                    all_disk_usage = status_info.get('disk_usage', []) # Corrected key from all_disk_usage to disk_usage

                    formatted_disks = {}
                    for disk in all_disk_usage:
                        path = disk.get('mount_point', 'N/A')
                        label = disk.get('mount_point', 'N/A') # Use mount_point as label initially

                        if disk.get('status') == 'Error':
                            formatted_disks[path] = f"• {COLOR_RED}Disk Usage ('{label}'):{COLOR_RESET} Error - {disk.get('error_message', 'N/A')}"
                        else:
                            total_gb = round(disk.get('total_gb', 0), 2)
                            used_gb = round(disk.get('used_gb', 0), 2)
                            percent_used = disk.get('percent_used', 'N/A')
                            percent_color = get_color_for_percentage(percent_used)
                            formatted_disks[path] = {
                                'string': f"• {COLOR_BLUE}Disk Usage ('{label}'):{COLOR_RESET} {total_gb} GB total, {used_gb} GB used ({percent_color}{percent_used}% used{COLOR_RESET})",
                                'data': disk
                            }

                    desired_order = [
                        {'path': '/boot', 'new_label': 'Boot'},
                        {'path': '/', 'new_label': 'Root'},
                        {'path': '/home', 'new_label': 'Home'}
                    ]

                    for item in desired_order:
                        path = item['path']
                        new_label = item['new_label']
                        if path in formatted_disks:
                            disk_data = formatted_disks[path]['data']
                            if disk_data.get('status') == 'Error':
                                msg_parts.append(f"• {COLOR_RED}Disk Usage ('{new_label}'):{COLOR_RESET} Error - {disk_data.get('error_message', 'N/A')}")
                            else:
                                total_gb = round(disk_data.get('total_gb', 0), 2)
                                used_gb = round(disk_data.get('used_gb', 0), 2)
                                percent_used = disk_data.get('percent_used', 'N/A')
                                percent_color = get_color_for_percentage(percent_used)
                                msg_parts.append(f"• {COLOR_BLUE}Disk Usage ('{new_label}'):{COLOR_RESET} {total_gb} GB total, {used_gb} GB used ({percent_color}{percent_used}% used{COLOR_RESET})")

                            del formatted_disks[path]

                    for path in formatted_disks:
                        msg_parts.append(formatted_disks[path]['string'])

                    if not all_disk_usage:
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
                        msg_parts.append(f"• {COLOR_BLUE}Database:{COLOR_RESET} Connected (Tables: {', '.join(db_status.get('tables', []))})")
                    else:
                        msg_parts.append(f"• {COLOR_BLUE}Database:{COLOR_RESET} Not Connected ({db_status.get('error', 'Unknown error')})")

                    response = "\n".join(msg_parts)
                    response_type = "system_status"
                    print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}")
                    print(f"{COLOR_GREEN}┌── System Status ──┐{COLOR_RESET}")
                    print(response)
                    print(f"{COLOR_GREEN}└───────────────────┘{COLOR_RESET}")
                # FIX: Fallback for other direct commands moved to an else block
                else:
                    print(f"\n{COLOR_BLUE}Kaia (Direct Command Mode):{COLOR_RESET}")
                    success, stdout, stderr = cli.execute_command(cmd_query)
                    if success:
                        response = f"Command executed successfully. Output:\n{stdout}"
                        print(f"{COLOR_GREEN}{response}{COLOR_RESET}")
                        if stderr: print(f"{COLOR_YELLOW}Stderr:\n{stderr}{COLOR_RESET}")
                    else:
                        response = f"Command failed. Stderr:\n{stderr}\nStdout:\n{stdout}"
                        print(f"{COLOR_RED}{response}{COLOR_RESET}")
                    response_type = "direct_command"

                database_utils.log_interaction(user_id=user_id, user_query=query, kaia_response=response, response_type=response_type)
                continue

            # Process user query through action planner
            plan = generate_action_plan(query)
            action = plan.get("action", "chat")
            content = plan.get("content", query)

            # FIX: Refactored main loop to use explicit `elif` blocks for each action, preventing incorrect routing.
            if action == "store_data":
                if isinstance(content, list): content = ' '.join(content)
                storage_handled, storage_response = database_utils.handle_memory_storage(user_id, content)
                response = storage_response
                response_type = "store_data"
                print(f"\n{COLOR_BLUE}Kaia: {response}{COLOR_RESET}")

            elif action == "command":
                print(f"\n{COLOR_BLUE}Kaia (Command Mode):{COLOR_RESET}")
                # FIX: Pass DEFAULT_COMMAND_MODEL to cli.generate_command
                command, error = cli.generate_command(str(content), DEFAULT_COMMAND_MODEL)
                if error:
                    response = f"Command generation failed: {error}"
                    print(f"{COLOR_RED}{response}{COLOR_RESET}")
                else:
                    print(f"\n{COLOR_YELLOW}┌── Proposed Command ──┐{COLOR_RESET}")
                    print(f"{COLOR_BLUE}{command}{COLOR_RESET}")
                    print(f"{COLOR_YELLOW}└──────────────────────┘{COLOR_RESET}")
                    confirm = input(f"{COLOR_YELLOW}Execute? (y/N): {COLOR_RESET}").lower().strip()
                    if confirm == 'y':
                        print(f"{COLOR_GREEN}Executing...{COLOR_RESET}")
                        success, stdout, stderr = cli.execute_command(command)
                        if success:
                            response = f"Command executed successfully. Output:\n{stdout}"
                            print(f"{COLOR_GREEN}{response}{COLOR_RESET}")
                            if stderr: print(f"{COLOR_YELLOW}Stderr:\n{stderr}{COLOR_RESET}")
                        else:
                            response = f"Command failed. Stderr:\n{stderr}\nStdout:\n{stdout}"
                            print(f"{COLOR_RED}{response}{COLOR_RESET}")
                    else:
                        response = f"Command cancelled: {command}"
                        print(f"{COLOR_BLUE}{response}{COLOR_RESET}")
                response_type = "command"

            elif action == "system_status":
                status_info = cli.get_system_status()
                # Restore full printing logic for system status
                msg_parts = [
                    f"• {COLOR_BLUE}Date & Time:{COLOR_RESET} {datetime.fromisoformat(status_info.get('timestamp', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')}",
                    f"• {COLOR_BLUE}Uptime:{COLOR_RESET} {status_info.get('uptime', 'N/A')}",
                    f"• {COLOR_BLUE}Board:{COLOR_RESET} {status_info.get('board_info', 'N/A')}",
                    f"• {COLOR_BLUE}OS:{COLOR_RESET} {status_info.get('os_info', 'N/A')}",
                    f"• {COLOR_BLUE}Kernel:{COLOR_RESET} {status_info.get('kernel_info', 'N/A')}",
                    f"• {COLOR_BLUE}Python Version:{COLOR_RESET} {platform.python_version()}",
                ]

                cpu_info = status_info.get('cpu_info', {})
                if cpu_info:
                    cpu_name = cpu_info.get('name', 'N/A')
                    cpu_speed = cpu_info.get('speed', 'N/A')
                    cpu_cores = cpu_info.get('logical_cores', 'N/A')
                    msg_parts.append(f"• {COLOR_BLUE}CPU:{COLOR_RESET} {cpu_name} ({cpu_cores}) @ {cpu_speed}")

                mem_info = status_info.get('memory_info', {})
                if mem_info:
                    total_gb = round(mem_info.get('total_gb', 0), 2)
                    available_gb = round(mem_info.get('available_gb', 0), 2)
                    percent_used = mem_info.get('percent_used', 'N/A')
                    percent_color = get_color_for_percentage(percent_used)
                    msg_parts.append(f"• {COLOR_BLUE}Memory:{COLOR_RESET} {total_gb} GB total, {available_gb} GB available ({percent_color}{percent_used}% used{COLOR_RESET})")

                all_disk_usage = status_info.get('disk_usage', []) # Corrected key from all_disk_usage to disk_usage

                formatted_disks = {}
                for disk in all_disk_usage:
                    path = disk.get('mount_point', 'N/A')
                    label = disk.get('mount_point', 'N/A')

                    if disk.get('status') == 'Error':
                        formatted_disks[path] = f"• {COLOR_RED}Disk Usage ('{label}'):{COLOR_RESET} Error - {disk.get('error_message', 'N/A')}"
                    else:
                        total_gb = round(disk.get('total_gb', 0), 2)
                        used_gb = round(disk.get('used_gb', 0), 2)
                        percent_used = disk.get('percent_used', 'N/A')
                        percent_color = get_color_for_percentage(percent_used)
                        formatted_disks[path] = {
                            'string': f"• {COLOR_BLUE}Disk Usage ('{label}'):{COLOR_RESET} {total_gb} GB total, {used_gb} GB used ({percent_color}{percent_used}% used{COLOR_RESET})",
                            'data': disk
                        }

                desired_order = [
                    {'path': '/boot', 'new_label': 'Boot'},
                    {'path': '/', 'new_label': 'Root'},
                    {'path': '/home', 'new_label': 'Home'}
                ]

                for item in desired_order:
                    path = item['path']
                    new_label = item['new_label']
                    if path in formatted_disks:
                        disk_data = formatted_disks[path]['data']
                        if disk_data.get('status') == 'Error':
                            msg_parts.append(f"• {COLOR_RED}Disk Usage ('{new_label}'):{COLOR_RESET} Error - {disk_data.get('error_message', 'N/A')}")
                        else:
                            total_gb = round(disk_data.get('total_gb', 0), 2)
                            used_gb = round(disk_data.get('used_gb', 0), 2)
                            percent_used = disk_data.get('percent_used', 'N/A')
                            percent_color = get_color_for_percentage(percent_used)
                            msg_parts.append(f"• {COLOR_BLUE}Disk Usage ('{new_label}'):{COLOR_RESET} {total_gb} GB total, {used_gb} GB used ({percent_color}{percent_used}% used{COLOR_RESET})")

                        del formatted_disks[path]

                for path in formatted_disks:
                    msg_parts.append(formatted_disks[path]['string'])

                if not all_disk_usage:
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
                    msg_parts.append(f"• {COLOR_BLUE}Database:{COLOR_RESET} Connected (Tables: {', '.join(db_status.get('tables', []))})")
                else:
                    msg_parts.append(f"• {COLOR_BLUE}Database:{COLOR_RESET} Not Connected ({db_status.get('error', 'Unknown error')})")

                response = "\n".join(msg_parts)
                response_type = "system_status"
                print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}")
                print(f"{COLOR_GREEN}┌── System Status ──┐{COLOR_RESET}")
                print(response) # Ensure full response is printed
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
                if isinstance(content, list): content = ' '.join(content)
                result = database_utils.handle_data_retrieval(user_id, content)
                response = result['message']
                response_type = result['response_type']
                print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}")
                if isinstance(result['data'], list) and result['data']:
                    print(f"{COLOR_GREEN}┌── {result['message']} ──┐{COLOR_RESET}")
                    for item in result['data']:
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

            elif action == "knowledge_query":
                print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}", end=" ", flush=True)
                response_stream = rag_chat_engine.stream_chat(content)
                response = stream_and_print_response(response_stream, start_time)
                response_type = "knowledge_query"

            else: # Default to pure chat for "chat" action or any unhandled cases
                print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}", end=" ", flush=True)
                response_stream = pure_chat_engine.stream_chat(content)
                response = stream_and_print_response(response_stream, start_time)
                response_type = "chat"

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
