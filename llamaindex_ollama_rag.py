import json
import logging
import os
import subprocess
import sys
import time
import requests
import database_utils
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage,
    SQLDatabase,
)
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from sqlalchemy import create_engine
from kaia_cli import KaiaCLI

# --- Config ---
LLM_MODEL = "mistral:instruct"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_COMMAND_MODEL = "mistral:instruct"

GENERAL_KNOWLEDGE_DIR = "./data"
PERSONAL_CONTEXT_DIR = "./personal_context"
PERSONA_DIR = "./data"
PERSIST_DIR = "./storage"
CHROMA_DB_PATH = "./storage/chroma_db"
LLAMA_INDEX_METADATA_PATH = "./storage/llama_index_metadata"
TTS_ENABLED = False
SQL_DATABASE_PATH = "./storage/kaia_database.db"

# --- ANSI Color Codes ---
COLOR_GREEN = "\033[92m"
COLOR_BLUE = "\033[94m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"

# --- Kaia's Core System Prompt ---
KAIA_SYSTEM_PROMPT = """
[Your system prompt content here]
"""

# --- Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.ERROR)
for noisy in ["httpx", "httpcore", "llama_index", "chromadb", "fsspec", "urllib3"]:
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

# --- Initialize Models ---
session = requests.Session()
try:
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=360.0, stream=True)
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
    _ = Settings.llm.complete("Hello")
    cli = KaiaCLI()
    print(f"{COLOR_GREEN}LLM initialized and warmed up.{COLOR_RESET}")
except Exception as e:
    logging.error(f"{COLOR_RED}Failed to initialize Ollama: {e}{COLOR_RESET}")
    sys.exit(1)

# --- Load Persona ---
persona_doc_path = os.path.join(PERSONA_DIR, "Kaia_Desktop_Persona.md")
try:
    kaia_persona_content = SimpleDirectoryReader(input_files=[persona_doc_path]).load_data()[0].text
    kaia_persona_content = kaia_persona_content.replace('\x00', '').replace('\u0000', '')
    print(f"{COLOR_GREEN}Kaia persona loaded.{COLOR_RESET}")
except Exception as e:
    kaia_persona_content = ""
    print(f"{COLOR_YELLOW}Could not load persona: {e}{COLOR_RESET}")

# --- Initialize ChromaDB ---
try:
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    if not os.access(CHROMA_DB_PATH, os.W_OK):
        logging.error(f"Insufficient permissions for ChromaDB path: {CHROMA_DB_PATH}")
        sys.exit(1)

    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection("kaia_documents")
    print(f"{COLOR_GREEN}ChromaDB initialized at '{CHROMA_DB_PATH}'.{COLOR_RESET}")
except Exception as e:
    logging.error(f"{COLOR_RED}Failed to initialize ChromaDB: {e}{COLOR_RESET}")
    sys.exit(1)

# --- Configure LlamaIndex ---
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

if not os.path.exists(LLAMA_INDEX_METADATA_PATH):
    print(f"{COLOR_BLUE}Building index...{COLOR_RESET}")
    general_docs = [doc for doc in SimpleDirectoryReader(GENERAL_KNOWLEDGE_DIR).load_data()
                   if "Kaia_Desktop_Persona.md" not in doc.id_]
    personal_docs = SimpleDirectoryReader(PERSONAL_CONTEXT_DIR).load_data()
    all_docs = general_docs + personal_docs

    if not all_docs:
        print(f"{COLOR_YELLOW}No documents found to build index.{COLOR_RESET}")
        sys.exit(1)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(all_docs, storage_context=storage_context)
    index.storage_context.persist(persist_dir=LLAMA_INDEX_METADATA_PATH)
    print(f"{COLOR_GREEN}Index built and saved.{COLOR_RESET}")
else:
    print(f"{COLOR_BLUE}Loading index...{COLOR_RESET}")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=LLAMA_INDEX_METADATA_PATH
    )
    index = load_index_from_storage(storage_context=storage_context)
    print(f"{COLOR_GREEN}Index loaded.{COLOR_RESET}")

# --- Chat Engine ---
def get_chat_engine():
    if not hasattr(get_chat_engine, '_cached_engine'):
        combined_system_prompt = KAIA_SYSTEM_PROMPT + "\n\n" + kaia_persona_content
        get_chat_engine._cached_engine = index.as_chat_engine(
            chat_mode="condense_plus_context",
            memory=ChatMemoryBuffer.from_defaults(token_limit=1000),
            system_prompt=combined_system_prompt,
            similarity_top_k=2
        )
    return get_chat_engine._cached_engine

chat_engine = get_chat_engine()

# --- SQL Database ---
DB_CONNECTION_STRING = "postgresql://kaiauser@localhost/kaiadb"

try:
    if not database_utils.initialize_db(DB_CONNECTION_STRING):
        logging.error("Failed to initialize database.")
        sys.exit(1)

    sql_database = SQLDatabase(database_utils.engine)
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        llm=Settings.llm,
        tables=["facts", "interaction_history", "user_preferences", "tools", "kaia_persona_details"]
    )
    print(f"{COLOR_GREEN}SQL Database initialized.{COLOR_RESET}")
    database_utils.insert_default_persona_details()
except Exception as e:
    logging.error(f"{COLOR_RED}Failed to initialize SQL Database: {e}{COLOR_RESET}")
    sys.exit(1)

# --- Helper Functions ---
def speak_text_async(text):
    if not TTS_ENABLED:
        return
    try:
        subprocess.run(["spd-say", "--wait", text], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    except Exception as e:
        logging.error(f"TTS failed: {str(e)}")

def get_dynamic_system_status_with_gpu():
    def _get_output(cmd, default="N/A"):
        try:
            return subprocess.check_output(cmd, shell=True, text=True, timeout=3).strip()
        except Exception:
            return default

    lines = [
        "System Info:",
        f"OS: {_get_output('uname -o')} {_get_output('uname -m')}",
        f"Kernel: {_get_output('uname -r')}",
        f"CPU: {_get_output('grep model.name /proc/cpuinfo | head -n1 | cut -d: -f2-').strip()}",
        f"Terminal: {os.environ.get('TERM', 'N/A')}"
    ]

    mem = _get_output("free -h | grep Mem:")
    if mem and len(mem.split()) >= 3:
        lines.append(f"Memory: {mem.split()[2]} used / {mem.split()[1]} total")

    gpu_info = _get_output("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
    if gpu_info and "N/A" not in gpu_info:
        lines.append("GPU: " + gpu_info)

    return "\n".join(lines)

def confirm_and_execute_command(command):
    if any(blocked in command for blocked in [';', '&&', '||', '`', '$(', '>', '<', '|']):
        print(f"{COLOR_RED}Dangerous command detected - aborting.{COLOR_RESET}")
        return

    print(f"\n{COLOR_YELLOW}Execute command? (y/n)\n{command}{COLOR_RESET}")
    if input("Your choice: ").lower().strip() == 'y':
        try:
            result = subprocess.run(command, shell=True, text=True, capture_output=True)
            print(f"{COLOR_GREEN}Output:\n{result.stdout}{COLOR_RESET}")
            if result.stderr:
                print(f"{COLOR_RED}Error:\n{result.stderr}{COLOR_RESET}")
            database_utils.log_interaction(command, f"Executed: {command}", "command_executed")
        except Exception as e:
            print(f"{COLOR_RED}Command failed: {e}{COLOR_RESET}")

def generate_action_plan(user_input):
    system_prompt = """You are an AI assistant that classifies user intents. Respond ONLY with valid JSON:
    {
        "action": "command"|"chat"|"sql"|"persona_detail"|"retrieve_data",
        "content": "appropriate content for the action"
    }

    Special cases:
    - For file/directory operations (list, find, search), use "command" action
    - For system information requests, use "command" action"""

    payload = {
        "model": DEFAULT_COMMAND_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(user_input)}
        ],
        "stream": False,
        "format": "json"
    }

    try:
        response = session.post("http://localhost:11434/api/chat", json=payload, timeout=360)
        result = response.json()["message"]["content"].strip()
        return json.loads(result)
    except Exception as e:
        logging.error(f"Action plan error: {e}")
        if any(keyword in str(user_input).lower() for keyword in ['list', 'find', 'search', 'show']):
            return {"action": "command", "content": user_input}
        return {"action": "chat", "content": user_input}

# --- Main Loop ---
print("""\033[36m
██╗  ██╗ █████╗ ██╗ █████╗
██║ ██╔╝██╔══██╗██║██╔══██╗
█████╔╝ ███████║██║███████║
██╔═██╗ ██╔══██║██║██╔══██║
██║  ██╗██║  ██║██║██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝\033[0m
Kaia (Personal AI Assistant)""")

while True:
    try:
        query = input("\nQuery (type 'exit' to quit): ").strip()
        if query == 'exit':
            print(f"{COLOR_BLUE}Kaia: Session ended. Until next time!{COLOR_RESET}")
            break
        if not query:
            continue

        # Handle slash commands
        if query.startswith('/'):
            cmd = query[1:].strip().lower()
            if cmd == 'help':
                msg = """Available commands:
/help - Show this help message
/status - Show system status
/exit - End the session"""
            elif cmd == 'status':
                msg = f"{get_dynamic_system_status_with_gpu()}\n\nDatabase: {'Connected' if database_utils.engine else 'Disconnected'}"
                if database_utils.engine:
                    try:
                        with database_utils.engine.connect() as conn:
                            tables = conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'").fetchall()
                            msg += f"\nTables: {', '.join(t[0] for t in tables)}"
                    except Exception as e:
                        msg += f"\nDatabase Info: Error checking ({str(e)})"
            else:
                msg = f"Unknown command: /{cmd}"
            print(f"{COLOR_BLUE}{msg}{COLOR_RESET}")
            if TTS_ENABLED:
                speak_text_async(msg.replace("\n", " "))
            continue

        # Handle memory/preference storage
        if database_utils.handle_memory_storage(query):
            continue

        # Generate and execute action plan
        plan = generate_action_plan(query)
        action = plan.get("action", "chat")
        content = plan.get("content", query)

        if action == "command":
            try:
                if 'cli' not in globals():
                    cli = KaiaCLI()

                print(f"\n{COLOR_BLUE}Kaia (Command Mode):{COLOR_RESET}")

                # Handle directory listing requests
                if 'list' in query.lower() or 'files' in query.lower() or 'dir' in query.lower():
                    if 'home' in query.lower() or '~' in query.lower():
                        command = "ls -lh ~"
                    elif 'current' in query.lower() or 'this' in query.lower():
                        command = "ls -lh ."
                    else:
                        command = "ls -lh ."
                    error = None
                else:
                    if isinstance(content, list):
                        content = ' '.join(content)
                    command, error = cli.generate_command(str(content))

                if error:
                    print(f"{COLOR_RED}Command generation failed: {error}{COLOR_RESET}")
                    database_utils.log_interaction(query, f"Command generation failed: {error}", "command_error")
                    continue

                print(f"\n{COLOR_YELLOW}┌── Proposed Command ──┐{COLOR_RESET}")
                print(f"{COLOR_BLUE}{command}{COLOR_RESET}")
                print(f"{COLOR_YELLOW}└──────────────────────┘{COLOR_RESET}")

                confirm = input(f"{COLOR_YELLOW}Execute? (y/N): {COLOR_RESET}").lower().strip()
                if confirm != 'y':
                    print(f"{COLOR_BLUE}Command execution cancelled.{COLOR_RESET}")
                    database_utils.log_interaction(query, f"Command cancelled: {command}", "command_cancelled")
                    continue

                print(f"\n{COLOR_GREEN}Executing...{COLOR_RESET}")
                try:
                    process = subprocess.Popen(
                        command,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=os.environ
                    )

                    while True:
                        output = process.stdout.readline()
                        error = process.stderr.readline()

                        if output == '' and error == '' and process.poll() is not None:
                            break

                        if output:
                            print(f"{COLOR_GREEN}{output.strip()}{COLOR_RESET}")
                        if error:
                            print(f"{COLOR_YELLOW}{error.strip()}{COLOR_RESET}")

                    exit_code = process.wait()
                    if exit_code == 0:
                        print(f"{COLOR_GREEN}✔ Command completed successfully{COLOR_RESET}")
                    else:
                        print(f"{COLOR_RED}✖ Command failed with exit code {exit_code}{COLOR_RESET}")

                    database_utils.log_interaction(query, f"Executed: {command}", "command_executed")

                except Exception as e:
                    print(f"{COLOR_RED}Command execution failed: {e}{COLOR_RESET}")
                    database_utils.log_interaction(query, f"Command execution failed: {e}", "command_error")

            except Exception as e:
                print(f"{COLOR_RED}Command processing error: {e}{COLOR_RESET}")
                database_utils.log_interaction(query, f"Command processing error: {e}", "command_error")
                continue

        elif action == "chat":
            print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}", end=" ", flush=True)
            start_time = time.time()
            full_response = ""
            line_length = 0

            if not any(char in query for char in ['?', '!', '.']):
                print("...", end="", flush=True)

            response_stream = chat_engine.stream_chat(content)
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

            end_time = time.time()
            print(f"\n\n{COLOR_YELLOW}⏱ Total time: {end_time - start_time:.2f}s", end=" ")
            if first_token_time:
                print(f"(First token: {first_token_time - start_time:.2f}s)", end="")
            print(f"{COLOR_RESET}")

            if TTS_ENABLED:
                speak_text_async(full_response)
            database_utils.log_interaction(content, full_response, "chat")

        elif action == "sql":
            try:
                print(f"\n{COLOR_BLUE}Kaia (Querying Database):{COLOR_RESET}")
                sql_response = sql_query_engine.query(content)
                print(f"{COLOR_GREEN}┌── Query Results ──┐{COLOR_RESET}")
                print(f"{sql_response}")
                print(f"{COLOR_GREEN}└───────────────────┘{COLOR_RESET}")
                if TTS_ENABLED:
                    speak_text_async(str(sql_response))
                database_utils.log_interaction(content, str(sql_response), "info_retrieved")
            except Exception as e:
                print(f"{COLOR_RED}Database Error: {e}{COLOR_RESET}")

        elif action == "persona_detail":
            detail_value = database_utils.get_persona_detail(content)
            if detail_value:
                print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET} {detail_value}")
                if TTS_ENABLED:
                    speak_text_async(detail_value)
                database_utils.log_interaction(content, detail_value, "persona_detail")
            else:
                msg = f"I don't have information about '{content}' in my persona details"
                print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET} {msg}")
                if TTS_ENABLED:
                    speak_text_async(msg)
                database_utils.log_interaction(content, msg, "persona_detail")

        elif action == "retrieve_data":
            result = database_utils.handle_data_retrieval(content)
            print(f"\n{COLOR_BLUE}Kaia:{COLOR_RESET}")
            if isinstance(result['data'], list) and result['data']:
                print(f"{COLOR_GREEN}┌── {result['message']} ──┐{COLOR_RESET}")
                for item in result['data']:
                    print(f"• {str(item)}")
                print(f"{COLOR_GREEN}└──────────────────────┘{COLOR_RESET}")
            else:
                print(result['message'])

            if TTS_ENABLED:
                speak_text_async(result['message'])
            database_utils.log_interaction(content, result['message'], result['response_type'])

    except KeyboardInterrupt:
        print(f"\n{COLOR_BLUE}Kaia: Received interrupt signal. Exiting gracefully...{COLOR_RESET}")
        break
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"{COLOR_RED}Kaia: An unexpected error occurred - {e}{COLOR_RESET}")
