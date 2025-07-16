import logging
import sys
import os
import subprocess
import time
import json
import requests
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.base.llms.types import ChatMessage

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

# --- Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
for noisy in ["httpx", "httpcore", "llama_index.core.chat_engine", "llama_index.core.llms", "llama_index.core.indices"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --- Initialize Models ---
session = requests.Session()
try:
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=360.0, stream=True)
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
    _ = Settings.llm.complete("Hello")
    print("LLM initialized and warmed up.")
except Exception as e:
    logging.error(f"Failed to initialize Ollama: {e}")
    sys.exit(1)

# --- Load Persona ---
persona_doc_path = os.path.join(PERSONA_DIR, "Kaia_Desktop_Persona.md")
try:
    kaia_persona_content = SimpleDirectoryReader(input_files=[persona_doc_path]).load_data()[0].text
    kaia_persona_content = kaia_persona_content.replace('\x00', '').replace('\u0000', '')
    print("Kaia persona loaded.")
except Exception as e:
    kaia_persona_content = ""
    print(f"Could not load persona: {e}")

# --- Initialize ChromaDB Client ---
try:
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    db = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    chroma_collection = db.get_or_create_collection("kaia_documents")
    print(f"ChromaDB client initialized at '{CHROMA_DB_PATH}' with collection 'kaia_documents'.")
except Exception as e:
    logging.error(f"Failed to initialize ChromaDB: {e}")
    sys.exit(1)

# --- Configure LlamaIndex to use ChromaDB ---
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

if not os.path.exists(LLAMA_INDEX_METADATA_PATH):
    print("LlamaIndex metadata not found. Building index...")
    general_docs = [doc for doc in SimpleDirectoryReader(GENERAL_KNOWLEDGE_DIR).load_data() if "Kaia_Desktop_Persona.md" not in doc.id_]
    personal_docs = SimpleDirectoryReader(PERSONAL_CONTEXT_DIR).load_data()
    all_docs = general_docs + personal_docs

    if not all_docs:
        print("No documents found to build the index. Please check your data directories.")
        sys.exit(1)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    index = VectorStoreIndex.from_documents(
        all_docs,
        storage_context=storage_context
    )
    index.storage_context.persist(persist_dir=LLAMA_INDEX_METADATA_PATH)
    print(f"Index built and LlamaIndex metadata saved to '{LLAMA_INDEX_METADATA_PATH}'.")
else:
    print(f"Loading index from '{LLAMA_INDEX_METADATA_PATH}' (using ChromaDB for vectors)...")
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=LLAMA_INDEX_METADATA_PATH
    )
    index = load_index_from_storage(storage_context=storage_context)
    print("Index loaded.")

chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=ChatMemoryBuffer.from_defaults(token_limit=1000),
    system_prompt=kaia_persona_content,
    similarity_top_k=2
)

# --- Helpers ---
def speak_text_async(text):
    try:
        subprocess.Popen(["spd-say", text])
    except Exception:
        pass

def get_dynamic_system_status_with_gpu():
    def _get_output(cmd, default="N/A"):
        try:
            return subprocess.check_output(cmd, shell=True, text=True, timeout=3).strip()
        except Exception:
            return default

    lines = [
        "System Info:",
        f"OS: {_get_output('uname -o')} {_get_output('uname -m')}",
        f"Kernel: {_get_output('uname -r')}"
    ]
    cpu_model = _get_output("grep 'model name' /proc/cpuinfo | head -n1 | cut -d: -f2-").strip()
    cpu_cores = _get_output("grep 'cpu cores' /proc/cpuinfo | head -n1 | cut -d: -f2-").strip()
    lines.append(f"CPU: {cpu_model} ({cpu_cores} cores)")
    mem = _get_output("free -h | grep Mem:")
    if mem:
        parts = mem.split()
        if len(parts) >= 3:
            lines.append(f"Memory: {parts[2]} used / {parts[1]} total")
    lines.append(f"Terminal: {os.environ.get('TERM', 'N/A')}")

    gpu_info = _get_output("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
    if gpu_info and "N/A" not in gpu_info:
        lines.append("GPU: " + gpu_info)
    else:
        lines.append("GPU: Not available or unsupported")

    return "\n".join(lines)

def generate_action_plan(user_input):
    system_prompt = """
    You are an AI assistant named Kaia. Your task is to classify the user's intent to determine the next action.
    Respond ONLY with a JSON object.
    The JSON object MUST have an "action" key, and its value MUST be either "command" or "chat".
    The JSON object MUST also have a "content" key, with the appropriate content for that action.
    Do NOT include any markdown formatting (e.g., ```json) or extra text outside the JSON object.

    If the user's query clearly indicates an intent to run a shell command (e.g., asking to 'list files', 'check disk space', 'show running processes', 'install a package'), set "action" to "command" and "content" to the actual Linux command.
    For all other queries, including general questions, requests for information, or conversational chat, set "action" to "chat" and "content" to the user's original query or a rephrased version suitable for the chat engine.

    Examples:
    User: list files in current directory
    Response: {"action": "command", "content": "ls -F"}

    User: check my disk usage
    Response: {"action": "command", "content": "df -h"}

    User: tell me about yourself
    Response: {"action": "chat", "content": "tell me about yourself"}

    User: What is the capital of France?
    Response: {"action": "chat", "content": "What is the capital of France?"}
    """
    payload = {
        "model": DEFAULT_COMMAND_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "stream": False,
        "format": "json" # <--- IMPORTANT: This tells Ollama to try to force JSON output
    }
    try:
        response = session.post("http://localhost:11434/api/chat", json=payload, timeout=360)
        result = response.json()["message"]["content"].strip()
        # The raw LLM response debug print has been moved to the main loop.
        return json.loads(result)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from action plan: {e}. Raw response was: '{result}'")
        return {"action": "chat", "content": f"Sorry, something went wrong parsing the action plan: {e}"}
    except Exception as e:
        print(f"Error generating action plan: {e}")
        return {"action": "chat", "content": f"Sorry, something went wrong during action planning: {e}"}

# --- Main Loop ---
print("""\033[36m
██╗  ██╗ █████╗ ██╗ █████╗
██║ ██╔╝██╔══██╗██║██╔══██╗
█████╔╝ ███████║██║███████║
██╔═██╗ ██╔══██║██║██╔══██║
██║  ██╗██║  ██║██║██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝\033[0m
Kaia (Personal AI Assistant)""")

tts_enabled = input("Enable TTS? (y/n): ").lower().strip() == 'y'

while True:
    try:
        query = input("\nQuery (type 'exit' to quit): ").strip()
        if query == 'exit':
            break


        if not query:
            continue

        if query.startswith('/'):
            cmd = query[1:].strip().lower()
            if cmd == 'help':
                msg = "/help, /status, /exit"
            elif cmd == 'status':
                msg = get_dynamic_system_status_with_gpu()
            else:
                msg = f"Unknown command: /{cmd}".replace('\x00', '').replace('\u0000', '')
            print(f"\033[36m{msg}\033[0m")
            if tts_enabled: speak_text_async(msg.replace("\n", " "))
            continue

        plan = generate_action_plan(query)
        # print(f"\nDEBUG: Action Plan (parsed): {plan}") # Commented out debug output
        action = plan.get("action")
        content = plan.get("content", "")

        if action == "command":
            print(f"\n\033[36mKaia (Command): {content}\033[0m")
            confirm_and_execute_command(content)
        elif action == "chat":
            print("\n\033[36mKaia: ", end="", flush=True)
            start = time.time()

            response_stream = chat_engine.stream_chat(query)
            full_response = ""
            first_token_time = None
            for token in response_stream.response_gen:
                if first_token_time is None:
                    first_token_time = time.time()
                    print(f"\n⏱ First token in {first_token_time - start:.2f}s\n", end="", flush=True)
                print(token, end="", flush=True)
                full_response += token
            print("\033[0m\n")

            if tts_enabled:
                clean_speech_text = full_response.replace("\\", "").replace("\n", " ").replace("\t", " ")
                speak_text_async(clean_speech_text)

        else:
            msg = "Kaia: I could not determine the appropriate action for your request. Please try rephrasing."
            print(f"\033[36m{msg}\033[0m")
            if tts_enabled:
                speak_text_async(msg.replace("\n", " "))

    except Exception as e:
        print(f"Kaia: Unexpected error: {e}")
