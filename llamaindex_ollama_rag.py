import logging
import sys
import os
import subprocess
import time
import json
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# --- Config ---
LLM_MODEL = "mistral:instruct"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_COMMAND_MODEL = "mistral:instruct"

GENERAL_KNOWLEDGE_DIR = "./data"
PERSONAL_CONTEXT_DIR = "./personal_context"
PERSONA_DIR = "./data"
PERSIST_DIR = "./storage"

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

# --- Load or Build Index ---
if not os.path.exists(PERSIST_DIR):
    print("Index not found. Building...")
    general_docs = [doc for doc in SimpleDirectoryReader(GENERAL_KNOWLEDGE_DIR).load_data() if "Kaia_Desktop_Persona.md" not in doc.id_]
    personal_docs = SimpleDirectoryReader(PERSONAL_CONTEXT_DIR).load_data()
    index = VectorStoreIndex.from_documents(general_docs + personal_docs)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=PERSIST_DIR))
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

    # GPU (optional: NVIDIA only)
    gpu_info = _get_output("nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
    if gpu_info and "N/A" not in gpu_info:
        lines.append("GPU: " + gpu_info)
    else:
        lines.append("GPU: Not available or unsupported")

    return "\n".join(lines)

def generate_action_plan(user_input):
    system_prompt = """
    You are an AI assistant named Kaia. Classify the user's intent.
    Respond ONLY with JSON like {"action": "command", "content": "..."} or {"action": "chat", "content": "..."}.
    Do not include markdown.
    """
    payload = {
        "model": DEFAULT_COMMAND_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "stream": False
    }
    try:
        response = session.post("http://localhost:11434/api/chat", json=payload, timeout=360)
        result = response.json()["message"]["content"].strip()
        return json.loads(result)
    except Exception as e:
        return {"action": "chat", "content": f"Sorry, something went wrong: {e}"}

def confirm_and_execute_command(command):
    if not command or command.lower().startswith("error"):
        print(f"Kaia: Invalid or unsafe command. {command}")
        return
    print(f"\nProposed command:\n{command}\n")
    if input("Execute? (y/N): ").lower() == 'y':
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Error:", result.stderr)
        except Exception as e:
            print(f"Execution failed: {e}")

# --- Main Loop ---
print("""\033[36m
██╗  ██╗ █████╗ ██╗ █████╗
██║ ██╔╝██╔══██╗██║██╔══██╗
█████╔╝ ███████║██║███████║
██╔═██╗ ██╔══██║██║██╔══██║
██║  ██╗██║  ██║██║██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝\033[0m
Kaia (Personal AI Assistant)""")

tts_enabled = input("Enable TTS? (y/n): ").lower().strip() == 'y'

while True:
    try:
        query = input("\nQuery (type 'exit' to quit): ").strip()
        if query == 'exit': break
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
        action = plan.get("action")
        content = plan.get("content", "")

        if action == "command":
            print(f"\033[36mKaia (Command): {content}\033[0m")
            confirm_and_execute_command(content)
        elif action == "chat":
            stream = chat_engine.stream_chat(query)
            full_response = ""
            print("\033[36mKaia: ", end="", flush=True)
            for token in stream.response_gen:
                print(token, end="", flush=True)
                full_response += token
            print("\033[0m\n")
            if tts_enabled: speak_text_async(full_response.replace("\n", " "))
        else:
            print("Kaia: I couldn't understand the request.")
    except Exception as e:
        print(f"Kaia: Unexpected error: {e}")
