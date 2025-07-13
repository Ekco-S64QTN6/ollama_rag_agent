import logging
import sys
import os
import subprocess
import time

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# Suppress verbose logs
for noisy in ["httpx", "httpcore", "llama_index.core.chat_engine", "llama_index.core.llms", "llama_index.core.indices"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# --- Ollama Configuration ---
LLM_MODEL = "llama2:13b-chat-q4_0"  # Consider q5_K_M or q3_K_M for quality/speed tradeoff
EMBEDDING_MODEL = "nomic-embed-text"

# --- Paths ---
GENERAL_KNOWLEDGE_DIR = "./data"
PERSONAL_CONTEXT_DIR = "./personal_context"
PERSONA_DIR = "./data"
PERSIST_DIR = "./storage"

# --- Initialize Ollama Models ---
try:
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=30.0, stream=True)
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
    print(f"Ollama LLM ('{LLM_MODEL}') and Embedding Model ('{EMBEDDING_MODEL}') initialized successfully.")

    # --- Warm-up LLM to reduce cold-start latency ---
    _ = Settings.llm.complete("Hello")
    print("LLM warm-up complete.\n")
except Exception as e:
    logging.error(f"Failed to initialize Ollama: {e}")
    sys.exit(1)

# --- Load Persona ---
kaia_persona_content = ""
persona_doc_path = os.path.join(PERSONA_DIR, "Kaia_Desktop_Persona.md")
try:
    persona_docs = SimpleDirectoryReader(input_files=[persona_doc_path]).load_data()
    kaia_persona_content = persona_docs[0].text if persona_docs else ""
    print("Kaia persona profile loaded.")
except Exception as e:
    print(f"Failed to load persona: {e}")
    kaia_persona_content = ""

# --- Index Load/Build ---
index = None
if not os.path.exists(PERSIST_DIR):
    print("Index not found. Building from scratch...")
    general_docs = SimpleDirectoryReader(GENERAL_KNOWLEDGE_DIR).load_data()
    general_docs = [doc for doc in general_docs if "Kaia_Desktop_Persona.md" not in doc.id_]

    personal_docs = SimpleDirectoryReader(PERSONAL_CONTEXT_DIR).load_data()
    all_docs = general_docs + personal_docs

    if not all_docs:
        print("No documents found. Exiting.")
        sys.exit(1)

    index = VectorStoreIndex.from_documents(all_docs)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index created and saved to '{PERSIST_DIR}'.")
else:
    print(f"Loading index from '{PERSIST_DIR}'...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("Index loaded.\n")

# --- Chat Engine Setup ---
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=ChatMemoryBuffer.from_defaults(token_limit=1000),  # Lower memory context = faster prompt processing
    system_prompt=kaia_persona_content,
    similarity_top_k=2  # Reduce retrieval context
)

# --- TTS Function ---
def speak_text(text):
    try:
        subprocess.run(["spd-say", text], check=True)
    except FileNotFoundError:
        print("spd-say not found.")
    except subprocess.CalledProcessError as e:
        print(f"TTS error: {e}")
    except Exception as e:
        print(f"Unexpected TTS error: {e}")

# --- System Status ---
def get_dynamic_system_status():
    def _get_output(command, default="N/A"):
        try:
            return subprocess.check_output(command, shell=True, text=True, timeout=3).strip()
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
        else:
            lines.append("Memory: Unable to parse")
    else:
        lines.append("Memory: N/A")

    lines.append(f"Terminal: {os.environ.get('TERM', 'N/A')}")
    lines.append("Note: Board/BIOS/Vulkan/OpenCL info needs tools like `dmidecode` or `vulkaninfo`.")

    return "\n".join(lines)



print(f"\033[36m" + """
██╗  ██╗ █████╗ ██╗ █████╗
██║ ██╔╝██╔══██╗██║██╔══██╗
█████╔╝ ███████║██║███████║
██╔═██╗ ██╔══██║██║██╔══██║
██║  ██╗██║  ██║██║██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
""" + "\033[0m")
print("Kaia (Personal AI Assistant)\n")


tts_enabled = False
while True:
    tts_choice = input("Enable Text-to-Speech (y/n)? ").lower().strip()
    if tts_choice == 'y':
        tts_enabled = True
        print("TTS enabled.\n")
        break
    elif tts_choice == 'n':
        print("TTS disabled.\n")
        break
    else:
        print("Please type 'y' or 'n'.")

# --- Main Loop ---
while True:
    try:
        query = input("Query (type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break

        if query.startswith('/'):
            command = query[1:].lower().split()[0]
            if command == 'help':
                msg = "/help - Show commands\n/status - Show system info\n/exit - Quit app"
            elif command == 'status':
                msg = get_dynamic_system_status()
            else:
                msg = f"Unknown command: /{command}"

            print(f"\033[36m{msg}\033[0m\n")
            if tts_enabled:
                speak_text(msg.replace("\n", " "))
            continue

        # --- Time the response ---
        print(f"\033[36mKaia: ", end="", flush=True)
        start = time.time()
        response = chat_engine.stream_chat(query)

        first_token_time = None
        full_response = ""
        buffer = ""

        for token in response.response_gen:
            if first_token_time is None:
                first_token_time = time.time()
                print(f"\n⏱ First token in {first_token_time - start:.2f}s\n", end="", flush=True)

            print(token, end="", flush=True)
            full_response += token

        print("\033[0m\n")

        if tts_enabled:
            speak_text(full_response.replace("\n", " "))

    except Exception as e:
        print(f"Error during query: {e}")
