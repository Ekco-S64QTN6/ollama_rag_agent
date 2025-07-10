import logging
import sys
import os
import subprocess

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.prompts import PromptTemplate

# Configure logging to show info messages for your script, but suppress others
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# --- Suppress verbose logging from specific LlamaIndex and HTTP client loggers ---
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("llama_index.core.chat_engine").setLevel(logging.WARNING)
logging.getLogger("llama_index.core.llms").setLevel(logging.WARNING)
logging.getLogger("llama_index.core.indices").setLevel(logging.WARNING)
# --------------------------------------------------------

# --- Ollama Model Configuration ---
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text"

# --- Directory Paths ---
GENERAL_KNOWLEDGE_DIR = "./data"
PERSONAL_CONTEXT_DIR = "./personal_context"
PERSONA_DIR = "./data"
PERSIST_DIR = "./storage"

# --- Initialize Ollama LLM and Embedding Model ---
try:
    Settings.llm = Ollama(model=LLM_MODEL, request_timeout=360.0)
    Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)
    print(f"Ollama LLM ('{LLM_MODEL}') and Embedding Model ('{EMBEDDING_MODEL}') initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Ollama LLM or Embedding Model: {e}")
    logging.error("Please ensure the Ollama server is running and the specified models are downloaded.")
    logging.error("You can check Ollama status with 'ollama list' and 'ollama serve'.")
    sys.exit(1) # Graceful exit if LLM/Embedding models cannot be initialized

# --- Load Persona document directly ---
kaia_persona_content = ""
try:
    persona_doc_path = os.path.join(PERSONA_DIR, "Kaia_Desktop_Persona.md")
    persona_docs_list = SimpleDirectoryReader(input_files=[persona_doc_path]).load_data()
    kaia_persona_content = persona_docs_list[0].text if persona_docs_list else ""
    if kaia_persona_content:
        print("Found Kaia's desktop persona profile.")
    else:
        print("Warning: Kaia's persona document found but empty.")
except Exception as e:
    kaia_persona_content = ""
    print(f"Error loading Kaia's persona document from {persona_doc_path}: {e}. Ensure the file exists and is readable.")

if not kaia_persona_content:
    print("Warning: Kaia's persona content is missing. Responses may be generic.")

# --- Index Creation/Loading Logic ---
index = None
if not os.path.exists(PERSIST_DIR):
    print("Building index from documents (first time or storage missing)...")
    general_docs = SimpleDirectoryReader(GENERAL_KNOWLEDGE_DIR).load_data()
    general_docs = [doc for doc in general_docs if "Kaia_Desktop_Persona.md" not in doc.id_]
    print(f"Loaded {len(general_docs)} general document(s) from {GENERAL_KNOWLEDGE_DIR}.")

    personal_docs = SimpleDirectoryReader(PERSONAL_CONTEXT_DIR).load_data()
    print(f"Loaded {len(personal_docs)} personal document(s) from {PERSONAL_CONTEXT_DIR}.")

    all_documents = general_docs + personal_docs
    print(f"Total {len(all_documents)} document(s) loaded for indexing.")

    # --- NEW: Exit if no documents found for indexing ---
    if not all_documents:
        logging.error("No documents (general knowledge or personal context) found to build the index. Exiting.")
        sys.exit(1)

    index = VectorStoreIndex.from_documents(all_documents)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index created and persisted to {PERSIST_DIR}.")
else:
    print(f"Loading index from {PERSIST_DIR}...")
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context=storage_context)
    print("Index loaded.")

# --- Configure Chat Engine with Persona ---
chat_engine = index.as_chat_engine(
    chat_mode="condense_plus_context",
    memory=ChatMemoryBuffer.from_defaults(token_limit=3000),
    system_prompt=kaia_persona_content,
)

# Function to speak text using Speech Dispatcher
def speak_text(text):
    try:
        subprocess.run(["spd-say", text], check=True)
    except FileNotFoundError:
        print("Warning: spd-say command not found. Speech output will not work. Is Speech Dispatcher installed?")
    except subprocess.CalledProcessError as e:
        print(f"Error calling spd-say: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during speech output: {e}")

# --- Dynamic System Status Function ---
def get_dynamic_system_status():
    """Gathers basic system information dynamically on Linux."""
    info_lines = []
    info_lines.append("Your current system configuration is:")

    # Helper function to safely run shell commands
    def _get_output(command, default_value="N/A"):
        try:
            # Use shell=True for pipes, but be cautious with untrusted input (not an issue here)
            # Timeout helps prevent hanging if a command is slow or interactive
            return subprocess.check_output(command, shell=True, text=True, timeout=3).strip()
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return default_value

    # OS and Kernel
    os_name = _get_output('uname -o')
    machine_arch = _get_output('uname -m')
    info_lines.append(f"OS: {os_name} {machine_arch}")
    info_lines.append(f"Kernel: {_get_output('uname -r')}")

    # CPU
    cpu_model = _get_output("grep 'model name' /proc/cpuinfo | head -n 1 | cut -d: -f2- | xargs", "N/A")
    cpu_cores = _get_output("grep 'cpu cores' /proc/cpuinfo | head -n 1 | cut -d: -f2- | xargs", "N/A")
    info_lines.append(f"CPU: {cpu_model} ({cpu_cores} cores)")

    # Memory (using free -h for human-readable format)
    mem_info_raw = _get_output("free -h | grep Mem:", "")
    if mem_info_raw:
        parts = mem_info_raw.split()
        if len(parts) >= 3: # Expected format: Mem: Total Used Free ...
            total_mem = parts[1]
            used_mem = parts[2]
            info_lines.append(f"Memory: {used_mem} / {total_mem}")
        else:
            info_lines.append(f"Memory: {mem_info_raw} (parse error)")
    else:
        info_lines.append("Memory: N/A")

    # Terminal
    info_lines.append(f"Terminal: {os.environ.get('TERM', 'N/A')}")

    # Note on info requiring elevated privileges or specialized tools
    info_lines.append("Board: (Requires 'dmidecode' or similar, often sudo)")
    info_lines.append("BIOS (UEFI): (Requires 'dmidecode' or similar, often sudo)")
    info_lines.append("Vulkan: (Requires 'vulkaninfo' or GPU driver tools)")
    info_lines.append("OpenCL: (Requires 'clinfo' or GPU driver tools)")

    return "\n".join(info_lines)
# --- End Dynamic System Status Function ---
print(f"\033[36m" + """
██╗  ██╗ █████╗ ██╗ █████╗
██║ ██╔╝██╔══██╗██║██╔══██╗
█████╔╝ ███████║██║███████║
██╔═██╗ ██╔══██║██║██╔══██║
██║  ██╗██║  ██║██║██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
""" + "\033[0m")
print("\nKaia (Personal AI Assistant)\n")
# --- TTS Toggle ---
tts_enabled = False
while True:
    tts_choice = input("Enable Text-to-Speech (y/n)? ").lower().strip()
    if tts_choice == 'y':
        tts_enabled = True
        print("Text-to-Speech is ON.")
        break
    elif tts_choice == 'n':
        tts_enabled = False
        print("Text-to-Speech is OFF.")
        break
    else:
        print("Invalid input. Please enter 'y' or 'n'.")

print("\nQuerying the index...")


# --- Interactive Query Loop ---
while True:
    try:
        query = input("Query (type 'exit' to quit): ").strip()

        if query.lower() == 'exit':
            break

        # --- Command Parsing Logic ---
        if query.startswith('/'):
            command = query[1:].lower().split(' ')[0]

            output_message = ""
            if command == 'help':
                output_message = (
                    "Available commands:\n"
                    "  /help  - Display this help message.\n"
                    "  /status - Display current system configuration details.\n"
                    "  /exit  - Exit the application.\n"
                    "You can also just type your question to query Kaia's knowledge."
                )
            elif command == 'status':
                output_message = get_dynamic_system_status()
            else:
                output_message = f"Unknown command: /{command}. Type /help for available commands."

            # Corrected Indentation for these lines:
            print(f"\033[36m{output_message}\033[0m\n")
            if tts_enabled:
                clean_speech_text = output_message.replace("\\", "").replace("\n", " ").replace("\t", " ")
                speak_text(clean_speech_text)

            continue
        # --- End Command Parsing Logic ---

        response = chat_engine.chat(query)
        clean_response_text = response.response

        print(f"\033[36m{clean_response_text}\033[0m\n")

        clean_speech_text = clean_response_text.replace("\\", "").replace("\n", " ").replace("\t", " ")

        if tts_enabled:
            speak_text(clean_speech_text)

    except Exception as e:
        print(f"An error occurred during query processing: {e}")
