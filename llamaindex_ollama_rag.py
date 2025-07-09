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
Settings.llm = Ollama(model=LLM_MODEL, request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)

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

# --- System Status Information (for /status command) ---
SYSTEM_STATUS_INFO = """
Your current system configuration is:
Board: ROG STRIX B650-A GAMING WIFI (Rev 1.xx)
BIOS (UEFI): 3257 (32.57)
OS: Arch Linux x86_64
Kernel: Linux 6.15.4-arch2-1
CPU: AMD Ryzen 5 9600X (12) @ 5.49 GHz
Memory: 3.22 GiB / 31.00 GiB (10%)
Vulkan: 1.4.303 - NVIDIA [575.64.03]
OpenCL: 3.0 CUDA 12.9.90
Terminal: konsole 25.4.3
"""

# --- Interactive Query Loop (MODIFIED for Command Parsing) ---
while True:
    try:
        query = input("Query (type 'exit' to quit): ").strip() # Added .strip() for cleaner input

        if query.lower() == 'exit':
            break

        # --- Command Parsing Logic (NEW) ---
        if query.startswith('/'):
            command = query[1:].lower().split(' ')[0] # Get command (e.g., "help" from "/help extra")

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
                output_message = SYSTEM_STATUS_INFO
            else:
                output_message = f"Unknown command: /{command}. Type /help for available commands."

            print(f"{output_message}\n")
            if tts_enabled:
                clean_speech_text = output_message.replace("\\", "").replace("\n", " ").replace("\t", " ")
                speak_text(clean_speech_text)

            continue # Skip to the next loop iteration after handling a command
        # --- End Command Parsing Logic ---

        # If it's not a command, process with the chat engine
        response = chat_engine.chat(query)
        clean_response_text = response.response

        print(f"{clean_response_text}\n")

        clean_speech_text = clean_response_text.replace("\\", "").replace("\n", " ").replace("\t", " ")

        if tts_enabled:
            speak_text(clean_speech_text)

    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        # Optionally, you can add more specific error handling or just break
        # break
git status
