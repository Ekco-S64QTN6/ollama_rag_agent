import logging
import sys
import os
import subprocess

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.prompts import PromptTemplate

# Configure logging to show info messages for your script, but suppress others
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

# --- Suppress verbose logging from httpx and httpcore ---
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
# --------------------------------------------------------

# --- Ollama Model Configuration ---
# You can change these to other models available via Ollama (e.g., "llama3", "mixtral", "gemma")
LLM_MODEL = "mistral"
EMBEDDING_MODEL = "nomic-embed-text" # Used for generating embeddings for your documents and queries

# --- Directory Paths ---
# Directory for general Linux knowledge base
GENERAL_KNOWLEDGE_DIR = "./data"
# Directory for personal context files
PERSONAL_CONTEXT_DIR = "./personal_context"
# Directory where your persona definition is
PERSONA_DIR = "./data"
# Directory for persisting the index
PERSIST_DIR = "./storage"

# --- Initialize Ollama LLM and Embedding Model ---
Settings.llm = Ollama(model=LLM_MODEL, request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)

# --- Load Persona document directly by specifying its file path ---
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
    # Load general Linux knowledge base documents
    general_docs = SimpleDirectoryReader(GENERAL_KNOWLEDGE_DIR).load_data()
    # Filter out the persona document from the general docs if it's meant to be loaded separately
    general_docs = [doc for doc in general_docs if "Kaia_Desktop_Persona.md" not in doc.id_]
    print(f"Loaded {len(general_docs)} general document(s) from {GENERAL_KNOWLEDGE_DIR}.")

    # Load personal context documents
    personal_docs = SimpleDirectoryReader(PERSONAL_CONTEXT_DIR).load_data()
    print(f"Loaded {len(personal_docs)} personal document(s) from {PERSONAL_CONTEXT_DIR}.")

    # Combine all documents for the main index
    all_documents = general_docs + personal_docs
    print(f"Total {len(all_documents)} document(s) loaded for indexing.")

    # Create Index
    index = VectorStoreIndex.from_documents(all_documents)
    # Persist the index to disk
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print(f"Index created and persisted to {PERSIST_DIR}.")
else:
    print(f"Loading index from {PERSIST_DIR}...")
    # Load the existing index from disk
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context=storage_context)
    print("Index loaded.")

# --- Configure Query Engine with Persona ---
query_engine = index.as_query_engine(
    similarity_top_k=3 # You can adjust this value to retrieve more or fewer relevant chunks
)

# Custom prompt template to inject the persona
# Ensure the persona instructions are concise and directly impact response style.
# The `context_str` will contain the retrieved knowledge.
template = (
    "You are an AI assistant named Kaia. Your persona is defined by the following guidelines:\n"
    "Persona Guidelines:\n"
    "------\n"
    f"{kaia_persona_content}\n"
    "------\n"
    "Given the context information and the persona guidelines, answer the query.\n"
    "Context Information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Query: {query_str}\n"
)
qa_template = PromptTemplate(template)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})

# Function to speak text using Speech Dispatcher
def speak_text(text):
    try:
        # Use spd-say with -e to ensure it waits for the text to be spoken
        # and -r for rate if you want to control speed (optional)
        # You might also add -o for output module (e.g., "piper") if spd-say isn't defaulting correctly
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

# --- Interactive Query Loop ---
while True:
    try:
        query = input("Query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        response = query_engine.query(query)
        print(f"{response}\n")

        # Clean the response text for speech output
        clean_response_text = str(response).replace("\\", "").replace("\n", " ").replace("\t", " ")

        if tts_enabled:
            speak_text(clean_response_text)

    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        # Optionally, you can add more specific error handling or just break
        # break
