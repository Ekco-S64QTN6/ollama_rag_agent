import logging
import sys
import os

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
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
# Directory for personal context files (add this to your .gitignore!)
PERSONAL_CONTEXT_DIR = "./personal_context"
# Directory where your persona definition is (now correctly pointing to ./data)
PERSONA_DIR = "./data"

# --- Initialize Ollama LLM and Embedding Model ---
Settings.llm = Ollama(model=LLM_MODEL, request_timeout=360.0)
Settings.embed_model = OllamaEmbedding(model_name=EMBEDDING_MODEL)

print(f"Loading documents from {GENERAL_KNOWLEDGE_DIR} and {PERSONAL_CONTEXT_DIR}...")

# --- Load Documents ---
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

# Load Persona document directly by specifying its file path
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
    print("CRITICAL WARNING: Kaia's persona content is missing. Responses may be generic.")


# --- Create Index ---
print("Creating index from all loaded documents (this might take a moment)...")
index = VectorStoreIndex.from_documents(all_documents)
print("Index created.")

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
    "Kaia's Response: "
)
qa_template = PromptTemplate(template)
query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_template})


print("\nQuerying the index...")

# --- Interactive Query Loop ---
while True:
    try:
        query = input("Query (type 'exit' to quit): ")
        if query.lower() == 'exit':
            break

        response = query_engine.query(query)
        print(f"Kaia's Response: {response}\n")
    except Exception as e:
        print(f"An error occurred during query processing: {e}")
        # Optionally, you can add more specific error handling or just break
        break
