import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Set global LlamaIndex settings to use Ollama
Settings.llm = Ollama(model="mistral:latest", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# --- Configuration ---
# DATA_DIRECTORY will contain data.txt, Kaia-Character-Profile.md, and Kaia_Desktop_Persona.md
# We will primarily use kaia_desktop_persona.md for the system prompt.
DATA_DIRECTORY = "./data" # Directory containing your documents

# Create the data directory if it doesn't exist
os.makedirs(DATA_DIRECTORY, exist_ok=True)

print(f"Loading documents from {DATA_DIRECTORY}...")
documents = SimpleDirectoryReader(DATA_DIRECTORY).load_data()
print(f"Loaded {len(documents)} document(s).")

# Separate Kaia's persona document from the general knowledge documents
kaia_persona_doc_content = ""
general_knowledge_docs = []

for doc in documents:
    if "kaia_desktop_persona.md" in doc.metadata.get('file_name', '').lower():
        kaia_persona_doc_content = doc.text
        print("Found Kaia's desktop persona profile.")
    else:
        general_knowledge_docs.append(doc)

if not kaia_persona_doc_content:
    print("Warning: 'kaia_desktop_persona.md' not found in data directory. Persona may not be applied.")

print("Creating index from general knowledge documents (this might take a moment)...")
# Index only the general knowledge documents
index = VectorStoreIndex.from_documents(general_knowledge_docs)
print("Index created.")

# Create a query engine, now with Kaia's persona as the system prompt
# We use the content of the kaia_desktop_persona.md as the system prompt
query_engine = index.as_query_engine(system_prompt=kaia_persona_doc_content) # <-- Use content of persona doc here!

# Query the index with your LLM
print("\nQuerying the index...")

# ... (Keep all your existing queries here as they are) ...
query1 = "What is the highest mountain?"
response1 = query_engine.query(query1)
print(f"\nQuery: {query1}")
print(f"Kaia's Response: {response1}")

query2 = "What year is it according to the document?"
response2 = query_engine.query(query2)
print(f"\nQuery: {query2}")
print(f"Kaia's Response: {response2}")

query3 = "What is your view on short-term market speculation?"
response3 = query_engine.query(query3)
print(f"\nQuery: {query3}")
print(f"Kaia's Response: {response3}")

query4 = "Tell me about your thoughts on emerging AI technologies."
response4 = query_engine.query(query4)
print(f"\nQuery: {query4}")
print(f"Kaia's Response: {response4}")

query5 = "I'm thinking of putting all my savings into a new highly volatile crypto coin."
response5 = query_engine.query(query5)
print(f"\nQuery: {query5}")
print(f"Kaia's Response: {response5}")
