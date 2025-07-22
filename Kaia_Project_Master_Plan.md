Kaia AI Assistant: Project Master Plan

Version: 1.1
Last Updated: July 21, 2025

This document serves as the single source of truth for the Kaia AI Assistant project, consolidating its vision, architecture, features, progress, and future roadmap.
Table of Contents

    Project Vision & Core Goal

    Core Architecture & Technology Stack

    Key Features & Implementation

    Development Progress Log

    Future Roadmap & Planned Features

    Setup & Configuration Summary

1. Project Vision & Core Goal

The core goal is to build Kaia, a local, intelligent AI assistant designed to enhance the Linux desktop experience. Kaia is envisioned to be a powerful conversational partner capable of leveraging a personal knowledge base (RAG) and executing Linux commands to perform tasks, answer questions, and streamline workflows. The project prioritizes local execution, user control, and a distinct persona.
2. Core Architecture & Technology Stack

Kaia's foundation is built on a stack of local, open-source tools that ensure privacy and control.
2.1. LLM and Embedding Engine: Ollama

    Primary LLM (Chat & RAG): llama2:13b-chat-q4_0

    Command Generation LLM: mixtral:8x7b-instruct-v0.1-q4_K_M (A more powerful model chosen for accurate command generation).

    Embedding Model: nomic-embed-text for document chunking and vectorization.

    Configuration: Models are initialized with a request_timeout=360.0 seconds and stream=True for real-time conversational output.

2.2. RAG and Indexing: LlamaIndex with ChromaDB

    Function: LlamaIndex is used to build the Retrieval-Augmented Generation (RAG) system, allowing Kaia to access and reason about a personal knowledge base.

    Vector Store: ChromaDB is integrated as the persistent vector store for efficient storage and retrieval of document embeddings.

    Process:

        Document Loading: SimpleDirectoryReader loads documents from specified knowledge directories.

        Vector Indexing: VectorStoreIndex is built using the Ollama embedding model and stored in ChromaDB.

        Persistent Storage: The index is persisted to and loaded from a ./storage directory using StorageContext and ChromaVectorStore to avoid re-indexing on every launch, significantly improving startup time.

2.3. Persistent Memory: PostgreSQL

    Function: A PostgreSQL database is used for structured, long-term memory, enabling Kaia to remember user-specific facts, preferences, and interaction history.

    Key Tables: users, user_preferences, facts, and interaction_history.

    Operations: Utilizes SQLAlchemy for robust database interactions, including efficient upsert operations (ON CONFLICT) for user preferences.

    Initialization: The database connection includes retry logic for enhanced resilience during startup.

2.4. Python Environment

    The project is built in Python and relies on key libraries:

        llama-index-core: For the core RAG pipeline.

        llama-index-llms-ollama, llama-index-embeddings-ollama, llama-index-vector-stores-chroma: For specific LlamaIndex integrations.

        ollama: For interfacing with the local LLMs.

        SQLAlchemy, psycopg2: For PostgreSQL database interactions.

        psutil, platform: For system monitoring.

        subprocess: For executing shell commands.

        requests: For API interactions.

3. Key Features & Implementation
3.1. Dual-Mode Operation: Conversational Chat & Command Execution

Kaia intelligently routes user input into one of two primary modes:

    Conversational Chat: For general queries, Kaia uses its RAG-powered chat engine to provide context-aware responses based on the knowledge base and its persona. Distinct RAG and pure chat engines are employed for optimized conversational flow.

    Command Execution: For task-based requests, Kaia uses a dedicated model and a strict confirmation workflow to generate and execute Linux shell commands.

3.2. RAG System for Knowledge & Persona

    Persona Loading: Kaia's distinct persona is loaded from a dedicated markdown file (Kaia_Desktop_Persona.md) to ensure a consistent and defined conversational style.

    Knowledge Base: Documents from ./data and ./personal_context are loaded into a vector index (ChromaDB), providing Kaia with a long-term memory and a base of information to draw from.

    Chat Engine: The chat engine is configured with chat_mode="condense_plus_context" and a ChatMemoryBuffer (8192 token limit) to maintain conversational history.

3.3. Command Generation & Execution Framework

This is a critical safety and functionality component.

    Dedicated Command Model: A more powerful model (mixtral) is used specifically for translating natural language into accurate shell commands.

    Strict Prompting: A carefully crafted system_prompt instructs the model to generate only the raw command, with no extra text, markdown, or explanations.

    Safety Confirmation: No command is executed without explicit user approval. The generated command is presented to the user with a y/N prompt to prevent unintended actions.

    Execution & Output: The subprocess module is used to run the confirmed command, capturing and displaying stdout and stderr for user feedback.

3.4. Enhanced System Monitoring

    Kaia now includes comprehensive system monitoring capabilities, providing real-time information on:

        CPU usage and core count.

        Memory usage (total, available, used, percentage).

        Disk usage for the root partition.

        GPU utilization and memory (for NVIDIA GPUs via nvidia-smi).

        Status of Ollama server.

        PostgreSQL database connection status and available tables.

3.5. Robust Data Retrieval and Natural Language Processing (NLP)

    User-Specific Memory Retrieval: Kaia can retrieve user preferences, facts, and interaction history from the PostgreSQL database.

    Advanced Query Matching: Implemented normalize_query and match_query_category functions to process user input, allowing for more flexible and natural language queries (e.g., "what do you know about me", "list my preferences", "what facts do you know", "show interaction history").

    Standardized Responses: Retrieval functions consistently return structured dictionaries, ensuring predictable output.

4. Development Progress Log
V1.0: Foundational System (Initial State)

    Architecture: Successfully integrated Ollama and LlamaIndex to create the core RAG and command execution pipelines.

    Models: Established the dual-model approach with llama2 for chat and mixtral for commands.

    RAG: Implemented persistent vector indexing, persona loading, and conversational memory.

    Command Execution: Built the ask_ollama_for_command and confirm_and_execute_command functions, creating a safe and functional command execution loop.

    Intelligent Routing: The main loop could differentiate between conversational queries and command requests.

V1.1: Enhanced Memory, System, and Robustness (Current State)

    Persistent Memory: Fully integrated PostgreSQL for storing user preferences, facts, and interaction history.

        Resolved psycopg2.errors.InvalidColumnReference via UniqueConstraint on user_preferences_table.

        Implemented handle_memory_storage for intelligent parsing and storage of user input.

        Ensured consistent user_id handling across all database operations.

    System Monitoring: Added comprehensive system status reporting (CPU, memory, disk, GPU, Ollama, PostgreSQL) via kaia_cli.py.

    Improved Data Retrieval: Enhanced NLP for memory retrieval, allowing more natural phrasing (e.g., "what do you know about me?", "what preferences do you know?", "what facts do you know?", "list history?").

    Application Stability:

        Implemented robust error handling for database connections and LLM interactions.

        Refined logging to suppress verbose library output during startup, providing a cleaner user experience.

        Corrected variable typos and ensured proper ANSI color code interpretation in activate_kaia_env.sh.

    Licensing: Added LICENSE.md (MIT License) and NOTICE.md for third-party component acknowledgments.

5. Future Roadmap & Planned Features

This section outlines the next steps for the project.
Phase 1: Enhancing Core Functionality

    [ ] Implement Multi-step Command Sequences: Develop an action plan generator that allows Kaia to chain multiple commands together to complete more complex tasks.

    [ ] Refine Error Handling: Improve Kaia's ability to understand and recover from failed commands or unexpected output.

    [ ] Expand Tool Integration: Integrate more tools beyond shell commands, such as file I/O (read_file, write_file) and web search capabilities.

Phase 2: Advanced Memory Management & Learning

    [ ] Integrate LangChain: Explore using LangChain for more advanced agentic workflows and structured memory solutions beyond current SQL capabilities.

    [ ] Self-Correction: Develop a mechanism for Kaia to learn from user corrections and improve its command generation and data interpretation over time.

Phase 3: User Interface & Experience

    [ ] Web UI: Create a simple web interface using Gradio or Streamlit to provide a richer user experience beyond the command line.

    [ ] Advanced RAG: Explore more sophisticated RAG techniques, such as different chunking strategies, vector stores (beyond Chroma, e.g., FAISS), and query transformations for better context retrieval.

6. Setup & Configuration Summary

    Knowledge Base Directories:

        General Knowledge: ./data

        Personal Context: ./personal_context

        Persona File: Kaia_Desktop_Persona.md (located in ./data)

    Persistent Index Storage: ./storage (for LlamaIndex and ChromaDB)

    Database: PostgreSQL (kaiadb) for structured memory.

    Logging: Configured to suppress verbose output from httpx, httpcore, fsspec, urllib3, llama_index.core.storage.kvstore, chromadb, and llama_index during startup, showing only critical errors and custom informative messages.

    Licensing Files: LICENSE.md (MIT License for Kaia's code) and NOTICE.md (for third-party component licenses).
