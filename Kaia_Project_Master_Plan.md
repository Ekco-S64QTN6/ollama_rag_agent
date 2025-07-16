# Kaia AI Assistant: Project Master Plan

**Version:** 1.0
**Last Updated:** July 16, 2025

This document serves as the single source of truth for the Kaia AI Assistant project, consolidating its vision, architecture, features, progress, and future roadmap.

---

## Table of Contents
1.  [Project Vision & Core Goal](#1-project-vision--core-goal)
2.  [Core Architecture & Technology Stack](#2-core-architecture--technology-stack)
3.  [Key Features & Implementation](#3-key-features--implementation)
4.  [Development Progress Log](#4-development-progress-log)
5.  [Future Roadmap & Planned Features](#5-future-roadmap--planned-features)
6.  [Setup & Configuration Summary](#6-setup--configuration-summary)

---

## 1. Project Vision & Core Goal

The core goal is to build **Kaia**, a local, intelligent AI assistant designed to enhance the Linux desktop experience. Kaia is envisioned to be a powerful conversational partner capable of leveraging a personal knowledge base (RAG) and executing Linux commands to perform tasks, answer questions, and streamline workflows. The project prioritizes local execution, user control, and a distinct persona.

---

## 2. Core Architecture & Technology Stack

Kaia's foundation is built on a stack of local, open-source tools that ensure privacy and control.

### **2.1. LLM and Embedding Engine: Ollama**
*   **Primary LLM (Chat & RAG):** `llama2:13b-chat-q4_0`
*   **Command Generation LLM:** `mixtral:8x7b-instruct-v0.1-q4_K_M` (A more powerful model chosen for accurate command generation).
*   **Embedding Model:** `nomic-embed-text` for document chunking and vectorization.
*   **Configuration:** Models are initialized with a `request_timeout=30.0` seconds and `stream=True` for real-time conversational output.

### **2.2. RAG and Indexing: LlamaIndex**
*   **Function:** LlamaIndex is used to build the Retrieval-Augmented Generation (RAG) system, allowing Kaia to access and reason about a personal knowledge base.
*   **Process:**
    1.  **Document Loading:** `SimpleDirectoryReader` loads documents from specified knowledge directories.
    2.  **Vector Indexing:** `VectorStoreIndex` is built using the Ollama embedding model.
    3.  **Persistent Storage:** The index is persisted to and loaded from a `./storage` directory using `StorageContext` to avoid re-indexing on every launch, significantly improving startup time.

### **2.3. Python Environment**
*   The project is built in Python and relies on key libraries:
    *   `llama-index`: For the core RAG pipeline.
    *   `ollama`: For interfacing with the local LLMs.
    *   `subprocess`: For executing shell commands.
    *   `requests`: For API interactions.

---

## 3. Key Features & Implementation

### **3.1. Dual-Mode Operation: Conversational Chat & Command Execution**
Kaia intelligently routes user input into one of two primary modes:
*   **Conversational Chat:** For general queries, Kaia uses its RAG-powered chat engine to provide context-aware responses based on the knowledge base and its persona.
*   **Command Execution:** For task-based requests, Kaia uses a dedicated model and a strict confirmation workflow to generate and execute Linux shell commands.

### **3.2. RAG System for Knowledge & Persona**
*   **Persona Loading:** Kaia's distinct persona is loaded from a dedicated markdown file (`Kaia_Desktop_Persona.md`) to ensure a consistent and defined conversational style.
*   **Knowledge Base:** Documents from `./data` and `./personal_context` are loaded into a vector index, providing Kaia with a long-term memory and a base of information to draw from.
*   **Chat Engine:** The chat engine is configured with `chat_mode="condense_plus_context"` and a `ChatMemoryBuffer` (1024 token limit) to maintain conversational history.

### **3.3. Command Generation & Execution Framework**
This is a critical safety and functionality component.
*   **Dedicated Command Model:** A more powerful model (`mixtral`) is used specifically for translating natural language into accurate shell commands.
*   **Strict Prompting:** A carefully crafted `tool_prompt` instructs the model to generate *only* the raw command, with no extra text, markdown, or explanations.
*   **Safety Confirmation:** **No command is executed without explicit user approval.** The generated command is presented to the user with a `y/N` prompt to prevent unintended actions.
*   **Execution & Output:** The `subprocess` module is used to run the confirmed command, capturing and displaying `stdout` and `stderr` for user feedback.

---

## 4. Development Progress Log

### **V1.0: Foundational System (Current State)**
*   **Architecture:** Successfully integrated Ollama and LlamaIndex to create the core RAG and command execution pipelines.
*   **Models:** Established the dual-model approach with `llama2` for chat and `mixtral` for commands.
*   **RAG:** Implemented persistent vector indexing, persona loading, and conversational memory.
*   **Command Execution:** Built the `ask_ollama_for_command` and `confirm_and_execute_command` functions, creating a safe and functional command execution loop.
*   **Intelligent Routing:** The main loop can now differentiate between conversational queries and command requests without needing explicit flags like `/exec`.

---

## 5. Future Roadmap & Planned Features

This section outlines the next steps for the project.

### **Phase 1: Enhancing Core Functionality**
*   [ ] **Implement Multi-step Command Sequences:** Develop an action plan generator that allows Kaia to chain multiple commands together to complete more complex tasks.
*   [ ] **Refine Error Handling:** Improve Kaia's ability to understand and recover from failed commands or unexpected output.
*   [ ] **Expand Tool Integration:** Integrate more tools beyond shell commands, such as file I/O (`read_file`, `write_file`) and web search capabilities.

### **Phase 2: Long-Term Memory & Learning**
*   [ ] **Integrate LangChain:** Explore using LangChain for more advanced agentic workflows and structured memory solutions.
*   [ ] **SQL Database for Facts:** Implement a persistent SQL database (e.g., SQLite or PostgreSQL) to allow Kaia to remember specific facts, user preferences, and past interactions permanently.
*   [ ] **Self-Correction:** Develop a mechanism for Kaia to learn from user corrections and improve its command generation over time.

### **Phase 3: User Interface & Experience**
*   [ ] **Web UI:** Create a simple web interface using Gradio or Streamlit to provide a richer user experience beyond the command line.
*   [ ] **Advanced RAG:** Explore more sophisticated RAG techniques, such as different chunking strategies, vector stores (Chroma, FAISS), and query transformations for better context retrieval.

---

## 6. Setup & Configuration Summary

*   **Knowledge Base Directories:**
    *   General Knowledge: `./data`
    *   Personal Context: `./personal_context`
    *   Persona File: `Kaia_Desktop_Persona.md` (located in `./data`)
*   **Persistent Index Storage:** `./storage`
*   **Logging:** Configured to stream to standard output, with verbose logs from `httpx` and `LlamaIndex` suppressed for clarity.
