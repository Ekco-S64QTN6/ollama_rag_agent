# My Project File System Map - Kaia AI Assistant (`ollama_dev`)

This document provides a detailed overview of the directory structure and key files *within the `ollama_dev` project*, which serves as the primary development environment for the Kaia AI assistant. This structure is designed to categorize various types of knowledge and project components for efficient access and retrieval by Kaia.

The `ollama_dev` project resides under the user's personal projects directory, specifically at `~/projects/personal/ollama_dev/`.

## Top-Level Directories and Files within `ollama_dev/`

* `data/`: This core directory houses Kaia's general knowledge base, primarily organized into specialized subdirectories.
    * **Direct File:** `Kaia_Desktop_Persona.md`: Defines the core persona and operational guidelines for the Kaia AI.
    * **Subdirectories:** `ai_development/`, `books/`, `command_cheatsheets/`, `development_guides/`, `hardware_compatibility/`, `linux_troubleshooting_guides/`, `security/`, `software_definitions/`, `system_knowledge/`. (Detailed contents follow in subsequent sections).
* `images/`: Stores images related to the project, such as screenshots or diagrams.
* `kaia_cli.py`: A Python script likely containing command-line interface functionalities for interacting with Kaia.
* `Kaia_Project_Master_Plan.md`: The central document outlining the project's vision, architecture, features, progress, and roadmap.
* `llamaindex_ollama_rag.py`: The main Python script responsible for initializing the LLM, loading Kaia's persona, managing the ChromaDB vector store, and handling the RAG (Retrieval-Augmented Generation) process. It builds and loads vector indexes for Kaia's knowledge base.
* `Personal Context/`: This directory contains personal Markdown documents providing user-specific context to Kaia, including this `my_project_filesystem_map.md` itself. (Detailed contents follow in the "Personal Context Subdirectories" section).
* `README.md`: The project's main README file, providing installation instructions, configuration details, and usage examples.
* `requirements.txt`: Lists the Python dependencies required for the project.
* `storage/`: This directory persists Kaia's operational data, including:
    * `chroma_db/`: The physical location for the ChromaDB vector store.
    * `llama_index_metadata/`: Stores LlamaIndex's internal metadata for the indexed documents, facilitating quicker loading on subsequent runs.

---

## Detailed Knowledge Base Subdirectories within `data/`

### `data/ai_development/`
This directory contains documentation related to AI platforms, frameworks, and setup guides.
* **Typical contents:** `ollama_llamaindex_langchain_setup.md`, `AI-Development.md`.

### `data/books/`
This directory is dedicated to storing comprehensive long-form documents such as technical books and extensive guides.
* **Typical contents:** PDF documents and Markdown files converted from EPUBs (e.g., AI textbooks, Linux guides, programming books).

### `data/command_cheatsheets/`
Contains quick reference guides and practical examples for common command-line utilities.
* **Typical contents:** Markdown files for `pacman`, `rsync`, `systemd`, `linux_dev_commands.md`.

### `data/development_guides/`
Provides guides and workflows for general software development practices.
* **Typical contents:** `git_workflow_guide.md`.

### `data/hardware_compatibility/`
Stores information regarding hardware compatibility, particularly for Linux systems.
* **Typical contents:** Markdown files for `nvidia_gpus.md`, `wifi_cards.md`.

### `data/linux_troubleshooting_guides/`
Provides structured guides for diagnosing and resolving common Linux system issues.
* **Typical contents:** Markdown files for `disk_errors.md`, `network_issues.md`, `service_failures.md`.

### `data/security/`
Contains information security best practices, tool cheat sheets, and related security documentation.
* **Typical contents:** `infosec_tools_cheat_sheet.md`, `security_best_practices.md`.

### `data/software_definitions/`
Contains definitions, integration details, and specific information about software components.
* **Typical contents:** `chromadb_info.md` (details on ChromaDB integration and vector storage).

### `data/system_knowledge/`
Stores general and in-depth knowledge about Linux systems.
* **Typical contents:** `linux_knowledge_base.md`.

---

## Detailed Personal Context Subdirectories within `Personal Context/`

This directory stores personal and user-specific context that helps Kaia provide tailored assistance.

* `archlinux_system_config.md`: Personal Arch Linux system configuration details.
* `creative_prompts_campaign_notes.md`: Notes and prompts for personal creative projects (e.g., AI image prompts, D&D campaign notes).
* `my_daily_workflow.md`: Documentation of daily routines and workflows.
* `my_project_directories.md`: A general overview of the user's Linux system directory conventions and data organization.
* `my_project_filesystem_map.md`: (This document itself) A detailed map of the `ollama_dev` project's internal structure.
* `my_software_preferences.md`: User's software preferences and configurations.
* `my_system_config.md`: General personal system configuration.
* `system_activity_logs.md`: Logs of system activities specific to the user's environment.
