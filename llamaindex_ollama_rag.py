import json
import logging
import os
import platform
import subprocess
import sys
import time
import requests
import config
import database_utils
import utils
import chromadb
from kaia_cli import KaiaCLI
from contextlib import redirect_stdout, contextmanager
from typing import Optional, List, Dict, Union, Any
from datetime import datetime
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    SQLDatabase,
    SimpleDirectoryReader,
)
from llama_index.core.chat_engine.simple import SimpleChatEngine
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

# Local imports for new modules
from toolbox import video_converter # Updated import path


# Logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
for noisy in ["httpx", "httpcore", "fsspec", "urllib3", "llama_index.core.storage.kvstore", "chromadb", "llama_index"]:
    logging.getLogger(noisy).setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

# Context Manager to suppress stdout
@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Main Function
def main():
    sql_rag_enabled_local = config.SQL_RAG_ENABLED
    cli = KaiaCLI()

    # Initialize Models
    print(f"{config.COLOR_BLUE}Initializing LLM and Embedding Model...{config.COLOR_RESET}")
    start_time_init_models = time.time()
    try:
        llm_model_to_use, llm_error = utils.check_ollama_model_availability(config.LLM_MODEL, config.DEFAULT_COMMAND_MODEL)
        if llm_error:
            raise RuntimeError(f"LLM initialization failed: {llm_error}")

        embed_model_to_use, embed_error = utils.check_ollama_model_availability(config.EMBEDDING_MODEL)
        if embed_error:
            raise RuntimeError(f"Embedding model initialization failed: {embed_error}")

        with suppress_stdout():
            Settings.llm = Ollama(model=llm_model_to_use, request_timeout=config.TIMEOUT_SECONDS, stream=True)
            Settings.embed_model = OllamaEmbedding(model_name=embed_model_to_use)
            _ = Settings.llm.complete("Hello")
            test_embedding = Settings.embed_model.get_query_embedding("test")
            embedding_dim = len(test_embedding)

        logger.info("LLM and embedding models initialized successfully")
        print(f"{config.COLOR_GREEN}LLM and embedding models initialized successfully in {time.time() - start_time_init_models:.2f}s.{config.COLOR_RESET}")

        logger.info(f"Embedding model dimension: {embedding_dim}")
        print(f"{config.COLOR_BLUE}Embedding dimension: {embedding_dim}{config.COLOR_RESET}")

    except Exception as e:
        logger.error(f"Failed to initialize Ollama models: {e}", exc_info=True)
        print(f"{config.COLOR_RED}Failed to initialize Ollama models: {e}{config.COLOR_RESET}")
        sys.exit(1)

    # Load Persona
    print(f"{config.COLOR_BLUE}Loading Kaia persona...{config.COLOR_RESET}")
    persona_doc_path = os.path.join(config.PERSONA_DIR, "Kaia_Desktop_Persona.md")
    kaia_persona_content = config.KAIA_SYSTEM_PROMPT
    try:
        if os.path.exists(persona_doc_path):
            kaia_persona_content = SimpleDirectoryReader(input_files=[persona_doc_path]).load_data()[0].text
            kaia_persona_content = kaia_persona_content.replace('\x00', '').replace('\u0000', '')
            logger.info("Kaia persona loaded successfully")
            print(f"{config.COLOR_GREEN}Kaia persona loaded successfully.{config.COLOR_RESET}")
        else:
            logger.warning(f"Persona file not found: {persona_doc_path}. Using default system prompt.")
            print(f"{config.COLOR_YELLOW}Warning: Persona file not found. Responses may be generic.{config.COLOR_RESET}")
    except Exception as e:
        logger.warning(f"Could not load persona: {e}. Using default system prompt.", exc_info=True)
        print(f"{config.COLOR_YELLOW}Warning: Could not load persona: {e}. Responses may be generic.{config.COLOR_RESET}")

    # Initialize ChromaDB
    print(f"{config.COLOR_BLUE}Initializing ChromaDB...{config.COLOR_RESET}")
    chroma_recreated = False

    try:
        Path(config.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True, mode=0o755)
        if not os.access(config.CHROMA_DB_PATH, os.W_OK):
            logger.error(f"Insufficient permissions for ChromaDB path: {config.CHROMA_DB_PATH}")
            print(f"{config.COLOR_RED}Error: Insufficient permissions for ChromaDB path: {config.CHROMA_DB_PATH}{config.COLOR_RESET}")
            sys.exit(1)

        try:
            requests.get(f"http://{config.CHROMA_SERVER_HOST}:{config.CHROMA_SERVER_PORT}/api/v1/heartbeat", timeout=1)
            logger.warning(f"Detected a ChromaDB server running at http://{config.CHROMA_SERVER_HOST}:{config.CHROMA_SERVER_PORT}. "
                           "This might conflict with the embedded ChromaDB client. "
                           "If you intend to use the embedded client, please stop the external server.")
            print(f"{config.COLOR_YELLOW}Warning: Detected an external ChromaDB server. This might cause conflicts. "
                  "If you intend to use the embedded database, please stop the external server.{config.COLOR_RESET}")
        except requests.exceptions.ConnectionError:
            logger.info("No external ChromaDB server detected, proceeding with embedded client.")
        except Exception as e:
            logger.warning(f"Could not check external ChromaDB server status: {e}")


        chroma_client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        chroma_collection_name = "kaia_documents"
        chroma_collection = None

        try:
            logger.info(f"Attempting to get ChromaDB collection '{chroma_collection_name}' from path: {config.CHROMA_DB_PATH}")
            chroma_collection = chroma_client.get_collection(chroma_collection_name)
            logger.info(f"ChromaDB collection '{chroma_collection_name}' found. Count: {chroma_collection.count()}")

            collection_dim = None
            if chroma_collection.count() > 0:
                peek_result = chroma_collection.peek()
                embeddings_data = peek_result.get("embeddings")

                if embeddings_data is not None and len(embeddings_data) > 0:
                    first_embedding = embeddings_data[0]
                    if hasattr(first_embedding, 'shape'):
                        collection_dim = first_embedding.shape[0]
                    elif isinstance(first_embedding, (list, tuple)):
                        collection_dim = len(first_embedding)
                    else:
                        logger.warning(f"Unrecognized embedding type: {type(first_embedding)}. Forcing recreation.")
                        raise RuntimeError("Unrecognized embedding type during peek.")
                else:
                    logger.warning("Peeked embeddings are empty/None despite collection count > 0. Forcing recreation.")
                    raise RuntimeError("Inconsistent collection state during peek.")

            if collection_dim is not None and collection_dim != embedding_dim:
                logger.warning(f"ChromaDB collection '{chroma_collection_name}' dimension mismatch. Expected {embedding_dim}, got {collection_dim}. Recreating...")
                chroma_client.delete_collection(chroma_collection_name)
                chroma_collection = chroma_client.create_collection(chroma_collection_name)
                logger.info(f"Recreated ChromaDB collection '{chroma_collection_name}' due to dimension mismatch.")
                chroma_recreated = True
            else:
                logger.info(f"ChromaDB collection '{chroma_collection_name}' found and dimension matches or is empty.")

        except (chromadb.errors.NotFoundError, RuntimeError, ValueError) as e:
            logger.warning(f"ChromaDB collection '{chroma_collection_name}' needs recreation: {e}. Deleting and recreating collection...")
            try:
                chroma_client.delete_collection(chroma_collection_name)
                logger.info(f"Deleted existing ChromaDB collection '{chroma_collection_name}'.")
            except chromadb.errors.NotFoundError:
                logger.info(f"ChromaDB collection '{chroma_collection_name}' not found during deletion attempt (already gone).")
                pass
            chroma_collection = chroma_client.create_collection(chroma_collection_name)
            logger.info(f"Created new ChromaDB collection '{chroma_collection_name}'.")
            chroma_recreated = True


        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        logger.info(f"ChromaDB persistent client initialized at '{config.CHROMA_DB_PATH}'.")
        print(f"{config.COLOR_GREEN}ChromaDB persistent client initialized.{config.COLOR_RESET}")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB persistent client: {e}", exc_info=True)
        print(f"{config.COLOR_RED}Failed to initialize ChromaDB persistent client: {e}{config.COLOR_RESET}")
        print(f"{config.COLOR_YELLOW}Please ensure write permissions for ChromaDB path: {config.CHROMA_DB_PATH}{config.COLOR_RESET}")
        sys.exit(1)

    # Configure LlamaIndex
    print(f"{config.COLOR_BLUE}Loading/Building LlamaIndex...{config.COLOR_RESET}")
    index = None
    try:
        if chroma_recreated or not os.path.exists(config.LLAMA_INDEX_METADATA_PATH):
            print(f"{config.COLOR_BLUE}Building index...{config.COLOR_RESET}")

            all_docs = []

            # Load General Knowledge Documents
            general_files = []
            for root, _, files in os.walk(config.GENERAL_KNOWLEDGE_DIR):
                for file in files:
                    file_path = os.path.join(root, file)
                    if not Path(file_path).name.startswith("Kaia_Desktop_Persona"):
                        general_files.append(file_path)

            loaded_general_count = 0
            for f_path in general_files:
                try:
                    with suppress_stdout():
                        docs = SimpleDirectoryReader(input_files=[f_path]).load_data()
                        all_docs.extend(docs)
                        loaded_general_count += len(docs)
                    logger.info(f"Successfully loaded general document: {f_path}")
                except Exception as e:
                    logger.error(f"Failed to load general document '{f_path}': {e}", exc_info=True)
                    print(f"{config.COLOR_YELLOW}Warning: Skipping problematic general document: {f_path} ({e}){config.COLOR_RESET}")

            logger.info(f"Loaded {loaded_general_count} general documents in total.")

            # Load Personal Context Documents
            personal_files = []
            for root, _, files in os.walk(config.PERSONAL_CONTEXT_DIR):
                for file in files:
                    personal_files.append(os.path.join(root, file))

            loaded_personal_count = 0
            for f_path in personal_files:
                try:
                    with suppress_stdout():
                        docs = SimpleDirectoryReader(input_files=[f_path]).load_data()
                        all_docs.extend(docs)
                        loaded_personal_count += len(docs)
                    logger.info(f"Successfully loaded personal document: {f_path}")
                except Exception as e:
                    logger.error(f"Failed to load personal document '{f_path}': {e}", exc_info=True)
                    print(f"{config.COLOR_YELLOW}Warning: Skipping problematic personal document: {f_path} ({e}){config.COLOR_RESET}")

            logger.info(f"Loaded {loaded_personal_count} personal documents in total.")

            if not all_docs:
                print(f"{config.COLOR_RED}ERROR: No documents found in knowledge directories to build index!{config.COLOR_RESET}")
                print(f"{config.COLOR_YELLOW}Knowledge directories checked:{config.COLOR_RESET}")
                print(f"- General: {config.GENERAL_KNOWLEDGE_DIR}")
                print(f"- Personal: {config.PERSONAL_CONTEXT_DIR}")
            else:
                with suppress_stdout():
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex.from_documents(all_docs, storage_context=storage_context)
                    index.storage_context.persist(persist_dir=config.LLAMA_INDEX_METADATA_PATH)
                logger.info("Index built and saved.")
                print(f"{config.COLOR_GREEN}Index built and saved.{config.COLOR_RESET}")
        else:
            print(f"{config.COLOR_BLUE}Loading index...{config.COLOR_RESET}")
            with suppress_stdout():
                storage_context = StorageContext.from_defaults(
                    vector_store=vector_store,
                    persist_dir=config.LLAMA_INDEX_METADATA_PATH
                )
                index = load_index_from_storage(storage_context=storage_context)
            logger.info("Index loaded.")
            print(f"{config.COLOR_GREEN}Index loaded.{config.COLOR_RESET}")
    except Exception as e:
        logger.error(f"Failed to initialize LlamaIndex: {e}", exc_info=True)
        print(f"{config.COLOR_RED}Failed to initialize LlamaIndex: {e}{config.COLOR_RESET}")
        index = None

    # Chat Engines
    def get_rag_chat_engine(index_param=None):
        if not hasattr(get_rag_chat_engine, '_cached_engine') or index_param != get_rag_chat_engine._cached_index:
            if index_param:
                combined_system_prompt = (
                    "You are Kaia, a helpful and knowledgeable AI assistant. "
                    "Your responses should be concise and directly answer the user's question. "
                    "You MUST ONLY use the provided context to answer the question. "
                    "If the answer is NOT explicitly explicitly present in the provided context, you MUST state clearly that you do not have enough information to answer based on the available data. "
                    "Do NOT invent information or use your general knowledge if it contradicts the context. "
                    "NEVER refer to yourself as a language model or AI assistant in the context of answering a factual question from the provided data, unless the question is explicitly about your own nature or capabilities. "
                    "Do NOT output internal document metadata like 'page_label', 'file_path', or 'doc_id' unless explicitly asked for document source details. "
                    "When providing factual answers from the context, avoid conversational filler, emotional expressions, or persona-driven text. Be direct and to the point. "
                    "Maintain your core persona: strategic precision, intellectual curiosity, dry, often sarcastic wit. "
                    "Persona details: " + kaia_persona_content + "\n" + config.KAIA_SYSTEM_PROMPT
                )
                get_rag_chat_engine._cached_engine = index_param.as_chat_engine(
                    chat_mode="condense_plus_context",
                    memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
                    system_prompt=combined_system_prompt,
                    similarity_top_k=4
                )
                get_rag_chat_engine._cached_index = index_param
            else:
                print(f"{config.COLOR_YELLOW}Warning: RAG index not available. Using basic chat engine for RAG queries.{config.COLOR_RESET}")
                get_rag_chat_engine._cached_engine = SimpleChatEngine.from_defaults(
                    llm=Settings.llm,
                    memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
                    system_prompt=config.KAIA_SYSTEM_PROMPT + "\n\n" + kaia_persona_content,
                )
                get_rag_chat_engine._cached_index = None
        return get_rag_chat_engine._cached_engine

    def get_pure_chat_engine():
        if not hasattr(get_pure_chat_engine, '_cached_engine'):
            get_pure_chat_engine._cached_engine = SimpleChatEngine.from_defaults(
                llm=Settings.llm,
                chat_mode="best",
                memory=ChatMemoryBuffer.from_defaults(token_limit=8192),
                system_prompt=config.KAIA_SYSTEM_PROMPT + "\n\n" + kaia_persona_content,
            )
        return get_pure_chat_engine._cached_engine

    if index is None:
        rag_chat_engine = get_pure_chat_engine()
    else:
        rag_chat_engine = get_rag_chat_engine(index)
    pure_chat_engine = get_pure_chat_engine()

    # SQL Database
    sql_query_engine = None
    if sql_rag_enabled_local:
        print(f"{config.COLOR_BLUE}Initializing PostgreSQL database...{config.COLOR_RESET}")
        try:
            with suppress_stdout():
                database_utils.initialize_db()
                sql_database = SQLDatabase(database_utils.engine)
                sql_query_engine = NLSQLTableQueryEngine(
                    sql_database=sql_database,
                    llm=Settings.llm,
                    tables=["facts", "interaction_history", "user_preferences"]
                )
            logger.info("PostgreSQL initialized successfully")
            print(f"{config.COLOR_GREEN}PostgreSQL initialized successfully.{config.COLOR_RESET}")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL Database: {e}", exc_info=True)
            print(f"{config.COLOR_RED}Failed to initialize PostgreSQL Database: {e}{config.COLOR_RESET}")
            sql_rag_enabled_local = False

    # Helper Functions
    def speak_text_async(text):
        if not config.TTS_ENABLED:
            return
        try:
            subprocess.run(["spd-say", "--wait", text], check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        except Exception as e:
            logger.error(f"TTS failed: {str(e)}")

    def generate_action_plan(user_input):
        payload = {
            "model": config.DEFAULT_COMMAND_MODEL,
            "messages": [
                {"role": "system", "content": config.ACTION_PLAN_SYSTEM_PROMPT},
            ] + config.ACTION_PLAN_EXAMPLES + [
                {"role": "user", "content": str(user_input)}
            ],
            "stream": False,
            "format": "json"
        }

        try:
            model_to_use, error_msg = utils.check_ollama_model_availability(config.DEFAULT_COMMAND_MODEL, config.LLM_MODEL)
            if error_msg:
                raise RuntimeError(error_msg)

            payload["model"] = model_to_use

            response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=config.TIMEOUT_SECONDS)
            response.raise_for_status()
            result = response.json()["message"]["content"].strip()
            return json.loads(result)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from action planner: {e}", exc_info=True)
            return {"action": "chat", "content": user_input}
        except Exception as e:
            logger.error(f"Action plan generation failed: {e}", exc_info=True)
            user_input_lower = user_input.lower()
            if any(keyword in user_input_lower for keyword in ['what is', 'who is', 'explain', 'tell me about', 'according to', 'summarize', 'list all the books', 'pull text from', 'synopsis of']):
                return {"action": "knowledge_query", "content": user_input}
            if any(keyword in user_input_lower for keyword in ['list files', 'show contents', 'run command', 'ls ', 'cd ']):
                return {"action": "command", "content": user_input}
            if any(keyword in user_input_lower for keyword in ['list my facts', 'list history', 'show interaction history', 'what do you know about me', 'my preferences']):
                return {"action": "retrieve_data", "content": user_input}
            if any(keyword in user_input_lower for keyword in ['status', 'how is my computer doing', 'system info', 'show system status', 'display system status', 'status kaia', 'kaia status']):
                return {"action": "system_status", "content": user_input}
            if any(keyword in user_input_lower for keyword in ['run ', 'execute ']) and '.sh' in user_input_lower or '.py' in user_input_lower:
                return {"action": "run_script", "content": user_input.replace('run ', '').replace('execute ', '').strip()}
            return {"action": "chat", "content": user_input}

    def stream_and_print_response(response_stream, start_time):
        full_response = ""
        first_token_received = False
        first_token_time = None
        current_line_length = 0
        word_buffer = []

        for token in response_stream.response_gen:
            if time.time() - start_time > config.TIMEOUT_SECONDS:
                raise TimeoutError("Response streaming timed out")

            if not first_token_received:
                first_token_received = True
                first_token_time = time.time()
                print(f"\n{config.COLOR_YELLOW}⚡ First token in {first_token_time - start_time:.2f}s{config.COLOR_RESET}\n{config.COLOR_BLUE}", end="", flush=True)
                sys.stdout.write("\033[K")

            for char in token:
                word_buffer.append(char)
                if char.isspace() or char in ['.', ',', '!', '?', ';', ':'] or len(word_buffer) > config.MAX_LINE_WIDTH:
                    word_to_print = "".join(word_buffer)
                    word_buffer = []

                    if current_line_length + len(word_to_print) > config.MAX_LINE_WIDTH and current_line_length > 0:
                        sys.stdout.write("\n")
                        current_line_length = 0

                    sys.stdout.write(word_to_print)
                    current_line_length += len(word_to_print)
                    full_response += word_to_print
                    sys.stdout.flush()

        if word_buffer:
            word_to_print = "".join(word_buffer)
            if current_line_length + len(word_to_print) > config.MAX_LINE_WIDTH and current_line_length > 0:
                sys.stdout.write("\n")
            sys.stdout.write(word_to_print)
            full_response += word_to_print
            sys.stdout.flush()


        response = full_response
        response_type = "chat"
        print(f"\n\n{config.COLOR_YELLOW}⏱ Total time: {time.time() - start_time:.2f}s", end=" ")
        if first_token_time:
            print(f"(First token: {first_token_time - start_time:.2f}s)", end="")
        print(f"{config.COLOR_RESET}")
        return full_response

    # Main Loop
    print(f"""{config.COLOR_BLUE}
██╗  ██╗ █████╗ ██╗ █████╗
██║ ██╔╝██╔══██╗██║██╔══██╗
█████╔╝ ███████║██║███████║
██╔═██╗ ██╔══██║██║██╔══██║
██║  ██╗██║  ██║██║██║  ██║
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═╝
{config.COLOR_RESET}
Kaia (Personal AI Assistant) - Ready

{config.COLOR_GREEN}Welcome to Kaia, your AI assistant.{config.COLOR_RESET}
Type 'exit' or 'quit' to end the session.
You can also ask about system status (e.g., 'show disk usage').
To store memories, use phrases like 'remember I prefer...' or 'remember that...'
To retrieve memories, use 'list my preferences' or 'show facts'.
To run a command, type 'run command <your command>' or '!<your command>'.
""")

    current_working_directory = Path.cwd()

    while True:
        try:
            user_id = database_utils.get_current_user()
            database_utils.ensure_user(user_id)

            query = input("\nYou: ").strip()

            if query.lower() in ['exit', 'quit', '/exit', '/quit']:
                print(f"{config.COLOR_BLUE}Kaia: Session ended. Until next time!{config.COLOR_RESET}")
                break
            if not query:
                continue

            start_time = time.time()
            response = ""
            response_type = "unclassified_query"

            if query.startswith('/') or query.startswith('!'):
                cmd_query = query[1:].strip()
                if cmd_query.lower() == 'help':
                    response = "Help: Use /status, /exit, or natural language."
                    response_type = "help"
                    print(f"{config.COLOR_BLUE}Kaia: {response}{config.COLOR_RESET}")
                elif cmd_query.lower() == 'status':
                    status_info = cli.get_system_status()
                    status_info['db_status'] = database_utils.get_database_status()
                    response = cli.format_system_status_output(status_info)
                    response_type = "system_status"
                    print(f"\n{config.COLOR_BLUE}Kaia:{config.COLOR_RESET}")
                    print(f"{config.COLOR_GREEN}┌── System Status ──┐{config.COLOR_RESET}")
                    print(response)
                    print(f"{config.COLOR_GREEN}└───────────────────┘{config.COLOR_RESET}")
                else:
                    action = "command"
                    content = cmd_query

                if response_type != "unclassified_query":
                    try:
                        database_utils.log_interaction(user_id=user_id, user_query=query, kaia_response=response, response_type=response_type)
                    except Exception as log_e:
                        logger.error(f"Failed to log interaction for direct command: {log_e}", exc_info=True)
                    continue


            plan = generate_action_plan(query)
            if not isinstance(plan, dict) or 'action' not in plan:
                plan = {"action": "chat", "content": query}

            action = plan.get("action", "chat")
            content = plan.get("content", query)

            if action == "store_data":
                if isinstance(content, list): content = ' '.join(content)
                storage_handled, storage_response = database_utils.handle_memory_storage(user_id, content)
                response = storage_response
                response_type = "store_data"
                print(f"\n{config.COLOR_BLUE}Kaia: {response}{config.COLOR_RESET}")

            elif action == "command":
                print(f"\n{config.COLOR_BLUE}Kaia (Command Mode):{config.COLOR_RESET}")
                command, error = cli.generate_command(str(content))
                if error:
                    response = f"Command generation failed: {error}"
                    print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                else:
                    print(f"\n{config.COLOR_YELLOW}┌── Proposed Command ──┐{config.COLOR_RESET}")
                    print(f"{config.COLOR_BLUE}{command}{config.COLOR_RESET}")
                    print(f"{config.COLOR_YELLOW}└──────────────────────┘{config.COLOR_RESET}")
                    confirm = input(f"{config.COLOR_YELLOW}Execute? (y/N): {config.COLOR_RESET}").lower().strip()
                    if confirm == 'y':
                        if command.strip().startswith("cd "):
                            target_dir = command.strip()[3:].strip()
                            new_path = (current_working_directory / target_dir).resolve()
                            if new_path.is_dir():
                                current_working_directory = new_path
                                response = f"Changed directory to: {current_working_directory}"
                                print(f"{config.COLOR_GREEN}{response}{config.COLOR_RESET}")
                            else:
                                response = f"Error: Directory not found: {target_dir}"
                                print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                            success = True
                        else:
                            success, stdout, stderr = cli.execute_command(command, cwd=str(current_working_directory))
                            if success:
                                response = f"Command executed successfully. Output:\n{stdout}"
                                print(f"{config.COLOR_GREEN}{response}{config.COLOR_RESET}")
                                if stderr: print(f"{config.COLOR_YELLOW}Stderr:\n{stderr}{config.COLOR_RESET}")
                            else:
                                response = f"Command failed. Stderr:\n{stderr}\nStdout:\n{stdout}"
                                print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                    else:
                        response = f"Command cancelled: {command}"
                        print(f"{config.COLOR_BLUE}{response}{config.COLOR_RESET}")
                response_type = "command"

            elif action == "run_script":
                script_name = content
                script_path = os.path.expanduser(os.path.join("~", script_name))

                if script_name in config.SCRIPT_ALLOWLIST:
                    if script_name in INTERACTIVE_SCRIPTS:
                        response = (
                            f"Error: Script '{script_name}' is interactive and cannot be run directly by Kaia. "
                            "Please run it manually in your terminal."
                        )
                        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                        response_type = "script_interactive_error"
                    elif os.path.exists(script_path) and os.path.isfile(script_path) and os.access(script_path, os.X_OK):
                        print(f"\n{config.COLOR_BLUE}Kaia (Running Script):{config.COLOR_RESET}")
                        print(f"{config.COLOR_YELLOW}┌── Executing Script ──┐{config.COLOR_RESET}")
                        print(f"{config.COLOR_BLUE}{script_path}{config.COLOR_RESET}")
                        print(f"{config.COLOR_YELLOW}└──────────────────────┘{config.COLOR_RESET}")
                        success, stdout, stderr = cli.execute_command(script_path)
                        if success:
                            response = f"Script executed successfully. Output:\n{stdout}"
                            print(f"{config.COLOR_GREEN}{response}{config.COLOR_RESET}")
                            if stderr: print(f"{config.COLOR_YELLOW}Stderr:\n{stderr}{config.COLOR_RESET}")
                        else:
                            response = f"Script failed. Stderr:\n{stderr}\nStdout:\n{stdout}"
                            print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                        response_type = "script_execution"
                    else:
                        response = f"Error: Script '{script_name}' not found at '{script_path}', not a file, or not executable."
                        print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                        response_type = "script_error"
                else:
                    response = f"Error: Script '{script_name}' is not in the allowlist for direct execution."
                    print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
                    response_type = "script_error"

            elif action == "convert_video_to_gif": # Call the new function
                conversion_result = video_converter.convert_video_to_gif_interactive(cli, user_id)
                response = conversion_result['response']
                response_type = conversion_result['response_type']

            elif action == "system_status":
                status_info = cli.get_system_status()
                status_info['db_status'] = database_utils.get_database_status()
                response = cli.format_system_status_output(status_info)
                response_type = "system_status"
                print(f"\n{config.COLOR_BLUE}Kaia:{config.COLOR_RESET}")
                print(f"{config.COLOR_GREEN}┌── System Status ──┐{config.COLOR_RESET}")
                print(response)
                print(f"{config.COLOR_GREEN}└───────────────────┘{config.COLOR_RESET}")

            elif action == "sql" and sql_rag_enabled_local and sql_query_engine:
                try:
                    print(f"\n{config.COLOR_BLUE}Kaia (Querying Database):{config.COLOR_RESET}")
                    sql_response = sql_query_engine.query(content)
                    response = str(sql_response)
                    response_type = "sql_query"
                    print(f"{config.COLOR_GREEN}┌── Query Results ──┐{config.COLOR_RESET}")
                    print(f"{response}")
                    print(f"{config.COLOR_GREEN}└───────────────────┘{config.COLOR_RESET}")
                except Exception as e:
                    response = f"Database Error: {e}"
                    response_type = "sql_error"
                    print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")

            elif action == "retrieve_data":
                if isinstance(content, list): content = ' '.join(content)
                result = database_utils.handle_data_retrieval(user_id, content)
                response = result['message']
                response_type = result['response_type']
                print(f"\n{config.COLOR_BLUE}Kaia:{config.COLOR_RESET}")
                if isinstance(result['data'], list) and result['data']:
                    print(f"{config.COLOR_GREEN}┌── {result['message']} ──┐{config.COLOR_RESET}")
                    for item in result['data']:
                        print(f"• {str(item)}")
                    print(f"{config.COLOR_GREEN}└──────────────────────┘{config.COLOR_RESET}")
                else:
                    print(response)

            elif action == "get_persona_content":
                print(f"\n{config.COLOR_BLUE}Kaia:{config.COLOR_RESET}")
                print(f"{config.COLOR_GREEN}┌── Kaia's Persona ──┐{config.COLOR_RESET}")
                print(kaia_persona_content)
                print(f"{config.COLOR_GREEN}└────────────────────┘{config.COLOR_RESET}")
                response = kaia_persona_content
                response_type = "persona_retrieved"

            elif action == "knowledge_query":
                print(f"\n{config.COLOR_BLUE}Kaia:{config.COLOR_RESET}", end=" ", flush=True)
                logger.info(f"Processing knowledge query: '{content}'")

                try:
                    if index is None:
                        print(f"{config.COLOR_RED}ERROR: RAG index not available{config.COLOR_RESET}")
                        response = "My knowledge base is not currently available."
                    else:
                        response_stream = rag_chat_engine.stream_chat(content)
                        response = stream_and_print_response(response_stream, start_time)

                except Exception as e:
                    logger.error(f"RAG query failed: {e}", exc_info=True)
                    response = "Error retrieving information from my knowledge base."
                    print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")

                response_type = "knowledge_query"

            else:
                logger.debug(f"No specific action matched for query: '{query}'. Falling back to pure chat.")
                print(f"\n{config.COLOR_BLUE}Kaia:{config.COLOR_RESET}", end=" ", flush=True)
                response_stream = pure_chat_engine.stream_chat(content)
                response = stream_and_print_response(response_stream, start_time)
                response_type = "chat"

            if config.TTS_ENABLED:
                speak_text_async(response)

            try:
                database_utils.log_interaction(
                    user_id=user_id,
                    user_query=query,
                    kaia_response=response,
                    response_type=response_type
                )
            except Exception as log_e:
                logger.error(f"Failed to log interaction: {log_e}", exc_info=True)

        except KeyboardInterrupt:
            print(f"\n{config.COLOR_BLUE}Kaia: Exiting gracefully...{config.COLOR_RESET}")
            break
        except Exception as e:
            logger.exception("Unexpected error in main loop")
            response = f"System error: {e}"
            response_type = "system_error"
            print(f"{config.COLOR_RED}{response}{config.COLOR_RESET}")
            try:
                database_utils.log_interaction(
                    user_id=user_id,
                    user_query=query,
                    kaia_response=f"System error: {str(e)[:200]}",
                    response_type="system_error"
                )
            except Exception as log_e:
                logger.error(f"Failed to log error: {log_e}")

if __name__ == "__main__":
    main()
