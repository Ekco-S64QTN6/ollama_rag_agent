import logging
import sys
import os
import subprocess
import time
import requests
import json # Import the json module

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
LLM_MODEL = "llama2:13b-chat-q4_0"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_COMMAND_MODEL = "mixtral:8x7b-instruct-v0.1-q4_K_M"

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

# --- Ollama API Interaction Function for Command Generation ---
def ask_ollama_for_command(user_query, model=DEFAULT_COMMAND_MODEL):
    messages = [
        {"role": "system", "content": "You are a highly intelligent and precise Linux expert named Kaia. Your primary goal is to provide *only* the exact shell command required to fulfill a user's request. Do not include any explanations, introductory text, or extra characters. The command should be ready to execute directly. Your response MUST NOT contain any Markdown formatting, like backticks (```), or language indicators (e.g., `bash`, `sh`). If a task requires multiple commands, provide them separated by ' && '. If a task is impossible, ambiguous, or highly dangerous, respond *only* with 'ERROR: Cannot generate a safe command for this request.'"},
        {"role": "user", "content": user_query}
    ]

    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }

    print(f"\nKaia: Consulting with '{model}' model for command generation...")
    try:
        api_url = getattr(Settings.llm, '_api_base', "http://localhost:11434/api") + "/chat"
        response = requests.post(api_url, json=payload, timeout=300)
        response.raise_for_status()
        full_response = response.json()
        command = full_response['message']['content'].strip()

        if command.startswith("```") and command.endswith("```"):
            command = command[3:-3].strip()
        elif command.startswith("`") and command.endswith("`"):
            command = command[1:-1].strip()
        else:
            command = command.strip()

        return command
    except requests.exceptions.Timeout:
        print("Kaia: The Ollama model took too long to respond.")
        return "ERROR: Model response timed out."
    except requests.exceptions.ConnectionError:
        print("Kaia: Could not connect to the Ollama server. Is it running?")
        return "ERROR: Ollama server not reachable."
    except requests.exceptions.RequestException as e:
        print(f"Kaia: An error occurred communicating with Ollama: {e}")
        return "ERROR: Ollama communication error."
    except KeyError:
        print("Kaia: Error: Unexpected response format from Ollama. The model might not have generated a 'message' or 'content'.")
        return "ERROR: Unexpected model response format."

# --- Command Confirmation and Execution Function ---
def confirm_and_execute_command(command):
    if not command or command.lower().startswith("error:"):
        print(f"\nKaia: I could not generate a valid command for that request, or it was deemed too dangerous.")
        if command:
            print(f"     Reason: {command.replace('ERROR: ', '')}")
        return

    print("\nKaia: I propose to execute the following command:")
    print(f"--------------------------------------------------")
    print(f"  {command}")
    print(f"--------------------------------------------------")

    confirm = input("Kaia: Do you want me to execute this command? (y/N) ").strip().lower()

    if confirm == 'y':
        print("\nKaia: Executing command...")
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True, env=os.environ)
            print("--- Command Output ---")
            print(result.stdout)
            if result.stderr:
                print("--- Command Error Output ---")
                print(result.stderr)
            print("--- Command Finished ---")
        except subprocess.CalledProcessError as e:
            print(f"Kaia: Error during command execution. Exit code: {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
        except FileNotFoundError:
            print("Kaia: Error: Command not found. Is it in your PATH?")
        except Exception as e:
            print(f"Kaia: An unexpected error occurred: {e}")
    else:
        print("Kaia: Command execution cancelled.")

# --- Generate Action Plan (Command or Chat) ---
def generate_action_plan(user_input):
    system_prompt = """
    You are an AI assistant named Kaia, focused on system architecture, workflow design, and technical assistance.
    Your task is to analyze the user's request and determine if it requires a shell command to be executed or a conversational response.
    Respond ONLY with a JSON object. Do NOT include any other text, explanations, or markdown outside the JSON.

    The JSON object MUST have two keys:
    1. "action": "command" if a shell command is required, or "chat" if a conversational response is appropriate.
    2. "content": If "action" is "command", this should be the precise Linux shell command to fulfill the request. If "action" is "chat", this should be a direct conversational response to the user.

    Examples for "action": "command":
    - User: "List files in the current directory"
      Response: {"action": "command", "content": "ls -F"}
    - User: "Show my disk usage"
      Response: {"action": "command", "content": "df -h"}
    - User: "What's my kernel version?"
      Response: {"action": "command", "content": "uname -r"}

    Examples for "action": "chat":
    - User: "Tell me about large language models"
      Response: {"action": "chat", "content": "Large Language Models (LLMs) are advanced AI models trained on vast amounts of text data to understand, generate, and process human language."}
    - User: "How are you today?"
      Response: {"action": "chat", "content": "As an AI, I don't experience feelings, but I'm ready to assist you. How can I help?"}

    If a request is ambiguous, dangerous, or cannot be safely translated into a command, default to "chat" and explain the limitation.
    Always prioritize safety and clarity for commands.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]

    payload = {
        "model": DEFAULT_COMMAND_MODEL,
        "messages": messages,
        "stream": False,
        "format": "json"
    }

    print(f"\nKaia: Analyzing intent with '{DEFAULT_COMMAND_MODEL}' model...")
    try:
        api_url = getattr(Settings.llm, '_api_base', "http://localhost:11434/api") + "/chat"
        response = requests.post(api_url, json=payload, timeout=300)
        response.raise_for_status()
        raw_response_content = response.json()['message']['content'].strip()

        try:
            action_plan = json.loads(raw_response_content)
            if "action" in action_plan and "content" in action_plan:
                return action_plan
            else:
                raise ValueError("JSON response missing 'action' or 'content' keys.")
        except json.JSONDecodeError as e:
            print(f"Kaia: Error parsing JSON from action planner: {e}")
            print(f"Raw response: {raw_response_content}")
            # Fallback to chat if JSON parsing fails
            return {"action": "chat", "content": "I had trouble interpreting your request for an action. Could you please rephrase it?"}
        except ValueError as e:
            print(f"Kaia: Invalid action plan structure: {e}")
            print(f"Raw response: {raw_response_content}")
            return {"action": "chat", "content": "I received an malformed action plan. Could you please rephrase your request?"}


    except requests.exceptions.Timeout:
        print("Kaia: The action planner model took too long to respond.")
        return {"action": "chat", "content": "The action planner model timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        print("Kaia: Could not connect to the Ollama server for action planning. Is it running?")
        return {"action": "chat", "content": "Could not connect to the Ollama server for action planning. Please ensure Ollama is running."}
    except requests.exceptions.RequestException as e:
        print(f"Kaia: An error occurred communicating with Ollama for action planning: {e}")
        return {"action": "chat", "content": "An error occurred during action planning. Please try again."}
    except KeyError:
        print("Kaia: Unexpected response format from action planner. No 'message' or 'content'.")
        return {"action": "chat", "content": "The action planner returned an unexpected format. Please try again."}


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

        # Handle explicit slash commands first (help, status)
        if query.startswith('/'):
            command_name = query[1:].lower().split(' ')[0]
            command_args = query[len(command_name) + 2:].strip() if len(query) > len(command_name) + 1 else ""

            if command_name == 'help':
                msg = ("/help - Show commands\n"
                       "/status - Show system info\n"
                       "Just type your request in natural language for Kaia to interpret and act on.\n"
                       "/exit - Quit app")
                print(f"\033[36m{msg}\033[0m\n")
                if tts_enabled:
                    clean_speech_text = msg.replace("\\", "").replace("\n", " ").replace("\t", " ")
                    speak_text(clean_speech_text)
                continue

            elif command_name == 'status':
                msg = get_dynamic_system_status()
                print(f"\033[36m{msg}\033[0m\n")
                if tts_enabled:
                    clean_speech_text = msg.replace("\\", "").replace("\n", " ").replace("\t", " ")
                    speak_text(clean_speech_text)
                continue

            else:
                msg = f"Unknown command: /{command_name}. Type /help for available commands."
                print(f"\033[36m{msg}\033[0m\n")
                if tts_enabled:
                    clean_speech_text = msg.replace("\\", "").replace("\n", " ").replace("\t", " ")
                    speak_text(clean_speech_text)
                continue

        # --- Intelligent Action Planning ---
        action_plan = generate_action_plan(query)
        action_type = action_plan.get("action")
        content = action_plan.get("content")

        if action_type == "command":
            if content and not content.lower().startswith("error:"):
                print(f"\033[36mKaia: I interpreted your request as a command. Proposing to execute: '{content}'\033[0m")
                confirm_and_execute_command(content)
            else:

                print(f"\033[36mKaia: I tried to generate a command, but encountered an issue: {content}\033[0m\n")
                if tts_enabled:
                    speak_text(content.replace("\n", " ").replace("\t", " "))

        elif action_type == "chat":
            print(f"\033[36mKaia: ", end="", flush=True)
            start = time.time()

            if content:
                full_response = content
                print(full_response, end="", flush=True)
            else:
                response_stream = chat_engine.stream_chat(query)
                full_response = ""
                first_token_time = None
                for token in response_stream.response_gen:
                    if first_token_time is None:
                        first_token_time = time.time()
                        print(f"\n⏱ First token in {first_token_time - start:.2f}s\n", end="", flush=True)
                    print(token, end="", flush=True)
                    full_response += token
            print("\033[0m\n")

            if tts_enabled:
                clean_speech_text = full_response.replace("\\", "").replace("\n", " ").replace("\t", " ")
                speak_text(clean_speech_text)

        else:
            msg = "Kaia: I could not determine the appropriate action for your request. Please try rephrasing."
            print(f"\033[36m{msg}\033[0m\n")
            if tts_enabled:
                speak_text(msg.replace("\n", " ").replace("\t", " "))


    except Exception as e:
        print(f"Error during query: {e}")
