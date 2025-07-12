import requests
import json
import subprocess
import os

# --- Kaia CLI Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_COMMAND_MODEL = "mixtral:8x7b-instruct-v0.1-q4_K_M"

# --- Ollama API Interaction Function ---
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
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300) # 5 min timeout
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        full_response = response.json()
        command = full_response['message']['content'].strip()

        if command.startswith("```"):

            command = command.split('\n', 1)[1]
        if command.endswith("```"):

            command = command[:-3]
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
            print(f"       Reason: {command.replace('ERROR: ', '')}")
        return

    print("\nKaia: I propose to execute the following command:")
    print(f"--------------------------------------------------")
    print(f"  {command}")
    print(f"--------------------------------------------------")

    # Safety confirmation
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

# --- Main Interaction Loop --- #
def main():
    print("Welcome to Kaia's Command Execution Interface!")
    print(f"I will use the '{DEFAULT_COMMAND_MODEL}' model to generate commands.")
    print("Type your request. Type 'exit' to quit.")

    while True:
        user_input = input("\nKaia: How may I assist with command generation? ").strip()

        if user_input.lower() == 'exit':
            print("Kaia: Farewell. May your binaries be swift and your logs concise.")
            break

        if not user_input:
            continue

        generated_command = ask_ollama_for_command(user_input)
        if generated_command:
            confirm_and_execute_command(generated_command)

if __name__ == "__main__":
    main()
