import requests
import subprocess
import os
import logging
from typing import Optional, Tuple

# --- Configuration ---
OLLAMA_API_URL = "http://localhost:11434/api/chat"
DEFAULT_COMMAND_MODEL = "mistral:instruct"
TIMEOUT_SECONDS = 300  # 5 minutes
DANGEROUS_PATTERNS = [';', '&&', '||', '`', '$(', '>', '<', '|']

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('kaia_cli.log')]
)
logger = logging.getLogger(__name__)

class KaiaCLI:
    """Core command generation and execution functionality."""

    def __init__(self, model: str = DEFAULT_COMMAND_MODEL):
        self.model = model
        self.last_command = None
        self.last_output = None

    def generate_command(self, user_query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate a Linux command from natural language.

        Args:
            user_query: Natural language description of desired command

        Returns:
            tuple: (generated_command, error_message) - one will be None
        """
        system_prompt = """You are Kaia, a precise Linux expert. Provide ONLY the exact shell command needed.
Rules:
1. No explanations, intros, or extra text
2. No Markdown formatting (``` or language indicators)
3. For multiple commands, use ' && ' between them
4. If unsafe/ambiguous, respond ONLY with: 'ERROR: Cannot generate safe command'
"""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            "stream": False
        }

        try:
            response = requests.post(
                OLLAMA_API_URL,
                json=payload,
                timeout=TIMEOUT_SECONDS
            )
            response.raise_for_status()

            command = response.json()['message']['content'].strip()

            # Clean any markdown formatting
            if command.startswith("```"):
                command = command.split('\n', 1)[1]
            if command.endswith("```"):
                command = command[:-3]
            command = command.strip()

            if any(pattern in command for pattern in DANGEROUS_PATTERNS):
                return None, "Command contains dangerous patterns"

            self.last_command = command
            return command, None

        except requests.exceptions.Timeout:
            return None, "Command generation timed out"
        except requests.exceptions.ConnectionError:
            return None, "Could not connect to Ollama server"
        except requests.exceptions.RequestException as e:
            return None, f"Ollama communication error: {str(e)}"
        except (KeyError, ValueError) as e:
            return None, f"Unexpected response format: {str(e)}"

    def execute_command(self, command: str) -> Tuple[bool, str, str]:
        """
        Execute a shell command with safety checks.

        Args:
            command: The command to execute

        Returns:
            tuple: (success: bool, stdout: str, stderr: str)
        """
        if not command:
            return False, "", "No command provided"

        if any(pattern in command for pattern in DANGEROUS_PATTERNS):
            return False, "", "Command contains dangerous patterns"

        try:
            result = subprocess.run(
                command,
                shell=True,
                check=False,  # We'll handle errors ourselves
                capture_output=True,
                text=True,
                env=os.environ
            )

            self.last_output = (result.stdout, result.stderr)
            return (
                result.returncode == 0,
                result.stdout,
                result.stderr
            )

        except Exception as e:
            return False, "", f"Execution error: {str(e)}"

def get_command_for_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    """Convenience function for quick command generation."""
    cli = KaiaCLI()
    return cli.generate_command(query)

def execute_user_command(command: str) -> Tuple[bool, str, str]:
    """Convenience function for quick command execution."""
    cli = KaiaCLI()
    return cli.execute_command(command)

# For backward compatibility with existing imports
if __name__ == "__main__":
    print("This module is meant to be imported, not run directly.")
    print("Use the provided functions or the KaiaCLI class.")
