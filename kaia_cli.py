import platform
import psutil
import subprocess
import json
import logging
import os
import requests
from datetime import datetime
from typing import Dict, List, Union, Any, Optional, Tuple

# --- Logging ---
logger = logging.getLogger(__name__)

class KaiaCLI:
    """
    Provides command-line interface functionalities for Kaia.
    This class is responsible for two main tasks:
    1.  Generating and executing Linux shell commands based on natural language.
    2.  Retrieving detailed system status information.
    It acts as a service layer, abstracting the underlying system calls and API interactions
    from the main application logic in `llamaindex_ollama_rag.py`.
    """

    def __init__(self):
        """Initializes the KaiaCLI."""
        # The __init__ method is kept simple as this class is stateless.
        pass

    def get_system_status(self) -> Dict[str, Any]:
        """
        Retrieves a comprehensive snapshot of the system's current status.
        This method aggregates data from multiple helper functions.

        Returns:
            A dictionary containing structured system metrics.
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'os_info': self._get_os_info(),
            'kernel_info': self._get_kernel_info(),
            'uptime': self._get_uptime(),
            'board_info': self._get_board_info(),
            'cpu_info': self._get_cpu_info_detailed(),
            'memory_info': self._get_memory_info(),
            'disk_usage': self._get_all_disk_usage(),
            'gpu_info': self._get_gpu_details(),
            'python_version': platform.python_version(),
            'ollama_status': self._check_ollama_status(),
        }
        return status

    def _get_os_info(self) -> str:
        """Retrieves OS information dynamically."""
        try:
            # Use freedesktop standard for better compatibility across Linux distros
            release_info = psutil.os.uname()
            return f"{release_info.sysname} {release_info.release} ({release_info.machine})"
        except Exception:
             # Fallback for other systems
            return f"{platform.system()} {platform.release()}"

    def _get_kernel_info(self) -> str:
        """Retrieves the kernel version."""
        return platform.release()

    def _get_uptime(self) -> str:
        """Calculates and returns system uptime in a human-readable format."""
        try:
            boot_time_timestamp = psutil.boot_time()
            uptime_seconds = datetime.now().timestamp() - boot_time_timestamp
            days, remainder = divmod(uptime_seconds, 86400)
            hours, remainder = divmod(remainder, 3600)
            minutes, _ = divmod(remainder, 60)

            parts = []
            if days > 0:
                parts.append(f"{int(days)} day{'s' if days != 1 else ''}")
            if hours > 0:
                parts.append(f"{int(hours)} hr{'s' if hours != 1 else ''}")
            if minutes > 0:
                parts.append(f"{int(minutes)} min{'s' if minutes != 1 else ''}")

            return ", ".join(parts) if parts else "Less than a minute"
        except Exception as e:
            logger.warning(f"Could not determine uptime: {e}")
            return "N/A"

    def _get_board_info(self) -> str:
        """
        Retrieves motherboard/board information.
        NOTE: This is difficult to retrieve reliably across all systems programmatically.
        A hardcoded value is used as a practical fallback. For dynamic retrieval,
        one might parse `dmidecode -t baseboard`, which requires root privileges.
        """
        return "ROG STRIX B650-A GAMING WIFI (Rev 1.xx)"

    def _get_memory_info(self) -> Dict[str, Union[int, float]]:
        """Retrieves system memory information."""
        mem = psutil.virtual_memory()
        return {
            'total_gb': round(mem.total / (1024**3), 2),
            'available_gb': round(mem.available / (1024**3), 2),
            'percent_used': mem.percent,
        }

    def _get_cpu_info_detailed(self) -> Dict[str, Any]:
        """Retrieves detailed CPU information."""
        cpu_name = "N/A"
        try:
            # Standard method for getting CPU model on Linux
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(':')[-1].strip()
                        break
        except Exception as e:
            logger.warning(f"Could not read CPU info from /proc/cpuinfo: {e}")

        return {
            'name': cpu_name,
            'percent_usage': psutil.cpu_percent(interval=1),
            'physical_cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
        }

    def _get_all_disk_usage(self) -> List[Dict[str, Any]]:
        """Retrieves disk usage for all physical partitions."""
        partitions = psutil.disk_partitions()
        disk_info = []
        for p in partitions:
            try:
                usage = psutil.disk_usage(p.mountpoint)
                disk_info.append({
                    'mount_point': p.mountpoint,
                    'total_gb': round(usage.total / (1024**3), 2),
                    'used_gb': round(usage.used / (1024**3), 2),
                    'percent_used': usage.percent,
                })
            except Exception as e:
                logger.warning(f"Could not get disk usage for {p.mountpoint}: {e}")
        return disk_info

    def _get_gpu_details(self) -> List[Dict[str, Any]]:
        """Retrieves GPU details, prioritizing nvidia-smi if available."""
        gpus = []
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.total,memory.used", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            for line in result.stdout.strip().split('\n'):
                name, util, mem_total, mem_used = [p.strip() for p in line.split(',')]
                gpus.append({
                    'name': name,
                    'type': 'Discrete (NVIDIA)',
                    'utilization_percent': float(util),
                    'memory_total_mb': float(mem_total),
                    'memory_used_mb': float(mem_used),
                })
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.info(f"nvidia-smi not found or failed. Skipping NVIDIA GPU info. Error: {e}")
            # As a fallback, you could add logic here to find other GPUs (e.g., AMD)
            # For now, we add the known integrated GPU if nvidia-smi fails.
            gpus.append({
                'name': 'AMD Radeon Graphics',
                'type': 'Integrated',
            })
        return gpus

    def _check_ollama_status(self) -> str:
        """Checks if the Ollama server is running and responsive."""
        try:
            # A more robust check that queries the API endpoint
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            response.raise_for_status()
            return "Running"
        except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as e:
            logger.warning(f"Ollama status check failed: {e}")
            return "Not Running or Unresponsive"

    def generate_command(self, query: str, model: str) -> Tuple[str, Optional[str]]:
        """
        Generates a Linux command from a natural language query using a specified Ollama model.

        Args:
            query: The user's natural language request.
            model: The name of the Ollama model to use for generation (e.g., "mistral:instruct").

        Returns:
            A tuple containing the generated command and an optional error message.
        """
        system_prompt = """You are an expert in the Linux command line. Your sole task is to convert the user's request into the most appropriate and direct shell command.
        - Respond ONLY with the raw command.
        - Do NOT provide explanations, markdown formatting, or any text other than the command itself.
        - If a command cannot be generated, respond with 'ERROR: Unable to generate command for the request.'
        """
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "list all files including hidden ones"},
                {"role": "assistant", "content": "ls -la"},
                {"role": "user", "content": "find all python files in my home directory"},
                {"role": "assistant", "content": "find ~ -type f -name \"*.py\""},
                {"role": "user", "content": query}
            ],
            "stream": False
        }
        try:
            response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=45)
            response.raise_for_status()
            command = response.json()["message"]["content"].strip()

            if command.startswith("ERROR:"):
                return "", command

            # Basic sanitation to prevent command chaining vulnerabilities from LLM hallucinations
            if any(op in command for op in ['&&', ';', '||', '`']):
                logger.warning(f"Potentially unsafe command detected and blocked: {command}")
                return "", "ERROR: Generated command contained unsafe operators."

            return command, None
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate command via Ollama API: {e}")
            return "", f"API Error: Failed to connect to Ollama."
        except Exception as e:
            logger.error(f"An unexpected error occurred during command generation: {e}")
            return "", f"Unexpected Error: {e}"

    def execute_command(self, command: str) -> Tuple[bool, str, str]:
        """
        Executes a given shell command safely.

        Args:
            command: The shell command to execute.

        Returns:
            A tuple containing success status (bool), stdout (str), and stderr (str).
        """
        try:
            # Using check=False to handle non-zero exit codes gracefully without raising an exception
            result = subprocess.run(command, shell=True, text=True, capture_output=True, timeout=120)
            success = result.returncode == 0
            return success, result.stdout.strip(), result.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", "Command execution timed out."
        except Exception as e:
            return False, "", f"Execution failed with an unexpected error: {e}"
