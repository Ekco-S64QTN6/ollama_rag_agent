import platform
import psutil
import subprocess
import json
from datetime import datetime, timedelta
import logging
import os
from typing import Dict, List, Union, Any, Optional, Tuple

# --- Constants ---
DEFAULT_COMMAND_MODEL = "mixtral:8x7b-instruct-v0.1-q4_K_M"

# --- Logging ---
logger = logging.getLogger(__name__)

class KaiaCLI:
    """
    Provides command-line interface functionalities for Kaia,
    including system status retrieval and command generation/execution.
    """

    def __init__(self):
        """Initializes the KaiaCLI."""
        pass

    def get_system_status(self) -> Dict[str, Any]:
        """
        Retrieves comprehensive system status information.

        Returns:
            Dict[str, Any]: A dictionary containing various system metrics.
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'os_info': self._get_os_info(),
            'kernel_info': self._get_kernel_info(),
            'python_version': platform.python_version(),
            'cpu_info': self._get_cpu_info_detailed(),
            'memory_info': self._get_memory_info(),
            'all_disk_usage': self._get_all_disk_usage(),
            'gpu_info': self._get_gpu_details(),
            'vulkan_info': self._get_vulkan_info(), # New: Vulkan information
            'opencl_info': self._get_opencl_info(), # New: OpenCL information
            'uptime': self._get_uptime(), # New: Uptime information
            'board_info': self._get_board_info(), # New: Board information
            'ollama_status': self._check_ollama_status(),
        }
        return status

    def _get_os_info(self) -> str:
        """Retrieves formatted OS information."""
        # Based on user's provided info: "OS: Arch Linux x86_64"
        return f"Arch Linux {platform.machine()}"

    def _get_kernel_info(self) -> str:
        """Retrieves formatted Kernel information."""
        # Based on user's provided info: "Kernel: Linux 6.15.7-arch1-1"
        return f"Linux {platform.release()}"

    def _get_uptime(self) -> str:
        """
        Calculates and returns system uptime in a human-readable format.
        """
        boot_time_timestamp = psutil.boot_time()
        boot_datetime = datetime.fromtimestamp(boot_time_timestamp)
        current_datetime = datetime.now()
        uptime_delta = current_datetime - boot_datetime

        days = uptime_delta.days
        hours, remainder = divmod(uptime_delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        uptime_parts = []
        if days > 0:
            uptime_parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            uptime_parts.append(f"{hours} hr{'s' if hours != 1 else ''}")
        if minutes > 0:
            uptime_parts.append(f"{minutes} min{'s' if minutes != 1 else ''}")

        if not uptime_parts: # Less than a minute
            return "Less than a minute"

        return ", ".join(uptime_parts)

    def _get_board_info(self) -> str:
        """
        Retrieves motherboard/board information.
        """
        # Based on user's provided info: "Board: ROG STRIX B650-A GAMING WIFI (Rev 1.xx)"
        return "ROG STRIX B650-A GAMING WIFI (Rev 1.xx)"


    def _get_memory_info(self) -> Dict[str, Union[int, float]]:
        """
        Retrieves system memory information.

        Returns:
            Dict[str, Union[int, float]]: Dictionary with total, available, used memory in bytes and percentage.
        """
        mem = psutil.virtual_memory()
        return {
            'total': mem.total,
            'available': mem.available,
            'percent': mem.percent,
            'used': mem.used
        }

    def _get_cpu_info_detailed(self) -> Dict[str, Union[float, int, str]]:
        """
        Retrieves detailed CPU information including name and clock speed.

        Returns:
            Dict[str, Union[float, int, str]]: Dictionary with CPU percentage, core count, name, and speed.
        """
        cpu_name = "N/A"
        cpu_speed = "N/A"
        try:
            # Attempt to get CPU name and speed from /proc/cpuinfo
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(':')[-1].strip()
                    if "cpu MHz" in line:
                        mhz = float(line.split(':')[-1].strip())
                        cpu_speed = f"{mhz / 1000:.2f} GHz" if mhz >= 1000 else f"{mhz:.0f} MHz"
                    if cpu_name != "N/A" and cpu_speed != "N/A":
                        break # Found both, no need to read further
        except Exception as e:
            logger.warning(f"Could not read CPU info from /proc/cpuinfo: {e}")

        return {
            'name': cpu_name,
            'speed': cpu_speed,
            'percent': psutil.cpu_percent(interval=1),
            'cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True)
        }

    def _get_all_disk_usage(self) -> List[Dict[str, Any]]:
        """
        Retrieves disk usage information for specified partitions/mount points.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing disk usage
                                  details for a specific mount point.
        """
        # Define the mount points and their desired labels based on user's request.
        # These paths are hardcoded based on your previous output.
        # If your system's mount points change, this list will need adjustment.
        disk_mounts = [
            {'path': '/', 'label': '/'},
            {'path': '/boot', 'label': '/boot'},
            {'path': '/home', 'label': '/home'},
            {'path': '/run/media/ekco/KingSpec1', 'label': 'KingSpec1'},
            {'path': '/run/media/ekco/KingSpec2', 'label': 'KingSpec2'},
            {'path': '/run/media/ekco/D02B-11D2', 'label': 'Removable'} # USB stick
        ]

        all_disk_info = []
        for disk_mount in disk_mounts:
            path = disk_mount['path']
            label = disk_mount['label']
            try:
                usage = psutil.disk_usage(path)
                all_disk_info.append({
                    'mount_point': path,
                    'label': label,
                    'total': usage.total,
                    'used': usage.used,
                    'free': usage.free,
                    'percent': usage.percent
                })
            except Exception as e:
                logger.warning(f"Could not get disk usage for {path} ({label}): {e}")
                all_disk_info.append({
                    'mount_point': path,
                    'label': label,
                    'status': 'Error',
                    'error_message': str(e)
                })
        return all_disk_info

    def _get_gpu_details(self) -> List[Dict[str, Any]]:
        """
        Retrieves GPU information, including integrated/discrete labels.
        """
        gpu_data = []

        # Add the integrated AMD GPU as a fixed entry based on user's system info
        gpu_data.append({
            'name': 'AMD Radeon Graphics',
            'type': 'Integrated',
            'utilization_gpu_percent': 'N/A', # Cannot easily get live usage for integrated without more complex tools
            'memory_total_mb': 'N/A',
            'memory_used_mb': 'N/A',
            'memory_free_mb': 'N/A'
        })

        # Attempt to get NVIDIA GPU information using nvidia-smi
        try:
            cmd = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 5:
                    gpu_data.append({
                        'name': parts[0],
                        'type': 'Discrete', # Assume NVIDIA GPUs found by nvidia-smi are discrete
                        'utilization_gpu_percent': float(parts[1]),
                        'memory_total_mb': float(parts[2]),
                        'memory_used_mb': float(parts[3]),
                        'memory_free_mb': float(parts[4])
                    })
        except FileNotFoundError:
            logger.info("nvidia-smi not found. Skipping NVIDIA GPU info retrieval.")
        except Exception as e:
            logger.error(f"Error retrieving NVIDIA GPU info: {e}")

        return gpu_data

    def _get_vulkan_info(self) -> str:
        """
        Retrieves Vulkan information. Hardcoded based on user's provided output.
        """
        return "1.4.311 - NVIDIA [575.64.03] radv [Mesa 25.1.6-arch1.1]"

    def _get_opencl_info(self) -> str:
        """
        Retrieves OpenCL information. Hardcoded based on user's provided output.
        """
        return "3.0 CUDA 12.9.90"

    def _check_ollama_status(self) -> str:
        """
        Checks if the Ollama server is running by attempting to connect to its default port.

        Returns:
            str: "Running" if Ollama is accessible, "Not Running" otherwise.
        """
        try:
            subprocess.run(["nc", "-z", "localhost", "11434"], check=True, capture_output=True, timeout=5)
            return "Running"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return "Not Running"
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return "Error"

    def generate_command(self, query: str) -> Tuple[str, Optional[str]]:
        """
        Generates a Linux command based on the user's natural language query using Ollama.

        Args:
            query (str): The user's natural language request for a command.

        Returns:
            Tuple[str, Optional[str]]: A tuple containing the generated command string
                                       and an error message (if any).
        """
        system_prompt = """You are a Linux command-line expert. Your task is to generate
        the most appropriate Linux shell command for the given user request.
        Respond ONLY with the raw command, no explanations, no markdown, no extra text.
        If you cannot generate a command, respond with 'ERROR: [reason]'.
        Examples:
        - User: list files
        - Assistant: ls -l
        - User: check disk space
        - Assistant: df -h
        - User: create a directory named 'my_project'
        - Assistant: mkdir my_project
        """
        payload = {
            "model": DEFAULT_COMMAND_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": False
        }
        try:
            response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=30)
            response.raise_for_status()
            command = response.json()["message"]["content"].strip()
            if command.startswith("ERROR:"):
                return "", command
            return command, None
        except Exception as e:
            return "", f"Failed to generate command: {e}"

    def execute_command(self, command: str) -> Tuple[bool, str, str]:
        """
        Executes a given shell command.

        Args:
            command (str): The shell command to execute.

        Returns:
            Tuple[bool, str, str]: A tuple containing:
                                   - bool: True if command succeeded, False otherwise.
                                   - str: Standard output of the command.
                                   - str: Standard error of the command.
        """
        try:
            result = subprocess.run(command, shell=True, text=True, capture_output=True, check=True)
            return True, result.stdout.strip(), result.stderr.strip()
        except subprocess.CalledProcessError as e:
            return False, e.stdout.strip(), e.stderr.strip()
        except Exception as e:
            return False, "", str(e)

