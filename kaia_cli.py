import json
import logging
import os
import platform
import psutil
import re
import shlex
import subprocess
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import config
import utils

logger = logging.getLogger(__name__)

class KaiaCLI:
    def __init__(self):
        pass

    # System Status Retrieval
    def get_system_status(self) -> Dict[str, Any]:
        status = {
            'timestamp': datetime.now().isoformat(),
            'os_info': self._get_os_info(),
            'kernel_info': self._get_kernel_info(),
            'python_version': platform.python_version(),
            'cpu_info': self._get_cpu_info_detailed(),
            'memory_info': self._get_memory_info(),
            'all_disk_usage': self._get_all_disk_usage(),
            'gpu_info': self._get_gpu_details(),
            'vulkan_info': self._get_vulkan_info(),
            'opencl_info': self._get_opencl_info(),
            'uptime': self._get_uptime(),
            'board_info': self._get_board_info(),
            'ollama_status': self._check_ollama_status(),
        }
        return status

    # System Status Formatting
    def format_system_status_output(self, status_info: Dict[str, Any]) -> str:
        msg_parts = [
            f"• {config.COLOR_BLUE}Date & Time:{config.COLOR_RESET} {datetime.fromisoformat(status_info.get('timestamp', datetime.now().isoformat())).strftime('%Y-%m-%d %H:%M:%S')}",
            f"• {config.COLOR_BLUE}Uptime:{config.COLOR_RESET} {status_info.get('uptime', 'N/A')}",
            f"• {config.COLOR_BLUE}Board:{config.COLOR_RESET} {status_info.get('board_info', 'N/A')}",
            f"• {config.COLOR_BLUE}OS:{config.COLOR_RESET} {status_info.get('os_info', 'N/A')}",
            f"• {config.COLOR_BLUE}Kernel:{config.COLOR_RESET} {platform.release()}",
            f"• {config.COLOR_BLUE}Python Version:{config.COLOR_RESET} {platform.python_version()}",
        ]

        cpu_info = status_info.get('cpu_info', {})
        if cpu_info:
            cpu_name = cpu_info.get('name', 'N/A')
            cpu_speed = cpu_info.get('speed', 'N/A')
            cpu_cores = cpu_info.get('logical_cores', 'N/A')
            msg_parts.append(f"• {config.COLOR_BLUE}CPU:{config.COLOR_RESET} {cpu_name} ({cpu_cores}) @ {cpu_speed}")

        mem_info = status_info.get('memory_info', {})
        if mem_info:
            total_gb = round(mem_info.get('total', 0) / (1024**3), 2)
            available_gb = round(mem_info.get('available', 0) / (1024**3), 2)
            percent_used = mem_info.get('percent', 'N/A')
            percent_color = utils.get_color_for_percentage(percent_used)
            msg_parts.append(f"• {config.COLOR_BLUE}Memory:{config.COLOR_RESET} {total_gb} GB total, {available_gb} GB available ({percent_color}{percent_used}% used{config.COLOR_RESET})")

        all_disk_usage = status_info.get('all_disk_usage', [])

        formatted_disks = {}
        for disk in all_disk_usage:
            path = disk.get('mount_point', 'N/A')
            label = disk.get('label', 'N/A')

            if disk.get('status') == 'Error':
                formatted_disks[path] = f"• {config.COLOR_RED}Disk Usage ('{label}'):{config.COLOR_RESET} Error - {disk.get('error_message', 'N/A')}"
            else:
                total_gb = round(disk.get('total', 0) / (1024**3), 2)
                used_gb = round(disk.get('used', 0) / (1024**3), 2)
                percent_used = disk.get('percent', 'N/A')
                percent_color = utils.get_color_for_percentage(percent_used)
                formatted_disks[path] = {
                    'string': f"• {config.COLOR_BLUE}Disk Usage ('{label}'):{config.COLOR_RESET} {total_gb} GB total, {used_gb} GB used ({percent_color}{percent_used}% used{config.COLOR_RESET})",
                    'data': disk
                }

        desired_order = [
            {'path': '/', 'new_label': 'Root'},
            {'path': '/home', 'new_label': 'Home'},
            {'path': '/boot', 'new_label': 'Boot'},
        ]

        for item in desired_order:
            path = item['path']
            new_label = item['new_label']
            if path in formatted_disks:
                disk_data = formatted_disks[path]['data']
                if disk_data.get('status') == 'Error':
                    msg_parts.append(f"• {config.COLOR_RED}Disk Usage ('{new_label}'):{config.COLOR_RESET} Error - {disk_data.get('error_message', 'N/A')}")
                else:
                    total_gb = round(disk_data.get('total', 0) / (1024**3), 2)
                    used_gb = round(disk_data.get('used', 0) / (1024**3), 2)
                    percent_used = disk_data.get('percent', 'N/A')
                    percent_color = utils.get_color_for_percentage(percent_used)
                    msg_parts.append(f"• {config.COLOR_BLUE}Disk Usage ('{new_label}'):{config.COLOR_RESET} {total_gb} GB total, {used_gb} GB used ({percent_color}{percent_used}% used{config.COLOR_RESET})")

                del formatted_disks[path]

        for path in formatted_disks:
            msg_parts.append(formatted_disks[path]['string'])

        if not all_disk_usage:
            msg_parts.append(f"• {config.COLOR_BLUE}Disk Usage:{config.COLOR_RESET} N/A")

        gpu_info = status_info.get('gpu_info', [])
        if gpu_info:
            for i, gpu in enumerate(gpu_info):
                gpu_name = gpu.get('name', 'N/A')
                gpu_type = gpu.get('type', 'N/A')
                msg_parts.append(f"• {config.COLOR_BLUE}GPU {i+1}:{config.COLOR_RESET} {gpu_name} [{gpu_type}]")
        else:
            msg_parts.append(f"• {config.COLOR_BLUE}GPU:{config.COLOR_RESET} N/A")

        vulkan_info = status_info.get('vulkan_info', 'N/A')
        opencl_info = status_info.get('opencl_info', 'N/A')
        msg_parts.append(f"• {config.COLOR_BLUE}Vulkan:{config.COLOR_RESET} {vulkan_info}")
        msg_parts.append(f"• {config.COLOR_BLUE}OpenCL:{config.COLOR_RESET} {opencl_info}")

        ollama_status = status_info.get('ollama_status', 'N/A')
        msg_parts.append(f"• {config.COLOR_BLUE}Ollama Server:{config.COLOR_RESET} {ollama_status}")

        db_status = status_info.get('db_status', {})
        if db_status.get('connected'):
            msg_parts.append(f"• {config.COLOR_BLUE}Database:{config.COLOR_RESET} Connected (Tables: {', '.join(db_status.get('tables', []))})")
        else:
            msg_parts.append(f"• {config.COLOR_BLUE}Database:{config.COLOR_RESET} Not Connected ({db_status.get('error', 'Unknown error')})")

        return "\n".join(msg_parts)

    # OS Information
    def _get_os_info(self) -> str:
        return f"Arch Linux {platform.machine()}"

    # Kernel Information
    def _get_kernel_info(self) -> str:
        return f"Linux {platform.release()}"

    # Uptime Calculation
    def _get_uptime(self) -> str:
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

        if not uptime_parts:
            return "Less than a minute"

        return ", ".join(uptime_parts)

    # Board Information
    def _get_board_info(self) -> str:
        return "ROG STRIX B650-A GAMING WIFI (Rev 1.xx)"

    # Memory Information
    def _get_memory_info(self) -> Dict[str, Union[int, float]]:
        mem = psutil.virtual_memory()
        return {
            'total': mem.total,
            'available': mem.available,
            'percent': float(mem.percent),
            'used': mem.used
        }

    # Detailed CPU Information
    def _get_cpu_info_detailed(self) -> Dict[str, Union[float, int, str]]:
        cpu_name = "N/A"
        cpu_speed = "N/A"
        try:
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if "model name" in line:
                        cpu_name = line.split(':')[-1].strip()
                    if "cpu MHz" in line:
                        mhz = float(line.split(':')[-1].strip())
                        cpu_speed = f"{mhz / 1000:.2f} GHz" if mhz >= 1000 else f"{mhz:.0f} MHz"
                    if cpu_name != "N/A" and cpu_speed != "N/A":
                        break
        except Exception as e:
            logger.warning(f"Could not read CPU info from /proc/cpuinfo: {e}")

        return {
            'name': cpu_name,
            'speed': cpu_speed,
            'percent': psutil.cpu_percent(interval=1),
            'cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True)
        }

    # Disk Usage Information
    def _get_all_disk_usage(self) -> List[Dict[str, Any]]:
        disk_mounts = config.DISK_MOUNTS

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
                    'percent': float(usage.percent)
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

    # GPU Details
    def _get_gpu_details(self) -> List[Dict[str, Any]]:
        gpu_data = []

        gpu_data.append({
            'name': 'AMD Radeon Graphics',
            'type': 'Integrated',
            'utilization_gpu_percent': 'N/A',
            'memory_total_mb': 'N/A',
            'memory_used_mb': 'N/A',
            'memory_free_mb': 'N/A'
        })

        try:
            cmd = ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=config.TIMEOUT_SECONDS)
            lines = result.stdout.strip().split('\n')
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) == 5:
                    gpu_data.append({
                        'name': parts[0],
                        'type': 'Discrete',
                        'utilization_gpu_percent': float(parts[1]),
                        'memory_total_mb': float(parts[2]),
                        'memory_used_mb': float(parts[3]),
                        'memory_free_mb': float(parts[4])
                    })
        except FileNotFoundError:
            logger.info("nvidia-smi not found. Skipping NVIDIA GPU info retrieval.")
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi command timed out.")
        except Exception as e:
            logger.error(f"Error retrieving NVIDIA GPU info: {e}")

        return gpu_data

    # Vulkan Information
    def _get_vulkan_info(self) -> str:
        return "1.4.311 - NVIDIA [575.64.03] radv [Mesa 25.1.6-arch1.1]"

    # OpenCL Information
    def _get_opencl_info(self) -> str:
        return "3.0 CUDA 12.9.90"

    # Ollama Server Status Check
    def _check_ollama_status(self) -> str:
        try:
            subprocess.run(["nc", "-z", "localhost", "11434"], check=True, capture_output=True, timeout=5)
            return "Running"
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return "Not Running"
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return "Error"

    # Command Generation
    def generate_command(self, query: str) -> Tuple[str, Optional[str]]:
        system_prompt = config.COMMAND_GENERATION_SYSTEM_PROMPT

        payload = {
            "model": config.DEFAULT_COMMAND_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            "stream": False
        }
        try:
            model_to_use, error_msg = utils.check_ollama_model_availability(
                config.DEFAULT_COMMAND_MODEL,
                fallback_model="llama2:7b-chat"
            )
            if error_msg:
                raise RuntimeError(error_msg)

            payload["model"] = model_to_use

            response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=config.TIMEOUT_SECONDS)
            response.raise_for_status()
            raw_command = response.json()["message"]["content"].strip()

            logger.debug(f"Raw command from LLM: '{raw_command}'")

            code_block_match = re.search(r'```(?:bash|sh)?\n(.*?)```', raw_command, re.DOTALL)
            if code_block_match:
                clean_command = code_block_match.group(1).strip()
            else:
                clean_command = raw_command

            clean_command = re.sub(r'(User:|Assistant:)\s*', '', clean_command, flags=re.IGNORECASE).strip()
            clean_command = re.sub(r'\s*\n\s*', ' ', clean_command).strip()

            command_pattern_match = re.match(r'^\s*([a-zA-Z0-9_./-]+(?:\s+[^&;|`\n]*)*)', clean_command)
            if command_pattern_match:
                clean_command = command_pattern_match.group(1).strip()
            else:
                conversational_phrases = [
                    r'That covers.*',
                    r'Here is the command.*',
                    r'The command is.*',
                    r'Here\'s the command.*',
                    r'This is the command.*',
                    r'I can only provide raw shell commands.*',
                    r'Feel free to ask.*',
                    r'Just remember that I can only provide raw commands.*',
                    r'Keep in mind that I can only provide raw shell commands.*',
                    r'If you need help with more specific tasks.*',
                    r'Please find the command below.*',
                    r'The requested command is.*',
                    r'Here you go.*',
                    r'Here\'s what you asked for.*',
                    r'As per your request.*',
                    r'This should do the trick.*',
                    r'```[\s\S]*?```'
                ]
                for phrase in conversational_phrases:
                    clean_command = re.sub(phrase, '', clean_command, flags=re.IGNORECASE | re.DOTALL).strip()

            clean_command = clean_command.strip('"').strip("'").strip()

            if not clean_command:
                lines = raw_command.strip().split('\n')
                for line in reversed(lines):
                    stripped_line = line.strip()
                    if stripped_line and not re.match(r'(User:|Assistant:)', stripped_line, re.IGNORECASE):
                        clean_command = stripped_line
                        break
                clean_command = clean_command.strip('"').strip("'").strip()


            logger.debug(f"Cleaned command for validation: '{clean_command}'")

            if any(clean_command.startswith(cmd) for cmd in config.SAFE_COMMAND_ALLOWLIST):
                return clean_command, None

            unsafe_operators = ['&&', ';', '||', '`', '\n']
            if any(op in clean_command for op in unsafe_operators):
                logger.warning(f"Unsafe command filtered: {clean_command}")
                return "", "ERROR: Generated command contained unsafe operators."

            if not clean_command:
                return "", "ERROR: Empty command generated"

            return clean_command, None
        except Exception as e:
            return "", f"Failed to generate command: {e}"

    # Command Execution
    def execute_command(self, command: str, cwd: Optional[str] = None) -> Tuple[bool, str, str]:
        try:
            expanded_command = command.replace("~", os.path.expanduser("~"))
            expanded_command = expanded_command.replace("$HOME", os.path.expanduser("~"))
            expanded_command = expanded_command.replace("$USER", os.getenv('USER', ''))


            command_parts = shlex.split(expanded_command)
            result = subprocess.run(
                command_parts,
                text=True,
                capture_output=True,
                check=True,
                cwd=cwd,
                timeout=config.TIMEOUT_SECONDS
            )
            return True, result.stdout.strip(), result.stderr.strip()
        except subprocess.CalledProcessError as e:
            return False, e.stdout.strip(), e.stderr.strip()
        except subprocess.TimeoutExpired:
            return False, "", f"Command timed out after {config.TIMEOUT_SECONDS} seconds."
        except Exception as e:
            return False, "", str(e)
