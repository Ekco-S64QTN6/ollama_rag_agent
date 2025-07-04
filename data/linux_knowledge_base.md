# Linux Knowledge Base for AI Assistant

This document contains general Linux commands, concepts, and troubleshooting tips. It is designed to provide foundational knowledge for a Linux desktop assistant without including any personally identifiable information (PII) or user-specific configurations.

## 1. Basic Linux Commands & Navigation

### File System
-   `pwd`: Print working directory (show current directory path)
-   `ls [options]`: List directory contents (`-l` for long format, `-a` for all files including hidden)
-   `cd [directory]`: Change directory
-   `mkdir [directory_name]`: Create a new directory
-   `rm [options] [file/directory]`: Remove files or directories (`-r` for recursive, `-f` for force)
-   `cp [source] [destination]`: Copy files or directories
-   `mv [source] [destination]`: Move (rename) files or directories
-   `touch [file_name]`: Create an empty file or update timestamp

### File Content
-   `cat [file]`: Display file content
-   `less [file]`: View file content page by page
-   `head [options] [file]`: Display the beginning of a file (`-n <lines>`)
-   `tail [options] [file]`: Display the end of a file (`-n <lines>`, `-f` for follow)

### Permissions & Ownership
-   `chmod [permissions] [file]`: Change file permissions (e.g., `chmod +x script.sh`, `chmod 755 file`)
-   `chown [user]:[group] [file]`: Change file ownership

## 2. System Management & Troubleshooting

### Systemd (Service Management)
-   `systemctl start [service]`: Start a service
-   `systemctl stop [service]`: Stop a service
-   `systemctl restart [service]`: Restart a service
-   `systemctl enable [service]`: Enable service to start on boot
-   `systemctl disable [service]`: Disable service from starting on boot
-   `systemctl status [service]`: Check service status
-   `journalctl -u [service]`: View logs for a specific systemd service
-   `journalctl -f`: Follow system logs in real-time

### Process Management
-   `ps aux`: Display running processes
-   `top`: Real-time view of system processes
-   `htop`: Interactive process viewer (often needs installation: `sudo pacman -S htop`)
-   `kill [PID]`: Terminate a process by ID
-   `killall [process_name]`: Terminate processes by name

### Disk Usage & Storage
-   `df -h`: Display free disk space in human-readable format
-   `du -sh [directory]`: Summarize disk usage of a directory
-   `fdisk -l`: List disk partitions (requires root)
-   `lsblk`: List block devices

## 3. Networking Basics

### Network Configuration
-   `ip a`: Display network interface information (IP addresses, etc.)
-   `ping [hostname/IP]`: Test network connectivity
-   `ss -tuln`: List listening sockets (ports)
-   `netstat -tuln`: (Deprecated, but still common) List listening sockets

### Remote Access
-   `ssh [user]@[host]`: Secure Shell for remote login
-   `scp [source] [destination]`: Secure copy files over SSH

## 4. Package Management (Arch Linux Focus)

-   **Pacman:**
    -   `sudo pacman -Syu`: Synchronize package databases and update system
    -   `sudo pacman -S [package]`: Install a package
    -   `sudo pacman -R [package]`: Remove a package
    -   `sudo pacman -Qs [keyword]`: Query local database for installed packages matching keyword
    -   `sudo pacman -Ss [keyword]`: Search remote repositories for packages
-   **AUR Helpers (e.g., Yay):**
    -   `yay -S [package]`: Install/build package from AUR
    -   `yay -Syu`: Update system including AUR packages

## 5. Text Processing & Utilities

-   `grep [pattern] [file]`: Search for patterns in files
-   `find [path] [expression]`: Search for files and directories
-   `sed`: Stream editor for text transformations
-   `awk`: Pattern-scanning and processing language
-   `tar [options] [archive] [files]`: Create or extract archive files (e.g., `tar -cvf`, `tar -xvf`)
-   `rsync [options] [source] [destination]`: Efficiently copy and synchronize files remotely or locally

## 6. General Best Practices & Concepts

-   **Security:** Keep your system updated, use strong unique passwords, understand firewall basics.
-   **Backup Strategy:** Emphasize regular backups of important data.
-   **The Unix Philosophy:** "Do one thing and do it well."
-   **Dotfiles:** Mention the concept of managing configuration files (dotfiles) for system personalization.
