"""
Code utilities for the Dead Code Detection Agent.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

from config import AgentConfig

logger = logging.getLogger(__name__)

def clone_repository(repo: str, branch: str = "main") -> str:
    """
    Clone a GitHub repository to a temporary directory.
    
    Args:
        repo: Repository name in format 'username/repository'
        branch: Branch to clone
        
    Returns:
        Path to the cloned repository
        
    Raises:
        ValueError: If the repository doesn't exist or can't be cloned
    """
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp(prefix="dead_code_agent_")
        
        # Clone the repository
        cmd = ["git", "clone", f"https://github.com/{repo}.git", "--branch", branch, "--single-branch", temp_dir]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info(f"Cloned repository {repo} (branch: {branch}) to {temp_dir}")
        return temp_dir
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository {repo}: {e.stderr}")
        raise ValueError(f"Failed to clone repository {repo}: {e.stderr}")
    except Exception as e:
        logger.error(f"Error cloning repository {repo}: {e}")
        raise ValueError(f"Error cloning repository {repo}: {e}")

def filter_files(repo_path: str, config: AgentConfig) -> List[str]:
    """
    Filter files in the repository based on configuration.
    
    Args:
        repo_path: Path to the repository
        config: Agent configuration
        
    Returns:
        List of file paths to analyze
    """
    files_to_analyze = []
    
    # Get supported languages from configuration
    supported_languages = [
        lang for lang, lang_config in config.analysis.languages.items()
        if lang_config.enabled
    ]
    
    # Map languages to file extensions
    language_extensions = {
        "python": [".py"],
        "javascript": [".js", ".jsx"],
        "typescript": [".ts", ".tsx"],
        "java": [".java"],
        "cpp": [".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"]
    }
    
    # Get all extensions to look for
    extensions = []
    for lang in supported_languages:
        extensions.extend(language_extensions.get(lang, []))
    
    # Walk through the repository
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and common directories to ignore
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ["node_modules", "venv", "build", "dist"]]
        
        for file in files:
            # Check if the file has a supported extension
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                
                # Check if the file should be ignored based on language-specific patterns
                should_ignore = False
                for lang, lang_exts in language_extensions.items():
                    if any(file.endswith(ext) for ext in lang_exts):
                        lang_config = config.analysis.languages.get(lang)
                        if lang_config and lang_config.enabled:
                            import fnmatch
                            rel_path = os.path.relpath(file_path, repo_path)
                            for pattern in lang_config.ignore_patterns:
                                if fnmatch.fnmatch(rel_path, pattern):
                                    should_ignore = True
                                    break
                        break
                
                if not should_ignore:
                    # Check file size
                    if os.path.getsize(file_path) <= config.performance.max_file_size_kb * 1024:
                        files_to_analyze.append(file_path)
                        # Log the file name with relative path
                        rel_path = os.path.relpath(file_path, repo_path)
                        logger.info(f"Adding file for analysis: {rel_path} ({get_file_language(file_path)})")
                    else:
                        logger.info(f"Skipping large file: {file_path}")
    
    logger.info(f"Found {len(files_to_analyze)} files to analyze")
    return files_to_analyze

def get_file_language(file_path: str) -> Optional[str]:
    """
    Determine the programming language of a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Language name or None if not recognized
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    extension_to_language = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".cxx": "cpp",
        ".c": "cpp",
        ".h": "cpp",
        ".hpp": "cpp"
    }
    
    return extension_to_language.get(extension)
