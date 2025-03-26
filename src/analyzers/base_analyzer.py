"""
Base analyzer class for dead code detection.
"""

import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from config import AgentConfig, LanguageConfig

logger = logging.getLogger(__name__)

class BaseAnalyzer(ABC):
    """
    Base class for language-specific code analyzers.
    """
    
    def __init__(self, file_path: str, config: AgentConfig):
        """
        Initialize the analyzer.
        
        Args:
            file_path: Path to the file to analyze
            config: Agent configuration
        """
        self.file_path = file_path
        self.config = config
        self.language_config = self._get_language_config()
    
    def _get_language_config(self) -> Optional[LanguageConfig]:
        """
        Get the language configuration for this file.
        
        Returns:
            Language configuration or None if not supported
        """
        extension = os.path.splitext(self.file_path)[1].lower()
        
        extension_to_language = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".jsx": "javascript",
            ".java": "java",
            ".cpp": "cpp",
            ".cc": "cpp",
            ".cxx": "cpp",
            ".c": "cpp",
            ".h": "cpp",
            ".hpp": "cpp"
        }
        
        language = extension_to_language.get(extension)
        if not language:
            return None
        
        language_config = self.config.analysis.languages.get(language)
        if not language_config or not language_config.enabled:
            return None
        
        return language_config
    
    def should_ignore_file(self) -> bool:
        """
        Check if the file should be ignored based on configuration.
        
        Returns:
            True if the file should be ignored, False otherwise
        """
        if not self.language_config:
            return True
        
        import fnmatch
        
        for pattern in self.language_config.ignore_patterns:
            if fnmatch.fnmatch(self.file_path, pattern):
                return True
        
        return False
    
    @abstractmethod
    def analyze(self) -> List[Dict[str, Any]]:
        """
        Analyze the file for dead code.
        
        Returns:
            List of dead code findings
        """
        pass
