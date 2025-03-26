"""
Factory for creating language-specific analyzers.
"""

import logging
import os
from typing import Optional

from config import AgentConfig
from analyzers.base_analyzer import BaseAnalyzer
from analyzers.python_analyzer import PythonAnalyzer
from analyzers.javascript_analyzer import JavaScriptAnalyzer
from analyzers.typescript_analyzer import TypeScriptAnalyzer
from analyzers.java_analyzer import JavaAnalyzer
from analyzers.cpp_analyzer import CppAnalyzer

logger = logging.getLogger(__name__)

def create_analyzer(file_path: str, config: AgentConfig) -> Optional[BaseAnalyzer]:
    """
    Create a language-specific analyzer for the given file.
    
    Args:
        file_path: Path to the file to analyze
        config: Agent configuration
        
    Returns:
        Analyzer instance or None if the file type is not supported
    """
    extension = os.path.splitext(file_path)[1].lower()
    
    # Map file extensions to analyzer classes
    extension_map = {
        ".py": PythonAnalyzer,
        ".js": JavaScriptAnalyzer,
        ".jsx": JavaScriptAnalyzer,
        ".ts": TypeScriptAnalyzer,
        ".tsx": TypeScriptAnalyzer,
        ".java": JavaAnalyzer,
        ".cpp": CppAnalyzer,
        ".cc": CppAnalyzer,
        ".cxx": CppAnalyzer,
        ".c": CppAnalyzer,
        ".h": CppAnalyzer,
        ".hpp": CppAnalyzer
    }
    
    analyzer_class = extension_map.get(extension)
    if not analyzer_class:
        logger.debug(f"No analyzer available for file type: {extension}")
        return None
    
    analyzer = analyzer_class(file_path, config)
    
    # Check if the file should be ignored based on configuration
    if analyzer.should_ignore_file():
        logger.debug(f"Ignoring file based on configuration: {file_path}")
        return None
    
    return analyzer
