"""
Configuration module for the Dead Code Detection Agent.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class LanguageConfig(BaseModel):
    """Configuration for a specific programming language."""
    enabled: bool = True
    min_confidence: float = Field(0.8, ge=0.0, le=1.0)
    ignore_patterns: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)

class AnalysisConfig(BaseModel):
    """Configuration for code analysis."""
    languages: Dict[str, LanguageConfig] = Field(default_factory=dict)

class GitHubConfig(BaseModel):
    """Configuration for GitHub integration."""
    auto_create_pr: bool = True
    pr_title_template: str = "refactor: Remove dead code identified by AI analysis"
    pr_description_template: str = "This PR removes dead code identified through automated analysis."
    require_approval_before_pr: bool = True
    commit_message_template: str = "refactor: Remove dead code in {filename}"

class ReportingConfig(BaseModel):
    """Configuration for reporting."""
    detail_level: str = "high"
    include_context: bool = True
    context_lines: int = Field(5, ge=0)
    group_by: str = "file"
    
    @validator("detail_level")
    def validate_detail_level(cls, v):
        """Validate the detail level."""
        if v not in ["low", "medium", "high"]:
            raise ValueError(f"detail_level must be one of: low, medium, high. Got {v}")
        return v
    
    @validator("group_by")
    def validate_group_by(cls, v):
        """Validate the group by setting."""
        if v not in ["file", "type", "confidence"]:
            raise ValueError(f"group_by must be one of: file, type, confidence. Got {v}")
        return v

class PerformanceConfig(BaseModel):
    """Configuration for performance optimization."""
    parallel_processing: bool = True
    max_workers: int = Field(4, ge=1)
    timeout_seconds: int = Field(300, ge=1)
    max_file_size_kb: int = Field(1000, ge=1)
    batch_size: int = Field(50, ge=1)

class AgentConfig(BaseModel):
    """Main configuration for the Dead Code Detection Agent."""
    github: GitHubConfig = Field(default_factory=GitHubConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

def load_config(config_path: Path) -> AgentConfig:
    """
    Load and validate the configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Validated configuration object
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        ValueError: If the configuration is invalid
    """
    if not config_path.exists():
        # Check if example config exists and create a copy
        example_path = config_path.with_name(f"{config_path.name}.example")
        if example_path.exists():
            logger.warning(f"Configuration file {config_path} not found. Creating from example.")
            with open(example_path, "r") as example_file:
                example_content = example_file.read()
            
            with open(config_path, "w") as config_file:
                config_file.write(example_content)
        else:
            raise FileNotFoundError(f"Configuration file {config_path} not found")
    
    with open(config_path, "r") as file:
        config_dict = yaml.safe_load(file)
    
    try:
        return AgentConfig(**config_dict)
    except Exception as e:
        raise ValueError(f"Invalid configuration: {e}")

def get_language_config(config: AgentConfig, file_path: str) -> Optional[LanguageConfig]:
    """
    Get the language configuration for a specific file based on its extension.
    
    Args:
        config: Agent configuration
        file_path: Path to the file
        
    Returns:
        Language configuration or None if the language is not supported
    """
    extension = os.path.splitext(file_path)[1].lower()
    
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
    
    language_config = config.analysis.languages.get(language)
    if not language_config or not language_config.enabled:
        return None
    
    return language_config
