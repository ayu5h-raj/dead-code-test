"""
Core agent implementation for dead code detection using Langgraph.
"""

import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from config import AgentConfig
from github_integration import GitHubIntegration
from analyzers.analyzer_factory import create_analyzer
from utils.code_utils import clone_repository, filter_files
from utils.report_generator import generate_report

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """Result of the dead code analysis."""
    success: bool
    pr_url: Optional[str] = None
    error: Optional[str] = None
    modified_files: List[str] = field(default_factory=list)
    
    @dataclass
    class Stats:
        """Statistics about the analysis."""
        files_analyzed: int = 0
        dead_code_found: int = 0
        files_with_dead_code: int = 0
        lines_removable: int = 0
    
    stats: Stats = field(default_factory=Stats)

class AgentState(TypedDict):
    """State for the dead code detection agent."""
    repo: str
    branch: str
    local_path: Optional[str]
    files_to_analyze: List[str]
    dead_code_findings: List[Dict[str, Any]]
    changes_to_make: List[Dict[str, Any]]
    analysis_complete: bool
    pr_created: bool
    errors: List[str]
    stats: Dict[str, int]
    temp_dir_created: bool
    dry_run: bool

class DeadCodeAgent:
    """
    AI agent for detecting dead code and generating PRs using Langgraph.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the agent with the given configuration.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.llm = ChatOpenAI(
            model_name=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0.1,
        )
        self.github = GitHubIntegration(
            token=os.getenv("GITHUB_TOKEN"),
            config=config.github
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the Langgraph execution graph for the agent.
        
        Returns:
            Configured Langgraph graph
        """
        # Define the nodes in our graph
        def initialize_state(state: AgentState) -> AgentState:
            """Initialize the agent state with repository information."""
            # State is already initialized when calling run()
            return state
        
        def fetch_repository(state: AgentState) -> AgentState:
            """Fetch or use the local repository."""
            try:
                if not state.get("local_path"):
                    # Clone the repository to a temporary directory
                    if not state.get("repo"):
                        raise ValueError("Either repo or local_path must be provided")
                        
                    repo_path = clone_repository(
                        repo=state["repo"],
                        branch=state["branch"]
                    )
                    
                    return {
                        **state,
                        "local_path": repo_path,
                        "temp_dir_created": True
                    }
                else:
                    local_path = state["local_path"]
                    # Verify the local path exists and is a directory
                    if not os.path.isdir(local_path):
                        raise ValueError(f"Local path {local_path} is not a valid directory")
                        
                    # If repo is not provided, try to infer it from the git remote
                    if not state.get("repo"):
                        try:
                            # Try to get the repo name from git remote
                            import subprocess
                            result = subprocess.run(
                                ["git", "remote", "get-url", "origin"],
                                cwd=local_path,
                                capture_output=True,
                                text=True,
                                check=False
                            )
                            
                            if result.returncode == 0:
                                remote_url = result.stdout.strip()
                                # Extract repo name from URL (e.g., https://github.com/username/repo.git or git@github.com:username/repo.git)
                                if "github.com" in remote_url:
                                    if remote_url.startswith("https://"):
                                        repo_parts = remote_url.split("github.com/")[1].split(".git")[0]
                                    else:  # SSH format
                                        repo_parts = remote_url.split("github.com:")[1].split(".git")[0]
                                        
                                    logger.info(f"Inferred repo name from git remote: {repo_parts}")
                                    state = {**state, "repo": repo_parts}
                        except Exception as e:
                            logger.warning(f"Could not infer repo name from git remote: {e}")
                            # Continue without repo name, which means we'll operate in local-only mode
                            
                    logger.info(f"Using existing repository at {state['local_path']}")
                    
                    return {
                        **state,
                        "temp_dir_created": False
                    }
            except Exception as e:
                logger.error(f"Failed to fetch repository: {str(e)}")
                
                return {
                    **state,
                    "errors": state["errors"] + [f"Failed to fetch repository: {str(e)}"]
                }
        
        def discover_files(state: AgentState) -> AgentState:
            """Discover files to analyze in the repository."""
            try:
                repo_path = state["local_path"]
                files = filter_files(
                    repo_path=repo_path,
                    config=self.config
                )
                
                logger.info(f"Discovered {len(files)} files to analyze")
                
                return {
                    **state,
                    "files_to_analyze": files
                }
            except Exception as e:
                logger.error(f"Failed to discover files: {str(e)}")
                
                return {
                    **state,
                    "errors": state["errors"] + [f"Failed to discover files: {str(e)}"]
                }
        
        def analyze_code(state: AgentState) -> AgentState:
            """Analyze code to find dead code."""
            try:
                repo_path = state["local_path"]
                files = state["files_to_analyze"]
                
                all_findings = []
                files_with_dead_code = set()
                total_dead_code = 0
                total_lines_removable = 0
                
                # Process files in batches for better performance
                batch_size = self.config.performance.batch_size
                for i in range(0, len(files), batch_size):
                    batch = files[i:i+batch_size]
                    
                    for file_path in batch:
                        try:
                            relative_path = os.path.relpath(file_path, repo_path)
                            analyzer = create_analyzer(file_path, self.config)
                            
                            if analyzer:
                                findings = analyzer.analyze()
                                
                                if findings:
                                    all_findings.extend(findings)
                                    files_with_dead_code.add(relative_path)
                                    total_dead_code += len(findings)
                                    
                                    # Log details about each finding
                                    logger.info(f"Found {len(findings)} instances of dead code in {relative_path}")
                                    for idx, finding in enumerate(findings):
                                        # Handle both line_number and start_line formats
                                        line_number = finding.get("line_number")
                                        if line_number is None:
                                            line_number = finding.get("start_line", 0)
                                        
                                        logger.info(f"  Dead code #{idx+1}: {finding.get('type', 'unknown')} '{finding.get('name', 'unnamed')}' at line {line_number}")
                                        if finding.get("reason"):
                                            logger.info(f"    Reason: {finding.get('reason')}")
                                        if finding.get("code_context"):
                                            # Limit context to first 100 chars to avoid huge logs
                                            context = finding.get('code_context', '')[:100]
                                            if len(finding.get('code_context', '')) > 100:
                                                context += "..."
                                            logger.info(f"    Context: {context}")
                                    
                                    # Calculate removable lines
                                    for finding in findings:
                                        if finding.get("line_count"):
                                            total_lines_removable += finding["line_count"]
                            
                            state["stats"]["files_analyzed"] += 1
                            
                        except Exception as e:
                            logger.warning(f"Error analyzing {file_path}: {str(e)}")
                
                # Update statistics
                updated_stats = {
                    **state["stats"],
                    "files_analyzed": state["stats"]["files_analyzed"],
                    "dead_code_found": total_dead_code,
                    "files_with_dead_code": len(files_with_dead_code),
                    "lines_removable": total_lines_removable
                }
                
                logger.info(f"Analysis complete. Found {total_dead_code} instances of dead code in {len(files_with_dead_code)} files.")
                
                return {
                    **state,
                    "dead_code_findings": all_findings,
                    "stats": updated_stats,
                    "analysis_complete": True
                }
            except Exception as e:
                logger.error(f"Failed to analyze code: {str(e)}")
                
                return {
                    **state,
                    "errors": state["errors"] + [f"Failed to analyze code: {str(e)}"]
                }
        
        def generate_changes(state: AgentState) -> AgentState:
            """Generate changes to remove dead code using LLM."""
            try:
                if not state["dead_code_findings"]:
                    logger.info("No dead code found. No changes to make.")
                    return {
                        **state,
                        "changes_to_make": []
                    }
                
                findings = state["dead_code_findings"]
                repo_path = state["local_path"]
                
                # Group findings by file
                findings_by_file = {}
                for finding in findings:
                    file_path = finding["file_path"]
                    # Ensure file_path is relative to repo_path
                    if os.path.isabs(file_path):
                        try:
                            file_path = os.path.relpath(file_path, repo_path)
                        except ValueError:
                            # Keep the original path if relpath fails
                            pass
                    
                    # Normalize the path to avoid issues with relative paths
                    file_path = os.path.normpath(file_path)
                    
                    if file_path not in findings_by_file:
                        findings_by_file[file_path] = []
                    findings_by_file[file_path].append(finding)
                
                # Generate changes for each file
                all_changes = []
                
                for file_path, file_findings in findings_by_file.items():
                    try:
                        # Ensure we have the absolute path for reading the file
                        abs_file_path = os.path.join(repo_path, file_path)
                        abs_file_path = os.path.normpath(abs_file_path)
                        
                        if not os.path.exists(abs_file_path):
                            logger.warning(f"File not found: {abs_file_path}")
                            continue
                            
                        # Read the file content
                        with open(abs_file_path, "r", encoding="utf-8", errors="replace") as f:
                            file_content = f.read()
                        
                        # Prepare the prompt for the LLM
                        prompt = self._prepare_change_prompt(file_path, file_content, file_findings)
                        
                        # Call the LLM to generate changes
                        messages = [
                            SystemMessage(content="""You are an expert code refactoring assistant. 
                            Your task is to remove dead code from files while maintaining the functionality of the codebase.
                            Provide the exact changes needed to remove the dead code."""),
                            HumanMessage(content=prompt)
                        ]
                        
                        response = self.llm.invoke(messages)
                        
                        # Parse the response to get the changes
                        changes = self._parse_llm_response(file_path, response.content, file_findings)
                        
                        if changes:
                            all_changes.extend(changes)
                    
                    except Exception as e:
                        logger.warning(f"Error generating changes for {file_path}: {str(e)}")
                
                logger.info(f"Generated {len(all_changes)} changes to remove dead code.")
                
                return {
                    **state,
                    "changes_to_make": all_changes
                }
            except Exception as e:
                logger.error(f"Failed to generate changes: {str(e)}")
                
                return {
                    **state,
                    "errors": state["errors"] + [f"Failed to generate changes: {str(e)}"]
                }
        
        def create_pull_request(state: AgentState) -> AgentState:
            """Create a pull request with the changes."""
            try:
                if not state["changes_to_make"]:
                    logger.info("No changes to make. Skipping PR creation.")
                    return state
                
                if state.get("dry_run", False):
                    logger.info("Dry run mode. Skipping PR creation.")
                    return state
                
                # Apply changes to files
                repo_path = state["local_path"]
                changes = state["changes_to_make"]
                
                # Track modified files for reporting
                modified_files = []
                
                for change in changes:
                    file_path = os.path.join(repo_path, change["file_path"])
                    with open(file_path, "w") as f:
                        f.write(change["new_content"])
                    modified_files.append(change["file_path"])
                
                # Check if we have a GitHub repo to create a PR
                if state.get("repo"):
                    # Create a PR
                    pr_url = self.github.create_pull_request(
                        repo=state["repo"],
                        branch=state["branch"],
                        title="Remove Dead Code",
                        body=generate_report(state["dead_code_findings"], state["stats"]),
                        local_path=repo_path
                    )
                    
                    logger.info(f"Created PR: {pr_url}")
                    
                    return {
                        **state,
                        "pr_created": True,
                        "pr_url": pr_url
                    }
                else:
                    # Local-only mode - just apply changes without creating a PR
                    logger.info(f"Local-only mode: Applied changes to {len(modified_files)} files")
                    for file in modified_files:
                        logger.info(f"  - Modified: {file}")
                    
                    return {
                        **state,
                        "pr_created": False,
                        "modified_files": modified_files
                    }
            except Exception as e:
                logger.error(f"Failed to create pull request: {str(e)}")
                
                return {
                    **state,
                    "errors": state["errors"] + [f"Failed to create pull request: {str(e)}"]
                }
        
        def cleanup(state: AgentState) -> AgentState:
            """Clean up temporary resources."""
            try:
                if state.get("temp_dir_created") and state.get("local_path"):
                    # Remove the temporary directory
                    temp_dir = state["local_path"]
                    if os.path.exists(temp_dir):
                        import shutil
                        shutil.rmtree(temp_dir)
                        logger.info(f"Cleaned up temporary directory: {temp_dir}")
                
                return state
            except Exception as e:
                logger.warning(f"Failed to clean up: {str(e)}")
                return state
        
        # Build the graph
        graph_builder = StateGraph(AgentState)
        
        # Add nodes
        graph_builder.add_node("initialize", initialize_state)
        graph_builder.add_node("fetch_repository", fetch_repository)
        graph_builder.add_node("discover_files", discover_files)
        graph_builder.add_node("analyze_code", analyze_code)
        graph_builder.add_node("generate_changes", generate_changes)
        graph_builder.add_node("create_pull_request", create_pull_request)
        graph_builder.add_node("cleanup", cleanup)
        
        # Connect nodes
        graph_builder.add_edge("initialize", "fetch_repository")
        graph_builder.add_edge("fetch_repository", "discover_files")
        graph_builder.add_edge("discover_files", "analyze_code")
        graph_builder.add_edge("analyze_code", "generate_changes")
        graph_builder.add_edge("generate_changes", "create_pull_request")
        graph_builder.add_edge("create_pull_request", "cleanup")
        graph_builder.add_edge("cleanup", END)
        
        # Add conditional edges for error handling
        graph_builder.add_conditional_edges(
            "fetch_repository",
            lambda state: "cleanup" if state.get("errors") else "discover_files"
        )
        graph_builder.add_conditional_edges(
            "discover_files",
            lambda state: "cleanup" if state.get("errors") else "analyze_code"
        )
        graph_builder.add_conditional_edges(
            "analyze_code",
            lambda state: "cleanup" if state.get("errors") else "generate_changes"
        )
        graph_builder.add_conditional_edges(
            "generate_changes",
            lambda state: "cleanup" if state.get("errors") else "create_pull_request"
        )
        
        # Set the entry point
        graph_builder.set_entry_point("initialize")
        
        return graph_builder.compile()
    
    def _prepare_change_prompt(self, file_path: str, file_content: str, findings: List[Dict[str, Any]]) -> str:
        """Prepare a prompt for the LLM to generate changes."""
        prompt = f"File: {file_path}\n\n"
        prompt += "Original content:\n```\n" + file_content + "\n```\n\n"
        prompt += "Dead code findings:\n"
        
        for i, finding in enumerate(findings):
            # Handle both line_number and start_line formats
            line_number = finding.get("line_number")
            if line_number is None:
                line_number = finding.get("start_line", 0)
            
            prompt += f"{i+1}. {finding['type']} '{finding['name']}' at line {line_number}"
            if finding.get("context"):
                prompt += f"\nContext: {finding['context']}"
            elif finding.get("code_context"):
                prompt += f"\nContext: {finding['code_context']}"
            prompt += "\n"
        
        prompt += "\nPlease remove the dead code from this file. Provide the new file content with the dead code removed."
        return prompt
    
    def _parse_llm_response(self, file_path: str, response: str, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse the LLM response to extract changes."""
        # Extract the code block from the response
        import re
        code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", response, re.DOTALL)
        
        if not code_blocks:
            # Try alternative format without language specifier and with spaces
            code_blocks = re.findall(r"```\s*(.*?)\s*```", response, re.DOTALL)
            
        if not code_blocks:
            logger.warning(f"No code blocks found in LLM response for {file_path}")
            return []
        
        # Use the largest code block as the new content
        new_content = max(code_blocks, key=len)
        
        # Clean up the content if needed
        new_content = new_content.strip()
        
        # Verify the content is not empty
        if not new_content:
            logger.warning(f"Empty code block found in LLM response for {file_path}")
            return []
            
        logger.info(f"Successfully extracted code changes for {file_path}")
        
        return [{
            "file_path": file_path,
            "new_content": new_content,
            "findings": findings
        }]
    
    def _create_result_from_state(self, state: Dict[str, Any]) -> AnalysisResult:
        """Create an analysis result from the state."""
        success = not state.get("errors")
        error = "; ".join(state.get("errors", [])) if state.get("errors") else None
        
        result = AnalysisResult(
            success=success,
            pr_url=state.get("pr_url"),
            error=error,
            modified_files=state.get("modified_files", [])
        )
        
        result.stats.files_analyzed = state.get("stats", {}).get("files_analyzed", 0)
        result.stats.dead_code_found = state.get("stats", {}).get("dead_code_found", 0)
        result.stats.files_with_dead_code = state.get("stats", {}).get("files_with_dead_code", 0)
        result.stats.lines_removable = state.get("stats", {}).get("lines_removable", 0)
        
        return result
    
    def run(
        self,
        repo: str,
        branch: str = "main",
        local_path: Optional[str] = None,
        dry_run: bool = False
    ) -> AnalysisResult:
        """
        Run the dead code detection agent on a repository.
        
        Args:
            repo: GitHub repository in format 'username/repository'
            branch: Branch to analyze
            local_path: Path to local repository (if already cloned)
            dry_run: If True, don't create a PR
            
        Returns:
            Analysis result
        """
        try:
            # Initialize state
            initial_state = {
                "repo": repo,
                "branch": branch,
                "local_path": local_path,
                "files_to_analyze": [],
                "dead_code_findings": [],
                "changes_to_make": [],
                "analysis_complete": False,
                "pr_created": False,
                "errors": [],
                "stats": {
                    "files_analyzed": 0,
                    "dead_code_found": 0,
                    "files_with_dead_code": 0,
                    "lines_removable": 0
                },
                "temp_dir_created": False,
                "dry_run": dry_run,
                "pr_url": None
            }
            
            # Run the graph
            logger.info(f"Starting analysis of repository {repo} on branch {branch}")
            final_state = self.graph.invoke(initial_state)
            
            # Create and return the result
            return self._create_result_from_state(final_state)
            
        except Exception as e:
            logger.error(f"Error running agent: {str(e)}")
            return AnalysisResult(
                success=False,
                error=f"Error running agent: {str(e)}"
            )
