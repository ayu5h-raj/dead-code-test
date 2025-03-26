"""
GitHub integration module for the Dead Code Detection Agent.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

from github import Github, GithubException, InputGitAuthor
from github.Repository import Repository

from config import GitHubConfig

logger = logging.getLogger(__name__)

class GitHubIntegration:
    """
    Handles GitHub operations for the Dead Code Detection Agent.
    """
    
    def __init__(self, token: str, config: GitHubConfig):
        """
        Initialize the GitHub integration.
        
        Args:
            token: GitHub personal access token
            config: GitHub configuration
        """
        self.github = Github(token)
        self.config = config
    
    def get_repository(self, repo_name: str) -> Repository:
        """
        Get a GitHub repository.
        
        Args:
            repo_name: Repository name in format 'username/repository'
            
        Returns:
            GitHub repository object
            
        Raises:
            ValueError: If the repository doesn't exist
        """
        try:
            return self.github.get_repo(repo_name)
        except GithubException as e:
            logger.error(f"Failed to get repository {repo_name}: {e}")
            raise ValueError(f"Repository {repo_name} not found or access denied")
    
    def create_pull_request(
        self,
        repo: str,
        branch: str,
        changes: List[Dict[str, Any]],
        report: str,
        stats: Dict[str, Any]
    ) -> str:
        """
        Create a pull request with the changes.
        
        Args:
            repo: Repository name in format 'username/repository'
            branch: Base branch
            changes: List of changes to make
            report: Report to include in the PR description
            stats: Statistics about the analysis
            
        Returns:
            URL of the created pull request
            
        Raises:
            ValueError: If the PR creation fails
        """
        if not changes:
            raise ValueError("No changes to commit")
        
        repo_obj = self.get_repository(repo)
        
        # Create a new branch for the changes
        base_branch = repo_obj.get_branch(branch)
        new_branch_name = f"dead-code-removal-{base_branch.commit.sha[:7]}"
        
        try:
            # Check if branch already exists
            repo_obj.get_branch(new_branch_name)
            # If we get here, the branch exists, so we'll use a timestamp to make it unique
            import time
            new_branch_name = f"dead-code-removal-{int(time.time())}"
        except GithubException:
            # Branch doesn't exist, which is what we want
            pass
        
        try:
            # Create the new branch
            repo_obj.create_git_ref(
                ref=f"refs/heads/{new_branch_name}",
                sha=base_branch.commit.sha
            )
            
            # Get committer information
            committer = InputGitAuthor(
                name="Dead Code Detection Bot",
                email="noreply@deadcodebot.ai"
            )
            
            # Commit each change
            for change in changes:
                file_path = change["file_path"]
                modified_content = change["modified_content"]
                
                try:
                    # Get the current file content
                    contents = repo_obj.get_contents(file_path, ref=new_branch_name)
                    
                    # Create the commit message
                    commit_message = self.config.commit_message_template.format(
                        filename=file_path
                    )
                    
                    # Update the file
                    repo_obj.update_file(
                        path=contents.path,
                        message=commit_message,
                        content=modified_content,
                        sha=contents.sha,
                        branch=new_branch_name,
                        committer=committer
                    )
                    
                    logger.info(f"Committed changes to {file_path}")
                    
                except GithubException as e:
                    logger.error(f"Failed to update {file_path}: {e}")
                    # Continue with other files
            
            # Format the PR title and description
            pr_title = self.config.pr_title_template
            
            pr_description = self.config.pr_description_template.format(
                total_files=len(changes),
                removed_lines=stats.get("lines_removable", 0),
                analysis_summary=report
            )
            
            # Create the PR
            pr = repo_obj.create_pull(
                title=pr_title,
                body=pr_description,
                head=new_branch_name,
                base=branch
            )
            
            logger.info(f"Created PR: {pr.html_url}")
            return pr.html_url
            
        except GithubException as e:
            logger.error(f"Failed to create PR: {e}")
            raise ValueError(f"Failed to create PR: {str(e)}")
            
        except Exception as e:
            logger.exception(f"Error creating PR: {e}")
            raise ValueError(f"Error creating PR: {str(e)}")
