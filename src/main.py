#!/usr/bin/env python3
"""
Main entry point for the Dead Code Detection and PR Generation Agent.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from config import load_config
from agent import DeadCodeAgent
from utils.logger import setup_logger

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI agent for dead code detection and PR generation"
    )
    parser.add_argument(
        "--repo", 
        required=True, 
        help="GitHub repository in format 'username/repository'"
    )
    parser.add_argument(
        "--branch", 
        default="main", 
        help="Branch to analyze (default: main)"
    )
    parser.add_argument(
        "--config", 
        default="config.yaml", 
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--local-path", 
        help="Path to local repository (if already cloned)"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Analyze without creating a PR"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    return parser.parse_args()

def main():
    """Main function to run the dead code detection agent."""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    if args.verbose:
        log_level = "DEBUG"
    setup_logger(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Dead Code Detection Agent")
    
    # Validate environment variables
    if not os.getenv("GITHUB_TOKEN"):
        logger.error("GITHUB_TOKEN environment variable is required")
        sys.exit(1)
    
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    # Load configuration
    try:
        config_path = Path(args.config)
        config = load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Initialize and run the agent
    try:
        agent = DeadCodeAgent(config)
        result = agent.run(
            repo=args.repo,
            branch=args.branch,
            local_path=args.local_path,
            dry_run=args.dry_run
        )
        
        if result.success:
            logger.info(f"Analysis completed successfully")
            if result.pr_url and not args.dry_run:
                logger.info(f"Pull request created: {result.pr_url}")
            else:
                logger.info("No pull request was created (dry run mode)")
            
            print("\nSummary Report:")
            print(f"Files analyzed: {result.stats.files_analyzed}")
            print(f"Dead code instances found: {result.stats.dead_code_found}")
            print(f"Files with dead code: {result.stats.files_with_dead_code}")
            print(f"Total lines that could be removed: {result.stats.lines_removable}")
            
            return 0
        else:
            logger.error(f"Analysis failed: {result.error}")
            return 1
            
    except Exception as e:
        logger.exception(f"An error occurred during execution: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
