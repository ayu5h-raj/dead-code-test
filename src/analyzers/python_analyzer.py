"""
Python-specific analyzer for dead code detection.
"""

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from analyzers.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class PythonAnalyzer(BaseAnalyzer):
    """
    Analyzer for detecting dead code in Python files.
    """
    
    def analyze(self) -> List[Dict[str, Any]]:
        """
        Analyze the Python file for dead code.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        # Check if file exists
        if not os.path.exists(self.file_path):
            logger.warning(f"File not found: {self.file_path}")
            return []
        
        # Use multiple tools for better accuracy
        vulture_findings = self._analyze_with_vulture()
        pylint_findings = self._analyze_with_pylint()
        
        # Combine and deduplicate findings
        all_findings = vulture_findings + pylint_findings
        
        # Deduplicate based on line ranges
        deduplicated = self._deduplicate_findings(all_findings)
        
        # Filter by confidence threshold
        min_confidence = self.language_config.min_confidence if self.language_config else 0.8
        filtered_findings = [f for f in deduplicated if f.get("confidence", 0) >= min_confidence]
        
        return filtered_findings
    
    def _analyze_with_vulture(self) -> List[Dict[str, Any]]:
        """
        Analyze the file using Vulture.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        try:
            # Create a temporary file to store Vulture output
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Run Vulture on the file
            cmd = ["vulture", self.file_path, "--min-confidence", "60"]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse Vulture output
                if result.stdout:
                    lines = result.stdout.strip().split('\n')
                    
                    for line in lines:
                        if not line or ":" not in line:
                            continue
                        
                        try:
                            # Parse the line (format: file.py:line: item 'name' (confidence))
                            parts = line.split(":", 2)
                            if len(parts) < 3:
                                continue
                            
                            file_part, line_num, description = parts
                            
                            # Extract line number
                            line_num = int(line_num.strip())
                            
                            # Extract item type and name
                            description = description.strip()
                            item_type = "unknown"
                            item_name = ""
                            confidence = 0.6  # Default confidence
                            
                            if "function" in description:
                                item_type = "function"
                            elif "variable" in description:
                                item_type = "variable"
                            elif "class" in description:
                                item_type = "class"
                            elif "import" in description:
                                item_type = "import"
                            
                            # Extract name (between quotes)
                            import re
                            name_match = re.search(r"'([^']*)'", description)
                            if name_match:
                                item_name = name_match.group(1)
                            
                            # Extract confidence
                            confidence_match = re.search(r"\((\d+)%\)", description)
                            if confidence_match:
                                confidence = int(confidence_match.group(1)) / 100
                            
                            # Read the file to get the context
                            with open(self.file_path, 'r') as f:
                                file_lines = f.readlines()
                            
                            # Determine the end line (simple heuristic)
                            end_line = line_num
                            for i in range(line_num, min(line_num + 20, len(file_lines))):
                                if i >= len(file_lines):
                                    break
                                if file_lines[i].strip() == "" or file_lines[i].strip().startswith(("def ", "class ")):
                                    end_line = i - 1
                                    break
                            
                            # Get the code context
                            context_start = max(0, line_num - 1)
                            context_end = min(len(file_lines), end_line + 1)
                            code_context = "".join(file_lines[context_start:context_end])
                            
                            # Calculate line count
                            line_count = end_line - line_num + 1
                            
                            findings.append({
                                "file_path": os.path.relpath(self.file_path),
                                "type": item_type,
                                "name": item_name,
                                "start_line": line_num,
                                "end_line": end_line,
                                "confidence": confidence,
                                "reason": f"Detected as unused {item_type} by Vulture",
                                "code_context": code_context,
                                "line_count": line_count,
                                "tool": "vulture"
                            })
                            
                        except Exception as e:
                            logger.warning(f"Error parsing Vulture output line '{line}': {e}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Vulture analysis timed out for {self.file_path}")
            except subprocess.SubprocessError as e:
                logger.warning(f"Error running Vulture on {self.file_path}: {e}")
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
        except Exception as e:
            logger.warning(f"Error in Vulture analysis for {self.file_path}: {e}")
        
        return findings
    
    def _analyze_with_pylint(self) -> List[Dict[str, Any]]:
        """
        Analyze the file using Pylint.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        try:
            # Run Pylint on the file
            cmd = ["pylint", "--disable=all", "--enable=unused-import,unused-variable,unused-argument,unused-wildcard-import", 
                   "--output-format=json", self.file_path]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse Pylint output
                if result.stdout:
                    import json
                    try:
                        pylint_results = json.loads(result.stdout)
                        
                        # Read the file to get the context
                        with open(self.file_path, 'r') as f:
                            file_lines = f.readlines()
                        
                        for item in pylint_results:
                            message_id = item.get("message-id", "")
                            line = item.get("line", 0)
                            column = item.get("column", 0)
                            message = item.get("message", "")
                            
                            item_type = "unknown"
                            if "unused-import" in message_id:
                                item_type = "import"
                                confidence = 0.9
                            elif "unused-variable" in message_id:
                                item_type = "variable"
                                confidence = 0.85
                            elif "unused-argument" in message_id:
                                item_type = "parameter"
                                confidence = 0.8
                            else:
                                confidence = 0.7
                            
                            # Extract name from message
                            import re
                            name_match = re.search(r"'([^']*)'", message)
                            item_name = name_match.group(1) if name_match else ""
                            
                            # Determine the end line (simple heuristic)
                            end_line = line
                            
                            # Get the code context
                            context_start = max(0, line - 1)
                            context_end = min(len(file_lines), line + 1)
                            code_context = "".join(file_lines[context_start:context_end])
                            
                            findings.append({
                                "file_path": os.path.relpath(self.file_path),
                                "type": item_type,
                                "name": item_name,
                                "start_line": line,
                                "end_line": end_line,
                                "confidence": confidence,
                                "reason": message,
                                "code_context": code_context,
                                "line_count": 1,
                                "tool": "pylint"
                            })
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing Pylint JSON output for {self.file_path}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Pylint analysis timed out for {self.file_path}")
            except subprocess.SubprocessError as e:
                logger.warning(f"Error running Pylint on {self.file_path}: {e}")
                
        except Exception as e:
            logger.warning(f"Error in Pylint analysis for {self.file_path}: {e}")
        
        return findings
    
    def _deduplicate_findings(self, findings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate findings based on line ranges.
        
        Args:
            findings: List of findings to deduplicate
            
        Returns:
            Deduplicated list of findings
        """
        if not findings:
            return []
        
        # Sort by start line
        sorted_findings = sorted(findings, key=lambda x: (x.get("start_line", 0), x.get("end_line", 0)))
        
        deduplicated = []
        current = sorted_findings[0]
        
        for i in range(1, len(sorted_findings)):
            next_finding = sorted_findings[i]
            
            # Check if the next finding overlaps with the current one
            if (next_finding.get("start_line", 0) <= current.get("end_line", 0) + 1 and
                next_finding.get("file_path") == current.get("file_path")):
                # Merge the findings
                current["end_line"] = max(current.get("end_line", 0), next_finding.get("end_line", 0))
                current["confidence"] = max(current.get("confidence", 0), next_finding.get("confidence", 0))
                current["line_count"] = current["end_line"] - current["start_line"] + 1
                
                # Combine reasons
                if current.get("reason") != next_finding.get("reason"):
                    current["reason"] = f"{current.get('reason')}; {next_finding.get('reason')}"
                
                # Combine tools
                if current.get("tool") != next_finding.get("tool"):
                    current["tool"] = f"{current.get('tool')}, {next_finding.get('tool')}"
            else:
                deduplicated.append(current)
                current = next_finding
        
        deduplicated.append(current)
        return deduplicated
