"""
C++-specific analyzer for dead code detection.
"""

import logging
import os
import subprocess
import tempfile
import json
from typing import Any, Dict, List, Optional

from analyzers.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class CppAnalyzer(BaseAnalyzer):
    """
    Analyzer for detecting dead code in C++ files.
    """
    
    def analyze(self) -> List[Dict[str, Any]]:
        """
        Analyze the C++ file for dead code.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        # Check if file exists
        if not os.path.exists(self.file_path):
            logger.warning(f"File not found: {self.file_path}")
            return []
        
        # Use Clang static analyzer for dead code detection
        clang_findings = self._analyze_with_clang()
        
        # Use cppcheck for additional analysis
        cppcheck_findings = self._analyze_with_cppcheck()
        
        # Combine and deduplicate findings
        all_findings = clang_findings + cppcheck_findings
        
        # Deduplicate based on line ranges
        deduplicated = self._deduplicate_findings(all_findings)
        
        # Filter by confidence threshold
        min_confidence = self.language_config.min_confidence if self.language_config else 0.8
        filtered_findings = [f for f in deduplicated if f.get("confidence", 0) >= min_confidence]
        
        return filtered_findings
    
    def _analyze_with_clang(self) -> List[Dict[str, Any]]:
        """
        Analyze the file using Clang static analyzer.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        try:
            # Create a temporary file for Clang output
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as output_file:
                output_path = output_file.name
            
            # Run Clang static analyzer on the file
            cmd = [
                "clang-tidy",
                "-checks=clang-analyzer-deadcode.*,misc-unused-parameters,cppcoreguidelines-avoid-non-const-global-variables,readability-redundant-declaration",
                "-export-fixes=" + output_path,
                self.file_path,
                "--",
                "-std=c++17"
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse Clang output
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    try:
                        with open(output_path, 'r') as f:
                            clang_results = json.load(f)
                        
                        # Read the file to get the context
                        with open(self.file_path, 'r') as f:
                            file_lines = f.readlines()
                        
                        for diagnostic in clang_results.get("Diagnostics", []):
                            file_path = diagnostic.get("FilePath", "")
                            if not file_path or os.path.normpath(file_path) != os.path.normpath(self.file_path):
                                continue
                            
                            check_name = diagnostic.get("CheckName", "")
                            message = diagnostic.get("DiagnosticMessage", {}).get("Message", "")
                            
                            # Get line information
                            file_offset = diagnostic.get("FileOffset", 0)
                            
                            # This is a simplification - in a real implementation, we would need to
                            # map the file offset to line numbers more accurately
                            line = 1
                            offset = 0
                            for i, line_content in enumerate(file_lines):
                                if offset + len(line_content) > file_offset:
                                    line = i + 1
                                    break
                                offset += len(line_content)
                            
                            # Estimate end line (simple heuristic)
                            end_line = line
                            for i in range(line, min(line + 10, len(file_lines))):
                                if i >= len(file_lines):
                                    break
                                if file_lines[i].strip() == "" or file_lines[i].strip().startswith(("}",)):
                                    end_line = i
                                    break
                            
                            item_type = "unknown"
                            confidence = 0.7
                            item_name = ""
                            
                            if "deadcode" in check_name.lower():
                                item_type = "code"
                                confidence = 0.9
                            elif "unused-parameters" in check_name.lower():
                                item_type = "parameter"
                                confidence = 0.85
                                
                                # Extract parameter name from message
                                import re
                                name_match = re.search(r"'([^']*)'", message)
                                if name_match:
                                    item_name = name_match.group(1)
                            elif "non-const-global-variables" in check_name.lower():
                                item_type = "variable"
                                confidence = 0.8
                                
                                # Extract variable name from message
                                import re
                                name_match = re.search(r"'([^']*)'", message)
                                if name_match:
                                    item_name = name_match.group(1)
                            elif "redundant-declaration" in check_name.lower():
                                item_type = "declaration"
                                confidence = 0.85
                                
                                # Extract declaration name from message
                                import re
                                name_match = re.search(r"'([^']*)'", message)
                                if name_match:
                                    item_name = name_match.group(1)
                            
                            # Get the code context
                            context_start = max(0, line - 2)
                            context_end = min(len(file_lines), end_line + 2)
                            code_context = "".join(file_lines[context_start:context_end])
                            
                            findings.append({
                                "file_path": os.path.relpath(self.file_path),
                                "type": item_type,
                                "name": item_name,
                                "start_line": line,
                                "end_line": end_line,
                                "confidence": confidence,
                                "reason": f"Clang: {message} ({check_name})",
                                "code_context": code_context,
                                "line_count": end_line - line + 1,
                                "tool": "clang"
                            })
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing Clang JSON output for {self.file_path}")
                
                # Also parse the stderr output for additional findings
                if result.stderr:
                    for line in result.stderr.strip().split('\n'):
                        if self.file_path in line and ': warning:' in line:
                            try:
                                # Parse the line (format: file:line:col: warning: message [check-name])
                                parts = line.split(':', 3)
                                if len(parts) < 4:
                                    continue
                                
                                file_part, line_num, col_num, rest = parts
                                
                                # Extract line number
                                line = int(line_num.strip())
                                
                                # Extract message and check name
                                rest_parts = rest.split('[', 1)
                                message = rest_parts[0].strip().replace('warning: ', '')
                                check_name = rest_parts[1].strip(']') if len(rest_parts) > 1 else ""
                                
                                # Estimate end line (simple heuristic)
                                end_line = line
                                for i in range(line, min(line + 10, len(file_lines))):
                                    if i >= len(file_lines):
                                        break
                                    if file_lines[i].strip() == "" or file_lines[i].strip().startswith(("}",)):
                                        end_line = i
                                        break
                                
                                item_type = "unknown"
                                confidence = 0.7
                                item_name = ""
                                
                                if "deadcode" in check_name.lower() or "unreachable" in message.lower():
                                    item_type = "code"
                                    confidence = 0.9
                                elif "unused" in message.lower() and "parameter" in message.lower():
                                    item_type = "parameter"
                                    confidence = 0.85
                                    
                                    # Extract parameter name from message
                                    import re
                                    name_match = re.search(r"'([^']*)'", message)
                                    if name_match:
                                        item_name = name_match.group(1)
                                elif "unused" in message.lower() and "variable" in message.lower():
                                    item_type = "variable"
                                    confidence = 0.8
                                    
                                    # Extract variable name from message
                                    import re
                                    name_match = re.search(r"'([^']*)'", message)
                                    if name_match:
                                        item_name = name_match.group(1)
                                elif "redundant" in message.lower():
                                    item_type = "declaration"
                                    confidence = 0.85
                                    
                                    # Extract declaration name from message
                                    import re
                                    name_match = re.search(r"'([^']*)'", message)
                                    if name_match:
                                        item_name = name_match.group(1)
                                
                                # Get the code context
                                context_start = max(0, line - 2)
                                context_end = min(len(file_lines), end_line + 2)
                                code_context = "".join(file_lines[context_start:context_end])
                                
                                findings.append({
                                    "file_path": os.path.relpath(self.file_path),
                                    "type": item_type,
                                    "name": item_name,
                                    "start_line": line,
                                    "end_line": end_line,
                                    "confidence": confidence,
                                    "reason": f"Clang: {message} ({check_name})",
                                    "code_context": code_context,
                                    "line_count": end_line - line + 1,
                                    "tool": "clang"
                                })
                            
                            except Exception as e:
                                logger.warning(f"Error parsing Clang output line '{line}': {e}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Clang analysis timed out for {self.file_path}")
            except subprocess.SubprocessError as e:
                logger.warning(f"Error running Clang on {self.file_path}: {e}")
            
            # Clean up temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
                
        except Exception as e:
            logger.warning(f"Error in Clang analysis for {self.file_path}: {e}")
        
        return findings
    
    def _analyze_with_cppcheck(self) -> List[Dict[str, Any]]:
        """
        Analyze the file using Cppcheck.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        try:
            # Create a temporary file for Cppcheck output
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=False) as output_file:
                output_path = output_file.name
            
            # Run Cppcheck on the file
            cmd = [
                "cppcheck",
                "--enable=all",
                "--xml",
                "--output-file=" + output_path,
                self.file_path
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse Cppcheck output
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    try:
                        import xml.etree.ElementTree as ET
                        tree = ET.parse(output_path)
                        root = tree.getroot()
                        
                        # Read the file to get the context
                        with open(self.file_path, 'r') as f:
                            file_lines = f.readlines()
                        
                        for error in root.findall('.//error'):
                            error_id = error.get('id', '')
                            message = error.get('msg', '')
                            
                            # Only process errors related to dead code
                            if not any(keyword in error_id for keyword in [
                                'unusedFunction', 'unusedVariable', 'unusedStructMember',
                                'unreadVariable', 'redundantAssignment', 'unreachableCode'
                            ]):
                                continue
                            
                            # Get location information
                            location = error.find('location')
                            if location is None:
                                continue
                            
                            file_path = location.get('file', '')
                            if not file_path or os.path.normpath(file_path) != os.path.normpath(self.file_path):
                                continue
                            
                            line = int(location.get('line', '0'))
                            
                            # Estimate end line (simple heuristic)
                            end_line = line
                            for i in range(line, min(line + 10, len(file_lines))):
                                if i >= len(file_lines):
                                    break
                                if file_lines[i].strip() == "" or file_lines[i].strip().startswith(("}",)):
                                    end_line = i
                                    break
                            
                            item_type = "unknown"
                            confidence = 0.7
                            item_name = ""
                            
                            if "unusedFunction" in error_id:
                                item_type = "function"
                                confidence = 0.9
                                
                                # Extract function name from message
                                import re
                                name_match = re.search(r"'([^']*)'", message)
                                if name_match:
                                    item_name = name_match.group(1)
                            elif "unusedVariable" in error_id or "unreadVariable" in error_id:
                                item_type = "variable"
                                confidence = 0.85
                                
                                # Extract variable name from message
                                import re
                                name_match = re.search(r"'([^']*)'", message)
                                if name_match:
                                    item_name = name_match.group(1)
                            elif "unusedStructMember" in error_id:
                                item_type = "member"
                                confidence = 0.8
                                
                                # Extract member name from message
                                import re
                                name_match = re.search(r"'([^']*)'", message)
                                if name_match:
                                    item_name = name_match.group(1)
                            elif "redundantAssignment" in error_id:
                                item_type = "assignment"
                                confidence = 0.75
                                
                                # Extract variable name from message
                                import re
                                name_match = re.search(r"'([^']*)'", message)
                                if name_match:
                                    item_name = name_match.group(1)
                            elif "unreachableCode" in error_id:
                                item_type = "code"
                                confidence = 0.9
                            
                            # Get the code context
                            context_start = max(0, line - 2)
                            context_end = min(len(file_lines), end_line + 2)
                            code_context = "".join(file_lines[context_start:context_end])
                            
                            findings.append({
                                "file_path": os.path.relpath(self.file_path),
                                "type": item_type,
                                "name": item_name,
                                "start_line": line,
                                "end_line": end_line,
                                "confidence": confidence,
                                "reason": f"Cppcheck: {message} ({error_id})",
                                "code_context": code_context,
                                "line_count": end_line - line + 1,
                                "tool": "cppcheck"
                            })
                    
                    except ET.ParseError:
                        logger.warning(f"Error parsing Cppcheck XML output for {self.file_path}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Cppcheck analysis timed out for {self.file_path}")
            except subprocess.SubprocessError as e:
                logger.warning(f"Error running Cppcheck on {self.file_path}: {e}")
            
            # Clean up temp file
            if os.path.exists(output_path):
                os.unlink(output_path)
                
        except Exception as e:
            logger.warning(f"Error in Cppcheck analysis for {self.file_path}: {e}")
        
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
