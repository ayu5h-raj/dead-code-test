"""
JavaScript-specific analyzer for dead code detection.
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from analyzers.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class JavaScriptAnalyzer(BaseAnalyzer):
    """
    Analyzer for detecting dead code in JavaScript files.
    """
    
    def analyze(self) -> List[Dict[str, Any]]:
        """
        Analyze the JavaScript file for dead code.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        # Check if file exists
        if not os.path.exists(self.file_path):
            logger.warning(f"File not found: {self.file_path}")
            return []
        
        # Use ESLint for dead code detection
        eslint_findings = self._analyze_with_eslint()
        
        # Use static analysis with esprima
        esprima_findings = self._analyze_with_esprima()
        
        # Combine and deduplicate findings
        all_findings = eslint_findings + esprima_findings
        
        # Deduplicate based on line ranges
        deduplicated = self._deduplicate_findings(all_findings)
        
        # Filter by confidence threshold
        min_confidence = self.language_config.min_confidence if self.language_config else 0.8
        filtered_findings = [f for f in deduplicated if f.get("confidence", 0) >= min_confidence]
        
        return filtered_findings
    
    def _analyze_with_eslint(self) -> List[Dict[str, Any]]:
        """
        Analyze the file using ESLint.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        try:
            # Create a temporary ESLint configuration file
            eslint_config = {
                "env": {
                    "browser": True,
                    "es6": True,
                    "node": True
                },
                "extends": "eslint:recommended",
                "parserOptions": {
                    "ecmaVersion": 2020,
                    "sourceType": "module"
                },
                "rules": {
                    "no-unused-vars": "error",
                    "no-unreachable": "error",
                    "no-empty": "error"
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as config_file:
                json.dump(eslint_config, config_file)
                config_path = config_file.name
            
            # Run ESLint on the file
            cmd = ["npx", "eslint", "--no-eslintrc", "-c", config_path, "--format", "json", self.file_path]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse ESLint output
                if result.stdout:
                    try:
                        eslint_results = json.loads(result.stdout)
                        
                        # Read the file to get the context
                        with open(self.file_path, 'r') as f:
                            file_lines = f.readlines()
                        
                        for file_result in eslint_results:
                            for message in file_result.get("messages", []):
                                rule_id = message.get("ruleId", "")
                                
                                # Only process rules related to dead code
                                if rule_id not in ["no-unused-vars", "no-unreachable", "no-empty"]:
                                    continue
                                
                                line = message.get("line", 0)
                                end_line = message.get("endLine", line)
                                message_text = message.get("message", "")
                                
                                item_type = "unknown"
                                if rule_id == "no-unused-vars":
                                    item_type = "variable"
                                    confidence = 0.9
                                elif rule_id == "no-unreachable":
                                    item_type = "code"
                                    confidence = 0.85
                                elif rule_id == "no-empty":
                                    item_type = "block"
                                    confidence = 0.7
                                
                                # Extract name from message
                                import re
                                name_match = re.search(r"'([^']*)'", message_text)
                                item_name = name_match.group(1) if name_match else ""
                                
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
                                    "reason": f"ESLint: {message_text} ({rule_id})",
                                    "code_context": code_context,
                                    "line_count": end_line - line + 1,
                                    "tool": "eslint"
                                })
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing ESLint JSON output for {self.file_path}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"ESLint analysis timed out for {self.file_path}")
            except subprocess.SubprocessError as e:
                logger.warning(f"Error running ESLint on {self.file_path}: {e}")
            
            # Clean up temp file
            if os.path.exists(config_path):
                os.unlink(config_path)
                
        except Exception as e:
            logger.warning(f"Error in ESLint analysis for {self.file_path}: {e}")
        
        return findings
    
    def _analyze_with_esprima(self) -> List[Dict[str, Any]]:
        """
        Analyze the file using Esprima for static analysis.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        try:
            # Create a temporary Python script for Esprima analysis
            esprima_script = """
import sys
import json
import esprima
from esprima import nodes

def find_unused_functions(ast):
    # Find all function declarations and expressions
    functions = {}
    function_calls = set()
    
    # Visitor to collect functions and function calls
    def visit(node, parent=None):
        if isinstance(node, nodes.FunctionDeclaration):
            if node.id and node.id.name:
                functions[node.id.name] = {
                    'node': node,
                    'used': False,
                    'line': node.loc.start.line,
                    'end_line': node.loc.end.line
                }
        elif isinstance(node, nodes.CallExpression):
            if isinstance(node.callee, nodes.Identifier):
                function_calls.add(node.callee.name)
        
        # Recursively visit all child nodes
        for key, value in node.__dict__.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, nodes.Node):
                        visit(item, node)
            elif isinstance(value, nodes.Node):
                visit(value, node)
    
    # Start visiting from the root
    visit(ast)
    
    # Mark used functions
    for func_name in function_calls:
        if func_name in functions:
            functions[func_name]['used'] = True
    
    # Return unused functions
    unused_functions = []
    for name, data in functions.items():
        if not data['used']:
            unused_functions.append({
                'name': name,
                'line': data['line'],
                'end_line': data['end_line'],
                'type': 'function'
            })
    
    return unused_functions

def find_unused_variables(ast):
    # Find all variable declarations and usages
    variables = {}
    variable_usages = set()
    
    # Visitor to collect variables and their usages
    def visit(node, parent=None):
        if isinstance(node, nodes.VariableDeclarator):
            if isinstance(node.id, nodes.Identifier):
                variables[node.id.name] = {
                    'node': node,
                    'used': False,
                    'line': node.loc.start.line,
                    'end_line': node.loc.end.line
                }
        elif isinstance(node, nodes.Identifier):
            # Skip identifiers that are part of variable declarations
            if not (parent and isinstance(parent, nodes.VariableDeclarator) and parent.id == node):
                variable_usages.add(node.name)
        
        # Recursively visit all child nodes
        for key, value in node.__dict__.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, nodes.Node):
                        visit(item, node)
            elif isinstance(value, nodes.Node):
                visit(value, node)
    
    # Start visiting from the root
    visit(ast)
    
    # Mark used variables
    for var_name in variable_usages:
        if var_name in variables:
            variables[var_name]['used'] = True
    
    # Return unused variables
    unused_variables = []
    for name, data in variables.items():
        if not data['used']:
            unused_variables.append({
                'name': name,
                'line': data['line'],
                'end_line': data['end_line'],
                'type': 'variable'
            })
    
    return unused_variables

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        with open(file_path, 'r') as f:
            source = f.read()
        
        # Parse the JavaScript code
        ast = esprima.parseScript(source, {'loc': True, 'range': True})
        
        # Find unused functions and variables
        unused_functions = find_unused_functions(ast)
        unused_variables = find_unused_variables(ast)
        
        # Combine results
        results = unused_functions + unused_variables
        
        # Output as JSON
        print(json.dumps(results))
        
    except Exception as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)

if __name__ == '__main__':
    main()
            """
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.py', delete=False) as script_file:
                script_file.write(esprima_script)
                script_path = script_file.name
            
            # Run the Esprima analysis script
            cmd = ["python", script_path, self.file_path]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse the output
                if result.stdout:
                    try:
                        esprima_results = json.loads(result.stdout)
                        
                        # Read the file to get the context
                        with open(self.file_path, 'r') as f:
                            file_lines = f.readlines()
                        
                        for item in esprima_results:
                            item_type = item.get("type", "unknown")
                            name = item.get("name", "")
                            line = item.get("line", 0)
                            end_line = item.get("end_line", line)
                            
                            # Assign confidence based on type
                            if item_type == "function":
                                confidence = 0.85
                                reason = f"Unused function '{name}'"
                            elif item_type == "variable":
                                confidence = 0.8
                                reason = f"Unused variable '{name}'"
                            else:
                                confidence = 0.7
                                reason = f"Unused code element '{name}'"
                            
                            # Get the code context
                            context_start = max(0, line - 2)
                            context_end = min(len(file_lines), end_line + 2)
                            code_context = "".join(file_lines[context_start:context_end])
                            
                            findings.append({
                                "file_path": os.path.relpath(self.file_path),
                                "type": item_type,
                                "name": name,
                                "start_line": line,
                                "end_line": end_line,
                                "confidence": confidence,
                                "reason": reason,
                                "code_context": code_context,
                                "line_count": end_line - line + 1,
                                "tool": "esprima"
                            })
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing Esprima analysis output for {self.file_path}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Esprima analysis timed out for {self.file_path}")
            except subprocess.SubprocessError as e:
                logger.warning(f"Error running Esprima analysis on {self.file_path}: {e}")
            
            # Clean up temp file
            if os.path.exists(script_path):
                os.unlink(script_path)
                
        except Exception as e:
            logger.warning(f"Error in Esprima analysis for {self.file_path}: {e}")
        
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
