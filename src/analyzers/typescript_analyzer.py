"""
TypeScript-specific analyzer for dead code detection.
"""

import json
import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

from analyzers.javascript_analyzer import JavaScriptAnalyzer

logger = logging.getLogger(__name__)

class TypeScriptAnalyzer(JavaScriptAnalyzer):
    """
    Analyzer for detecting dead code in TypeScript files.
    Extends the JavaScript analyzer with TypeScript-specific functionality.
    """
    
    def analyze(self) -> List[Dict[str, Any]]:
        """
        Analyze the TypeScript file for dead code.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        # Check if file exists
        if not os.path.exists(self.file_path):
            logger.warning(f"File not found: {self.file_path}")
            return []
        
        # Use ESLint with TypeScript parser for dead code detection
        eslint_findings = self._analyze_with_typescript_eslint()
        
        # Use the TypeScript compiler API for additional analysis
        tsc_findings = self._analyze_with_typescript_compiler()
        
        # Combine with base JavaScript analysis (using esprima)
        esprima_findings = super()._analyze_with_esprima()
        
        # Combine and deduplicate findings
        all_findings = eslint_findings + tsc_findings + esprima_findings
        
        # Deduplicate based on line ranges
        deduplicated = self._deduplicate_findings(all_findings)
        
        # Filter by confidence threshold
        min_confidence = self.language_config.min_confidence if self.language_config else 0.8
        filtered_findings = [f for f in deduplicated if f.get("confidence", 0) >= min_confidence]
        
        return filtered_findings
    
    def _analyze_with_typescript_eslint(self) -> List[Dict[str, Any]]:
        """
        Analyze the file using ESLint with TypeScript parser.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        try:
            # Create a temporary ESLint configuration file for TypeScript
            eslint_config = {
                "env": {
                    "browser": True,
                    "es6": True,
                    "node": True
                },
                "extends": [
                    "eslint:recommended",
                    "plugin:@typescript-eslint/recommended"
                ],
                "parser": "@typescript-eslint/parser",
                "parserOptions": {
                    "ecmaVersion": 2020,
                    "sourceType": "module",
                    "project": "./tsconfig.json"
                },
                "plugins": [
                    "@typescript-eslint"
                ],
                "rules": {
                    "@typescript-eslint/no-unused-vars": "error",
                    "no-unreachable": "error",
                    "no-empty": "error",
                    "@typescript-eslint/no-empty-interface": "error",
                    "@typescript-eslint/no-empty-function": "error"
                }
            }
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as config_file:
                json.dump(eslint_config, config_file)
                config_path = config_file.name
            
            # Create a minimal tsconfig.json if needed
            tsconfig = {
                "compilerOptions": {
                    "target": "es2020",
                    "module": "commonjs",
                    "strict": true,
                    "esModuleInterop": true,
                    "skipLibCheck": true,
                    "forceConsistentCasingInFileNames": true
                },
                "include": [
                    self.file_path
                ]
            }
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tsconfig_file:
                json.dump(tsconfig, tsconfig_file)
                tsconfig_path = tsconfig_file.name
            
            # Run ESLint with TypeScript parser on the file
            cmd = [
                "npx", "eslint", 
                "--no-eslintrc", 
                "-c", config_path, 
                "--format", "json", 
                "--ext", ".ts,.tsx",
                self.file_path
            ]
            
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
                                if not any(rule in rule_id for rule in [
                                    "no-unused-vars", 
                                    "no-unreachable", 
                                    "no-empty", 
                                    "no-empty-interface", 
                                    "no-empty-function"
                                ]):
                                    continue
                                
                                line = message.get("line", 0)
                                end_line = message.get("endLine", line)
                                message_text = message.get("message", "")
                                
                                item_type = "unknown"
                                if "no-unused-vars" in rule_id:
                                    item_type = "variable"
                                    confidence = 0.9
                                elif "no-unreachable" in rule_id:
                                    item_type = "code"
                                    confidence = 0.85
                                elif "no-empty" in rule_id:
                                    item_type = "block"
                                    confidence = 0.7
                                elif "no-empty-interface" in rule_id:
                                    item_type = "interface"
                                    confidence = 0.85
                                elif "no-empty-function" in rule_id:
                                    item_type = "function"
                                    confidence = 0.8
                                
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
                                    "reason": f"TypeScript ESLint: {message_text} ({rule_id})",
                                    "code_context": code_context,
                                    "line_count": end_line - line + 1,
                                    "tool": "typescript-eslint"
                                })
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing TypeScript ESLint JSON output for {self.file_path}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"TypeScript ESLint analysis timed out for {self.file_path}")
            except subprocess.SubprocessError as e:
                logger.warning(f"Error running TypeScript ESLint on {self.file_path}: {e}")
            
            # Clean up temp files
            for path in [config_path, tsconfig_path]:
                if os.path.exists(path):
                    os.unlink(path)
                
        except Exception as e:
            logger.warning(f"Error in TypeScript ESLint analysis for {self.file_path}: {e}")
        
        return findings
    
    def _analyze_with_typescript_compiler(self) -> List[Dict[str, Any]]:
        """
        Analyze the file using the TypeScript compiler API.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        try:
            # Create a temporary TypeScript analysis script
            ts_script = """
const ts = require('typescript');
const fs = require('fs');

// Get the file path from command line arguments
const filePath = process.argv[2];

// Read the file content
const fileContent = fs.readFileSync(filePath, 'utf8');

// Create a source file
const sourceFile = ts.createSourceFile(
    filePath,
    fileContent,
    ts.ScriptTarget.Latest,
    true
);

// Find unused declarations
const unusedDeclarations = [];
const declaredSymbols = new Map();
const usedSymbols = new Set();

// Visit nodes to collect declarations and usages
function visit(node) {
    // Check for declarations
    if (ts.isVariableDeclaration(node) || ts.isParameter(node)) {
        if (ts.isIdentifier(node.name)) {
            const name = node.name.text;
            const startPos = node.getStart(sourceFile);
            const endPos = node.getEnd();
            const { line: startLine } = ts.getLineAndCharacterOfPosition(sourceFile, startPos);
            const { line: endLine } = ts.getLineAndCharacterOfPosition(sourceFile, endPos);
            
            declaredSymbols.set(name, {
                type: ts.isVariableDeclaration(node) ? 'variable' : 'parameter',
                startLine: startLine + 1,
                endLine: endLine + 1,
                used: false
            });
        }
    } else if (ts.isFunctionDeclaration(node) || ts.isMethodDeclaration(node)) {
        if (node.name && ts.isIdentifier(node.name)) {
            const name = node.name.text;
            const startPos = node.getStart(sourceFile);
            const endPos = node.getEnd();
            const { line: startLine } = ts.getLineAndCharacterOfPosition(sourceFile, startPos);
            const { line: endLine } = ts.getLineAndCharacterOfPosition(sourceFile, endPos);
            
            declaredSymbols.set(name, {
                type: 'function',
                startLine: startLine + 1,
                endLine: endLine + 1,
                used: false
            });
        }
    } else if (ts.isClassDeclaration(node)) {
        if (node.name && ts.isIdentifier(node.name)) {
            const name = node.name.text;
            const startPos = node.getStart(sourceFile);
            const endPos = node.getEnd();
            const { line: startLine } = ts.getLineAndCharacterOfPosition(sourceFile, startPos);
            const { line: endLine } = ts.getLineAndCharacterOfPosition(sourceFile, endPos);
            
            declaredSymbols.set(name, {
                type: 'class',
                startLine: startLine + 1,
                endLine: endLine + 1,
                used: false
            });
        }
    } else if (ts.isInterfaceDeclaration(node)) {
        if (node.name && ts.isIdentifier(node.name)) {
            const name = node.name.text;
            const startPos = node.getStart(sourceFile);
            const endPos = node.getEnd();
            const { line: startLine } = ts.getLineAndCharacterOfPosition(sourceFile, startPos);
            const { line: endLine } = ts.getLineAndCharacterOfPosition(sourceFile, endPos);
            
            declaredSymbols.set(name, {
                type: 'interface',
                startLine: startLine + 1,
                endLine: endLine + 1,
                used: false
            });
        }
    }
    
    // Check for usages
    if (ts.isIdentifier(node) && !ts.isPropertyAccessExpression(node.parent)) {
        const name = node.text;
        usedSymbols.add(name);
    }
    
    // Visit all children
    ts.forEachChild(node, visit);
}

// Start the traversal
visit(sourceFile);

// Mark used symbols
for (const symbol of usedSymbols) {
    if (declaredSymbols.has(symbol)) {
        declaredSymbols.get(symbol).used = true;
    }
}

// Collect unused declarations
for (const [name, info] of declaredSymbols.entries()) {
    if (!info.used) {
        unusedDeclarations.push({
            name,
            type: info.type,
            startLine: info.startLine,
            endLine: info.endLine
        });
    }
}

// Output the results as JSON
console.log(JSON.stringify(unusedDeclarations));
            """
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.js', delete=False) as script_file:
                script_file.write(ts_script)
                script_path = script_file.name
            
            # Run the TypeScript analysis script
            cmd = ["node", script_path, self.file_path]
            
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
                        ts_results = json.loads(result.stdout)
                        
                        # Read the file to get the context
                        with open(self.file_path, 'r') as f:
                            file_lines = f.readlines()
                        
                        for item in ts_results:
                            item_type = item.get("type", "unknown")
                            name = item.get("name", "")
                            start_line = item.get("startLine", 0)
                            end_line = item.get("endLine", start_line)
                            
                            # Assign confidence based on type
                            if item_type == "function":
                                confidence = 0.85
                                reason = f"Unused function '{name}'"
                            elif item_type == "variable":
                                confidence = 0.8
                                reason = f"Unused variable '{name}'"
                            elif item_type == "parameter":
                                confidence = 0.75
                                reason = f"Unused parameter '{name}'"
                            elif item_type == "class":
                                confidence = 0.9
                                reason = f"Unused class '{name}'"
                            elif item_type == "interface":
                                confidence = 0.85
                                reason = f"Unused interface '{name}'"
                            else:
                                confidence = 0.7
                                reason = f"Unused code element '{name}'"
                            
                            # Get the code context
                            context_start = max(0, start_line - 2)
                            context_end = min(len(file_lines), end_line + 2)
                            code_context = "".join(file_lines[context_start:context_end])
                            
                            findings.append({
                                "file_path": os.path.relpath(self.file_path),
                                "type": item_type,
                                "name": name,
                                "start_line": start_line,
                                "end_line": end_line,
                                "confidence": confidence,
                                "reason": reason,
                                "code_context": code_context,
                                "line_count": end_line - start_line + 1,
                                "tool": "typescript-compiler"
                            })
                    
                    except json.JSONDecodeError:
                        logger.warning(f"Error parsing TypeScript compiler analysis output for {self.file_path}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"TypeScript compiler analysis timed out for {self.file_path}")
            except subprocess.SubprocessError as e:
                logger.warning(f"Error running TypeScript compiler analysis on {self.file_path}: {e}")
            
            # Clean up temp file
            if os.path.exists(script_path):
                os.unlink(script_path)
                
        except Exception as e:
            logger.warning(f"Error in TypeScript compiler analysis for {self.file_path}: {e}")
        
        return findings
