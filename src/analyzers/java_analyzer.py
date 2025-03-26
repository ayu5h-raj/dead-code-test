"""
Java-specific analyzer for dead code detection.
"""

import logging
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from analyzers.base_analyzer import BaseAnalyzer

logger = logging.getLogger(__name__)

class JavaAnalyzer(BaseAnalyzer):
    """
    Analyzer for detecting dead code in Java files.
    """
    
    def analyze(self) -> List[Dict[str, Any]]:
        """
        Analyze the Java file for dead code.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        # Check if file exists
        if not os.path.exists(self.file_path):
            logger.warning(f"File not found: {self.file_path}")
            return []
        
        # Use PMD for dead code detection
        pmd_findings = self._analyze_with_pmd()
        
        # Use static analysis with JavaParser
        javaparser_findings = self._analyze_with_javaparser()
        
        # Combine and deduplicate findings
        all_findings = pmd_findings + javaparser_findings
        
        # Deduplicate based on line ranges
        deduplicated = self._deduplicate_findings(all_findings)
        
        # Filter by confidence threshold
        min_confidence = self.language_config.min_confidence if self.language_config else 0.8
        filtered_findings = [f for f in deduplicated if f.get("confidence", 0) >= min_confidence]
        
        return filtered_findings
    
    def _analyze_with_pmd(self) -> List[Dict[str, Any]]:
        """
        Analyze the file using PMD.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        try:
            # Create a temporary PMD ruleset XML file
            pmd_ruleset = """<?xml version="1.0"?>
<ruleset name="DeadCodeDetection"
         xmlns="http://pmd.sourceforge.net/ruleset/2.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://pmd.sourceforge.net/ruleset/2.0.0 https://pmd.sourceforge.io/ruleset_2_0_0.xsd">
    <description>Dead Code Detection Rules</description>
    
    <rule ref="category/java/bestpractices.xml/UnusedPrivateMethod"/>
    <rule ref="category/java/bestpractices.xml/UnusedPrivateField"/>
    <rule ref="category/java/bestpractices.xml/UnusedLocalVariable"/>
    <rule ref="category/java/bestpractices.xml/UnusedFormalParameter"/>
    <rule ref="category/java/bestpractices.xml/UnusedImports"/>
    <rule ref="category/java/codestyle.xml/UnnecessaryImport"/>
    <rule ref="category/java/design.xml/UselessOverridingMethod"/>
    <rule ref="category/java/errorprone.xml/EmptyIfStmt"/>
    <rule ref="category/java/errorprone.xml/EmptyWhileStmt"/>
    <rule ref="category/java/errorprone.xml/EmptyTryBlock"/>
    <rule ref="category/java/errorprone.xml/EmptyFinallyBlock"/>
    <rule ref="category/java/errorprone.xml/EmptySwitchStatements"/>
    <rule ref="category/java/errorprone.xml/EmptySynchronizedBlock"/>
    <rule ref="category/java/errorprone.xml/EmptyStatementBlock"/>
    <rule ref="category/java/errorprone.xml/UnreachableCode"/>
</ruleset>
            """
            
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=False) as ruleset_file:
                ruleset_file.write(pmd_ruleset)
                ruleset_path = ruleset_file.name
            
            # Create a temporary file for PMD output
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.xml', delete=False) as output_file:
                output_path = output_file.name
            
            # Run PMD on the file
            cmd = [
                "pmd", "check",
                "-d", self.file_path,
                "-R", ruleset_path,
                "-f", "xml",
                "-r", output_path
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                # Parse PMD output
                if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                    try:
                        tree = ET.parse(output_path)
                        root = tree.getroot()
                        
                        # Read the file to get the context
                        with open(self.file_path, 'r') as f:
                            file_lines = f.readlines()
                        
                        for file_element in root.findall('.//file'):
                            for violation in file_element.findall('.//violation'):
                                rule = violation.get('rule', '')
                                ruleset = violation.get('ruleset', '')
                                line = int(violation.get('beginline', '0'))
                                end_line = int(violation.get('endline', line))
                                description = violation.text.strip() if violation.text else ""
                                
                                item_type = "unknown"
                                confidence = 0.7
                                
                                if "UnusedPrivateMethod" in rule:
                                    item_type = "method"
                                    confidence = 0.9
                                elif "UnusedPrivateField" in rule:
                                    item_type = "field"
                                    confidence = 0.9
                                elif "UnusedLocalVariable" in rule:
                                    item_type = "variable"
                                    confidence = 0.85
                                elif "UnusedFormalParameter" in rule:
                                    item_type = "parameter"
                                    confidence = 0.8
                                elif "UnusedImports" in rule or "UnnecessaryImport" in rule:
                                    item_type = "import"
                                    confidence = 0.95
                                elif "UselessOverridingMethod" in rule:
                                    item_type = "method"
                                    confidence = 0.85
                                elif "Empty" in rule:
                                    item_type = "block"
                                    confidence = 0.75
                                elif "UnreachableCode" in rule:
                                    item_type = "code"
                                    confidence = 0.9
                                
                                # Extract name from description
                                import re
                                name_match = re.search(r"'([^']*)'", description)
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
                                    "reason": f"PMD: {description} ({rule})",
                                    "code_context": code_context,
                                    "line_count": end_line - line + 1,
                                    "tool": "pmd"
                                })
                    
                    except ET.ParseError:
                        logger.warning(f"Error parsing PMD XML output for {self.file_path}")
                
            except subprocess.TimeoutExpired:
                logger.warning(f"PMD analysis timed out for {self.file_path}")
            except subprocess.SubprocessError as e:
                logger.warning(f"Error running PMD on {self.file_path}: {e}")
            
            # Clean up temp files
            for path in [ruleset_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
                
        except Exception as e:
            logger.warning(f"Error in PMD analysis for {self.file_path}: {e}")
        
        return findings
    
    def _analyze_with_javaparser(self) -> List[Dict[str, Any]]:
        """
        Analyze the file using JavaParser.
        
        Returns:
            List of dead code findings
        """
        findings = []
        
        try:
            # Create a temporary Java program for JavaParser analysis
            javaparser_program = """
import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.*;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.expr.NameExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.io.File;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.util.*;

public class DeadCodeDetector {
    private static class SymbolInfo {
        String type;
        int startLine;
        int endLine;
        boolean used;

        public SymbolInfo(String type, int startLine, int endLine) {
            this.type = type;
            this.startLine = startLine;
            this.endLine = endLine;
            this.used = false;
        }
    }

    private static Map<String, SymbolInfo> declaredSymbols = new HashMap<>();
    private static Set<String> usedSymbols = new HashSet<>();

    public static void main(String[] args) {
        if (args.length < 1) {
            System.err.println("Usage: java DeadCodeDetector <file_path>");
            System.exit(1);
        }

        String filePath = args[0];
        File file = new File(filePath);

        try {
            // Parse the Java file
            JavaParser parser = new JavaParser();
            CompilationUnit cu = parser.parse(file).getResult().orElseThrow();

            // Collect declarations and usages
            cu.accept(new DeclarationVisitor(), null);
            cu.accept(new UsageVisitor(), null);

            // Mark used symbols
            for (String symbol : usedSymbols) {
                if (declaredSymbols.containsKey(symbol)) {
                    declaredSymbols.get(symbol).used = true;
                }
            }

            // Collect unused declarations
            List<Map<String, Object>> unusedDeclarations = new ArrayList<>();
            for (Map.Entry<String, SymbolInfo> entry : declaredSymbols.entrySet()) {
                String name = entry.getKey();
                SymbolInfo info = entry.getValue();
                
                if (!info.used) {
                    Map<String, Object> declaration = new HashMap<>();
                    declaration.put("name", name);
                    declaration.put("type", info.type);
                    declaration.put("startLine", info.startLine);
                    declaration.put("endLine", info.endLine);
                    unusedDeclarations.add(declaration);
                }
            }

            // Output as JSON
            StringBuilder json = new StringBuilder("[");
            for (int i = 0; i < unusedDeclarations.size(); i++) {
                Map<String, Object> decl = unusedDeclarations.get(i);
                json.append("{");
                json.append("\\\"name\\\":\\\"").append(decl.get("name")).append("\\\",");
                json.append("\\\"type\\\":\\\"").append(decl.get("type")).append("\\\",");
                json.append("\\\"startLine\\\":").append(decl.get("startLine")).append(",");
                json.append("\\\"endLine\\\":").append(decl.get("endLine"));
                json.append("}");
                if (i < unusedDeclarations.size() - 1) {
                    json.append(",");
                }
            }
            json.append("]");
            
            System.out.println(json.toString());

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            System.exit(1);
        }
    }

    private static class DeclarationVisitor extends VoidVisitorAdapter<Void> {
        @Override
        public void visit(MethodDeclaration n, Void arg) {
            super.visit(n, arg);
            
            if (n.isPrivate()) {
                String name = n.getNameAsString();
                int startLine = n.getBegin().get().line;
                int endLine = n.getEnd().get().line;
                
                declaredSymbols.put(name, new SymbolInfo("method", startLine, endLine));
            }
        }

        @Override
        public void visit(FieldDeclaration n, Void arg) {
            super.visit(n, arg);
            
            if (n.isPrivate()) {
                for (VariableDeclarator var : n.getVariables()) {
                    String name = var.getNameAsString();
                    int startLine = n.getBegin().get().line;
                    int endLine = n.getEnd().get().line;
                    
                    declaredSymbols.put(name, new SymbolInfo("field", startLine, endLine));
                }
            }
        }

        @Override
        public void visit(VariableDeclarator n, Void arg) {
            super.visit(n, arg);
            
            if (n.getParentNode().isPresent() && !(n.getParentNode().get() instanceof FieldDeclaration)) {
                String name = n.getNameAsString();
                int startLine = n.getBegin().get().line;
                int endLine = n.getEnd().get().line;
                
                declaredSymbols.put(name, new SymbolInfo("variable", startLine, endLine));
            }
        }

        @Override
        public void visit(Parameter n, Void arg) {
            super.visit(n, arg);
            
            String name = n.getNameAsString();
            int startLine = n.getBegin().get().line;
            int endLine = n.getEnd().get().line;
            
            declaredSymbols.put(name, new SymbolInfo("parameter", startLine, endLine));
        }
    }

    private static class UsageVisitor extends VoidVisitorAdapter<Void> {
        @Override
        public void visit(NameExpr n, Void arg) {
            super.visit(n, arg);
            usedSymbols.add(n.getNameAsString());
        }

        @Override
        public void visit(MethodCallExpr n, Void arg) {
            super.visit(n, arg);
            if (n.getScope().isEmpty()) {
                usedSymbols.add(n.getNameAsString());
            }
        }
    }
}
            """
            
            # Create a temporary directory for JavaParser analysis
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write the Java program to a file
                program_path = os.path.join(temp_dir, "DeadCodeDetector.java")
                with open(program_path, 'w') as f:
                    f.write(javaparser_program)
                
                # Compile the Java program
                compile_cmd = [
                    "javac", "-cp", ".:javaparser-core-3.24.0.jar", program_path
                ]
                
                try:
                    # Note: This will likely fail in most environments without JavaParser in classpath
                    # This is just a placeholder for the actual implementation
                    compile_result = subprocess.run(
                        compile_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=temp_dir
                    )
                    
                    # Run the Java program
                    run_cmd = [
                        "java", "-cp", ".:javaparser-core-3.24.0.jar", "DeadCodeDetector", self.file_path
                    ]
                    
                    run_result = subprocess.run(
                        run_cmd,
                        capture_output=True,
                        text=True,
                        timeout=30,
                        cwd=temp_dir
                    )
                    
                    # Parse the output
                    if run_result.stdout:
                        import json
                        try:
                            javaparser_results = json.loads(run_result.stdout)
                            
                            # Read the file to get the context
                            with open(self.file_path, 'r') as f:
                                file_lines = f.readlines()
                            
                            for item in javaparser_results:
                                item_type = item.get("type", "unknown")
                                name = item.get("name", "")
                                start_line = item.get("startLine", 0)
                                end_line = item.get("endLine", start_line)
                                
                                # Assign confidence based on type
                                if item_type == "method":
                                    confidence = 0.85
                                    reason = f"Unused method '{name}'"
                                elif item_type == "field":
                                    confidence = 0.9
                                    reason = f"Unused field '{name}'"
                                elif item_type == "variable":
                                    confidence = 0.8
                                    reason = f"Unused variable '{name}'"
                                elif item_type == "parameter":
                                    confidence = 0.75
                                    reason = f"Unused parameter '{name}'"
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
                                    "tool": "javaparser"
                                })
                        
                        except json.JSONDecodeError:
                            logger.warning(f"Error parsing JavaParser analysis output for {self.file_path}")
                
                except subprocess.TimeoutExpired:
                    logger.warning(f"JavaParser analysis timed out for {self.file_path}")
                except subprocess.SubprocessError as e:
                    logger.warning(f"Error running JavaParser analysis on {self.file_path}: {e}")
                
        except Exception as e:
            logger.warning(f"Error in JavaParser analysis for {self.file_path}: {e}")
        
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
