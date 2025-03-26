"""
Report generator for the Dead Code Detection Agent.
"""

import logging
from typing import Any, Dict, List

from config import ReportingConfig

logger = logging.getLogger(__name__)

def generate_report(
    findings: List[Dict[str, Any]],
    stats: Dict[str, Any],
    config: ReportingConfig
) -> str:
    """
    Generate a report of the dead code findings.
    
    Args:
        findings: List of dead code findings
        stats: Statistics about the analysis
        config: Reporting configuration
        
    Returns:
        Formatted report
    """
    if not findings:
        return "No dead code found in the analyzed files."
    
    # Group findings based on configuration
    if config.group_by == "file":
        grouped_findings = _group_by_file(findings)
    elif config.group_by == "type":
        grouped_findings = _group_by_type(findings)
    elif config.group_by == "confidence":
        grouped_findings = _group_by_confidence(findings)
    else:
        grouped_findings = _group_by_file(findings)
    
    # Generate report based on detail level
    if config.detail_level == "low":
        return _generate_low_detail_report(grouped_findings, stats, config)
    elif config.detail_level == "medium":
        return _generate_medium_detail_report(grouped_findings, stats, config)
    else:  # high
        return _generate_high_detail_report(grouped_findings, stats, config)

def _group_by_file(findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group findings by file path."""
    grouped = {}
    for finding in findings:
        file_path = finding.get("file_path", "unknown")
        if file_path not in grouped:
            grouped[file_path] = []
        grouped[file_path].append(finding)
    return grouped

def _group_by_type(findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group findings by dead code type."""
    grouped = {}
    for finding in findings:
        item_type = finding.get("type", "unknown")
        if item_type not in grouped:
            grouped[item_type] = []
        grouped[item_type].append(finding)
    return grouped

def _group_by_confidence(findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group findings by confidence level."""
    grouped = {
        "high (90-100%)": [],
        "medium (80-90%)": [],
        "low (< 80%)": []
    }
    
    for finding in findings:
        confidence = finding.get("confidence", 0)
        if confidence >= 0.9:
            grouped["high (90-100%)"].append(finding)
        elif confidence >= 0.8:
            grouped["medium (80-90%)"].append(finding)
        else:
            grouped["low (< 80%)"].append(finding)
    
    return grouped

def _generate_low_detail_report(
    grouped_findings: Dict[str, List[Dict[str, Any]]],
    stats: Dict[str, Any],
    config: ReportingConfig
) -> str:
    """Generate a low detail report."""
    report = []
    report.append("# Dead Code Analysis Report\n")
    
    # Add summary
    report.append("## Summary\n")
    report.append(f"- Files analyzed: {stats.get('files_analyzed', 0)}")
    report.append(f"- Dead code instances found: {stats.get('dead_code_found', 0)}")
    report.append(f"- Files with dead code: {stats.get('files_with_dead_code', 0)}")
    report.append(f"- Total lines that could be removed: {stats.get('lines_removable', 0)}\n")
    
    # Add findings by group
    report.append("## Findings\n")
    
    for group_name, group_findings in grouped_findings.items():
        report.append(f"### {group_name}\n")
        report.append(f"- Found {len(group_findings)} instances of dead code\n")
    
    return "\n".join(report)

def _generate_medium_detail_report(
    grouped_findings: Dict[str, List[Dict[str, Any]]],
    stats: Dict[str, Any],
    config: ReportingConfig
) -> str:
    """Generate a medium detail report."""
    report = []
    report.append("# Dead Code Analysis Report\n")
    
    # Add summary
    report.append("## Summary\n")
    report.append(f"- Files analyzed: {stats.get('files_analyzed', 0)}")
    report.append(f"- Dead code instances found: {stats.get('dead_code_found', 0)}")
    report.append(f"- Files with dead code: {stats.get('files_with_dead_code', 0)}")
    report.append(f"- Total lines that could be removed: {stats.get('lines_removable', 0)}\n")
    
    # Add findings by group
    report.append("## Findings\n")
    
    for group_name, group_findings in grouped_findings.items():
        report.append(f"### {group_name}\n")
        
        for finding in group_findings:
            item_type = finding.get("type", "unknown")
            name = finding.get("name", "")
            start_line = finding.get("start_line", 0)
            end_line = finding.get("end_line", 0)
            confidence = finding.get("confidence", 0) * 100
            
            name_str = f" '{name}'" if name else ""
            report.append(f"- {item_type.capitalize()}{name_str} at lines {start_line}-{end_line} (confidence: {confidence:.0f}%)")
        
        report.append("")
    
    return "\n".join(report)

def _generate_high_detail_report(
    grouped_findings: Dict[str, List[Dict[str, Any]]],
    stats: Dict[str, Any],
    config: ReportingConfig
) -> str:
    """Generate a high detail report."""
    report = []
    report.append("# Dead Code Analysis Report\n")
    
    # Add summary
    report.append("## Summary\n")
    report.append(f"- Files analyzed: {stats.get('files_analyzed', 0)}")
    report.append(f"- Dead code instances found: {stats.get('dead_code_found', 0)}")
    report.append(f"- Files with dead code: {stats.get('files_with_dead_code', 0)}")
    report.append(f"- Total lines that could be removed: {stats.get('lines_removable', 0)}\n")
    
    # Add findings by group
    report.append("## Findings\n")
    
    for group_name, group_findings in grouped_findings.items():
        report.append(f"### {group_name}\n")
        
        for finding in group_findings:
            item_type = finding.get("type", "unknown")
            name = finding.get("name", "")
            start_line = finding.get("start_line", 0)
            end_line = finding.get("end_line", 0)
            confidence = finding.get("confidence", 0) * 100
            reason = finding.get("reason", "")
            code_context = finding.get("code_context", "")
            line_count = finding.get("line_count", 0)
            tool = finding.get("tool", "")
            
            name_str = f" '{name}'" if name else ""
            report.append(f"#### {item_type.capitalize()}{name_str} (lines {start_line}-{end_line})\n")
            report.append(f"- **Confidence**: {confidence:.0f}%")
            report.append(f"- **Reason**: {reason}")
            report.append(f"- **Lines affected**: {line_count}")
            report.append(f"- **Detection tool**: {tool}")
            
            if config.include_context and code_context:
                report.append("\n**Code context**:")
                report.append("```")
                report.append(code_context)
                report.append("```")
            
            report.append("")
    
    return "\n".join(report)
