#!/usr/bin/env python
"""
Compare evaluation results between baseline and adaptive agents.

This script compares JSONL result files from different agent configurations
to analyze performance differences.

Usage:
    python scripts/compare_results.py baseline.jsonl adaptive.jsonl
    
    # With specific output format
    python scripts/compare_results.py baseline.jsonl adaptive.jsonl --output markdown
    
    # Save comparison to file
    python scripts/compare_results.py baseline.jsonl adaptive.jsonl --save comparison.txt
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

# Add parent directory to path for imports
root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, root)

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using basic comparison")

from src.metric import question_scorer


def load_results(filepath: str) -> List[Dict]:
    """Load results from a JSONL file."""
    results = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def calculate_accuracy(results: List[Dict]) -> Tuple[float, int, int]:
    """
    Calculate accuracy from results.
    
    Returns:
        Tuple of (accuracy, correct_count, total_count)
    """
    correct = 0
    total = 0
    
    for result in results:
        # Skip if no ground truth
        if result.get('true_answer') == '?':
            continue
            
        prediction = result.get('prediction')
        truth = result.get('true_answer')
        
        if prediction is None or prediction == "Unable to determine":
            total += 1
            continue
        
        total += 1
        if question_scorer(str(prediction), str(truth)):
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total


def calculate_accuracy_by_level(results: List[Dict]) -> Dict[str, Tuple[float, int, int]]:
    """
    Calculate accuracy grouped by task level/difficulty.
    
    Returns:
        Dict mapping level to (accuracy, correct, total)
    """
    by_level = defaultdict(list)
    
    for result in results:
        level = result.get('task', 'unknown')
        by_level[level].append(result)
    
    accuracies = {}
    for level, level_results in sorted(by_level.items()):
        accuracies[level] = calculate_accuracy(level_results)
    
    return accuracies


def compare_results(
    baseline_path: str,
    adaptive_path: str,
    output_format: str = 'text'
) -> str:
    """
    Compare baseline and adaptive results.
    
    Args:
        baseline_path: Path to baseline JSONL file
        adaptive_path: Path to adaptive JSONL file
        output_format: 'text', 'markdown', or 'json'
        
    Returns:
        Formatted comparison string
    """
    # Load results
    baseline = load_results(baseline_path)
    adaptive = load_results(adaptive_path)
    
    # Overall accuracy
    base_acc, base_correct, base_total = calculate_accuracy(baseline)
    adapt_acc, adapt_correct, adapt_total = calculate_accuracy(adaptive)
    
    # By level accuracy
    base_by_level = calculate_accuracy_by_level(baseline)
    adapt_by_level = calculate_accuracy_by_level(adaptive)
    
    # Per-question comparison
    base_by_id = {r.get('task_id'): r for r in baseline}
    adapt_by_id = {r.get('task_id'): r for r in adaptive}
    
    common_ids = set(base_by_id.keys()) & set(adapt_by_id.keys())
    
    improvements = []
    regressions = []
    unchanged = []
    
    for task_id in common_ids:
        base_result = base_by_id[task_id]
        adapt_result = adapt_by_id[task_id]
        
        truth = base_result.get('true_answer')
        if truth == '?':
            continue
        
        base_pred = base_result.get('prediction')
        adapt_pred = adapt_result.get('prediction')
        
        base_correct_q = base_pred and question_scorer(str(base_pred), str(truth))
        adapt_correct_q = adapt_pred and question_scorer(str(adapt_pred), str(truth))
        
        item = {
            'task_id': task_id,
            'question': base_result.get('question', '')[:100],
            'level': base_result.get('task', 'unknown'),
            'baseline_correct': base_correct_q,
            'adaptive_correct': adapt_correct_q,
        }
        
        if adapt_correct_q and not base_correct_q:
            improvements.append(item)
        elif base_correct_q and not adapt_correct_q:
            regressions.append(item)
        else:
            unchanged.append(item)
    
    # Format output
    if output_format == 'json':
        return json.dumps({
            'summary': {
                'baseline': {'accuracy': base_acc, 'correct': base_correct, 'total': base_total},
                'adaptive': {'accuracy': adapt_acc, 'correct': adapt_correct, 'total': adapt_total},
                'improvement': adapt_acc - base_acc,
            },
            'by_level': {
                'baseline': {k: {'accuracy': v[0], 'correct': v[1], 'total': v[2]} 
                            for k, v in base_by_level.items()},
                'adaptive': {k: {'accuracy': v[0], 'correct': v[1], 'total': v[2]} 
                            for k, v in adapt_by_level.items()},
            },
            'improvements': improvements,
            'regressions': regressions,
            'unchanged_count': len(unchanged),
        }, indent=2)
    
    elif output_format == 'markdown':
        lines = [
            "# Evaluation Comparison Report",
            "",
            "## Overall Results",
            "",
            "| Metric | Baseline | Adaptive | Change |",
            "|--------|----------|----------|--------|",
            f"| Accuracy | {base_acc:.2%} | {adapt_acc:.2%} | {adapt_acc - base_acc:+.2%} |",
            f"| Correct | {base_correct} | {adapt_correct} | {adapt_correct - base_correct:+d} |",
            f"| Total | {base_total} | {adapt_total} | - |",
            "",
            "## Results by Level",
            "",
            "| Level | Baseline | Adaptive | Change |",
            "|-------|----------|----------|--------|",
        ]
        
        all_levels = sorted(set(base_by_level.keys()) | set(adapt_by_level.keys()))
        for level in all_levels:
            base_l = base_by_level.get(level, (0, 0, 0))
            adapt_l = adapt_by_level.get(level, (0, 0, 0))
            change = adapt_l[0] - base_l[0]
            lines.append(
                f"| {level} | {base_l[0]:.2%} ({base_l[1]}/{base_l[2]}) | "
                f"{adapt_l[0]:.2%} ({adapt_l[1]}/{adapt_l[2]}) | {change:+.2%} |"
            )
        
        lines.extend([
            "",
            "## Question-Level Analysis",
            "",
            f"- **Improvements** (baseline wrong, adaptive correct): {len(improvements)}",
            f"- **Regressions** (baseline correct, adaptive wrong): {len(regressions)}",
            f"- **Unchanged**: {len(unchanged)}",
            "",
        ])
        
        if improvements:
            lines.extend([
                "### Improvements",
                "",
            ])
            for item in improvements[:10]:  # Show top 10
                lines.append(f"- [{item['level']}] {item['task_id']}: {item['question']}...")
        
        if regressions:
            lines.extend([
                "",
                "### Regressions",
                "",
            ])
            for item in regressions[:10]:
                lines.append(f"- [{item['level']}] {item['task_id']}: {item['question']}...")
        
        return "\n".join(lines)
    
    else:  # text format
        lines = [
            "=" * 60,
            "EVALUATION COMPARISON REPORT",
            "=" * 60,
            "",
            "OVERALL RESULTS",
            "-" * 40,
            f"Baseline:  {base_acc:.2%} ({base_correct}/{base_total})",
            f"Adaptive:  {adapt_acc:.2%} ({adapt_correct}/{adapt_total})",
            f"Change:    {adapt_acc - base_acc:+.2%} ({adapt_correct - base_correct:+d})",
            "",
            "RESULTS BY LEVEL",
            "-" * 40,
        ]
        
        all_levels = sorted(set(base_by_level.keys()) | set(adapt_by_level.keys()))
        for level in all_levels:
            base_l = base_by_level.get(level, (0, 0, 0))
            adapt_l = adapt_by_level.get(level, (0, 0, 0))
            change = adapt_l[0] - base_l[0]
            lines.append(
                f"  {level:15} Baseline: {base_l[0]:.2%} ({base_l[1]}/{base_l[2]})  "
                f"Adaptive: {adapt_l[0]:.2%} ({adapt_l[1]}/{adapt_l[2]})  "
                f"Change: {change:+.2%}"
            )
        
        lines.extend([
            "",
            "QUESTION-LEVEL ANALYSIS",
            "-" * 40,
            f"  Improvements (baseline wrong -> adaptive correct): {len(improvements)}",
            f"  Regressions (baseline correct -> adaptive wrong): {len(regressions)}",
            f"  Unchanged: {len(unchanged)}",
            "",
        ])
        
        if improvements:
            lines.extend([
                "TOP IMPROVEMENTS:",
            ])
            for item in improvements[:5]:
                lines.append(f"  - [{item['level']}] {item['task_id']}")
        
        if regressions:
            lines.extend([
                "",
                "REGRESSIONS:",
            ])
            for item in regressions[:5]:
                lines.append(f"  - [{item['level']}] {item['task_id']}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Compare evaluation results between baseline and adaptive agents'
    )
    parser.add_argument(
        'baseline',
        help='Path to baseline results JSONL file'
    )
    parser.add_argument(
        'adaptive',
        help='Path to adaptive results JSONL file'
    )
    parser.add_argument(
        '--output', '-o',
        choices=['text', 'markdown', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    parser.add_argument(
        '--save', '-s',
        help='Save comparison to file'
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not Path(args.baseline).exists():
        print(f"Error: Baseline file not found: {args.baseline}")
        sys.exit(1)
    if not Path(args.adaptive).exists():
        print(f"Error: Adaptive file not found: {args.adaptive}")
        sys.exit(1)
    
    # Run comparison
    result = compare_results(args.baseline, args.adaptive, args.output)
    
    # Output
    print(result)
    
    if args.save:
        with open(args.save, 'w', encoding='utf-8') as f:
            f.write(result)
        print(f"\nComparison saved to: {args.save}")


if __name__ == '__main__':
    main()
