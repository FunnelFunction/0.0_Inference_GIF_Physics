#!/usr/bin/env python3
"""
ARC-AGI Solver App
==================

A systematic approach to solving all 400 ARC challenges.

Features:
- Tracks solved vs unsolved puzzles
- Categorizes puzzles by rule type
- Saves progress to JSON
- Visual puzzle display
- Rule library for known patterns

Usage:
    python arc_solver_app.py                    # Interactive menu
    python arc_solver_app.py --status           # Show progress
    python arc_solver_app.py --solve <puzzle_id>  # Solve specific puzzle
    python arc_solver_app.py --category <type>  # List puzzles by category
"""

import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum

# Add core to path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.arc_loader import ARCLoader


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class RuleCategory(Enum):
    """Categories of ARC transformation rules."""
    GEOMETRIC = "geometric"           # flip, rotate, transpose
    COLOR_MAP = "color_map"           # pixel-wise color substitution
    FILL = "fill"                     # fill enclosed regions
    TILE = "tile"                     # repeat/tile patterns
    SCALE = "scale"                   # resize grid
    OBJECT = "object"                 # object manipulation
    COUNT = "count"                   # counting-based rules
    PATTERN = "pattern"               # pattern completion
    COMPOSITE = "composite"           # multiple rules combined
    UNKNOWN = "unknown"               # not yet categorized


@dataclass
class PuzzleSolution:
    """Record of a solved puzzle."""
    puzzle_id: str
    category: str
    rule_description: str
    solver_function: str
    solved_at: str
    test_accuracy: float
    notes: str = ""


@dataclass
class PuzzleInfo:
    """Information about a puzzle."""
    puzzle_id: str
    n_train: int
    n_test: int
    input_shapes: List[Tuple[int, int]]
    output_shapes: List[Tuple[int, int]]
    has_size_change: bool
    category: str = "unknown"
    is_solved: bool = False


class ProgressTracker:
    """Track solving progress across sessions."""
    
    def __init__(self, save_path: str = "./arc_progress.json"):
        self.save_path = Path(save_path)
        self.solutions: Dict[str, PuzzleSolution] = {}
        self.puzzle_info: Dict[str, PuzzleInfo] = {}
        self.categories: Dict[str, List[str]] = {}
        self.load()
    
    def load(self):
        """Load progress from file."""
        if self.save_path.exists():
            with open(self.save_path) as f:
                data = json.load(f)
                self.solutions = {
                    k: PuzzleSolution(**v) 
                    for k, v in data.get('solutions', {}).items()
                }
                self.categories = data.get('categories', {})
    
    def save(self):
        """Save progress to file."""
        data = {
            'solutions': {k: asdict(v) for k, v in self.solutions.items()},
            'categories': self.categories,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def mark_solved(self, solution: PuzzleSolution):
        """Mark a puzzle as solved."""
        self.solutions[solution.puzzle_id] = solution
        
        # Update category
        if solution.category not in self.categories:
            self.categories[solution.category] = []
        if solution.puzzle_id not in self.categories[solution.category]:
            self.categories[solution.category].append(solution.puzzle_id)
        
        self.save()
    
    def is_solved(self, puzzle_id: str) -> bool:
        return puzzle_id in self.solutions
    
    def get_stats(self) -> Dict:
        return {
            'total_solved': len(self.solutions),
            'by_category': {k: len(v) for k, v in self.categories.items()}
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVER LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

class SolverLibrary:
    """Library of solving functions."""
    
    def __init__(self):
        self.solvers: Dict[str, Callable] = {}
        self._register_builtin_solvers()
    
    def _register_builtin_solvers(self):
        """Register built-in solvers."""
        
        # Geometric transforms
        self.solvers['flip_h'] = lambda x: np.fliplr(x)
        self.solvers['flip_v'] = lambda x: np.flipud(x)
        self.solvers['rot90'] = lambda x: np.rot90(x, 1)
        self.solvers['rot180'] = lambda x: np.rot90(x, 2)
        self.solvers['rot270'] = lambda x: np.rot90(x, 3)
        self.solvers['transpose'] = lambda x: x.T
        self.solvers['identity'] = lambda x: x
        
        # Will add more as we solve them
    
    def add_colormap_solver(self, name: str, colormap: Dict[int, int]):
        """Add a color mapping solver."""
        def solver(grid):
            result = grid.copy()
            for c_in, c_out in colormap.items():
                result[grid == c_in] = c_out
            return result
        self.solvers[name] = solver
    
    def get(self, name: str) -> Optional[Callable]:
        return self.solvers.get(name)
    
    def list(self) -> List[str]:
        return list(self.solvers.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# PUZZLE ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class PuzzleAnalyzer:
    """Analyze puzzles to detect patterns."""
    
    def __init__(self, library: SolverLibrary):
        self.library = library
    
    def analyze(self, puzzle: Dict) -> Dict:
        """Analyze a puzzle and return insights."""
        train = puzzle['train']
        test = puzzle['test']
        
        # Basic stats
        input_shapes = [tuple(np.array(ex['input']).shape) for ex in train]
        output_shapes = [tuple(np.array(ex['output']).shape) for ex in train]
        
        has_size_change = any(i != o for i, o in zip(input_shapes, output_shapes))
        
        # Detect potential rule
        detected_rule = None
        
        if not has_size_change:
            # Try geometric transforms
            for name in ['flip_h', 'flip_v', 'rot90', 'rot180', 'rot270', 'transpose', 'identity']:
                solver = self.library.get(name)
                if self._test_solver(train, solver):
                    detected_rule = ('geometric', name)
                    break
            
            # Try color mapping
            if not detected_rule:
                cmap = self._detect_colormap(train)
                if cmap:
                    detected_rule = ('color_map', cmap)
        
        return {
            'n_train': len(train),
            'n_test': len(test),
            'input_shapes': input_shapes,
            'output_shapes': output_shapes,
            'has_size_change': has_size_change,
            'detected_rule': detected_rule,
            'colors_used': self._get_colors(train)
        }
    
    def _test_solver(self, examples: List[Dict], solver: Callable) -> bool:
        """Test if solver works on all examples."""
        for ex in examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            try:
                pred = solver(inp)
                if not np.array_equal(pred, out):
                    return False
            except:
                return False
        return True
    
    def _detect_colormap(self, examples: List[Dict]) -> Optional[Dict[int, int]]:
        """Detect if there's a consistent color mapping."""
        cmap = {}
        
        for ex in examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            if inp.shape != out.shape:
                return None
            
            for i in range(inp.shape[0]):
                for j in range(inp.shape[1]):
                    c_in = int(inp[i, j])
                    c_out = int(out[i, j])
                    if c_in in cmap:
                        if cmap[c_in] != c_out:
                            return None
                    else:
                        cmap[c_in] = c_out
        
        # Check non-identity
        if all(k == v for k, v in cmap.items()):
            return None
        
        return cmap
    
    def _get_colors(self, examples: List[Dict]) -> set:
        """Get all colors used in examples."""
        colors = set()
        for ex in examples:
            colors.update(np.unique(ex['input']))
            colors.update(np.unique(ex['output']))
        return colors


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# ANSI color codes for terminal display
COLORS = {
    0: '\033[40m  \033[0m',   # Black
    1: '\033[44m  \033[0m',   # Blue
    2: '\033[41m  \033[0m',   # Red
    3: '\033[42m  \033[0m',   # Green
    4: '\033[43m  \033[0m',   # Yellow
    5: '\033[47m  \033[0m',   # Gray
    6: '\033[45m  \033[0m',   # Magenta
    7: '\033[48;5;208m  \033[0m',  # Orange
    8: '\033[46m  \033[0m',   # Cyan
    9: '\033[48;5;52m  \033[0m',   # Maroon
}


def display_grid(grid: np.ndarray, label: str = ""):
    """Display a grid with colors."""
    if label:
        print(f"\n{label} ({grid.shape[0]}×{grid.shape[1]}):")
    
    for row in grid:
        line = ""
        for val in row:
            line += COLORS.get(int(val), f'[{val}]')
        print(f"  {line}")


def display_puzzle(puzzle: Dict, puzzle_id: str = ""):
    """Display a full puzzle."""
    print(f"\n{'═' * 60}")
    if puzzle_id:
        print(f"PUZZLE: {puzzle_id}")
    print(f"{'═' * 60}")
    
    print(f"\n📚 TRAINING EXAMPLES ({len(puzzle['train'])})")
    for i, ex in enumerate(puzzle['train']):
        print(f"\n--- Example {i+1} ---")
        display_grid(np.array(ex['input']), "Input")
        display_grid(np.array(ex['output']), "Output")
    
    print(f"\n🎯 TEST CASES ({len(puzzle['test'])})")
    for i, test in enumerate(puzzle['test']):
        print(f"\n--- Test {i+1} ---")
        display_grid(np.array(test['input']), "Input")
        if 'output' in test:
            display_grid(np.array(test['output']), "Expected Output")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════

class ARCSolverApp:
    """Main application for solving ARC puzzles."""
    
    def __init__(self, data_dir: str = "./data/arc"):
        self.loader = ARCLoader(data_dir=data_dir)
        self.tracker = ProgressTracker()
        self.library = SolverLibrary()
        self.analyzer = PuzzleAnalyzer(self.library)
        
        # Pre-register solved colormaps
        self._register_known_solutions()
    
    def _register_known_solutions(self):
        """Register solutions we've already found."""
        known = [
            ('67a3c6ac', 'geometric', 'flip_h', 'Horizontal Flip'),
            ('68b16354', 'geometric', 'flip_v', 'Vertical Flip'),
            ('ed36ccf7', 'geometric', 'rot90', 'Rotate 90°'),
            ('6150a2bd', 'geometric', 'rot180', 'Rotate 180°'),
            ('3c9b0459', 'geometric', 'rot180', 'Rotate 180°'),
            ('74dd1130', 'geometric', 'transpose', 'Matrix Transpose'),
            ('9dfd6313', 'geometric', 'transpose', 'Matrix Transpose'),
        ]
        
        colormaps = [
            ('0d3d703e', {3: 4, 1: 5, 2: 6, 8: 9}),
            ('b1948b0a', {6: 2, 7: 7}),
            ('c8f0f002', {1: 1, 7: 5, 8: 8}),
            ('d511f180', {5: 8, 8: 5, 1: 1, 2: 2, 3: 3, 4: 4, 6: 6, 7: 7, 9: 9}),
        ]
        
        for pid, category, solver, desc in known:
            if not self.tracker.is_solved(pid):
                self.tracker.mark_solved(PuzzleSolution(
                    puzzle_id=pid,
                    category=category,
                    rule_description=desc,
                    solver_function=solver,
                    solved_at=datetime.now().isoformat(),
                    test_accuracy=1.0
                ))
        
        for pid, cmap in colormaps:
            solver_name = f"colormap_{pid}"
            self.library.add_colormap_solver(solver_name, cmap)
            if not self.tracker.is_solved(pid):
                self.tracker.mark_solved(PuzzleSolution(
                    puzzle_id=pid,
                    category='color_map',
                    rule_description=f"Color map: {cmap}",
                    solver_function=solver_name,
                    solved_at=datetime.now().isoformat(),
                    test_accuracy=1.0
                ))
    
    def status(self):
        """Show current progress."""
        all_puzzles = self.loader.list_puzzles('training')
        stats = self.tracker.get_stats()
        
        print("\n" + "═" * 60)
        print("ARC-AGI SOLVER STATUS")
        print("═" * 60)
        print(f"\n✅ SOLVED: {stats['total_solved']}/400 ({100*stats['total_solved']/400:.1f}%)")
        print(f"❓ REMAINING: {400 - stats['total_solved']}")
        
        print("\n📊 BY CATEGORY:")
        for category, count in sorted(stats['by_category'].items(), key=lambda x: -x[1]):
            print(f"  {category}: {count}")
        
        print("\n🏆 SOLVED PUZZLES:")
        for pid, solution in sorted(self.tracker.solutions.items()):
            print(f"  ✓ {pid}: {solution.rule_description}")
    
    def analyze_puzzle(self, puzzle_id: str):
        """Analyze a specific puzzle."""
        puzzle = self.loader.load_puzzle(puzzle_id)
        if not puzzle:
            print(f"Puzzle '{puzzle_id}' not found")
            return
        
        display_puzzle(puzzle, puzzle_id)
        
        analysis = self.analyzer.analyze(puzzle)
        
        print(f"\n📋 ANALYSIS:")
        print(f"  Training examples: {analysis['n_train']}")
        print(f"  Test cases: {analysis['n_test']}")
        print(f"  Input shapes: {analysis['input_shapes']}")
        print(f"  Output shapes: {analysis['output_shapes']}")
        print(f"  Size change: {'Yes' if analysis['has_size_change'] else 'No'}")
        print(f"  Colors: {sorted(analysis['colors_used'])}")
        
        if analysis['detected_rule']:
            rule_type, rule_detail = analysis['detected_rule']
            print(f"\n🎯 DETECTED RULE: {rule_type}")
            print(f"   {rule_detail}")
        else:
            print(f"\n❓ No automatic rule detected")
        
        if self.tracker.is_solved(puzzle_id):
            solution = self.tracker.solutions[puzzle_id]
            print(f"\n✅ ALREADY SOLVED: {solution.rule_description}")
    
    def solve_puzzle(self, puzzle_id: str, solver_name: str = None):
        """Attempt to solve a puzzle."""
        puzzle = self.loader.load_puzzle(puzzle_id)
        if not puzzle:
            print(f"Puzzle '{puzzle_id}' not found")
            return False
        
        # Auto-detect if no solver specified
        if not solver_name:
            analysis = self.analyzer.analyze(puzzle)
            if analysis['detected_rule']:
                rule_type, rule_detail = analysis['detected_rule']
                if rule_type == 'geometric':
                    solver_name = rule_detail
                elif rule_type == 'color_map':
                    # Register the colormap solver
                    solver_name = f"colormap_{puzzle_id}"
                    self.library.add_colormap_solver(solver_name, rule_detail)
        
        if not solver_name:
            print(f"Could not auto-detect solver for {puzzle_id}")
            return False
        
        solver = self.library.get(solver_name)
        if not solver:
            print(f"Solver '{solver_name}' not found")
            return False
        
        # Test on training examples
        train = puzzle['train']
        for ex in train:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            try:
                pred = solver(inp)
                if not np.array_equal(pred, out):
                    print(f"❌ Solver failed on training example")
                    return False
            except Exception as e:
                print(f"❌ Solver error: {e}")
                return False
        
        # Test on test cases
        test = puzzle['test']
        correct = 0
        for t in test:
            inp = np.array(t['input'])
            out = np.array(t['output'])
            try:
                pred = solver(inp)
                if np.array_equal(pred, out):
                    correct += 1
            except:
                pass
        
        accuracy = correct / len(test)
        
        if accuracy == 1.0:
            print(f"✅ SOLVED: {puzzle_id} using {solver_name}")
            self.tracker.mark_solved(PuzzleSolution(
                puzzle_id=puzzle_id,
                category='geometric' if solver_name in ['flip_h', 'flip_v', 'rot90', 'rot180', 'rot270', 'transpose'] else 'color_map',
                rule_description=solver_name,
                solver_function=solver_name,
                solved_at=datetime.now().isoformat(),
                test_accuracy=accuracy
            ))
            return True
        else:
            print(f"❌ Failed: {correct}/{len(test)} test cases")
            return False
    
    def scan_unsolved(self, limit: int = 20):
        """Scan unsolved puzzles for patterns."""
        all_puzzles = self.loader.load_all('training')
        
        print("\n" + "═" * 60)
        print("SCANNING UNSOLVED PUZZLES")
        print("═" * 60)
        
        categories = {
            'auto_solvable': [],
            'same_size': [],
            'size_change': [],
        }
        
        count = 0
        for pid, puzzle in all_puzzles.items():
            if self.tracker.is_solved(pid):
                continue
            
            analysis = self.analyzer.analyze(puzzle)
            
            if analysis['detected_rule']:
                categories['auto_solvable'].append((pid, analysis['detected_rule']))
            elif analysis['has_size_change']:
                categories['size_change'].append(pid)
            else:
                categories['same_size'].append(pid)
            
            count += 1
            if count >= limit:
                break
        
        print(f"\n🎯 AUTO-SOLVABLE ({len(categories['auto_solvable'])}):")
        for pid, rule in categories['auto_solvable'][:10]:
            print(f"  {pid}: {rule}")
        
        print(f"\n📐 SAME SIZE - NEEDS ANALYSIS ({len(categories['same_size'])}):")
        for pid in categories['same_size'][:10]:
            print(f"  {pid}")
        
        print(f"\n📏 SIZE CHANGE ({len(categories['size_change'])}):")
        for pid in categories['size_change'][:10]:
            print(f"  {pid}")
    
    def auto_solve_all(self):
        """Attempt to auto-solve all puzzles."""
        all_puzzles = self.loader.load_all('training')
        
        print("\n" + "═" * 60)
        print("AUTO-SOLVING ALL PUZZLES")
        print("═" * 60)
        
        newly_solved = 0
        
        for pid, puzzle in all_puzzles.items():
            if self.tracker.is_solved(pid):
                continue
            
            if self.solve_puzzle(pid):
                newly_solved += 1
        
        print(f"\n✅ Newly solved: {newly_solved}")
        self.status()
    
    def interactive(self):
        """Interactive menu."""
        while True:
            print("\n" + "═" * 60)
            print("ARC-AGI SOLVER - MAIN MENU")
            print("═" * 60)
            print("1. Show status")
            print("2. Analyze puzzle")
            print("3. Solve puzzle")
            print("4. Scan unsolved")
            print("5. Auto-solve all")
            print("6. List solved")
            print("7. Exit")
            
            choice = input("\nChoice: ").strip()
            
            if choice == '1':
                self.status()
            elif choice == '2':
                pid = input("Puzzle ID: ").strip()
                self.analyze_puzzle(pid)
            elif choice == '3':
                pid = input("Puzzle ID: ").strip()
                self.solve_puzzle(pid)
            elif choice == '4':
                self.scan_unsolved()
            elif choice == '5':
                self.auto_solve_all()
            elif choice == '6':
                for pid, sol in sorted(self.tracker.solutions.items()):
                    print(f"  {pid}: {sol.rule_description}")
            elif choice == '7':
                print("Goodbye!")
                break


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='ARC-AGI Solver App')
    parser.add_argument('--status', action='store_true', help='Show progress')
    parser.add_argument('--solve', type=str, help='Solve specific puzzle')
    parser.add_argument('--analyze', type=str, help='Analyze specific puzzle')
    parser.add_argument('--auto', action='store_true', help='Auto-solve all')
    parser.add_argument('--scan', type=int, default=0, help='Scan N unsolved puzzles')
    
    args = parser.parse_args()
    
    app = ARCSolverApp()
    
    if args.status:
        app.status()
    elif args.solve:
        app.solve_puzzle(args.solve)
    elif args.analyze:
        app.analyze_puzzle(args.analyze)
    elif args.auto:
        app.auto_solve_all()
    elif args.scan > 0:
        app.scan_unsolved(args.scan)
    else:
        app.interactive()


if __name__ == '__main__':
    main()
