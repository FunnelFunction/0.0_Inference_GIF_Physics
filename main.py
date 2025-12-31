#!/usr/bin/env python3
"""
main.py - Entry Point for Inference GIF Physics

Run ARC-AGI solver from command line.

Usage:
    python main.py                    # Run on example puzzles
    python main.py --download         # Download ARC dataset
    python main.py --evaluate 10      # Evaluate on 10 random puzzles
    python main.py --puzzle 00d62c1b  # Solve specific puzzle
"""

import argparse
import sys
import numpy as np

# Add core to path
sys.path.insert(0, '.')

from core import InferenceSolver, SolverConfig
from experiments.arc_loader import (
    ARCLoader, EXAMPLE_PUZZLES, 
    get_example_puzzle, list_example_puzzles
)


def run_examples():
    """Run on built-in example puzzles."""
    print("=" * 60)
    print("INFERENCE GIF PHYSICS - Example Puzzles")
    print("=" * 60)
    
    solver = InferenceSolver()
    
    for name in list_example_puzzles():
        puzzle = get_example_puzzle(name)
        print(f"\n📦 Puzzle: {name}")
        print("-" * 40)
        
        results = solver.solve_with_results(puzzle)
        
        for i, result in enumerate(results):
            test = puzzle['test'][i]
            pred = result.predicted_grid
            
            print(f"  Test {i}:")
            print(f"    Method: {result.transformation_used}")
            if result.matched_attractor:
                print(f"    Attractor: {result.matched_attractor}")
            print(f"    Confidence: {result.confidence:.1%}")
            print(f"    Time: {result.solve_time*1000:.1f}ms")
            
            if 'output' in test:
                target = np.array(test['output'])
                correct = np.array_equal(pred, target)
                print(f"    Correct: {'✓' if correct else '✗'}")
                
                if not correct:
                    print(f"    Predicted:\n{pred}")
                    print(f"    Expected:\n{target}")


def download_arc():
    """Download ARC dataset."""
    loader = ARCLoader()
    success = loader.download()
    if success:
        print(f"✓ Downloaded ARC dataset to {loader.data_dir}")
        print(f"  Training puzzles: {len(loader.list_puzzles('training'))}")
        print(f"  Evaluation puzzles: {len(loader.list_puzzles('evaluation'))}")
    else:
        print("✗ Download failed")
        sys.exit(1)


def evaluate(n_puzzles: int, split: str = 'training'):
    """Evaluate solver on random puzzles."""
    from core.inference_solver import evaluate_solver
    
    loader = ARCLoader()
    
    if not loader.training_dir.exists():
        print("ARC dataset not found. Run with --download first.")
        sys.exit(1)
    
    print(f"Loading {n_puzzles} random puzzles from {split}...")
    puzzles = loader.load_sample(n_puzzles, split=split)
    
    print(f"\nEvaluating...")
    print("-" * 40)
    
    solver = InferenceSolver()
    results = evaluate_solver(solver, puzzles, verbose=True)
    
    print("\n" + "=" * 40)
    print(f"FINAL ACCURACY: {results['accuracy']:.1%}")
    print(f"({results['correct_predictions']}/{results['total_test_cases']} test cases)")


def solve_puzzle(puzzle_id: str, split: str = 'training'):
    """Solve a specific puzzle."""
    loader = ARCLoader()
    puzzle = loader.load_puzzle(puzzle_id, split=split)
    
    if puzzle is None:
        print(f"Puzzle '{puzzle_id}' not found in {split}")
        sys.exit(1)
    
    print(f"Solving puzzle: {puzzle_id}")
    print(f"Training examples: {len(puzzle['train'])}")
    print(f"Test cases: {len(puzzle['test'])}")
    print("-" * 40)
    
    solver = InferenceSolver()
    results = solver.solve_with_results(puzzle)
    
    for i, result in enumerate(results):
        test = puzzle['test'][i]
        pred = result.predicted_grid
        
        print(f"\nTest {i}:")
        print(f"  Method: {result.transformation_used}")
        if result.matched_attractor:
            print(f"  Attractor: {result.matched_attractor}")
        print(f"  Time: {result.solve_time*1000:.1f}ms")
        print(f"  Predicted:\n{pred}")
        
        if 'output' in test:
            target = np.array(test['output'])
            correct = np.array_equal(pred, target)
            print(f"  Expected:\n{target}")
            print(f"  Correct: {'✓' if correct else '✗'}")


def main():
    parser = argparse.ArgumentParser(
        description='Inference GIF Physics - ARC-AGI Solver'
    )
    
    parser.add_argument('--download', action='store_true',
                       help='Download ARC dataset')
    parser.add_argument('--evaluate', type=int, metavar='N',
                       help='Evaluate on N random puzzles')
    parser.add_argument('--puzzle', type=str, metavar='ID',
                       help='Solve specific puzzle by ID')
    parser.add_argument('--split', type=str, default='training',
                       choices=['training', 'evaluation'],
                       help='Dataset split to use')
    
    args = parser.parse_args()
    
    if args.download:
        download_arc()
    elif args.evaluate:
        evaluate(args.evaluate, args.split)
    elif args.puzzle:
        solve_puzzle(args.puzzle, args.split)
    else:
        # Default: run examples
        run_examples()


if __name__ == '__main__':
    main()
