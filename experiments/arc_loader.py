"""
arc_loader.py - Load ARC-AGI Puzzles

Downloads and loads puzzles from the ARC-AGI dataset.

Source: https://github.com/fchollet/ARC-AGI

The dataset contains:
    - 400 training puzzles
    - 400 evaluation puzzles
    - Each puzzle has 2-10 training examples + 1-3 test cases
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import zipfile
import io


# ARC-AGI release URL
ARC_RELEASE_URL = "https://github.com/fchollet/ARC-AGI/archive/refs/tags/v1.0.2.zip"
ARC_RELEASE_TAG = "v1.0.2"


class ARCLoader:
    """
    Load ARC-AGI puzzles.
    """
    
    def __init__(self, data_dir: str = "./data/arc"):
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "training"
        self.evaluation_dir = self.data_dir / "evaluation"
    
    def download(self, force: bool = False) -> bool:
        """
        Download ARC dataset from GitHub.
        
        Returns True if download was successful.
        """
        if self.training_dir.exists() and not force:
            print(f"ARC data already exists at {self.data_dir}")
            return True
        
        print(f"Downloading ARC {ARC_RELEASE_TAG}...")
        
        try:
            # Download zip
            response = urllib.request.urlopen(ARC_RELEASE_URL)
            zip_data = response.read()
            
            # Extract
            with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
                # Find the data directories
                for name in zf.namelist():
                    if '/data/training/' in name or '/data/evaluation/' in name:
                        # Extract with adjusted path
                        parts = name.split('/')
                        if len(parts) >= 3:
                            if 'training' in name:
                                target = self.training_dir / parts[-1]
                            else:
                                target = self.evaluation_dir / parts[-1]
                            
                            if parts[-1].endswith('.json'):
                                target.parent.mkdir(parents=True, exist_ok=True)
                                with zf.open(name) as src:
                                    target.write_bytes(src.read())
            
            print(f"Downloaded to {self.data_dir}")
            return True
        
        except Exception as e:
            print(f"Download failed: {e}")
            return False
    
    def load_puzzle(self, puzzle_id: str, 
                    split: str = 'training') -> Optional[Dict]:
        """
        Load a single puzzle by ID.
        
        Args:
            puzzle_id: The puzzle filename (without .json)
            split: 'training' or 'evaluation'
        
        Returns:
            Puzzle dictionary or None if not found
        """
        if split == 'training':
            path = self.training_dir / f"{puzzle_id}.json"
        else:
            path = self.evaluation_dir / f"{puzzle_id}.json"
        
        if not path.exists():
            return None
        
        with open(path) as f:
            return json.load(f)
    
    def load_all(self, split: str = 'training') -> Dict[str, Dict]:
        """
        Load all puzzles from a split.
        
        Returns:
            Dictionary of puzzle_id -> puzzle
        """
        if split == 'training':
            directory = self.training_dir
        else:
            directory = self.evaluation_dir
        
        if not directory.exists():
            return {}
        
        puzzles = {}
        for path in directory.glob("*.json"):
            puzzle_id = path.stem
            with open(path) as f:
                puzzles[puzzle_id] = json.load(f)
        
        return puzzles
    
    def load_sample(self, n: int = 10, 
                    split: str = 'training',
                    seed: int = 42) -> Dict[str, Dict]:
        """
        Load a random sample of puzzles.
        """
        import random
        random.seed(seed)
        
        all_puzzles = self.load_all(split)
        
        if n >= len(all_puzzles):
            return all_puzzles
        
        sample_ids = random.sample(list(all_puzzles.keys()), n)
        return {pid: all_puzzles[pid] for pid in sample_ids}
    
    def puzzle_stats(self, puzzle: Dict) -> Dict:
        """
        Get statistics about a puzzle.
        """
        train = puzzle['train']
        test = puzzle['test']
        
        return {
            'n_train_examples': len(train),
            'n_test_cases': len(test),
            'train_input_sizes': [
                (len(ex['input']), len(ex['input'][0]) if ex['input'] else 0)
                for ex in train
            ],
            'train_output_sizes': [
                (len(ex['output']), len(ex['output'][0]) if ex['output'] else 0)
                for ex in train
            ],
            'size_changes': any(
                len(ex['input']) != len(ex['output']) or
                (ex['input'] and ex['output'] and 
                 len(ex['input'][0]) != len(ex['output'][0]))
                for ex in train
            )
        }
    
    def list_puzzles(self, split: str = 'training') -> List[str]:
        """List all puzzle IDs in a split."""
        if split == 'training':
            directory = self.training_dir
        else:
            directory = self.evaluation_dir
        
        if not directory.exists():
            return []
        
        return sorted([p.stem for p in directory.glob("*.json")])


def load_puzzle_from_json(json_str: str) -> Dict:
    """Load a puzzle from JSON string."""
    return json.loads(json_str)


def create_simple_puzzle(input_grids: List[List], 
                         output_grids: List[List],
                         test_inputs: List[List]) -> Dict:
    """
    Create a puzzle from raw grids.
    
    Useful for testing.
    """
    train = [
        {'input': inp, 'output': out}
        for inp, out in zip(input_grids, output_grids)
    ]
    
    test = [{'input': inp} for inp in test_inputs]
    
    return {'train': train, 'test': test}


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE PUZZLES FOR TESTING
# ═══════════════════════════════════════════════════════════════════════════════

EXAMPLE_PUZZLES = {
    # Simple identity
    'identity': {
        'train': [
            {'input': [[1,2],[3,4]], 'output': [[1,2],[3,4]]},
            {'input': [[5,6,7],[8,9,0]], 'output': [[5,6,7],[8,9,0]]}
        ],
        'test': [
            {'input': [[1,1],[2,2]], 'output': [[1,1],[2,2]]}
        ]
    },
    
    # Horizontal flip
    'flip_h': {
        'train': [
            {'input': [[1,2,3]], 'output': [[3,2,1]]},
            {'input': [[1,0],[0,1]], 'output': [[0,1],[1,0]]}
        ],
        'test': [
            {'input': [[1,2],[3,4]], 'output': [[2,1],[4,3]]}
        ]
    },
    
    # Vertical flip
    'flip_v': {
        'train': [
            {'input': [[1],[2],[3]], 'output': [[3],[2],[1]]},
            {'input': [[1,2],[3,4]], 'output': [[3,4],[1,2]]}
        ],
        'test': [
            {'input': [[1,1],[2,2],[3,3]], 'output': [[3,3],[2,2],[1,1]]}
        ]
    },
    
    # 90° rotation
    'rotate_90': {
        'train': [
            {'input': [[1,2],[3,4]], 'output': [[2,4],[1,3]]},
            {'input': [[1,2,3],[4,5,6]], 'output': [[3,6],[2,5],[1,4]]}
        ],
        'test': [
            {'input': [[1,0,0],[0,1,0],[0,0,1]], 
             'output': [[0,0,1],[0,1,0],[1,0,0]]}
        ]
    },
    
    # Color inversion (swap 0 and 1)
    'invert_binary': {
        'train': [
            {'input': [[0,1],[1,0]], 'output': [[1,0],[0,1]]},
            {'input': [[0,0,1],[1,1,0]], 'output': [[1,1,0],[0,0,1]]}
        ],
        'test': [
            {'input': [[1,0,1],[0,1,0]], 'output': [[0,1,0],[1,0,1]]}
        ]
    }
}


def get_example_puzzle(name: str) -> Optional[Dict]:
    """Get a built-in example puzzle."""
    return EXAMPLE_PUZZLES.get(name)


def list_example_puzzles() -> List[str]:
    """List available example puzzles."""
    return list(EXAMPLE_PUZZLES.keys())
