"""
Intent Tensor Theory - Self-Resolving Tensor Field V10
Added: border pattern, colormap

The math: ∇²Φ = f(ΔΨ, κ)
"""

import numpy as np
from typing import List, Dict
from scipy import ndimage

class IntentTensorFieldV10:
    
    def __init__(self, examples: List[Dict], debug: bool = False):
        self.examples = examples
        self.debug = debug
        self.d0 = self.d1 = self.d2 = self.d3 = None
        self.transform = None
        self.solution_type = None
    
    def compute_d0(self):
        self.d0 = []
        for ex in self.examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            seed = {
                'inp': inp, 'out': out,
                'same_shape': inp.shape == out.shape,
                'h_ratio': out.shape[0] / inp.shape[0] if inp.shape[0] else 0,
                'w_ratio': out.shape[1] / inp.shape[1] if inp.shape[1] else 0,
            }
            if seed['same_shape']:
                seed['new'] = (inp == 0) & (out != 0)
                seed['removed'] = (inp != 0) & (out == 0)
                seed['changed'] = (inp != 0) & (out != 0) & (inp != out)
            self.d0.append(seed)
        return self
    
    def compute_d1(self):
        s = self.d0[0]
        self.d1 = {'same_shape': s['same_shape']}
        
        if s['same_shape']:
            total_new = sum(np.sum(d['new']) for d in self.d0)
            total_removed = sum(np.sum(d['removed']) for d in self.d0)
            total_changed = sum(np.sum(d['changed']) for d in self.d0)
            
            self.d1['total_new'] = total_new
            self.d1['total_removed'] = total_removed
            self.d1['total_changed'] = total_changed
            
            if total_new == 0 and total_removed == 0 and total_changed == 0:
                self.d1['type'] = 'identity'
            elif total_new > 0 and total_removed == 0 and total_changed == 0:
                self.d1['type'] = 'fill'
            elif total_changed > 0 and total_new == 0 and total_removed == 0:
                self.d1['type'] = 'recolor'
            else:
                self.d1['type'] = 'transform'
        else:
            h, w = s['h_ratio'], s['w_ratio']
            if h == w == 2: self.d1['type'] = 'double'
            elif h == w == 3: self.d1['type'] = 'triple'
            elif h == 1 and w == 2: self.d1['type'] = 'concat_h'
            elif h == 2 and w == 1: self.d1['type'] = 'concat_v'
            else: self.d1['type'] = 'expand'
        return self
    
    def compute_d2(self):
        self.d2 = {}
        
        if self.d1['same_shape']:
            # Always check geometric first
            geo = self._detect_geometric()
            if geo:
                self.d2['geo'] = geo
                return self
            
            # Check border pattern (input all zeros)
            border = self._detect_border()
            if border:
                self.d2['border'] = border
                return self
            
            # Check colormap (pure recolor)
            if self.d1['type'] == 'recolor':
                cmap = self._detect_colormap()
                if cmap:
                    self.d2['colormap'] = cmap
                    return self
            
            # Check fill patterns
            if self.d1['type'] == 'fill':
                fill = self._detect_fill()
                if fill:
                    self.d2['fill'] = fill
                    return self
        else:
            exp = self._detect_expansion()
            if exp:
                self.d2['expand'] = exp
        return self
    
    def _detect_geometric(self):
        tests = [
            ('flip_h', lambda x: np.fliplr(x)),
            ('flip_v', lambda x: np.flipud(x)),
            ('rot90', lambda x: np.rot90(x, 1)),
            ('rot180', lambda x: np.rot90(x, 2)),
            ('rot270', lambda x: np.rot90(x, 3)),
        ]
        if self.d0[0]['inp'].shape[0] == self.d0[0]['inp'].shape[1]:
            tests.append(('transpose', lambda x: x.T))
        
        for name, fn in tests:
            if all(np.array_equal(fn(d['inp']), d['out']) for d in self.d0):
                return {'type': name}
        return None
    
    def _detect_border(self):
        """Detect border pattern: input all zeros, output has colored border."""
        for seed in self.d0:
            inp, out = seed['inp'], seed['out']
            
            # Input must be all zeros
            if np.any(inp != 0):
                return None
            
            # Output should have border filled
            h, w = out.shape
            
            # Check if it's a border pattern
            border_mask = np.zeros_like(out, dtype=bool)
            border_mask[0, :] = True
            border_mask[-1, :] = True
            border_mask[:, 0] = True
            border_mask[:, -1] = True
            
            interior_mask = ~border_mask
            
            # Border should be non-zero (single color), interior should be zero
            border_vals = out[border_mask]
            interior_vals = out[interior_mask]
            
            if len(set(border_vals)) != 1 or border_vals[0] == 0:
                return None
            if np.any(interior_vals != 0):
                return None
        
        # Get the border color from first example
        border_color = int(self.d0[0]['out'][0, 0])
        return {'type': 'border', 'color': border_color}
    
    def _detect_colormap(self):
        """Detect consistent color mapping across examples."""
        global_mapping = {}
        
        for seed in self.d0:
            inp, out = seed['inp'], seed['out']
            
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    ic, oc = int(inp[r, c]), int(out[r, c])
                    if ic in global_mapping:
                        if global_mapping[ic] != oc:
                            return None  # Inconsistent mapping
                    else:
                        global_mapping[ic] = oc
        
        # Verify mapping works for all examples
        for seed in self.d0:
            inp, out = seed['inp'], seed['out']
            mapped = np.vectorize(lambda x: global_mapping.get(int(x), x))(inp)
            if not np.array_equal(mapped, out):
                return None
        
        return {'type': 'colormap', 'mapping': global_mapping}
    
    def _detect_fill(self):
        """Detect fill patterns with STRICT verification."""
        all_offsets = []
        pattern_results = []
        
        for seed in self.d0:
            inp, out = seed['inp'], seed['out']
            new_mask = seed['new']
            
            if not np.any(new_mask):
                pattern_results.append({'empty': True})
                continue
            
            nz_pos = np.argwhere(inp != 0)
            new_pos = np.argwhere(new_mask)
            new_set = set(tuple(p) for p in new_pos)
            
            fill_color = out[new_mask][0]
            anchor_color = inp[inp != 0][0] if np.any(inp != 0) else 0
            offset = int(fill_color) - int(anchor_color)
            all_offsets.append(offset)
            
            results = {}
            
            # 1. ENCLOSED (binary_fill_holes)
            if np.any(inp != 0):
                primary = int(np.bincount(inp[inp != 0].flatten()).argmax())
                filled = ndimage.binary_fill_holes(inp == primary)
                enclosed_set = set(tuple(p) for p in np.argwhere((filled) & (inp == 0)))
                results['enclosed'] = enclosed_set == new_set and len(new_set) > 0
            else:
                results['enclosed'] = False
            
            # 2. CROSSFILL
            results['crossfill'] = self._check_crossfill(inp, new_set)
            
            # 3. CONNECT_ROW
            row_expected = self._get_connect_row_positions(inp)
            results['connect_row'] = row_expected == new_set and len(new_set) > 0
            
            # 4. CONNECT_COL
            col_expected = self._get_connect_col_positions(inp)
            results['connect_col'] = col_expected == new_set and len(new_set) > 0
            
            # 5. CONNECT_ROW_OR_COL
            row_or_col_expected = row_expected | col_expected
            results['connect_row_or_col'] = row_or_col_expected == new_set and len(new_set) > 0
            
            # 6. BBOX
            if len(nz_pos) > 0:
                r_min, c_min = nz_pos.min(axis=0)
                r_max, c_max = nz_pos.max(axis=0)
                bbox_expected = set()
                for r in range(r_min, r_max + 1):
                    for c in range(c_min, c_max + 1):
                        if inp[r, c] == 0:
                            bbox_expected.add((r, c))
                results['bbox'] = bbox_expected == new_set and len(new_set) > 0
            else:
                results['bbox'] = False
            
            pattern_results.append(results)
        
        if not pattern_results:
            return None
        
        offset = all_offsets[0] if all_offsets else 0
        if not all(o == offset for o in all_offsets):
            return None
        
        for key in ['enclosed', 'crossfill', 'connect_row', 'connect_col', 'connect_row_or_col', 'bbox']:
            if all(r.get('empty') or r.get(key) for r in pattern_results):
                return {'type': key, 'color_offset': offset}
        
        return None
    
    def _get_connect_row_positions(self, inp):
        nz_pos = np.argwhere(inp != 0)
        expected = set()
        for r in range(inp.shape[0]):
            row_nz = nz_pos[nz_pos[:, 0] == r]
            if len(row_nz) >= 2:
                c_min, c_max = row_nz[:, 1].min(), row_nz[:, 1].max()
                for c in range(c_min + 1, c_max):
                    if inp[r, c] == 0:
                        expected.add((r, c))
        return expected
    
    def _get_connect_col_positions(self, inp):
        nz_pos = np.argwhere(inp != 0)
        expected = set()
        for c in range(inp.shape[1]):
            col_nz = nz_pos[nz_pos[:, 1] == c]
            if len(col_nz) >= 2:
                r_min, r_max = col_nz[:, 0].min(), col_nz[:, 0].max()
                for r in range(r_min + 1, r_max):
                    if inp[r, c] == 0:
                        expected.add((r, c))
        return expected
    
    def _check_crossfill(self, inp, new_set):
        nz_pos = np.argwhere(inp != 0)
        if len(nz_pos) < 2: return False
        
        row_counts = {}
        for p in nz_pos: row_counts[p[0]] = row_counts.get(p[0], 0) + 1
        header_row = max(row_counts.keys(), key=lambda r: row_counts[r])
        
        col_counts = {}
        for p in nz_pos:
            if p[0] != header_row:
                col_counts[p[1]] = col_counts.get(p[1], 0) + 1
        if not col_counts: return False
        
        header_col = max(col_counts.keys(), key=lambda c: col_counts[c])
        
        fill_cols = [p[1] for p in nz_pos if p[0] == header_row and p[1] != header_col]
        fill_rows = [p[0] for p in nz_pos if p[1] == header_col and p[0] != header_row]
        
        expected = {(r, c) for r in fill_rows for c in fill_cols if inp[r, c] == 0}
        return expected == new_set and len(expected) > 0
    
    def _detect_expansion(self):
        s = self.d0[0]
        inp, out = s['inp'], s['out']
        patterns = {}
        
        if s['h_ratio'] == s['w_ratio'] and s['h_ratio'] == int(s['h_ratio']):
            n = int(s['h_ratio'])
            if np.array_equal(np.repeat(np.repeat(inp, n, 0), n, 1), out):
                patterns['upscale'] = n
        
        if s['h_ratio'] == int(s['h_ratio']) and s['w_ratio'] == int(s['w_ratio']):
            h, w = int(s['h_ratio']), int(s['w_ratio'])
            if np.array_equal(np.tile(inp, (h, w)), out):
                patterns['tile'] = (h, w)
        
        if s['h_ratio'] == 1 and s['w_ratio'] == 2:
            if np.array_equal(np.hstack([inp, np.fliplr(inp)]), out):
                patterns['concat_h_flip'] = True
            elif np.array_equal(np.hstack([inp, inp]), out):
                patterns['concat_h'] = True
        
        if s['h_ratio'] == 2 and s['w_ratio'] == 1:
            if np.array_equal(np.vstack([inp, np.flipud(inp)]), out):
                patterns['concat_v_flip'] = True
            elif np.array_equal(np.vstack([inp, inp]), out):
                patterns['concat_v'] = True
        
        if s['h_ratio'] == 2 and s['w_ratio'] == 2:
            mirror_2x2 = np.vstack([
                np.hstack([inp, np.fliplr(inp)]),
                np.hstack([np.flipud(inp), np.rot90(inp, 2)])
            ])
            if np.array_equal(mirror_2x2, out):
                patterns['mirror_2x2'] = True
            
            tile_mirror_h = np.vstack([
                np.hstack([inp, np.fliplr(inp)]),
                np.hstack([inp, np.fliplr(inp)])
            ])
            if np.array_equal(tile_mirror_h, out):
                patterns['tile_mirror_h'] = True
        
        # Verify all examples
        for key in list(patterns.keys()):
            for seed in self.d0:
                i, o = seed['inp'], seed['out']
                if key == 'upscale':
                    exp = np.repeat(np.repeat(i, patterns[key], 0), patterns[key], 1)
                elif key == 'tile':
                    exp = np.tile(i, patterns[key])
                elif key == 'concat_h_flip':
                    exp = np.hstack([i, np.fliplr(i)])
                elif key == 'concat_h':
                    exp = np.hstack([i, i])
                elif key == 'concat_v_flip':
                    exp = np.vstack([i, np.flipud(i)])
                elif key == 'concat_v':
                    exp = np.vstack([i, i])
                elif key == 'mirror_2x2':
                    exp = np.vstack([np.hstack([i, np.fliplr(i)]), np.hstack([np.flipud(i), np.rot90(i, 2)])])
                elif key == 'tile_mirror_h':
                    exp = np.vstack([np.hstack([i, np.fliplr(i)]), np.hstack([i, np.fliplr(i)])])
                else:
                    continue
                if not np.array_equal(exp, o):
                    del patterns[key]
                    break
        
        return patterns if patterns else None
    
    def collapse(self):
        if self.d2.get('geo'): return self._collapse_geo()
        if self.d2.get('border'): return self._collapse_border()
        if self.d2.get('colormap'): return self._collapse_colormap()
        if self.d2.get('fill'): return self._collapse_fill()
        if self.d2.get('expand'): return self._collapse_expand()
        self.d3 = {'collapsed': False}
        return self
    
    def _collapse_geo(self):
        g = self.d2['geo']['type']
        fns = {
            'flip_h': lambda x: np.fliplr(np.array(x)),
            'flip_v': lambda x: np.flipud(np.array(x)),
            'rot90': lambda x: np.rot90(np.array(x), 1),
            'rot180': lambda x: np.rot90(np.array(x), 2),
            'rot270': lambda x: np.rot90(np.array(x), 3),
            'transpose': lambda x: np.array(x).T
        }
        self.transform = fns[g]
        self.solution_type = g
        self.d3 = {'collapsed': True}
        return self
    
    def _collapse_border(self):
        color = self.d2['border']['color']
        def fn(x, c=color):
            arr = np.array(x)
            result = np.zeros_like(arr)
            result[0, :] = c
            result[-1, :] = c
            result[:, 0] = c
            result[:, -1] = c
            return result
        self.transform = fn
        self.solution_type = 'border'
        self.d3 = {'collapsed': True}
        return self
    
    def _collapse_colormap(self):
        mapping = self.d2['colormap']['mapping']
        def fn(x, m=mapping):
            arr = np.array(x)
            return np.vectorize(lambda v: m.get(int(v), v))(arr)
        self.transform = fn
        self.solution_type = 'colormap'
        self.d3 = {'collapsed': True}
        return self
    
    def _collapse_fill(self):
        f = self.d2['fill']
        t, off = f['type'], f.get('color_offset', 0)
        
        if t == 'enclosed':
            def fn(x, off=off):
                arr = np.array(x)
                if not np.any(arr != 0): return arr
                primary = int(np.bincount(arr[arr != 0].flatten()).argmax())
                fc = primary + off
                filled = ndimage.binary_fill_holes(arr == primary)
                result = arr.copy()
                result[(filled) & (arr == 0)] = fc
                return result
            self.transform = fn
            
        elif t == 'crossfill':
            def fn(x, off=off):
                arr = np.array(x)
                result = arr.copy()
                primary = int(np.bincount(arr[arr != 0].flatten()).argmax()) if np.any(arr != 0) else 0
                fc = primary + off
                nz = np.argwhere(arr != 0)
                rc = {}
                for p in nz: rc[p[0]] = rc.get(p[0], 0) + 1
                hr = max(rc.keys(), key=lambda r: rc[r])
                cc = {}
                for p in nz:
                    if p[0] != hr: cc[p[1]] = cc.get(p[1], 0) + 1
                if cc:
                    hc = max(cc.keys(), key=lambda c: cc[c])
                    fcols = [p[1] for p in nz if p[0] == hr and p[1] != hc]
                    frows = [p[0] for p in nz if p[1] == hc and p[0] != hr]
                    for r in frows:
                        for c in fcols:
                            if result[r, c] == 0: result[r, c] = fc
                return result
            self.transform = fn
            
        elif t == 'connect_row':
            def fn(x, off=off):
                arr = np.array(x)
                result = arr.copy()
                primary = int(np.bincount(arr[arr != 0].flatten()).argmax()) if np.any(arr != 0) else 0
                fc = primary + off
                nz = np.argwhere(arr != 0)
                for r in range(arr.shape[0]):
                    row_nz = nz[nz[:, 0] == r]
                    if len(row_nz) >= 2:
                        for c in range(row_nz[:, 1].min() + 1, row_nz[:, 1].max()):
                            if result[r, c] == 0: result[r, c] = fc
                return result
            self.transform = fn
            
        elif t == 'connect_col':
            def fn(x, off=off):
                arr = np.array(x)
                result = arr.copy()
                primary = int(np.bincount(arr[arr != 0].flatten()).argmax()) if np.any(arr != 0) else 0
                fc = primary + off
                nz = np.argwhere(arr != 0)
                for c in range(arr.shape[1]):
                    col_nz = nz[nz[:, 1] == c]
                    if len(col_nz) >= 2:
                        for r in range(col_nz[:, 0].min() + 1, col_nz[:, 0].max()):
                            if result[r, c] == 0: result[r, c] = fc
                return result
            self.transform = fn
            
        elif t == 'connect_row_or_col':
            def fn(x, off=off):
                arr = np.array(x)
                result = arr.copy()
                primary = int(np.bincount(arr[arr != 0].flatten()).argmax()) if np.any(arr != 0) else 0
                fc = primary + off
                nz = np.argwhere(arr != 0)
                for r in range(arr.shape[0]):
                    row_nz = nz[nz[:, 0] == r]
                    if len(row_nz) >= 2:
                        for c in range(row_nz[:, 1].min() + 1, row_nz[:, 1].max()):
                            if result[r, c] == 0: result[r, c] = fc
                for c in range(arr.shape[1]):
                    col_nz = nz[nz[:, 1] == c]
                    if len(col_nz) >= 2:
                        for r in range(col_nz[:, 0].min() + 1, col_nz[:, 0].max()):
                            if result[r, c] == 0: result[r, c] = fc
                return result
            self.transform = fn
            
        elif t == 'bbox':
            def fn(x, off=off):
                arr = np.array(x)
                nz = np.argwhere(arr != 0)
                if len(nz) == 0: return arr
                primary = int(np.bincount(arr[arr != 0].flatten()).argmax())
                fc = primary + off
                r_min, c_min = nz.min(axis=0)
                r_max, c_max = nz.max(axis=0)
                result = arr.copy()
                for r in range(r_min, r_max + 1):
                    for c in range(c_min, c_max + 1):
                        if result[r, c] == 0: result[r, c] = fc
                return result
            self.transform = fn
        
        self.solution_type = t
        self.d3 = {'collapsed': True}
        return self
    
    def _collapse_expand(self):
        e = self.d2['expand']
        if 'upscale' in e:
            n = e['upscale']
            self.transform = lambda x, n=n: np.repeat(np.repeat(np.array(x), n, 0), n, 1)
            self.solution_type = f'upscale_{n}x'
        elif 'tile' in e:
            f = e['tile']
            self.transform = lambda x, f=f: np.tile(np.array(x), f)
            self.solution_type = f'tile_{f[0]}x{f[1]}'
        elif 'concat_h_flip' in e:
            self.transform = lambda x: np.hstack([np.array(x), np.fliplr(np.array(x))])
            self.solution_type = 'concat_h_flip'
        elif 'concat_h' in e:
            self.transform = lambda x: np.hstack([np.array(x), np.array(x)])
            self.solution_type = 'concat_h'
        elif 'concat_v_flip' in e:
            self.transform = lambda x: np.vstack([np.array(x), np.flipud(np.array(x))])
            self.solution_type = 'concat_v_flip'
        elif 'concat_v' in e:
            self.transform = lambda x: np.vstack([np.array(x), np.array(x)])
            self.solution_type = 'concat_v'
        elif 'mirror_2x2' in e:
            def fn(x):
                i = np.array(x)
                return np.vstack([np.hstack([i, np.fliplr(i)]), np.hstack([np.flipud(i), np.rot90(i, 2)])])
            self.transform = fn
            self.solution_type = 'mirror_2x2'
        elif 'tile_mirror_h' in e:
            def fn(x):
                i = np.array(x)
                return np.vstack([np.hstack([i, np.fliplr(i)]), np.hstack([i, np.fliplr(i)])])
            self.transform = fn
            self.solution_type = 'tile_mirror_h'
        
        self.d3 = {'collapsed': True}
        return self
    
    def resolve(self):
        return self.compute_d0().compute_d1().compute_d2().collapse()
    
    def apply(self, x):
        if not self.transform: raise ValueError("Not solved")
        return self.transform(x)
    
    def is_solved(self):
        return self.d3 and self.d3.get('collapsed', False)
    
    def summary(self):
        return self.solution_type if self.is_solved() else "UNSOLVED"

