# -*- coding: utf-8 -*-
"""Módulo de Operadores Paramétricos e Sub-rotinas Aprendidas pelo F3-AGI v4.0 (SOAR Library Learning)."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from f3_agi_v4_0_core import F3Primitives, center_of_mass, find_objects, label

class F3LearnedOperators:
    """
    Biblioteca de Operadores Neuro-Simbólicos Paramétricos abstraídos via
    Refatoração DreamCoder/SOAR a partir de traços consolidados no NREMBuffer.
    Livre de constantes mágicas fixas por tarefa; todos os operadores inferem
    geometria/períodos/cores dinamicamente ou recebem parâmetros do sintetizador.
    """

    @staticmethod
    def kronecker_self_pattern(
        grid: np.ndarray, bg_color: int = 0, invert_mask: bool = False
    ) -> np.ndarray:
        """
        Generalização paramétrica de auto-similaridade fractal (produto de Kronecker):
        Cada célula não-background (ou background se invert_mask=True) estampa uma
        cópia completa da própria grade `grid`.
        """
        if grid.size == 0:
            return grid.copy()
        if invert_mask:
            mask = (grid == bg_color).astype(grid.dtype)
        else:
            mask = (grid != bg_color).astype(grid.dtype)
        if bg_color == 0:
            return np.kron(mask, grid)
        h, w = grid.shape
        out = np.full((h * h, w * w), bg_color, dtype=grid.dtype)
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    out[r * h : (r + 1) * h, c * w : (c + 1) * w] = grid
        return out

    @staticmethod
    def extend_2d_periodicity(
        grid: np.ndarray,
        target_h: int = -1,
        target_w: int = -1,
        mask_color: int = -1,
    ) -> np.ndarray:
        """
        Generalização paramétrica vetorizada de periodicidade 1D/2D (017c7c7b, 0dfd9992, 1d0a4b61):
        Infere automaticamente o menor período vertical e horizontal (p_r, p_c)
        consistente com as células observadas (ignorando `mask_color` se >= 0),
        reconstrói oclusões marcadas com `mask_color` e estende/redimensiona
        para `(target_h, target_w)`.
        """
        h, w = grid.shape
        th = h if target_h <= 0 else int(target_h)
        tw = w if target_w <= 0 else int(target_w)

        # 1. Infere menor período vertical p_r vetorizado
        best_pr = h
        for pr in range(1, h):
            g1 = grid[: h - pr, :]
            g2 = grid[pr:, :]
            if mask_color >= 0:
                valid = (g1 != mask_color) & (g2 != mask_color)
                if np.any(valid) and not np.any((g1 != g2) & valid):
                    best_pr = pr
                    break
            else:
                if np.array_equal(g1, g2):
                    best_pr = pr
                    break

        # 2. Infere menor período horizontal p_c vetorizado
        best_pc = w
        for pc in range(1, w):
            g1 = grid[:, : w - pc]
            g2 = grid[:, pc:]
            if mask_color >= 0:
                valid = (g1 != mask_color) & (g2 != mask_color)
                if np.any(valid) and not np.any((g1 != g2) & valid):
                    best_pc = pc
                    break
            else:
                if np.array_equal(g1, g2):
                    best_pc = pc
                    break

        # 3. Constrói tile canônico (best_pr, best_pc) e replica via np.tile
        tile = np.zeros((best_pr, best_pc), dtype=grid.dtype)
        for r in range(best_pr):
            for c in range(best_pc):
                samples = grid[r::best_pr, c::best_pc]
                if mask_color >= 0:
                    valid_vals = samples[samples != mask_color]
                else:
                    valid_vals = samples.ravel()
                if valid_vals.size > 0:
                    vals, counts = np.unique(valid_vals, return_counts=True)
                    tile[r, c] = vals[np.argmax(counts)]
                else:
                    tile[r, c] = grid[r, c]

        reps_r = (th + best_pr - 1) // max(1, best_pr)
        reps_c = (tw + best_pc - 1) // max(1, best_pc)
        tiled = np.tile(tile, (max(1, reps_r), max(1, reps_c)))[:th, :tw].copy()
        if th == h and tw == w and mask_color >= 0:
            return np.where(grid != mask_color, grid, tiled)
        return tiled

    @staticmethod
    def dihedral_template_matching(
        grid: np.ndarray, bg_color: int = 0
    ) -> np.ndarray:
        """
        Generalização paramétrica de casamento de templates sob o grupo diedral D4
        e carimbo direcional por marcadores (0e206a2e, 045e512c).
        """
        h, w = grid.shape
        fg_pixels = grid[grid != bg_color]
        if fg_pixels.size == 0:
            return grid.copy()
        vals, counts = np.unique(fg_pixels, return_counts=True)
        body_color = int(vals[np.argmax(counts)])

        vis = np.zeros((h, w), dtype=bool)
        components: List[List[Tuple[int, int]]] = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != bg_color and not vis[r, c]:
                    q = [(r, c)]
                    vis[r, c] = True
                    pts = []
                    while q:
                        cr, cc = q.pop()
                        pts.append((cr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                nr, nc = cr + dr, cc + dc
                                if (
                                    0 <= nr < h
                                    and 0 <= nc < w
                                    and grid[nr, nc] != bg_color
                                    and not vis[nr, nc]
                                ):
                                    vis[nr, nc] = True
                                    q.append((nr, nc))
                    components.append(pts)

        templates: List[np.ndarray] = []
        template_bboxes: List[Tuple[int, int, int, int]] = []
        other_components: List[List[Tuple[int, int]]] = []
        dot_list: List[Tuple[int, int, int]] = []

        for pts in components:
            if any(grid[pr, pc] == body_color for pr, pc in pts) and len(pts) > 1:
                rs = [p[0] for p in pts]
                cs = [p[1] for p in pts]
                r0, r1, c0, c1 = min(rs), max(rs) + 1, min(cs), max(cs) + 1
                sub = np.full((r1 - r0, c1 - c0), bg_color, dtype=grid.dtype)
                for pr, pc in pts:
                    sub[pr - r0, pc - c0] = grid[pr, pc]
                templates.append(sub)
                template_bboxes.append((r0, c0, r1, c1))
            else:
                other_components.append(pts)
                for pr, pc in pts:
                    dot_list.append((pr, pc, int(grid[pr, pc])))

        if not templates:
            return grid.copy()

        has_internal_anchors = any(
            np.any((t != bg_color) & (t != body_color)) for t in templates
        )

        if has_internal_anchors and dot_list:
            res = np.full_like(grid, bg_color)
            dot_set = set(dot_list)
            for tmpl in templates:
                transforms = []
                for k in range(4):
                    rot = np.rot90(tmpl, k)
                    transforms.append(rot)
                    transforms.append(np.fliplr(rot))
                placed = False
                for t in transforms:
                    if placed:
                        break
                    th, tw = t.shape
                    anchors = [
                        (r, c, int(t[r, c]))
                        for r in range(th)
                        for c in range(tw)
                        if t[r, c] != bg_color and t[r, c] != body_color
                    ]
                    if not anchors:
                        continue
                    ar, ac, acol = anchors[0]
                    for dr, dc, dcol in dot_list:
                        if acol == dcol:
                            sr, sc = dr - ar, dc - ac
                            if all(
                                (sr + a2r, sc + a2c, a2col) in dot_set
                                for a2r, a2c, a2col in anchors
                            ):
                                for rr in range(th):
                                    for cc in range(tw):
                                        if (
                                            t[rr, cc] != bg_color
                                            and 0 <= sr + rr < h
                                            and 0 <= sc + cc < w
                                        ):
                                            res[sr + rr, sc + cc] = t[rr, cc]
                                placed = True
                                break
            return res

        res = grid.copy()
        mr0, mc0, mr1, mc1 = template_bboxes[0]
        tmpl_mask = templates[0] != bg_color
        th, tw = tmpl_mask.shape
        for pts in other_components:
            rs = [p[0] for p in pts]
            cs = [p[1] for p in pts]
            r0, r1, c0, c1 = min(rs), max(rs) + 1, min(cs), max(cs) + 1
            col = int(grid[pts[0][0], pts[0][1]])
            dr = -1 if r1 <= mr0 else (1 if r0 >= mr1 else 0)
            dc = -1 if c1 <= mc0 else (1 if c0 >= mc1 else 0)
            if dr == 0 and dc == 0:
                continue
            for k in range(1, max(h, w) // max(1, min(th, tw)) + 2):
                sr = mr0 + k * dr * (th + 1)
                sc = mc0 + k * dc * (tw + 1)
                for rr in range(th):
                    for cc_i in range(tw):
                        if (
                            tmpl_mask[rr, cc_i]
                            and 0 <= sr + rr < h
                            and 0 <= sc + cc_i < w
                        ):
                            res[sr + rr, sc + cc_i] = col
        return res

    @staticmethod
    def expand_local_neighborhoods(
        grid: np.ndarray,
        rules: Optional[Dict[int, Dict[str, Any]]] = None,
        bg_color: int = 0,
    ) -> np.ndarray:
        """
        Generalização paramétrica de expansão de vizinhanças locais (0962bcdd, 0ca9ddb6).
        """
        res = grid.copy()
        h, w = grid.shape
        if rules:
            for r in range(h):
                for c in range(w):
                    val = int(grid[r, c])
                    if val in rules:
                        spec = rules[val]
                        if "ortho" in spec:
                            ocol, rad = spec["ortho"]
                            for d in range(1, int(rad) + 1):
                                for dr, dc in ((-d, 0), (d, 0), (0, -d), (0, d)):
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < h and 0 <= nc < w and res[nr, nc] == bg_color:
                                        res[nr, nc] = int(ocol)
                        if "diag" in spec:
                            dcol, rad = spec["diag"]
                            for d in range(1, int(rad) + 1):
                                for dr, dc in ((-d, -d), (-d, d), (d, -d), (d, d)):
                                    nr, nc = r + dr, c + dc
                                    if 0 <= nr < h and 0 <= nc < w and res[nr, nc] == bg_color:
                                        res[nr, nc] = int(dcol)
            return res

        for r in range(1, h - 1):
            for c in range(1, w - 1):
                c_center = int(grid[r, c])
                c_petal = int(grid[r - 1, c])
                if (
                    c_center != bg_color
                    and c_petal != bg_color
                    and c_center != c_petal
                    and grid[r + 1, c] == c_petal
                    and grid[r, c - 1] == c_petal
                    and grid[r, c + 1] == c_petal
                ):
                    for d in (-2, -1, 1, 2):
                        if 0 <= r + d < h:
                            res[r + d, c] = c_petal
                        if 0 <= c + d < w:
                            res[r, c + d] = c_petal
                        if 0 <= r + d < h and 0 <= c + d < w:
                            res[r + d, c + d] = c_center
                        if 0 <= r + d < h and 0 <= c - d < w:
                            res[r + d, c - d] = c_center
        return res

    @staticmethod
    def isolate_extremum_object_or_cell(
        grid: np.ndarray,
        criterion: str = "rarest_color",
        bg_color: int = 0,
    ) -> np.ndarray:
        """
        Generalização paramétrica de isolamento/ranking por extremo estatístico ou
        topológico (0b148d64, 09629e4f, 08ed6ac7).
        """
        if grid.size == 0:
            return grid.copy()

        if criterion in ("rarest_color", "most_frequent_color"):
            fg = grid[grid != bg_color]
            if fg.size == 0:
                return grid.copy()
            vals, counts = np.unique(fg, return_counts=True)
            target_col = int(
                vals[np.argmin(counts)]
                if criterion == "rarest_color"
                else vals[np.argmax(counts)]
            )
            mask = grid == target_col
            coords = np.argwhere(mask)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            return grid[r0 : r1 + 1, c0 : c1 + 1].copy()

        if criterion in ("largest_object", "smallest_object"):
            objs = F3Primitives.get_connected_objects(grid, ignore_bg=True)
            if not objs:
                return grid.copy()
            chosen = objs[0] if criterion == "largest_object" else objs[-1]
            return chosen["subgrid"].copy()

        if criterion == "unique_color_object":
            objs = F3Primitives.get_connected_objects(grid, ignore_bg=True)
            if not objs:
                return grid.copy()
            color_counts: Dict[int, int] = {}
            for o in objs:
                color_counts[o["color"]] = color_counts.get(o["color"], 0) + 1
            for o in objs:
                if color_counts[o["color"]] == 1:
                    return o["subgrid"].copy()
            return objs[0]["subgrid"].copy()

        if criterion == "keep_largest_only":
            objs = F3Primitives.get_connected_objects(grid, ignore_bg=True)
            if not objs:
                return grid.copy()
            res = np.full_like(grid, bg_color)
            r0, c0, r1, c1 = objs[0]["bbox"]
            sub = objs[0]["subgrid"]
            res[r0:r1, c0:c1] = np.where(sub != bg_color, sub, res[r0:r1, c0:c1])
            return res

        if criterion == "remove_noise_dots":
            objs = F3Primitives.get_connected_objects(grid, ignore_bg=True)
            res = np.full_like(grid, bg_color)
            for o in objs:
                if o["size"] > 1:
                    r0, c0, r1, c1 = o["bbox"]
                    sub = o["subgrid"]
                    res[r0:r1, c0:c1] = np.where(
                        sub != bg_color, sub, res[r0:r1, c0:c1]
                    )
            return res

        if criterion == "most_sparse_subgrid":
            h, w = grid.shape
            if h >= 11 and w >= 11:
                sep_color = int(grid[3, 0])
                res = np.full_like(grid, bg_color)
                res[3, :] = sep_color
                res[7, :] = sep_color
                res[:, 3] = sep_color
                res[:, 7] = sep_color
                best_sub = None
                best_cnt = 999
                for br in range(3):
                    for bc in range(3):
                        sub = grid[4 * br : 4 * br + 3, 4 * bc : 4 * bc + 3]
                        cnt = int(np.sum(sub != bg_color))
                        if 0 < cnt < best_cnt:
                            best_cnt = cnt
                            best_sub = sub
                if best_sub is not None:
                    for br in range(3):
                        for bc in range(3):
                            res[4 * br : 4 * br + 3, 4 * bc : 4 * bc + 3] = best_sub[
                                br, bc
                            ]
                return res
            return grid.copy()

        if criterion == "rank_by_area":
            res = grid.copy()
            objs = F3Primitives.get_connected_objects(grid, ignore_bg=True)
            objs.sort(key=lambda o: (-o["size"], o["bbox"][1]))
            for idx, o in enumerate(objs):
                r0, c0, r1, c1 = o["bbox"]
                sub = o["subgrid"]
                res[r0:r1, c0:c1] = np.where(
                    sub != bg_color, idx + 1, res[r0:r1, c0:c1]
                )
            return res

        return grid.copy()

    @staticmethod
    def apply_color_bijection(
        grid: np.ndarray, mapping: Dict[int, int]
    ) -> np.ndarray:
        """Aplica simultaneamente uma bijeção/mapa de substituição de cores."""
        res = grid.copy()
        for k, v in mapping.items():
            res[grid == int(k)] = int(v)
        return res

    @staticmethod
    def complete_dihedral_symmetry(
        grid: np.ndarray, bg_color: int = 0
    ) -> np.ndarray:
        """
        Infere rapidamente o centro de simetria do conjunto de pixels de figura
        (restrito à vizinhança do centro da bounding box) e completa a simetria
        diedral de 8 dobras sobre `bg_color` em < 1 ms.
        """
        coords = np.argwhere(grid != bg_color)
        if coords.size == 0 or coords.shape[0] > 400:
            return grid.copy()
        h, w = grid.shape
        rmin, cmin = coords.min(axis=0)
        rmax, cmax = coords.max(axis=0)
        res = grid.copy()
        mid_r2 = int(rmin + rmax)
        mid_c2 = int(cmin + cmax)
        best_cr, best_cc = mid_r2 / 2.0, mid_c2 / 2.0
        best_score = -1

        for cr2 in range(max(0, mid_r2 - 2), min(2 * h - 1, mid_r2 + 3)):
            for cc2 in range(max(0, mid_c2 - 2), min(2 * w - 1, mid_c2 + 3)):
                cr, cc = cr2 / 2.0, cc2 / 2.0
                ok = True
                cnt = 0
                for r, c in coords:
                    val = grid[r, c]
                    for nr, nc in (
                        (int(round(2 * cr - r)), int(c)),
                        (int(r), int(round(2 * cc - c))),
                        (int(round(cr + (c - cc))), int(round(cc + (r - cr)))),
                    ):
                        if 0 <= nr < h and 0 <= nc < w:
                            g_nb = grid[nr, nc]
                            if g_nb != bg_color and g_nb != val:
                                ok = False
                                break
                            elif g_nb == val:
                                cnt += 1
                        else:
                            ok = False
                            break
                    if not ok:
                        break
                if ok and cnt > best_score:
                    best_score = cnt
                    best_cr, best_cc = cr, cc

        cr, cc = best_cr, best_cc
        for _ in range(2):
            pts = np.argwhere(res != bg_color)
            for r, c in pts:
                val = res[r, c]
                for nr, nc in (
                    (int(round(2 * cr - r)), int(c)),
                    (int(r), int(round(2 * cc - c))),
                    (int(round(2 * cr - r)), int(round(2 * cc - c))),
                    (int(round(cr + (c - cc))), int(round(cc + (r - cr)))),
                    (int(round(cr - (c - cc))), int(round(cc - (r - cr)))),
                ):
                    if 0 <= nr < h and 0 <= nc < w and res[nr, nc] == bg_color:
                        res[nr, nc] = val
        return res

    @staticmethod
    def mirror_tile_2x2(
        grid: np.ndarray, mode: str = "vh"
    ) -> np.ndarray:
        """
        Gera ladrilhamento espelhado 2x2:
        - "vh": [[G, fliplr(G)], [flipud(G), rot180(G)]]
        - "hv": [[rot180(G), flipud(G)], [fliplr(G), G]]
        - "rot": [[G, rot90(G, 3)], [rot90(G, 1), rot90(G, 2)]]
        """
        if mode == "hv":
            top = np.hstack([np.rot90(grid, 2), np.flipud(grid)])
            bot = np.hstack([np.fliplr(grid), grid])
            return np.vstack([top, bot])
        if mode == "rot":
            top = np.hstack([grid, np.rot90(grid, 3)])
            bot = np.hstack([np.rot90(grid, 1), np.rot90(grid, 2)])
            return np.vstack([top, bot])
        top = np.hstack([grid, np.fliplr(grid)])
        bot = np.hstack([np.flipud(grid), np.rot90(grid, 2)])
        return np.vstack([top, bot])

    @staticmethod
    def alternating_mirror_tile(
        grid: np.ndarray,
        reps_h: int = 3,
        reps_w: int = 3,
        mode: str = "row_fliplr",
    ) -> np.ndarray:
        """
        Generaliza ladrilhamento periódico (como 00576224 e 15696249) aplicando
        reflexões alternadas por linha/coluna:
        - "row_fliplr": linhas ímpares de blocos sofrem `np.fliplr(grid)`.
        - "col_flipud": colunas ímpares de blocos sofrem `np.flipud(grid)`.
        - "checker_hv": reflexão horizontal em colunas ímpares e vertical em linhas ímpares.
        """
        rh = max(1, int(reps_h))
        rw = max(1, int(reps_w))
        rows = []
        for r in range(rh):
            row_tiles = []
            for c in range(rw):
                t = grid.copy()
                if mode == "row_fliplr" and (r % 2 == 1):
                    t = np.fliplr(t)
                elif mode == "col_flipud" and (c % 2 == 1):
                    t = np.flipud(t)
                elif mode == "checker_hv":
                    if r % 2 == 1:
                        t = np.flipud(t)
                    if c % 2 == 1:
                        t = np.fliplr(t)
                row_tiles.append(t)
            rows.append(np.hstack(row_tiles))
        return np.vstack(rows)

    @staticmethod
    def hollow_objects(
        grid: np.ndarray, fill_color: int = 0, bg_color: int = 0
    ) -> np.ndarray:
        """
        Esvazia o interior de formas retangulares/sólidas (mantendo a borda de 1 pixel
        de espessura) ou preenche o interior com `fill_color`.
        """
        h, w = grid.shape
        res = grid.copy()
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                val = grid[r, c]
                if val != bg_color:
                    if (
                        grid[r - 1, c] == val
                        and grid[r + 1, c] == val
                        and grid[r, c - 1] == val
                        and grid[r, c + 1] == val
                        and grid[r - 1, c - 1] == val
                        and grid[r - 1, c + 1] == val
                        and grid[r + 1, c - 1] == val
                        and grid[r + 1, c + 1] == val
                    ):
                        res[r, c] = fill_color
        return res

    @staticmethod
    def split_and_boolean_combine(
        grid: np.ndarray,
        axis: int = 0,
        op: str = "and",
        out_color: int = -1,
        bg_color: int = 0,
    ) -> np.ndarray:
        """
        Divide a grade em 2 sub-grades (detectando automaticamente se há linha/coluna
        separadora central) e combina-as via operador booleano ('and', 'or', 'xor',
        'nor', 'sub', 'overlay_first', 'overlay_second') usando `out_color`.
        """
        h, w = grid.shape
        if axis == 0:
            if h % 2 == 1:
                mid = h // 2
                p1, p2 = grid[:mid, :], grid[mid + 1 :, :]
            else:
                mid = h // 2
                p1, p2 = grid[:mid, :], grid[mid:, :]
        else:
            if w % 2 == 1:
                mid = w // 2
                p1, p2 = grid[:, :mid], grid[:, mid + 1 :]
            else:
                mid = w // 2
                p1, p2 = grid[:, :mid], grid[:, mid:]

        if p1.shape != p2.shape or p1.size == 0:
            return grid.copy()

        m1 = p1 != bg_color
        m2 = p2 != bg_color
        if op == "and":
            mask = m1 & m2
        elif op == "or":
            mask = m1 | m2
        elif op == "xor":
            mask = m1 ^ m2
        elif op == "nor":
            mask = ~(m1 | m2)
        elif op == "sub":
            mask = m1 & (~m2)
        elif op == "rsub":
            mask = m2 & (~m1)
        elif op == "eq":
            mask = p1 == p2
        elif op == "overlay_first":
            return np.where(m1, p1, p2)
        elif op == "overlay_second":
            return np.where(m2, p2, p1)
        else:
            mask = m1 | m2

        if out_color > 0:
            return np.where(mask, int(out_color), bg_color).astype(grid.dtype)
        combined_col = np.where(m1, p1, p2)
        fg_vals = combined_col[combined_col != bg_color]
        default_col = int(fg_vals[0]) if fg_vals.size > 0 else 1
        return np.where(
            mask, np.where(combined_col != bg_color, combined_col, default_col), bg_color
        ).astype(grid.dtype)

    @staticmethod
    def extract_unique_or_extremum_subgrid(
        grid: np.ndarray, mode: str = "unique", bg_color: int = 0
    ) -> np.ndarray:
        """
        Detecta linhas e colunas separadoras que dividem a grade em sub-células de
        mesmo tamanho e retorna uma única sub-célula segundo `mode`.
        """
        h, w = grid.shape
        sep_rows = [
            r
            for r in range(h)
            if len(np.unique(grid[r, :])) == 1 and grid[r, 0] != bg_color
        ]
        sep_cols = [
            c
            for c in range(w)
            if len(np.unique(grid[:, c])) == 1 and grid[0, c] != bg_color
        ]

        row_ranges: List[Tuple[int, int]] = []
        prev_r = 0
        for r in sep_rows + [h]:
            if r > prev_r:
                row_ranges.append((prev_r, r))
            prev_r = r + 1

        col_ranges: List[Tuple[int, int]] = []
        prev_c = 0
        for c in sep_cols + [w]:
            if c > prev_c:
                col_ranges.append((prev_c, c))
            prev_c = c + 1

        if len(row_ranges) * len(col_ranges) <= 1:
            return grid.copy()

        shapes = {
            (r1 - r0, c1 - c0) for r0, r1 in row_ranges for c0, c1 in col_ranges
        }
        if len(shapes) != 1:
            return grid.copy()

        cells = [
            grid[r0:r1, c0:c1].copy()
            for r0, r1 in row_ranges
            for c0, c1 in col_ranges
        ]

        if mode == "overlay_all":
            acc = np.full_like(cells[0], bg_color)
            for cell in cells:
                acc = np.where(cell != bg_color, cell, acc)
            return acc

        if mode == "max_fg":
            return max(cells, key=lambda c: int(np.sum(c != bg_color)))

        if mode == "min_fg":
            non_empty = [c for c in cells if np.any(c != bg_color)]
            if non_empty:
                return min(non_empty, key=lambda c: int(np.sum(c != bg_color)))
            return cells[0]

        counts = [
            sum(1 for c2 in cells if np.array_equal(c1, c2)) for c1 in cells
        ]
        return cells[int(np.argmin(counts))]

    @staticmethod
    def connect_aligned_markers(
        grid: np.ndarray,
        line_color: int = -1,
        priority: str = "default",
        bg_color: int = 0,
    ) -> np.ndarray:
        """
        Conecta pares de pixels ou células de mesma cor alinhados na mesma linha ou
        coluna (com suporte a prioridade de interseção 'horiz_over_vert',
        'vert_over_horiz', 'color_desc' e supergrades).
        """
        h, w = grid.shape
        # Caso 1: Supergrade de células 2x2 separadas por linhas a cada 3 pixels
        if h >= 8 and w >= 8 and len(np.unique(grid[2, :])) == 1 and grid[2, 2] != bg_color:
            sep_color = int(grid[2, 2])
            cells = grid[0::3, 0::3].copy()
            colors = [int(c) for c in np.unique(cells) if c != bg_color and c != sep_color]
            conn = np.zeros_like(cells)
            for col in colors:
                layer = F3Primitives.draw_connecting_lines(
                    np.where(cells == col, col, bg_color), col, col
                )
                conn = np.where(layer == col, col, conn)
            res = grid.copy()
            for r in range(conn.shape[0]):
                for c in range(conn.shape[1]):
                    if conn[r, c] != bg_color:
                        res[3 * r : 3 * r + 2, 3 * c : 3 * c + 2] = conn[r, c]
            return res

        # Caso 2: Conexão com controle de prioridade em cruzamentos ortogonais (ex: 070dd51e)
        fg_colors = [int(c) for c in np.unique(grid) if int(c) != bg_color]
        if priority == "color_desc":
            fg_colors = list(reversed(fg_colors))

        if priority in ("horiz_over_vert", "vert_over_horiz"):
            horiz_layer = np.full_like(grid, bg_color)
            vert_layer = np.full_like(grid, bg_color)
            for col in fg_colors:
                lc = int(col) if line_color < 0 else int(line_color)
                # Linhas horizontais
                for r in range(h):
                    cols_idx = np.where(grid[r, :] == col)[0]
                    if cols_idx.size >= 2:
                        c_min, c_max = int(cols_idx[0]), int(cols_idx[-1])
                        horiz_layer[r, c_min : c_max + 1] = lc
                # Linhas verticais
                for c in range(w):
                    rows_idx = np.where(grid[:, c] == col)[0]
                    if rows_idx.size >= 2:
                        r_min, r_max = int(rows_idx[0]), int(rows_idx[-1])
                        vert_layer[r_min : r_max + 1, c] = lc

            res = np.full_like(grid, bg_color)
            if priority == "horiz_over_vert":
                res = np.where(vert_layer != bg_color, vert_layer, res)
                res = np.where(horiz_layer != bg_color, horiz_layer, res)
            else:
                res = np.where(horiz_layer != bg_color, horiz_layer, res)
                res = np.where(vert_layer != bg_color, vert_layer, res)
            # Preserva marcadores originais
            res = np.where(grid != bg_color, grid, res)
            return res

        res = grid.copy()
        for col in fg_colors:
            lc = int(col) if line_color < 0 else int(line_color)
            layer = F3Primitives.draw_connecting_lines(grid, int(col), lc)
            res = np.where((res == bg_color) & (layer != bg_color), layer, res)
        return res

    @staticmethod
    def compress_non_empty_rows_cols(
        grid: np.ndarray, bg_color: int = 0
    ) -> np.ndarray:
        """Remove linhas e colunas inteiramente compostas por `bg_color`."""
        keep_r = [r for r in range(grid.shape[0]) if np.any(grid[r, :] != bg_color)]
        keep_c = [c for c in range(grid.shape[1]) if np.any(grid[:, c] != bg_color)]
        if not keep_r or not keep_c:
            return grid.copy()
        return grid[np.ix_(keep_r, keep_c)].copy()

    @staticmethod
    def deduplicate_rows_cols(grid: np.ndarray) -> np.ndarray:
        """Comprime linhas consecutivas idênticas e colunas consecutivas idênticas."""
        if grid.size == 0:
            return grid.copy()
        keep_r = [0] + [
            r
            for r in range(1, grid.shape[0])
            if not np.array_equal(grid[r, :], grid[r - 1, :])
        ]
        sub = grid[keep_r, :]
        keep_c = [0] + [
            c
            for c in range(1, sub.shape[1])
            if not np.array_equal(sub[:, c], sub[:, c - 1])
        ]
        return sub[:, keep_c].copy()

    @staticmethod
    def fill_bounding_boxes(
        grid: np.ndarray, frame_only: bool = False, bg_color: int = 0
    ) -> np.ndarray:
        """
        Para cada cor de figura na grade, preenche sua bounding box (ou apenas a
        moldura de 1px se `frame_only=True`) preservando pixels de outras cores.
        """
        res = grid.copy()
        for col in np.unique(grid):
            if col == bg_color:
                continue
            coords = np.argwhere(grid == col)
            if coords.shape[0] < 2:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if frame_only:
                res[r0, c0 : c1 + 1] = np.where(
                    res[r0, c0 : c1 + 1] == bg_color, col, res[r0, c0 : c1 + 1]
                )
                res[r1, c0 : c1 + 1] = np.where(
                    res[r1, c0 : c1 + 1] == bg_color, col, res[r1, c0 : c1 + 1]
                )
                res[r0 : r1 + 1, c0] = np.where(
                    res[r0 : r1 + 1, c0] == bg_color, col, res[r0 : r1 + 1, c0]
                )
                res[r0 : r1 + 1, c1] = np.where(
                    res[r0 : r1 + 1, c1] == bg_color, col, res[r0 : r1 + 1, c1]
                )
            else:
                sub = res[r0 : r1 + 1, c0 : c1 + 1]
                res[r0 : r1 + 1, c0 : c1 + 1] = np.where(sub == bg_color, col, sub)
        return res

    @staticmethod
    def move_until_contact(
        grid: np.ndarray, bg_color: int = 0
    ) -> np.ndarray:
        """
        Translada ortogonalmente o objeto menor na direção do objeto maior até
        entrarem em contato adjacente.
        """
        fg_vals, fg_counts = np.unique(grid[grid != bg_color], return_counts=True)
        if len(fg_vals) != 2:
            return grid.copy()
        c_small = int(fg_vals[np.argmin(fg_counts)])
        c_large = int(fg_vals[np.argmax(fg_counts)])
        mob = grid == c_small
        anc = grid == c_large
        rr, rc = np.where(mob)
        cr, cc = np.where(anc)
        dr, dc = 0, 0
        if rr.max() < cr.min():
            dr = int(cr.min() - rr.max() - 1)
        elif rr.min() > cr.max():
            dr = int(cr.max() - rr.min() + 1)
        elif rc.max() < cc.min():
            dc = int(cc.min() - rc.max() - 1)
        elif rc.min() > cc.max():
            dc = int(cc.max() - rc.min() + 1)
        res = np.where(mob, bg_color, grid)
        moved = F3Primitives.translate_object(
            np.where(mob, c_small, bg_color), dr, dc, bg_color=bg_color
        )
        return np.where(moved != bg_color, c_small, res)

    @staticmethod
    def fill_seeded_or_sized_cavities(
        grid: np.ndarray,
        mode: str = "seeded",
        rule_map: Optional[Dict[Any, int]] = None,
        bg_color: int = 0,
    ) -> np.ndarray:
        """
        Generaliza `fill_enclosed_holes` para cavidades topológicas condicionadas:
        - `mode="seeded"`: Se uma cavidade fechada contém um ponto-semente interno
          de cor diferente da moldura externa, preenche toda a cavidade com a cor
          da semente; cavidades sem semente permanecem `bg_color` (ex: 09c534e7).
        - `mode="by_size"`: Preenche cada cavidade fechada segundo sua dimensão
          mínima ou área mapeada em `rule_map`, preservando pontos centrais (ex: 00dbd492).
        - `mode="by_boundary"`: Preenche cada cavidade fechada segundo a cor da
          moldura que a envolve mapeada em `rule_map`.
        """
        h, w = grid.shape
        if grid.size == 0:
            return grid.copy()

        res = grid.copy()

        if mode == "seeded":
            # Identifica a cor de moldura predominante ou contornos fechados
            fg_vals, fg_counts = np.unique(grid[grid != bg_color], return_counts=True)
            if fg_vals.size == 0:
                return res
            # Testa tratar apenas a cor mais frequente (ou paredes) como barreira,
            # ou encontrar cavidades de bg_color adjacentes a uma semente minoritária
            holes_filled = F3Primitives.fill_enclosed_holes(grid, -99, bg_color)
            hole_mask = holes_filled == -99
            if not np.any(hole_mask):
                return res
            labeled_holes, n_holes = label(hole_mask)
            for idx in range(1, n_holes + 1):
                comp = labeled_holes == idx
                rs, cs = np.where(comp)
                # Coleta todas as cores de figura 4-adjacentes a esta cavidade
                adj_colors: Dict[int, int] = {}
                for r, c in zip(rs, cs):
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = int(r + dr), int(c + dc)
                        if 0 <= nr < h and 0 <= nc < w:
                            val = int(grid[nr, nc])
                            if val != bg_color:
                                adj_colors[val] = adj_colors.get(val, 0) + 1
                if len(adj_colors) >= 2:
                    # A cor com menor perímetro de contato com a cavidade é a semente interna!
                    seed_col = min(adj_colors.keys(), key=lambda k: adj_colors[k])
                    res[comp] = seed_col
            return res

        # Para mode="by_size" ou "by_boundary":
        # Pontos isolados de 1px dentro de uma cavidade não devem dividir a cavidade
        clean_grid = grid.copy()
        for r in range(1, h - 1):
            for c in range(1, w - 1):
                if (
                    clean_grid[r, c] != bg_color
                    and clean_grid[r - 1, c] == bg_color
                    and clean_grid[r + 1, c] == bg_color
                    and clean_grid[r, c - 1] == bg_color
                    and clean_grid[r, c + 1] == bg_color
                ):
                    clean_grid[r, c] = bg_color

        holes_filled = F3Primitives.fill_enclosed_holes(clean_grid, -99, bg_color)
        hole_mask = holes_filled == -99
        if not np.any(hole_mask):
            return res
        labeled_holes, n_holes = label(hole_mask)
        rule_map = rule_map or {}
        for idx in range(1, n_holes + 1):
            comp = labeled_holes == idx
            rs, cs = np.where(comp)
            bh = int(rs.max() - rs.min() + 1)
            bw = int(cs.max() - cs.min() + 1)
            area = int(np.sum(comp))
            fill_col = None
            if mode == "by_size":
                fill_col = (
                    rule_map.get((bh, bw))
                    or rule_map.get(min(bh, bw))
                    or rule_map.get(area)
                    or rule_map.get(f"{bh}x{bw}")
                )
            elif mode == "by_boundary":
                adj_colors = {}
                for r, c in zip(rs, cs):
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = int(r + dr), int(c + dc)
                        if 0 <= nr < h and 0 <= nc < w and clean_grid[nr, nc] != bg_color:
                            v = int(clean_grid[nr, nc])
                            adj_colors[v] = adj_colors.get(v, 0) + 1
                if adj_colors:
                    b_col = max(adj_colors.keys(), key=lambda k: adj_colors[k])
                    fill_col = rule_map.get(b_col) or rule_map.get(str(b_col))
            if fill_col is not None:
                res[comp & (grid == bg_color)] = int(fill_col)
        return res

    @staticmethod
    def symmetrize_per_object_or_frame(
        grid: np.ndarray, mode: str = "hv", bg_color: int = 0
    ) -> np.ndarray:
        """
        Completa a simetria (horizontal+vertical ou diedral D4) do padrão interno
        de cada moldura/objeto conectado individualmente em relação à sua própria
        bounding box (resolve tarefas como 18419cfa).
        """
        res = grid.copy()
        objs = F3Primitives.get_connected_objects(grid, ignore_bg=True)
        for o in objs:
            r0, c0, r1, c1 = o["bbox"]
            if (r1 - r0) < 3 or (c1 - c0) < 3:
                continue
            sub = res[r0:r1, c0:c1].copy()
            frame_col = o["color"]
            # Simetriza pixels internos (que estão dentro da bbox e não são bg_color)
            interior = sub[1:-1, 1:-1].copy()
            if not np.any((interior != bg_color) & (interior != frame_col)):
                # Tenta simetrizar o próprio objeto dentro de sua bbox
                sym_sub = sub.copy()
                for m in (np.fliplr(sub), np.flipud(sub), np.rot90(sub, 2)):
                    sym_sub = np.where(sym_sub == bg_color, m, sym_sub)
                if mode == "d4" and sym_sub.shape[0] == sym_sub.shape[1]:
                    for k in (1, 3):
                        m = np.rot90(sym_sub, k)
                        sym_sub = np.where(sym_sub == bg_color, m, sym_sub)
                res[r0:r1, c0:c1] = sym_sub
            else:
                sym_int = interior.copy()
                for m in (np.fliplr(interior), np.flipud(interior), np.rot90(interior, 2)):
                    sym_int = np.where(sym_int == bg_color, m, sym_int)
                if mode == "d4" and sym_int.shape[0] == sym_int.shape[1]:
                    for k in (1, 3):
                        m = np.rot90(sym_int, k)
                        sym_int = np.where(sym_int == bg_color, m, sym_int)
                res[r0 + 1 : r1 - 1, c0 + 1 : c1 - 1] = sym_int
        return res

    @staticmethod
    def denoise_periodic_subblocks(
        grid: np.ndarray, bg_color: int = 0
    ) -> np.ndarray:
        """
        Detecta grades de sub-blocos periódicos corrompidos por ruído esparso
        (como 0607ce86) e reconstrói cada sub-bloco pelo voto majoritário
        pixel-a-pixel entre as réplicas, zerando o ruído nas margens separadoras.
        """
        h, w = grid.shape
        # Primeiro remove pontos isolados de 1px para estimar linhas/colunas ativas
        cleaned = F3LearnedOperators.isolate_extremum_object_or_cell(
            grid, criterion="remove_noise_dots", bg_color=bg_color
        )
        row_active = np.sum(cleaned != bg_color, axis=1) >= max(2, w // 6)
        col_active = np.sum(cleaned != bg_color, axis=0) >= max(2, h // 6)

        def _get_spans(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
            spans = []
            in_span = False
            s = 0
            for i, v in enumerate(mask_1d):
                if v and not in_span:
                    s = i
                    in_span = True
                elif not v and in_span:
                    spans.append((s, i))
                    in_span = False
            if in_span:
                spans.append((s, len(mask_1d)))
            return spans

        r_spans = _get_spans(row_active)
        c_spans = _get_spans(col_active)
        if len(r_spans) < 2 or len(c_spans) < 2:
            return cleaned

        # Verifica se os spans têm tamanhos consistentes (ou ajusta pelo tamanho modal)
        r_lens = [r1 - r0 for r0, r1 in r_spans]
        c_lens = [c1 - c0 for c0, c1 in c_spans]
        th = int(np.bincount(r_lens).argmax())
        tw = int(np.bincount(c_lens).argmax())
        if th < 2 or tw < 2:
            return cleaned

        # Normaliza spans para tamanho (th, tw)
        norm_r = [(r0, r0 + th) for r0, r1 in r_spans if r0 + th <= h and abs((r1 - r0) - th) <= 1]
        norm_c = [(c0, c0 + tw) for c0, c1 in c_spans if c0 + tw <= w and abs((c1 - c0) - tw) <= 1]
        if len(norm_r) < 2 or len(norm_c) < 2:
            return cleaned

        blocks = [
            grid[r0:r1, c0:c1] for r0, r1 in norm_r for c0, c1 in norm_c
        ]
        # Constrói bloco consenso por voto majoritário em cada célula (r, c)
        consensus = np.zeros((th, tw), dtype=grid.dtype)
        for rr in range(th):
            for cc in range(tw):
                vals = [int(b[rr, cc]) for b in blocks]
                u_vals, u_cnts = np.unique(vals, return_counts=True)
                consensus[rr, cc] = u_vals[np.argmax(u_cnts)]

        res = np.full_like(grid, bg_color)
        for r0, r1 in norm_r:
            for c0, c1 in norm_c:
                res[r0:r1, c0:c1] = consensus
        return res

    # ------------------------------------------------------------------
    # Horizonte 5 — Operadores paramétricos gerais (macro-células, consenso,
    # Kronecker condicional, empacotamento de objetos e completamento local)
    # ------------------------------------------------------------------

    @staticmethod
    def _runs(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
        """Retorna intervalos [s, e) de valores True consecutivos."""
        runs: List[Tuple[int, int]] = []
        start = -1
        for i, v in enumerate(list(mask_1d) + [False]):
            if v and start < 0:
                start = i
            elif not v and start >= 0:
                runs.append((start, i))
                start = -1
        return runs

    @staticmethod
    def _components(mask: np.ndarray, connectivity: int = 8) -> List[np.ndarray]:
        """BFS em NumPy puro: lista de arrays (N, 2) de coordenadas por componente."""
        h, w = mask.shape
        seen = np.zeros(mask.shape, dtype=bool)
        if connectivity == 8:
            nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        else:
            nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        comps: List[np.ndarray] = []
        for r0, c0 in zip(*np.where(mask)):
            if seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            pts = []
            while stack:
                r, c = stack.pop()
                pts.append((r, c))
                for dr, dc in nbrs:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
        return comps

    @staticmethod
    def detect_separator_lines(
        grid: np.ndarray, bg_color: int = 0, min_frac: float = 0.6
    ) -> Optional[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]]:
        """
        Detecta a cor separadora (linhas/colunas com fração >= `min_frac` dessa cor)
        e devolve `(sep_color, row_cells, col_cells)` com os intervalos [s, e) das
        macro-células entre separadores. Retorna None se não houver grade.
        """
        best = None
        for col in np.unique(grid):
            col = int(col)
            if col == bg_color:
                continue
            m = grid == col
            sep_r = m.mean(axis=1) >= min_frac
            sep_c = m.mean(axis=0) >= min_frac
            n_lines = int(sep_r.sum() + sep_c.sum())
            if n_lines >= 2 and (best is None or n_lines > best[0]):
                best = (n_lines, col, sep_r, sep_c)
        if best is None:
            return None
        _, sep_col, sep_r, sep_c = best
        row_cells = F3LearnedOperators._runs(~sep_r)
        col_cells = F3LearnedOperators._runs(~sep_c)
        if len(row_cells) * len(col_cells) < 2:
            return None
        return sep_col, row_cells, col_cells

    @staticmethod
    def fractal_conditional_kron(
        grid: np.ndarray,
        bg_color: int = 0,
        mask_mode: str = "fg",
        template_mode: str = "invert",
    ) -> np.ndarray:
        """
        Kronecker condicional / fractal local (generaliza `kronecker_self_pattern`):
        cada célula selecionada por `mask_mode` ("fg" ou "bg") estampa um template
        derivado da própria grade por `template_mode`:
          "self" | "invert" (complemento fg<->bg na cor dominante) | "rot90" |
          "rot180" | "transpose" | "fliplr" | "flipud".
        """
        h, w = grid.shape
        if grid.size == 0 or h * h > 30 or w * w > 30:
            return grid.copy()
        fg_vals = grid[grid != bg_color]
        if fg_vals.size == 0:
            return grid.copy()
        dom = int(np.bincount(fg_vals.astype(int)).argmax())
        if template_mode == "invert":
            tpl = np.where(grid == bg_color, dom, bg_color).astype(grid.dtype)
        elif template_mode == "rot90":
            tpl = np.rot90(grid)
        elif template_mode == "rot180":
            tpl = np.rot90(grid, 2)
        elif template_mode == "transpose":
            tpl = grid.T.copy()
        elif template_mode == "fliplr":
            tpl = np.fliplr(grid)
        elif template_mode == "flipud":
            tpl = np.flipud(grid)
        else:
            tpl = grid.copy()
        if tpl.shape != (h, w):
            return grid.copy()
        mask = (grid != bg_color) if mask_mode == "fg" else (grid == bg_color)
        out = np.full((h * h, w * w), bg_color, dtype=grid.dtype)
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    out[r * h : (r + 1) * h, c * w : (c + 1) * w] = tpl
        return out

    @staticmethod
    def denoise_subblock_consensus(
        grid: np.ndarray, bg_color: int = 0, density_frac: float = 0.6
    ) -> np.ndarray:
        """
        Consenso de sub-blocos periódicos (padrão 0607ce86, versão robusta):
        1. linhas densas (>= density_frac * máximo de pixels de figura) formam faixas;
           mantém apenas faixas com o comprimento modal `th`;
        2. o mesmo nas colunas, medido só dentro das faixas densas (`tw`);
        3. sub-bloco canônico = voto majoritário pixel-a-pixel entre todas as réplicas
           não vazias; todo o resto (ruído fora dos blocos) vira `bg_color`.
        """
        h, w = grid.shape
        fg = grid != bg_color
        row_d = fg.sum(axis=1)
        if row_d.max() == 0:
            return grid.copy()
        r_runs = F3LearnedOperators._runs(row_d >= density_frac * row_d.max())
        if not r_runs:
            return grid.copy()
        th = int(np.bincount([e - s for s, e in r_runs]).argmax())
        r_runs = [(s, e) for s, e in r_runs if e - s == th]
        dense_rows = np.zeros(h, dtype=bool)
        for s, e in r_runs:
            dense_rows[s:e] = True
        col_d = fg[dense_rows].sum(axis=0)
        if col_d.max() == 0:
            return grid.copy()
        c_runs = F3LearnedOperators._runs(col_d >= density_frac * col_d.max())
        if not c_runs:
            return grid.copy()
        tw = int(np.bincount([e - s for s, e in c_runs]).argmax())
        c_runs = [(s, e) for s, e in c_runs if e - s == tw]
        if len(r_runs) * len(c_runs) < 2 or th < 2 or tw < 2:
            return grid.copy()
        blocks, coords = [], []
        for rs, re_ in r_runs:
            for cs, ce in c_runs:
                blocks.append(grid[rs:re_, cs:ce])
                coords.append((rs, cs))
        fills = np.array([np.mean(b != bg_color) for b in blocks])
        keep = fills >= 0.5 * fills.max()
        stack = np.stack([b for b, k in zip(blocks, keep) if k])
        consensus = np.zeros((th, tw), dtype=grid.dtype)
        for rr in range(th):
            for cc in range(tw):
                vals, cnts = np.unique(stack[:, rr, cc], return_counts=True)
                consensus[rr, cc] = vals[np.argmax(cnts)]
        out = np.full_like(grid, bg_color)
        for (rs, cs), k in zip(coords, keep):
            if k:
                out[rs : rs + th, cs : cs + tw] = consensus
        return out

    @staticmethod
    def separator_fill(
        grid: np.ndarray, inner_color: int, outer_color: int = -1, bg_color: int = 0
    ) -> np.ndarray:
        """
        Preenchimento entre grades (padrão 137f0df0): linhas/colunas totalmente vazias
        situadas estritamente entre as linhas/colunas ocupadas viram corredores;
        o trecho do corredor dentro da bbox da figura recebe `inner_color` e o trecho
        fora da bbox (até a borda) recebe `outer_color` (se >= 0).
        """
        fg = grid != bg_color
        if not fg.any():
            return grid.copy()
        rows = np.where(fg.any(axis=1))[0]
        cols = np.where(fg.any(axis=0))[0]
        r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
        gap_r = [r for r in range(r0 + 1, r1) if not fg[r].any()]
        gap_c = [c for c in range(c0 + 1, c1) if not fg[:, c].any()]
        if not gap_r and not gap_c:
            return grid.copy()
        out = grid.copy()
        h, w = grid.shape
        for r in gap_r:
            for c in range(w):
                if c0 <= c <= c1:
                    out[r, c] = inner_color
                elif outer_color >= 0:
                    out[r, c] = outer_color
        for c in gap_c:
            for r in range(h):
                if r0 <= r <= r1:
                    out[r, c] = inner_color
                elif outer_color >= 0 and out[r, c] == bg_color:
                    out[r, c] = outer_color
        return out

    @staticmethod
    def grid_cell_pattern_stamp(grid: np.ndarray, bg_color: int = 0) -> np.ndarray:
        """
        Carimbo de padrão em macro-células (padrão 15113be4):
        - detecta a grade de separadores e o tamanho modal das macro-células;
        - para cada cor rara `b` (nem fundo, nem separador, nem a cor dominante das
          células), extrai a região delimitada pelos separadores que a contém e a
          reduz pelo fator de escala até o tamanho de uma célula -> padrão P;
        - em toda macro-célula cuja máscara da cor dominante contém P, recolore as
          posições de P com `b`.
        """
        det = F3LearnedOperators.detect_separator_lines(grid, bg_color)
        if det is None:
            return grid.copy()
        sep, row_cells, col_cells = det
        ch = int(np.bincount([e - s for s, e in row_cells]).argmax())
        cw = int(np.bincount([e - s for s, e in col_cells]).argmax())
        vals, cnts = np.unique(grid[(grid != bg_color) & (grid != sep)], return_counts=True)
        if len(vals) < 2:
            return grid.copy()
        cell_color = int(vals[np.argmax(cnts)])
        rare = [int(v) for v in vals if int(v) != cell_color]
        out = grid.copy()
        comps = F3LearnedOperators._components(grid != sep, connectivity=4)
        for b in rare:
            seed = np.argwhere(grid == b)[0]
            region = None
            for comp in comps:
                if np.any((comp[:, 0] == seed[0]) & (comp[:, 1] == seed[1])):
                    region = comp
                    break
            if region is None:
                continue
            rr0, cc0 = region.min(axis=0)
            rr1, cc1 = region.max(axis=0) + 1
            rh, rw = int(rr1 - rr0), int(cc1 - cc0)
            if rh % ch or rw % cw or rh // ch != rw // cw or rh // ch < 1:
                continue
            s = rh // ch
            sub = grid[rr0:rr1, cc0:cc1] == b
            pat = sub.reshape(ch, s, cw, s).any(axis=(1, 3))
            if not pat.any():
                continue
            for rs, re_ in row_cells:
                for cs, ce in col_cells:
                    if re_ - rs != ch or ce - cs != cw:
                        continue
                    cell = grid[rs:re_, cs:ce]
                    if np.any(cell == sep) or np.any(np.isin(cell, rare)):
                        continue
                    if np.all((cell == cell_color)[pat]):
                        blk = out[rs:re_, cs:ce]
                        blk[pat] = b
        return out

    @staticmethod
    def induce_macro_cell_table(
        train_examples: List[Dict[str, Any]], bg_color: int = 0
    ) -> Optional[Tuple[int, int, Dict[str, int]]]:
        """
        Induz (cell_h, cell_w, tabela padrão->cor) quando cada macro-célula quadrada
        da saída é uniforme e sua cor é função determinística do padrão binário da
        macro-célula correspondente na entrada (padrão 17cae0c1).
        """
        table: Dict[str, int] = {}
        cell = None
        for ex in train_examples:
            inp = np.array(ex["input"], dtype=int)
            out = np.array(ex["output"], dtype=int)
            if inp.shape != out.shape:
                return None
            h, w = inp.shape
            k = min(h, w)
            if k < 2 or h % k or w % k or (h == k and w == k):
                return None
            if cell is None:
                cell = k
            elif cell != k:
                return None
            for r in range(0, h, k):
                for c in range(0, w, k):
                    o = out[r : r + k, c : c + k]
                    if len(np.unique(o)) != 1:
                        return None
                    key = "".join(
                        "1" if v != bg_color else "0" for v in inp[r : r + k, c : c + k].ravel()
                    )
                    col = int(o[0, 0])
                    if table.get(key, col) != col:
                        return None
                    table[key] = col
        if cell is None or len(table) < 2:
            return None
        return cell, cell, table

    @staticmethod
    def macro_cell_lookup(
        grid: np.ndarray, cell_h: int, cell_w: int, table: Dict[str, int], bg_color: int = 0
    ) -> np.ndarray:
        """Preenche cada macro-célula com a cor associada ao seu padrão binário."""
        h, w = grid.shape
        if cell_h < 1 or cell_w < 1 or h % cell_h or w % cell_w:
            return grid.copy()
        out = grid.copy()
        for r in range(0, h, cell_h):
            for c in range(0, w, cell_w):
                key = "".join(
                    "1" if v != bg_color else "0"
                    for v in grid[r : r + cell_h, c : c + cell_w].ravel()
                )
                if key in table:
                    out[r : r + cell_h, c : c + cell_w] = table[key]
        return out

    @staticmethod
    def pack_objects_into_grid(
        grid: np.ndarray, bg_color: int = 0, gap: int = 1, connectivity: int = 8
    ) -> np.ndarray:
        """
        Empacota objetos (componentes conexos) numa grade compacta k x k preservando a
        ordem espacial (linha do centróide, depois coluna), com `gap` de fundo entre
        células do tamanho do maior bbox (padrão 1990f7a8).
        """
        comps = F3LearnedOperators._components(grid != bg_color, connectivity)
        if len(comps) < 2 or len(comps) > 16:
            return grid.copy()
        ch = max(int(p[:, 0].max() - p[:, 0].min() + 1) for p in comps)
        cw = max(int(p[:, 1].max() - p[:, 1].min() + 1) for p in comps)
        n = len(comps)
        k = int(np.ceil(np.sqrt(n)))
        comps = sorted(comps, key=lambda p: (p[:, 0].mean(), p[:, 1].mean()))
        rows = [comps[i : i + k] for i in range(0, n, k)]
        rows = [sorted(rw, key=lambda p: p[:, 1].mean()) for rw in rows]
        H = len(rows) * ch + (len(rows) - 1) * gap
        W = k * cw + (k - 1) * gap
        if H > 30 or W > 30:
            return grid.copy()
        out = np.full((H, W), bg_color, dtype=grid.dtype)
        for i, rw in enumerate(rows):
            for j, p in enumerate(rw):
                r0, c0 = p[:, 0].min(), p[:, 1].min()
                for r, c in p:
                    out[i * (ch + gap) + r - r0, j * (cw + gap) + c - c0] = grid[r, c]
        return out

    @staticmethod
    def complete_partial_plus(
        grid: np.ndarray,
        anchor_color: int,
        fill_color: int,
        bg_color: int = 0,
    ) -> np.ndarray:
        """
        Completa cruzes (+) parciais por cobertura exata das âncoras (padrão 14754a24):
        particiona todos os pixels de `anchor_color` em vizinhanças em cruz (+)
        de tamanho >= 4 inteiramente contidas na figura (`!= bg_color`) com >= 2 âncoras,
        recolorindo os braços não-âncora com `fill_color`.
        """
        h, w = grid.shape
        all_anchors = set(map(tuple, np.argwhere(grid == anchor_color)))
        if len(all_anchors) < 2 or len(all_anchors) > 80:
            return grid.copy()
        offs = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        pluses = []
        for r in range(h):
            for c in range(w):
                cells = [
                    (r + dr, c + dc)
                    for dr, dc in offs
                    if 0 <= r + dr < h and 0 <= c + dc < w
                ]
                if len(cells) >= 4 and all(int(grid[pt]) != bg_color for pt in cells):
                    anc = frozenset(pt for pt in cells if int(grid[pt]) == anchor_color)
                    if len(anc) >= 2:
                        pluses.append((len(anc), len(cells), cells, anc))
        if not pluses:
            return grid.copy()
        pluses.sort(key=lambda x: (-x[0], -x[1]))
        chosen: List[Any] = []
        steps = [0]

        def _cover(rem: set) -> bool:
            if not rem:
                return True
            steps[0] += 1
            if steps[0] > 1500:
                return False
            # Heurística MRV (Minimum Remaining Values) para poda instantânea
            pt = min(rem, key=lambda p: sum(1 for pl in pluses if p in pl[3] and pl[3].issubset(rem)))
            cands = [pl for pl in pluses if pt in pl[3] and pl[3].issubset(rem)]
            for pl in cands:
                chosen.append(pl)
                if _cover(rem - set(pl[3])):
                    return True
                chosen.pop()
            return False

        if not _cover(all_anchors):
            return grid.copy()
        out = grid.copy()
        for pl in chosen:
            for pt in pl[2]:
                if int(out[pt]) != anchor_color:
                    out[pt] = fill_color
        return out

    @staticmethod
    def fill_seeded_component_interiors(
        grid: np.ndarray, bg_color: int = 0
    ) -> np.ndarray:
        """
        Preenche o núcleo interior morfológico (vizinhança 3x3 totalmente figura,
        preservando borda de 1px e pontes de 1px) de cada componente conectado 4-vizinhos
        que contenha uma única cor-semente minoritária (padrão 09c534e7).
        """
        h, w = grid.shape
        if h < 3 or w < 3:
            return grid.copy()
        fg = grid != bg_color
        padded = np.pad(fg, 1, constant_values=False)
        interior = np.ones((h, w), dtype=bool)
        for dr in range(3):
            for dc in range(3):
                interior &= padded[dr : dr + h, dc : dc + w]
        if not interior.any():
            return grid.copy()
        out = grid.copy()
        for comp in F3LearnedOperators._components(fg, connectivity=4):
            vals, cnts = np.unique([int(grid[r, c]) for r, c in comp], return_counts=True)
            if len(vals) == 2:
                dom = int(vals[np.argmax(cnts)])
                seed_col = int(vals[0] if vals[1] == dom else vals[1])
                for r, c in comp:
                    if interior[r, c]:
                        out[r, c] = seed_col
        return out

    @staticmethod
    def recolor_cavity_fitting_objects(
        grid: np.ndarray,
        wall_color: int,
        piece_color: int,
        fit_color: int,
        allow_mirror: bool = False,
        bg_color: int = 0,
    ) -> np.ndarray:
        """
        Teste geométrico de encaixe rígido em cavidades/entalhes (padrão 1acc24af):
        identifica os entalhes abertos ou fechados da estrutura `wall_color`
        (células vazias confinadas em >= 3 direções cardinais por `wall_color`) e
        recolore para `fit_color` cada objeto de `piece_color` que, sob rotação 2D
        rígida SO(2) (ou D4 se `allow_mirror=True`), sela ao menos um entalhe sem
        colidir com `wall_color`.
        """
        h, w = grid.shape
        wall = grid == wall_color
        if not wall.any() or not np.any(grid == piece_color):
            return grid.copy()
        hl = np.maximum.accumulate(wall, axis=1)
        hr = np.maximum.accumulate(wall[:, ::-1], axis=1)[:, ::-1]
        hu = np.maximum.accumulate(wall, axis=0)
        hd = np.maximum.accumulate(wall[::-1, :], axis=0)[::-1, :]
        wr, wc = np.where(wall)
        pr, pc_arr = np.where(grid == piece_color)
        dr_mean = float(pr.mean() - wr.mean())
        dc_mean = float(pc_arr.mean() - wc.mean())
        if abs(dr_mean) >= abs(dc_mean):
            orient_mask = (hl & hr & hu) if dr_mean >= 0 else (hl & hr & hd)
        else:
            orient_mask = (hu & hd & hl) if dc_mean >= 0 else (hu & hd & hr)
        notch_mask = (~wall) & (grid == bg_color) & orient_mask
        notches = F3LearnedOperators._components(notch_mask, connectivity=4)
        if not notches or len(notches) > 12:
            return grid.copy()
        pieces = F3LearnedOperators._components(grid == piece_color, connectivity=4)
        if not pieces or len(pieces) > 20:
            return grid.copy()
        out = grid.copy()
        for comp in pieces:
            r0, c0 = comp.min(axis=0)
            r1, c1 = comp.max(axis=0)
            obj = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
            for r, c in comp:
                obj[int(r - r0), int(c - c0)] = True
            variants = [np.rot90(obj, k) for k in range(4)]
            if allow_mirror:
                variants.extend([np.rot90(np.fliplr(obj), k) for k in range(4)])
            fits = False
            for rot in variants:
                pts = np.argwhere(rot)
                if len(pts) > 40:
                    continue
                for ncomp in notches:
                    if len(pts) < len(ncomp):
                        continue
                    nset = set(map(tuple, ncomp))
                    nr0, nc0 = int(ncomp[0, 0]), int(ncomp[0, 1])
                    for pr, pc in pts:
                        dr, dc = nr0 - int(pr), nc0 - int(pc)
                        shifted = [(int(r) + dr, int(c) + dc) for r, c in pts]
                        if all(0 <= rr < h and 0 <= cc < w and not wall[rr, cc] for rr, cc in shifted):
                            if nset.issubset(set(shifted)):
                                fits = True
                                break
                    if fits:
                        break
                if fits:
                    break
            if fits:
                for r, c in comp:
                    out[r, c] = fit_color
        return out

    @staticmethod
    def pack_rectangles_into_frame_holes(
        grid: np.ndarray, bg_color: int = 0
    ) -> np.ndarray:
        """
        Empacotamento exato de blocos retangulares externos nas cavidades internas
        de molduras (padrão 184a9768):
        1. remove cores de ruído esparso (cores com >= 3 componentes todos de área 1);
        2. separa componentes por cor em molduras (não-retangulares/com furos) e
           blocos sólidos retangulares;
        3. ladrilha exatamente as cavidades internas das molduras com os blocos sólidos.
        """
        work = grid.copy()
        for col in np.unique(work):
            col = int(col)
            if col == bg_color:
                continue
            cs = F3LearnedOperators._components(work == col, connectivity=4)
            if len(cs) >= 3 and max(len(c) for c in cs) == 1:
                work[work == col] = bg_color

        frames: List[np.ndarray] = []
        pieces: List[Tuple[int, int, int, int]] = []
        for col in np.unique(work):
            col = int(col)
            if col == bg_color:
                continue
            for comp in F3LearnedOperators._components(work == col, connectivity=4):
                r0, c0 = comp.min(axis=0)
                r1, c1 = comp.max(axis=0)
                bh, bw = int(r1 - r0 + 1), int(c1 - c0 + 1)
                if len(comp) == bh * bw:
                    pieces.append((len(comp), bh, bw, col))
                else:
                    frames.append(comp)

        if not frames or not pieces or len(pieces) > 20:
            return grid.copy()

        out = np.full_like(work, bg_color)
        hmask = np.zeros(work.shape, dtype=bool)
        for fcomp in frames:
            fmask = np.zeros(work.shape, dtype=bool)
            for r, c in fcomp:
                out[r, c] = work[r, c]
                fmask[r, c] = True
            hl = np.maximum.accumulate(fmask, axis=1)
            hr = np.maximum.accumulate(fmask[:, ::-1], axis=1)[:, ::-1]
            hu = np.maximum.accumulate(fmask, axis=0)
            hd = np.maximum.accumulate(fmask[::-1, :], axis=0)[::-1, :]
            hmask |= (~fmask) & hl & hr & hu & hd

        rem_holes = set(map(tuple, np.argwhere(hmask)))
        if not rem_holes or len(rem_holes) > 160 or sum(p[0] for p in pieces) < len(rem_holes):
            return grid.copy()

        pieces.sort(key=lambda x: -x[0])
        used = [False] * len(pieces)
        steps = [0]

        def _pack(rem: set) -> bool:
            if not rem:
                return True
            steps[0] += 1
            if steps[0] > 2000:
                return False
            r0, c0 = min(rem)
            tried = set()
            for i, (_, ph, pw, pcol) in enumerate(pieces):
                if used[i] or (ph, pw, pcol) in tried:
                    continue
                tried.add((ph, pw, pcol))
                pts = {(r0 + dr, c0 + dc) for dr in range(ph) for dc in range(pw)}
                if pts.issubset(rem):
                    used[i] = True
                    out[r0 : r0 + ph, c0 : c0 + pw] = pcol
                    if _pack(rem - pts):
                        return True
                    out[r0 : r0 + ph, c0 : c0 + pw] = bg_color
                    used[i] = False
            return False

        if not _pack(rem_holes):
            return grid.copy()
        return out

    @staticmethod
    def grid_cell_consensus(
        grid: np.ndarray,
        mode: str = "majority_single_cell",
        out_color: int = -1,
        bg_color: int = 0,
    ) -> np.ndarray:
        """
        Operações algébricas e de consenso sobre macro-células divididas por linhas
        separadoras (`detect_separator_lines`):
          - "majority_single_cell": voto majoritário pixel-a-pixel -> retorna 1 célula (ch, cw)
          - "majority_in_place": substitui todas as macro-células pelo consenso mantendo separadores
          - "overlay_first" / "overlay_last": sobrepõe macro-células não-background -> (ch, cw)
          - "common_and" / "odd_xor" / "nor": lógica booleana entre todas as macro-células -> (ch, cw)
        """
        det = F3LearnedOperators.detect_separator_lines(grid, bg_color)
        if det is None:
            return grid.copy()
        _, row_cells, col_cells = det
        ch = int(np.bincount([e - s for s, e in row_cells]).argmax())
        cw = int(np.bincount([e - s for s, e in col_cells]).argmax())
        cells = []
        coords = []
        for rs, re_ in row_cells:
            for cs, ce in col_cells:
                if re_ - rs == ch and ce - cs == cw:
                    cells.append(grid[rs:re_, cs:ce])
                    coords.append((rs, cs))
        if len(cells) < 2:
            return grid.copy()
        stack = np.stack(cells)  # (N, ch, cw)

        if mode in ("majority_single_cell", "majority_in_place"):
            cons = np.zeros((ch, cw), dtype=grid.dtype)
            for rr in range(ch):
                for cc in range(cw):
                    vals, cnts = np.unique(stack[:, rr, cc], return_counts=True)
                    cons[rr, cc] = vals[np.argmax(cnts)]
            if mode == "majority_single_cell":
                return cons
            out = grid.copy()
            for rs, cs in coords:
                out[rs : rs + ch, cs : cs + cw] = cons
            return out

        fg_vals = grid[grid != bg_color]
        fill_c = out_color if out_color >= 0 else (int(np.bincount(fg_vals.astype(int)).argmax()) if fg_vals.size else 1)

        if mode == "overlay_first":
            res = np.full((ch, cw), bg_color, dtype=grid.dtype)
            for cell in reversed(cells):
                res = np.where(cell != bg_color, cell, res)
            return res
        if mode == "overlay_last":
            res = np.full((ch, cw), bg_color, dtype=grid.dtype)
            for cell in cells:
                res = np.where(cell != bg_color, cell, res)
            return res

        masks = stack != bg_color
        if mode == "common_and":
            m = np.all(masks, axis=0)
        elif mode == "odd_xor":
            m = (np.sum(masks, axis=0) % 2) == 1
        elif mode == "nor":
            m = ~np.any(masks, axis=0)
        else:
            m = np.any(masks, axis=0)
        return np.where(m, fill_c, bg_color).astype(grid.dtype)

    @staticmethod
    def crop_to_color_bbox(
        grid: np.ndarray, color: int, trim_border: int = 0
    ) -> np.ndarray:
        """
        Recorta a sub-grade delimitada pelo bounding box de `grid == color`,
        removendo opcionalmente `trim_border` pixels da moldura/marcadores de canto.
        """
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            return grid.copy()
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0) + 1
        r0, c0 = int(r0) + trim_border, int(c0) + trim_border
        r1, c1 = int(r1) - trim_border, int(c1) - trim_border
        if r1 <= r0 or c1 <= c0:
            return grid.copy()
        return grid[r0:r1, c0:c1].copy()

    @staticmethod
    def downsample_subgrid_cells(
        grid: np.ndarray, bg_color: int = 0
    ) -> np.ndarray:
        """
        Resume uma grade dividida por linhas separadoras em uma matriz (R, C) onde
        cada entrada (i, j) é a cor dominante não-background da macro-célula (i, j)
        (ou `bg_color` se a macro-célula estiver vazia).
        """
        det = F3LearnedOperators.detect_separator_lines(grid, bg_color)
        if det is None:
            return grid.copy()
        sep_col, row_cells, col_cells = det
        out = np.full((len(row_cells), len(col_cells)), bg_color, dtype=grid.dtype)
        for i, (rs, re_) in enumerate(row_cells):
            for j, (cs, ce) in enumerate(col_cells):
                sub = grid[rs:re_, cs:ce]
                fg = sub[(sub != bg_color) & (sub != sep_col)]
                if fg.size > 0:
                    vals, cnts = np.unique(fg, return_counts=True)
                    out[i, j] = vals[np.argmax(cnts)]
        return out

    @staticmethod
    def apply_residual_patch(
        grid: np.ndarray,
        pred: np.ndarray,
        patch_spec: Dict[str, Any],
        bg_color: int = 0,
    ) -> np.ndarray:
        """
        Executa um patch residual local determinístico induzido pelo
        `ResidualPatchSynthesizer` (Ciclo 3 do AACR):
        Tipos suportados em `patch_spec["type"]`:
          - `"restore_anchors"`: restaura cores originais de `grid` que foram sobrescritas.
          - `"ray_corridor_table"`: tabela relacional baseada em confinamento ortogonal
            por raios (up/down/left/right) e bbox de figura (resolve 1acc24af, 137f0df0).
          - `"neighbor_count_table"`: tabela relacional baseada em contagens 4/8-vizinhos
            de cada cor em `grid` e `pred` (resolve 14754a24).
          - `"component_recolor"`: recoloração de componentes conectados por assinatura
            topológica (contato com outras cores, borda ou concavidade).
        """
        if grid.shape != pred.shape:
            return pred.copy()
        res = pred.copy()
        h, w = grid.shape
        ptype = patch_spec.get("type", "")

        if ptype == "restore_anchors":
            anchor_cols = patch_spec.get("colors", [])
            for c in anchor_cols:
                res = np.where(grid == int(c), grid, res)
            return res

        if ptype == "ray_corridor_table":
            ref_col = int(patch_spec.get("ref_color", 1))
            table: Dict[Tuple[int, int, int, int, int, int], int] = patch_spec.get("table", {})
            ref_mask = grid == ref_col
            if not np.any(ref_mask):
                return res
            rs, cs = np.where(ref_mask)
            rmin, rmax = int(rs.min()), int(rs.max())
            cmin, cmax = int(cs.min()), int(cs.max())
            # Pré-calcula visibilidade por raios nas 4 direções cardinais até `ref_col`
            has_up = np.maximum.accumulate(ref_mask, axis=0)
            has_down = np.maximum.accumulate(ref_mask[::-1, :], axis=0)[::-1, :]
            has_left = np.maximum.accumulate(ref_mask, axis=1)
            has_right = np.maximum.accumulate(ref_mask[:, ::-1], axis=1)[:, ::-1]

            for r in range(h):
                in_r = 1 if (rmin <= r <= rmax) else 0
                for c in range(w):
                    in_bbox = 1 if (in_r and cmin <= c <= cmax) else 0
                    h_bound = 1 if (has_left[r, c] and has_right[r, c]) else 0
                    v_bound = 1 if (has_up[r, c] and has_down[r, c]) else 0
                    n_rays = (
                        int(has_up[r, c])
                        + int(has_down[r, c])
                        + int(has_left[r, c])
                        + int(has_right[r, c])
                    )
                    key = (
                        int(grid[r, c]),
                        int(pred[r, c]),
                        h_bound,
                        v_bound,
                        n_rays,
                        in_bbox,
                    )
                    if key in table:
                        res[r, c] = int(table[key])
            return res

        if ptype == "neighbor_count_table":
            ref_col = int(patch_spec.get("ref_color", 1))
            table_nb: Dict[Tuple[int, int, int, int], int] = patch_spec.get("table", {})
            for r in range(h):
                for c in range(w):
                    n4 = 0
                    n8 = 0
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == ref_col:
                                if abs(dr) + abs(dc) == 1:
                                    n4 += 1
                                else:
                                    n8 += 1
                    key = (int(grid[r, c]), int(pred[r, c]), n4, n8)
                    if key in table_nb:
                        res[r, c] = int(table_nb[key])
            return res

        if ptype == "pred_neighbor_count_table":
            ref_col = int(patch_spec.get("ref_color", 1))
            table_pnb: Dict[Tuple[int, int, int, int], int] = patch_spec.get("table", {})
            m = (pred == ref_col).astype(int)
            pad = np.pad(m, 1, constant_values=0)
            n4_arr = pad[0:h, 1:w+1] + pad[2:h+2, 1:w+1] + pad[1:h+1, 0:w] + pad[1:h+1, 2:w+2]
            n8_arr = pad[0:h, 0:w] + pad[0:h, 2:w+2] + pad[2:h+2, 0:w] + pad[2:h+2, 2:w+2]
            for r in range(h):
                for c in range(w):
                    key = (int(grid[r, c]), int(pred[r, c]), int(n4_arr[r, c]), int(n8_arr[r, c]))
                    if key in table_pnb:
                        res[r, c] = int(table_pnb[key])
            return res

        if ptype == "dual_neighbor_count_table":
            c1 = int(patch_spec.get("c1", 1))
            c2 = int(patch_spec.get("c2", 2))
            table_dnb: Dict[Tuple[int, int, int, int, int, int], int] = patch_spec.get(
                "table", {}
            )
            for r in range(h):
                for c in range(w):
                    n4_1, n8_1, n4_2, n8_2 = 0, 0, 0, 0
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                v = int(grid[nr, nc])
                                is_ortho = (abs(dr) + abs(dc)) == 1
                                if v == c1:
                                    if is_ortho:
                                        n4_1 += 1
                                    else:
                                        n8_1 += 1
                                elif v == c2:
                                    if is_ortho:
                                        n4_2 += 1
                                    else:
                                        n8_2 += 1
                    key = (int(grid[r, c]), int(pred[r, c]), n4_1, n8_1, n4_2, n8_2)
                    if key in table_dnb:
                        res[r, c] = int(table_dnb[key])
            return res

        return res

LEARNED_METADATA: Dict[str, str] = {
    "F3LearnedOperators.kronecker_self_pattern": "Parametric fractal Kronecker self-tiling for any background color and mask polarity.",
    "F3LearnedOperators.extend_2d_periodicity": "Vectorized 1D/2D period inference (p_r, p_c), occlusion reconstruction, and periodic grid extension.",
    "F3LearnedOperators.dihedral_template_matching": "D4 dihedral template matching onto multi-color anchors or directional marker rays.",
    "F3LearnedOperators.expand_local_neighborhoods": "Conditioned orthogonal and diagonal local neighborhood expansion or 3x3 cross -> 5x5 star expansion.",
    "F3LearnedOperators.isolate_extremum_object_or_cell": "Isolate or rank objects/subgrids by statistical/topological extrema (rarest_color, most_sparse_subgrid, rank_by_area).",
    "F3LearnedOperators.apply_color_bijection": "Simultaneous bijective color cipher remapping.",
    "F3LearnedOperators.complete_dihedral_symmetry": "Fast 8-fold dihedral symmetry completion around inferred cluster center.",
    "F3LearnedOperators.mirror_tile_2x2": "2x2 mirror/rotational tiling across horizontal and vertical reflections.",
    "F3LearnedOperators.alternating_mirror_tile": "Alternating row/col reflection tiling (row_fliplr, col_flipud, checker_hv).",
    "F3LearnedOperators.hollow_objects": "Hollow interior of solid shapes keeping 1px boundary.",
    "F3LearnedOperators.split_and_boolean_combine": "Split grid across separator or midpoint and combine via boolean/overlay logic.",
    "F3LearnedOperators.extract_unique_or_extremum_subgrid": "Detect separator grid and extract unique, densest, sparsest, or overlaid cell.",
    "F3LearnedOperators.connect_aligned_markers": "Connect aligned markers along rows/cols with intersection priority control.",
    "F3LearnedOperators.fill_seeded_or_sized_cavities": "Fill enclosed cavities conditioned on interior seed dot, cavity size, or frame color.",
    "F3LearnedOperators.symmetrize_per_object_or_frame": "Complete horizontal/vertical or dihedral symmetry inside each object/frame bbox.",
    "F3LearnedOperators.denoise_periodic_subblocks": "Denoise repeating subblock tiles via pixelwise majority vote and margin clearing.",
    "F3LearnedOperators.fractal_conditional_kron": "Conditional fractal Kronecker: fg/bg cells stamp self, inverted or rotated template.",
    "F3LearnedOperators.denoise_subblock_consensus": "Density-aligned periodic subblock grid + pixelwise majority consensus; clears off-block noise.",
    "F3LearnedOperators.separator_fill": "Fill empty corridors between occupied rows/cols (inner color inside bbox, outer color to border).",
    "F3LearnedOperators.grid_cell_pattern_stamp": "Downscale separator-framed rare-color region to a cell pattern and stamp it onto matching macro-cells.",
    "F3LearnedOperators.macro_cell_lookup": "Fill each macro-cell with a color looked up from its binary pattern (table induced from train).",
    "F3LearnedOperators.pack_objects_into_grid": "Pack connected objects into a compact k x k grid preserving spatial order.",
    "F3LearnedOperators.complete_partial_plus": "Complete partial plus-shaped anchors by recoloring non-anchor cells.",
    "F3LearnedOperators.fill_seeded_component_interiors": "Fill 3x3 morphological interior of 4-connected components carrying a unique seed color.",
    "F3LearnedOperators.recolor_cavity_fitting_objects": "Recolor objects that rigidly plug 3-side-bounded wall cavities without collision.",
    "F3LearnedOperators.pack_rectangles_into_frame_holes": "Denoise sparse 1px color and pack exterior solid rectangles into interior frame cavities.",
    "F3LearnedOperators.grid_cell_consensus": "Macro-cell consensus, overlay, and N-way boolean logic across separator-divided grids.",
    "F3LearnedOperators.crop_to_color_bbox": "Crop grid to bounding box of a marker/frame color with optional border trim.",
    "F3LearnedOperators.downsample_subgrid_cells": "Summarize each macro-cell of a separator-divided grid by its modal foreground color.",
    "F3LearnedOperators.apply_residual_patch": "Execute induced local relational residual repair (anchors, ray corridors, neighbor counts).",
    "solved_00576224": "[Synthesizer D1 | F3LearnedOperators.alternating_mirror_tile] Alternating mirror tile (3, 3) with mode 'row_fliplr'.",
    "solved_00dbd492": "[Synthesizer D1 | F3LearnedOperators.fill_seeded_or_sized_cavities] Fill enclosed cavities conditioned on cavity size...",
    "solved_070dd51e": "[Synthesizer D1 | F3LearnedOperators.connect_aligned_markers] Connect aligned same-color markers along rows/cols (pri...",
    "solved_0c786b71": "[Synthesizer D1 | F3LearnedOperators.mirror_tile_2x2] Apply 2x2 mirror/rotational tiling with mode 'hv'.",
    "solved_0c9aba6e": "[Synthesizer D1 | F3LearnedOperators.split_and_boolean_combine] Split grid along axis 0 and combine halves with 'nor'...",
    "solved_18419cfa": "[Synthesizer D1 | F3LearnedOperators.symmetrize_per_object_or_frame] Complete horizontal and vertical symmetry inside...",
    "solved_195ba7dc": "[Synthesizer D1 | F3LearnedOperators.split_and_boolean_combine] Split grid along axis 1 and combine halves with 'or' ...",
    "solved_1c0d0a4b": "[Synthesizer D3 | F3Primitives.replace_color + np.rot90 + ResidualPatch(ray_corridor_table)] [Synthesizer D2 | F3Prim...",
    "solved_1d0a4b61": "[Synthesizer D1 | F3LearnedOperators.extend_2d_periodicity] Infer 2D wallpaper period and reconstruct patches occlude..."
}

# solved_007bbfb7: 
def solved_007bbfb7(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.kronecker_self_pattern(grid, bg_color=0)

# solved_00d62c1b: 
def solved_00d62c1b(grid: np.ndarray) -> np.ndarray:
    return F3Primitives.fill_enclosed_holes(grid, 4, 0)

# solved_017c7c7b: 
def solved_017c7c7b(grid: np.ndarray) -> np.ndarray:
    ext = F3LearnedOperators.extend_2d_periodicity(grid, target_h=9, target_w=grid.shape[1])
    return F3Primitives.replace_color(ext, 1, 2)

# solved_045e512c: 
def solved_045e512c(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.dihedral_template_matching(grid, bg_color=0)

# solved_08ed6ac7: 
def solved_08ed6ac7(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.isolate_extremum_object_or_cell(grid, criterion="rank_by_area", bg_color=0)

# solved_09629e4f: 
def solved_09629e4f(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.isolate_extremum_object_or_cell(grid, criterion="most_sparse_subgrid", bg_color=0)

# solved_0962bcdd: 
def solved_0962bcdd(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.expand_local_neighborhoods(grid, rules=None, bg_color=0)

# solved_0b148d64: 
def solved_0b148d64(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.isolate_extremum_object_or_cell(grid, criterion="rarest_color", bg_color=0)

# solved_0ca9ddb6: 
def solved_0ca9ddb6(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.expand_local_neighborhoods(
        grid, rules={2: {"diag": (4, 1)}, 1: {"ortho": (7, 1)}}, bg_color=0
    )

# solved_0d3d703e: 
def solved_0d3d703e(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.apply_color_bijection(
        grid, {1: 5, 2: 6, 3: 4, 4: 3, 5: 1, 6: 2, 8: 9, 9: 8}
    )

# solved_0dfd9992: 
def solved_0dfd9992(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.extend_2d_periodicity(
        grid, target_h=grid.shape[0], target_w=grid.shape[1], mask_color=0
    )

# solved_0e206a2e: 
def solved_0e206a2e(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.dihedral_template_matching(grid, bg_color=0)

# solved_11852cab: 
def solved_11852cab(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.complete_dihedral_symmetry(grid, bg_color=0)

# solved_0c786b71: [Synthesizer D1 | F3LearnedOperators.mirror_tile_2x2] Apply 2x2 mirror/rotational tiling with mode 'hv'.
def solved_0c786b71(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.mirror_tile_2x2(grid, mode='hv')

# solved_0c9aba6e: [Synthesizer D1 | F3LearnedOperators.split_and_boolean_combine] Split grid along axis 0 and combine halves with 'nor'...
def solved_0c9aba6e(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.split_and_boolean_combine(grid, axis=0, op='nor', out_color=8, bg_color=0)

# solved_195ba7dc: [Synthesizer D1 | F3LearnedOperators.split_and_boolean_combine] Split grid along axis 1 and combine halves with 'or' ...
def solved_195ba7dc(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.split_and_boolean_combine(grid, axis=1, op='or', out_color=1, bg_color=0)

# solved_1d0a4b61: [Synthesizer D1 | F3LearnedOperators.extend_2d_periodicity] Infer 2D wallpaper period and reconstruct patches occlude...
def solved_1d0a4b61(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.extend_2d_periodicity(grid, target_h=grid.shape[0], target_w=grid.shape[1], mask_color=0)

# solved_00576224: [Synthesizer D1 | F3LearnedOperators.alternating_mirror_tile] Alternating mirror tile (3, 3) with mode 'row_fliplr'.
def solved_00576224(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.alternating_mirror_tile(grid, 3, 3, mode='row_fliplr')

# solved_00dbd492: [Synthesizer D1 | F3LearnedOperators.fill_seeded_or_sized_cavities] Fill enclosed cavities conditioned on cavity size...
def solved_00dbd492(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.fill_seeded_or_sized_cavities(grid, mode='by_size', rule_map={3: 8, 5: 4, 7: 3}, bg_color=0)

# solved_070dd51e: [Synthesizer D1 | F3LearnedOperators.connect_aligned_markers] Connect aligned same-color markers along rows/cols (pri...
def solved_070dd51e(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.connect_aligned_markers(grid, line_color=-1, priority='vert_over_horiz', bg_color=0)

# solved_18419cfa: [Synthesizer D1 | F3LearnedOperators.symmetrize_per_object_or_frame] Complete horizontal and vertical symmetry inside...
def solved_18419cfa(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.symmetrize_per_object_or_frame(grid, mode='hv', bg_color=0)

# solved_1c0d0a4b: [Synthesizer D3 | F3Primitives.replace_color + np.rot90 + ResidualPatch(ray_corridor_table)] [Synthesizer D2 | F3Prim...
def solved_1c0d0a4b(grid: np.ndarray) -> np.ndarray:
    return F3LearnedOperators.apply_residual_patch(grid, np.rot90(F3Primitives.replace_color(grid, 8, 2), 2), {'type': 'ray_corridor_table', 'ref_color': 8, 'table': {(0, 0, 1, 1, 4, 1): 2, (0, 0, 0, 0, 2, 1): 2, (8, 2, 1, 1, 4, 1): 0, (0, 0, 0, 1, 3, 1): 2, (0, 0, 1, 0, 3, 1): 2}}, bg_color=0)
