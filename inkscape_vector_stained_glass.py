#!/usr/bin/env python3
import inkex
import numpy as np
import math
import io
import base64
import os
from PIL import Image, ImageFilter

from inkex.units import convert_unit

from image_helper import pil_quantize_indices, same_color_components, component_representative_points_dt, thread_pool
from geometry_helper import clip_polygon_halfplane, clip_polygon_rect, solve_linear_abc, voronoi_cell, simplify_poly, snap_points_to_rect,\
    dedupe_points, ray_polygon_intersections, polygon_area, polygon_centroid


# ---------- Bowyer-Watson Delaunay triangulation (numpy only) ----------

def delaunay_triangulate(points):
    """Bowyer-Watson incremental Delaunay triangulation.

    points : (N, 2) float array
    returns: list of (i, j, k) index triples into points (counter-clockwise)
    """
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return []

    # Super-triangle enclosing all points
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    center = (min_xy + max_xy) / 2.0
    span = max(max_xy - min_xy) * 20.0 + 1e-6
    st = np.array([
        [center[0] - span, center[1] - span],
        [center[0] + span, center[1] - span],
        [center[0], center[1] + span],
    ])
    all_pts = np.vstack([pts, st])
    n_all = len(all_pts)

    def circumcircle(p1, p2, p3):
        """Return (cx, cy, r2) of circumcircle. Degenerate -> r2 = -inf."""
        ax, ay = p1
        bx, by = p2
        cx_, cy_ = p3
        d = 2.0 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
        if abs(d) < 1e-12:
            return 0.0, 0.0, -np.inf
        ux = ((ax * ax + ay * ay) * (by - cy_) +
              (bx * bx + by * by) * (cy_ - ay) +
              (cx_ * cx_ + cy_ * cy_) * (ay - by)) / d
        uy = ((ax * ax + ay * ay) * (cx_ - bx) +
              (bx * bx + by * by) * (ax - cx_) +
              (cx_ * cx_ + cy_ * cy_) * (bx - ax)) / d
        r2 = (ax - ux) ** 2 + (ay - uy) ** 2
        return ux, uy, r2

    # Start with super-triangle
    tri = [(n, n + 1, n + 2)]
    cc = [circumcircle(all_pts[n], all_pts[n + 1], all_pts[n + 2])]

    for pi in range(n):
        p = all_pts[pi]
        bad = []
        for ti, (ux, uy, r2) in enumerate(cc):
            if r2 > 0 and (p[0] - ux) ** 2 + (p[1] - uy) ** 2 < r2 - 1e-9:
                bad.append(ti)
        if not bad:
            continue

        # Boundary edges of the cavity (edges appearing exactly once)
        edge_count = {}
        for ti in bad:
            a, b, c = tri[ti]
            for e in ((a, b), (b, c), (c, a)):
                key = (min(e), max(e))
                if key in edge_count:
                    edge_count[key][1] += 1
                else:
                    edge_count[key] = [e, 1]
        boundary = [v[0] for v in edge_count.values() if v[1] == 1]

        # Remove bad triangles
        for ti in sorted(bad, reverse=True):
            tri.pop(ti)
            cc.pop(ti)

        # Re-triangulate cavity with new point
        for (a, b) in boundary:
            tri.append((a, b, pi))
            cc.append(circumcircle(all_pts[a], all_pts[b], all_pts[pi]))

    # Drop triangles touching super-triangle vertices
    result = []
    for (a, b, c) in tri:
        if a >= n or b >= n or c >= n:
            continue
        result.append((a, b, c))
    return result


def _boundary_points(w, h, spacing):
    """Add points along the four rectangle edges at the given spacing (including corners, no duplicates)."""
    if spacing <= 0:
        spacing = max(w, h) / 8.0

    def lin(a, b):
        n = max(1, int(np.ceil(abs(b - a) / spacing)))
        return np.linspace(a, b, n + 1)

    xs = lin(0.0, w)
    ys = lin(0.0, h)

    pts = []
    # Bottom edge y=0, top edge y=h
    for x in xs:
        pts.append((x, 0.0))
        pts.append((x, h))
    # Left edge x=0, right edge x=w (corners will be duplicated, deduplicated later)
    for y in ys:
        pts.append((0.0, y))
        pts.append((w, y))
    return np.array(pts, dtype=np.float64)

def triangle_cells(w, h, pts, snap_ratio=0.6, boundary_spacing=None):
    """Generate triangle cells covering [0,w]x[0,h].

    Strategy:
      1. Snap points within snap_ratio * grid_step of a boundary onto it.
      2. Add uniformly spaced boundary points to keep edges well-sampled.
      3. Delaunay + clip each triangle to the rectangle.

    Returns list of polygons.
    """
    n_seeds = len(pts)
    # if n_seeds < 4:
    #     n_seeds = 4

    # # Grid dimensions ~ sqrt(n_seeds), aspect-correct
    aspect = w / max(h, 1e-6)
    nx = max(2, int(round(np.sqrt(n_seeds * aspect))))
    ny = max(2, int(round(n_seeds / nx)))
    if nx * ny < n_seeds:
        ny = int(np.ceil(n_seeds / nx))

    step_x = w / nx
    step_y = h / ny

    # # Jittered grid points
    # gx = (np.arange(nx) + 0.5) * step_x
    # gy = (np.arange(ny) + 0.5) * step_y
    # GX, GY = np.meshgrid(gx, gy)
    # pts = np.column_stack([GX.ravel(), GY.ravel()])
    # dx = step_x * jitter
    # dy = step_y * jitter
    # pts[:, 0] += rng.uniform(-dx, dx, size=len(pts))
    # pts[:, 1] += rng.uniform(-dy, dy, size=len(pts))

    # --- NEW: snap near-boundary points onto the rectangle edges ---
    snap_dist = snap_ratio * min(step_x, step_y)
    pts = snap_points_to_rect(pts, w, h, snap_dist)

    # --- NEW: add uniformly spaced boundary points ---
    if boundary_spacing is None:
        boundary_spacing = min(step_x, step_y) * 0.9
    bpts = _boundary_points(w, h, boundary_spacing)

    # Remove boundary points that are too close to existing points in pts
    # (so the jittered grid keeps priority and boundary points only fill gaps)
    if len(bpts) > 0 and len(pts) > 0:
        min_sep = min(step_x, step_y) * 0.5
        min_sep_sq = min_sep * min_sep
        keep_mask = np.ones(len(bpts), dtype=bool)
        # Chunked distance check to avoid a huge (len(bpts) x len(pts)) matrix
        chunk = 2048
        for s in range(0, len(bpts), chunk):
            e = min(s + chunk, len(bpts))
            sub = bpts[s:e]
            d2 = ((sub[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
            keep_mask[s:e] = d2.min(axis=1) >= min_sep_sq
        bpts = bpts[keep_mask]
    
    pts = np.vstack([pts, bpts])

    # Deduplicate (corners / snapped points may coincide with boundary points)
    pts = dedupe_points(pts, eps=1e-6)

    # Final safety clip
    pts[:, 0] = np.clip(pts[:, 0], 0.0, w)
    pts[:, 1] = np.clip(pts[:, 1], 0.0, h)

    tris = delaunay_triangulate(pts)
    cells = []
    for (a, b, c) in tris:
        tri_pts = [tuple(pts[a]), tuple(pts[b]), tuple(pts[c])]
        clipped = clip_polygon_rect(tri_pts, w, h)
        clipped = simplify_poly(clipped)
        if len(clipped) >= 3:
            cells.append(clipped)
    return cells


# ---------- Seed generation (A1: grid-accelerated Poisson-disk) ----------

def grid_seeds(w, h, n_seeds, min_dist=None, rng=None):
    """Grid-accelerated Poisson-disk approximation."""
    if rng is None:
        rng = np.random.default_rng()

    if min_dist is None:
        min_dist = np.sqrt(w * h / max(n_seeds, 1)) * 0.65
    if min_dist <= 0:
        min_dist = 1e-6

    cell = min_dist
    gw = max(1, int(np.ceil(w / cell)))
    gh = max(1, int(np.ceil(h / cell)))
    grid = np.full((gh, gw), -1, dtype=np.int32)

    seeds = np.empty((n_seeds, 2), dtype=np.float64)
    n_found = 0
    min_dist_sq = min_dist * min_dist
    attempts = 0
    max_attempts = n_seeds * 40

    while n_found < n_seeds and attempts < max_attempts:
        x = rng.uniform(0.0, w)
        y = rng.uniform(0.0, h)
        attempts += 1

        gx = int(x / cell)
        gy = int(y / cell)
        if gx >= gw:
            gx = gw - 1
        if gy >= gh:
            gy = gh - 1

        ok = True
        y0 = max(0, gy - 1)
        y1 = min(gh, gy + 2)
        x0 = max(0, gx - 1)
        x1 = min(gw, gx + 2)
        for ny in range(y0, y1):
            for nx in range(x0, x1):
                j = grid[ny, nx]
                if j >= 0:
                    dx = seeds[j, 0] - x
                    dy = seeds[j, 1] - y
                    if dx * dx + dy * dy < min_dist_sq:
                        ok = False
                        break
            if not ok:
                break

        if ok:
            seeds[n_found, 0] = x
            seeds[n_found, 1] = y
            grid[gy, gx] = n_found
            n_found += 1

    if n_found == 0:
        return np.array([[w / 2.0, h / 2.0]], dtype=np.float64)
    return seeds[:n_found]


def generate_seeds(w, h, n_seeds, img_small, rng, bias_type):
    if bias_type == 'edges':
        gray = np.array(img_small.convert("L"), dtype=np.float64)
        gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
        gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
        edge_map = gx + gy
        prob = edge_map.ravel() + edge_map.mean() * 0.5
        prob /= prob.sum()
        idx = rng.choice(w * h, n_seeds, replace=False, p=prob)
        return np.stack([idx % w, idx // w], axis=1).astype(np.float64)
    elif bias_type == 'brightness':
        gray = np.array(img_small.convert("L"), dtype=np.float64)
        prob = 256.0 - gray.ravel()
        prob /= prob.sum()
        idx = rng.choice(w * h, n_seeds, replace=False, p=prob)
        return np.stack([idx % w, idx // w], axis=1).astype(np.float64)
    
    return grid_seeds(w, h, n_seeds, rng=rng)
    


def _edge_map(gray):
    """Sobel-ish gradient magnitude, normalized to [0, 1]."""
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    e = gx + gy
    mx = e.max()
    return e / mx if mx > 0 else e


def _component_edge_energy(comp_info, labels, edge_map):
    """Mean edge strength inside each component's mask."""
    n = len(comp_info)
    out = np.zeros(n, dtype=np.float64)
    if edge_map is None:
        return out
    # Group pixels by component id once, then average edge_map over each group.
    # labels is (h, w); flat edge_map indexed the same way.
    flat_labels = labels.ravel()
    flat_edge = edge_map.ravel()
    # Sum edge per component
    sums = np.bincount(flat_labels, weights=flat_edge, minlength=n)
    counts = np.bincount(flat_labels, minlength=n)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    return out

# ---------- A3: dominant palette index per cell (bincount + argmax) ----------

def dominant_palette_index(region_labels, pal_indices, n_cells, n_palette):
    """For each cell, the most frequent palette index (ties -> smallest index)."""
    flat_cell = region_labels.ravel().astype(np.int64)
    flat_pal = pal_indices.ravel().astype(np.int64)
    combined = flat_cell * n_palette + flat_pal
    hist = np.bincount(combined, minlength=n_cells * n_palette)
    hist = hist.reshape(n_cells, n_palette)
    return hist.argmax(axis=1).astype(np.int32)


# ---------- knife cut ----------
def _component_edge_energy(comp_info, labels, edge_map):
    """Mean edge strength inside each component's mask."""
    n = len(comp_info)
    out = np.zeros(n, dtype=np.float64)
    if edge_map is None:
        return out
    # Group pixels by component id once, then average edge_map over each group.
    # labels is (h, w); flat edge_map indexed the same way.
    flat_labels = labels.ravel()
    flat_edge = edge_map.ravel()
    # Sum edge per component
    sums = np.bincount(flat_labels, weights=flat_edge, minlength=n)
    counts = np.bincount(flat_labels, minlength=n)
    nz = counts > 0
    out[nz] = sums[nz] / counts[nz]
    return out

def _pair_edge_score(pts, i, j, edge_map):
    """Average edge strength sampled at both endpoints of pair (i, j)."""
    if edge_map is None:
        return 0.0
    h, w = edge_map.shape
    xi = int(np.clip(pts[i][0], 0, w - 1))
    yi = int(np.clip(pts[i][1], 0, h - 1))
    xj = int(np.clip(pts[j][0], 0, w - 1))
    yj = int(np.clip(pts[j][1], 0, h - 1))
    return 0.5 * (edge_map[yi, xi] + edge_map[yj, xj])

def centroid_cut_sparse(centroids, w, h, n_seeds, rng, edge_map=None):
    """Each centroid only cuts against its single nearest neighbor.

    If edge_map is provided, the (random) subset of pairs is chosen with a
    bias toward pairs whose endpoints sit on strong image edges, so cuts
    tend to follow visual structure.
    """
    pts = np.asarray(centroids, dtype=np.float64)
    n = len(pts)
    if n < 2:
        return [[(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]]

    # Find nearest neighbor for each centroid
    d = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.argmin(axis=1)

    # Deduplicate undirected pairs (i, nn[i])
    i_arr = np.arange(n)
    lo = np.minimum(i_arr, nn)
    hi = np.maximum(i_arr, nn)
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)

    # Choose n//2 pairs. With an edge map, bias selection by edge strength.
    k = max(1, n // 2)
    if len(pairs) > k:
        if edge_map is not None:
            scores = np.array(
                [_pair_edge_score(pts, i, j, edge_map) for (i, j) in pairs],
                dtype=np.float64,
            )
            # Convert scores to sampling weights; add a small floor so that
            # non-edge pairs still have a chance (keeps coverage reasonable).
            weights = scores + 0.05
            weights /= weights.sum()
            pairs = rng.choice(pairs, size=k, replace=False, p=weights)
        else:
            pairs = rng.choice(pairs, size=k, replace=False)

    # Start with full rectangle, cut by each pair's bisector
    cells = [[(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]]
    ptsi = pts[pairs[:,0]]
    ptsj = pts[pairs[:,1]]
    abc = solve_linear_abc(ptsi, ptsj)
    for (a, b, c) in abc:
        if len(cells) > n_seeds:
            break
        new_cells = []
        for poly in cells:
            left = clip_polygon_halfplane(poly, a, b, c)
            right = clip_polygon_halfplane(poly, -a, -b, -c)
            left = simplify_poly(left)
            right = simplify_poly(right)
            if len(left) >= 3:
                new_cells.append(left)
            if len(right) >= 3:
                new_cells.append(right)
        cells = new_cells
    return cells

def _select_centroids_by_area_with_quota(comp_pal, comp_ids, comp_info,
                                         n_seeds, edge_energy=None,
                                         edge_bonus=0.0,
                                         palette=None,
                                         brightness_bonus=0.0):
    """Select up to n_seeds component indices:
       1. Guarantee at least one component per palette color (the largest one).
       2. Fill the remaining budget with the largest components overall.

    If edge_energy is provided and edge_bonus > 0, ranking adds
    edge_bonus * (normalized edge energy).

    If palette is provided and brightness_bonus > 0, each component's
    brightness is taken directly from its palette color (components are
    single-color by construction). Darker components get a higher score,
    so brighter areas have a lower selection probability.

    Returns a sorted list of indices into the centroid/comp_pal lists.
    """
    m = len(comp_pal)
    if m <= n_seeds:
        return list(range(m))

    areas = np.array([comp_info[k][1] for k in comp_ids], dtype=np.float64)
    amax = areas.max()
    a_norm = areas / amax if amax > 0 else areas

    score = a_norm.copy()

    if edge_energy is not None and edge_bonus > 0.0:
        ee = edge_energy[np.asarray(comp_ids)]
        emax = ee.max()
        e_norm = ee / emax if emax > 0 else ee
        score = score + edge_bonus * e_norm

    if palette is not None and brightness_bonus > 0.0:
        pal = np.array(comp_pal, dtype=np.int64)
        colors = np.asarray(palette, dtype=np.float64)[pal]  # (m, 3)
        # Perceptual-ish luma (Rec. 601)
        luma = 0.299 * colors[:, 0] + 0.587 * colors[:, 1] + 0.114 * colors[:, 2]
        lmax = luma.max()
        l_norm = luma / lmax if lmax > 0 else luma
        score = score + brightness_bonus * l_norm

    pal = np.array(comp_pal, dtype=np.int64)

    selected = set()

    # ---- Pass 1: one (largest-score) component per palette color ----
    order_desc = np.argsort(score)[::-1]
    seen_pal = set()
    for i in order_desc:
        p = int(pal[i])
        if p not in seen_pal:
            seen_pal.add(p)
            selected.add(int(i))
            if len(selected) >= n_seeds:
                break

    # ---- Pass 2: fill remaining budget with highest-score components ----
    if len(selected) < n_seeds:
        for i in order_desc:
            if len(selected) >= n_seeds:
                break
            i = int(i)
            if i not in selected:
                selected.add(i)

    return sorted(selected)

def knife_cut_cells(img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                    bias_type, bridge_ratio=0.01,
                    edge_weight=0.5, edge_bonus=0.5,
                    brightness_bonus=0.5):
    seed_from_edges = bias_type == 'edges'
    edge_map = None
    gray = None
    if seed_from_edges:
        gray = np.array(img_small.convert("L"), dtype=np.float64)
        edge_map = _edge_map(gray)

    if len(manual_seeds) == 0:
        bridge_r = int(round(max(w, h) * bridge_ratio))
        min_area = w * h / (n_seeds * n_palette)
        labels, comp_info = same_color_components(
            pal_indices, bridge_radius=bridge_r
        )

        edge_energy = None
        if seed_from_edges:
            edge_energy = _component_edge_energy(comp_info, labels, edge_map)

        centroids, comp_pal, comp_ids = component_representative_points_dt(
            labels, comp_info, min_area,
            edge_map=edge_map,
            edge_weight=edge_weight if seed_from_edges else 0.0,
        )
        if not centroids:
            return []

        if len(centroids) > 1:
            keep = _select_centroids_by_area_with_quota(
                comp_pal, comp_ids, comp_info, int(n_seeds),
                edge_energy=edge_energy,
                edge_bonus=edge_bonus if seed_from_edges else 0.0,
                palette=palette,
                brightness_bonus=brightness_bonus if bias_type == 'brightness' else 0.0,
            )
            centroids = [centroids[i] for i in keep]
    else:
        centroids = manual_seeds

    final_polys = centroid_cut_sparse(
        centroids, float(w), float(h), n_seeds, rng,
        edge_map=edge_map if seed_from_edges else None,
    )
    if not final_polys:
        return []
    return final_polys


# flower cut
def flower_seeds(center, poly, n_petals=12, seeds_per_petal=6, min_radius=2.0, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    C = np.asarray(center, dtype=np.float64)
    P = np.asarray(poly, dtype=np.float64)
    seeds = [C]
    for s in range(1, seeds_per_petal + 1):
        n_ring_petals = max(3, n_petals - seeds_per_petal // 2 + s) # grow out with 1 more pedal per step
        for i in range(n_ring_petals):
            th = 2*np.pi * (i / n_petals + rng.normal(0, 0.5 / n_ring_petals))  # jitter
            r_frac = (s / seeds_per_petal) ** 2
            _, hit = ray_polygon_intersections(C, np.array([[np.cos(th), np.sin(th)]]), P)
            rmax = np.linalg.norm(hit[0] - C) if np.isfinite(hit[0]).all() else 0.5
            r = rmax * r_frac * (0.9 + 0.2 * rng.random())
            if r > min_radius:
                seeds.append(C + r * np.array([np.cos(th), np.sin(th)]))
    return np.array(seeds)

def flowers_cut_cells(img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                    bias_type):
    is_manual_seed = len(manual_seeds) > 0
    seeds = manual_seeds if is_manual_seed else generate_seeds(w, h, rng.integers(1, 5, endpoint=True), img_small, rng, bias_type)
    rect = np.asarray([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])

    def _process_voronoi(args):
        i, seeds, is_manual_seed, rect = args
        poly = voronoi_cell(i, seeds, rect, None)
        if poly is None or len(poly) < 3:
            return
        poly = simplify_poly(poly)
        if len(poly) < 3:
            return

        lo = np.min(poly, axis=0)
        hi = np.max(poly, axis=0)

        if np.prod(hi - lo) < 4.0:
            return

        # use manual seed as flower center if provided
        centroid = seeds[i] if is_manual_seed else polygon_centroid(poly)

        return poly, centroid, polygon_area(poly)

    if len(seeds) == 0:
        seeds = np.round((w / 2, h / 2))
    
    if len(seeds) <= 1:
        poly_infos = [[rect, seeds[0], w * h]]
    else:
        poly_infos = [x for x in thread_pool.map(_process_voronoi, [(i, seeds, is_manual_seed, rect) for i in range(len(seeds))]) if x is not None]
    areas = np.asarray([x[2] for x in poly_infos])
    n_sub_seeds = np.ceil(areas / areas.sum() * n_seeds).astype(np.int32)
    final_polys = []

    def _process_poly(args):
        poly_info, n_sub_seedi = args
        poly, centroid, area = poly_info
        n_petals = 12 if n_sub_seedi > 12 else math.ceil(n_sub_seedi / 2)
        seeds_per_petal = math.ceil(n_sub_seedi / n_petals)
        sub_seeds = flower_seeds(centroid, poly, n_petals=n_petals, seeds_per_petal=seeds_per_petal)
        return [voronoi_cell(j, sub_seeds, poly, None) for j in range(len(sub_seeds))]
            
    for x in thread_pool.map(_process_poly, list(zip(poly_infos, n_sub_seeds))):
        final_polys.extend([p for p in x if p is not None])
    return final_polys


def flowers_soft_cut_cells(img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                    bias_type):
    is_manual_seed = len(manual_seeds) > 0
    seeds = manual_seeds if is_manual_seed else generate_seeds(w, h, rng.integers(1, 5, endpoint=True), img_small, rng, bias_type)
    rect = np.asarray([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])

    def _process_voronoi(args):
        i, seeds, is_manual_seed, rect = args
        poly = voronoi_cell(i, seeds, rect, None)
        if poly is None or len(poly) < 3:
            return
        poly = simplify_poly(poly)
        if len(poly) < 3:
            return

        lo = np.min(poly, axis=0)
        hi = np.max(poly, axis=0)

        if np.prod(hi - lo) < 4.0:
            return

        # use manual seed as flower center if provided
        centroid = seeds[i] if is_manual_seed else polygon_centroid(poly)

        return poly, centroid, polygon_area(poly)

    if len(seeds) == 0:
        seeds = np.round((w / 2, h / 2))
    
    if len(seeds) <= 1:
        poly_infos = [[rect, seeds[0], w * h]]
    else:
        poly_infos = [x for x in thread_pool.map(_process_voronoi, [(i, seeds, is_manual_seed, rect) for i in range(len(seeds))]) if x is not None]
    areas = np.asarray([x[2] for x in poly_infos])
    n_sub_seeds = np.ceil(areas / areas.sum() * n_seeds).astype(np.int32)

    final_seeds = []
    for (poly, centroid, area), n_sub_seedi in zip(poly_infos, n_sub_seeds):
        n_petals = 12 if n_sub_seedi > 12 else math.ceil(n_sub_seedi / 2)
        seeds_per_petal = math.ceil(n_sub_seedi / n_petals)
        final_seeds.append(flower_seeds(centroid, poly, n_petals=n_petals, seeds_per_petal=seeds_per_petal))

    final_seeds = dedupe_points(np.concatenate(final_seeds), eps=2)
    
    def _process_poly(args):
        i, seeds, rect = args
        poly = voronoi_cell(i, seeds, rect, None)
        if poly is None or len(poly) < 3:
            return
        poly = simplify_poly(poly)
        if len(poly) < 3:
            return

        lo = np.min(poly, axis=0)
        hi = np.max(poly, axis=0)

        if np.prod(hi - lo) < 4.0:
            return
        
        return poly
        
    final_polys = [x for x in thread_pool.map(_process_poly, [(i, final_seeds, rect) for i in range(len(final_seeds))]) if x is not None]

    return final_polys


# water flow
# todo


# ---------- Triangle-based per-cell sampling ----------

def dominant_palette_index_for_polygons(polygons, pal_indices, n_palette, w, h):
    """For each polygon, the dominant palette index of the pixels it covers.

    We rasterize each polygon separately using a simple even-odd point-in-polygon
    test on a per-polygon bounding box. For a few hundred cells this is fine.
    """
    def _process(args):
        pal_indices, poly = args
        h_img, w_img = pal_indices.shape
        xs = poly[:,0]
        ys = poly[:,1]
        x0 = max(0, int(np.floor(np.min(xs))))
        x1 = min(w_img, int(np.ceil(np.max(xs))) + 1)
        y0 = max(0, int(np.floor(np.min(ys))))
        y1 = min(h_img, int(np.ceil(np.max(ys))) + 1)
        if x1 <= x0 or y1 <= y0:
            return 0
        # Pixel centers in bbox
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        # Point-in-polygon (ray casting) — vectorized over pixels
        inside = np.zeros(gx.shape, dtype=bool)
        n = len(poly)
        for i in range(n):
            ax, ay = poly[i]
            bx, by = poly[(i + 1) % n]
            # Edge (ax,ay)-(bx,by); count crossings of horizontal ray to +x
            cond = ((ay > gy) != (by > gy))
            x_int = (bx - ax) * (gy - ay) / (by - ay + 1e-12) + ax
            inside ^= cond & (gx < x_int)
        if not inside.any():
            return 0
        idx = np.where(inside.ravel())[0]
        # Map flat local index back to global pixel index
        local_y, local_x = np.divmod(idx, (x1 - x0))
        glob_y = local_y + y0
        glob_x = local_x + x0
        vals = pal_indices[glob_y, glob_x]
        counts = np.bincount(vals, minlength=n_palette)
        return int(np.argmax(counts))

    results = list(thread_pool.map(_process, [(pal_indices, poly) for poly in polygons]))
    return np.array(results, dtype=np.int32)

# ---------- Inkscape extension ----------

class VectorStainedGlassExtension(inkex.EffectExtension):
    def add_arguments(self, pars):
        pars.add_argument(
            "--blur_size", type=int, default=0,
            help="Blur size applied before segmentation; 0 disables blur",
        )
        pars.add_argument(
            "--cell_shape", type=str, default="voronoi",
            choices=["voronoi", "triangles", "knife_cut", "flowers", "flowers_soft"],
            help="Cell shape: voronoi (default) or triangles (Delaunay-based) or knife_cut or flowers",
        )
        pars.add_argument("--n_colors", type=int, default=12)
        pars.add_argument("--n_seeds", type=int, default=600)
        pars.add_argument("--stroke_width", type=float, default=2.0)
        pars.add_argument("--bias_type", type=str, default="none",
            choices=["none", "edges", "brightness"],
        )
        pars.add_argument("--random_seed", type=int, default=42)
        pars.add_argument("--max_dim", type=int, default=600)

    def _load_image(self, image):
        xlink = image.get("{http://www.w3.org/1999/xlink}href")
        if xlink is None:
            xlink = image.get("href")
        if xlink is None:
            raise ValueError("Image element has no href")

        if xlink.startswith("data:image"):
            _, data = xlink.split(",", 1)
            return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")

        if not os.path.isabs(xlink):
            base = os.path.dirname(self.document_path) if self.document_path else os.getcwd()
            xlink = os.path.normpath(os.path.join(base, xlink))
        return Image.open(xlink).convert("RGB")

    @staticmethod
    def _parse_length(value):
        if value is None:
            raise ValueError("missing length")
        return float(convert_unit(str(value), "px"))

    def _image_geometry(self, image):
        x = self._parse_length(image.get("x", 0))
        y = self._parse_length(image.get("y", 0))
        w = self._parse_length(image.get("width"))
        h = self._parse_length(image.get("height"))
        transform = image.get("transform")
        return x, y, w, h, transform

    def effect(self):
        image = None
        manual_seeds = []
        for el in self.svg.selection.values():
            if image is None and isinstance(el, inkex.Image):
                image = el
            else:
                # use non image elements as manual seeds, replacing random generated seed
                manual_seeds.append([int(convert_unit(str(value), "px")) for value in el.bounding_box().center])
       
        if image is None:
            inkex.errormsg("Please select a bitmap image first.")
            return
        
        try:
            img = self._load_image(image)
            img_x, img_y, img_w, img_h, img_transform = self._image_geometry(image)
        except Exception as e:
            inkex.errormsg(f"Cannot read image: {e}")
            return
     
        orig_w, orig_h = img.size
        max_dim = self.options.max_dim
        if max(orig_w, orig_h) > max_dim:
            ratio = max_dim / max(orig_w, orig_h)
            w, h = int(orig_w * ratio), int(orig_h * ratio)
            img_small = img.resize((w, h))
        else:
            w, h = orig_w, orig_h
            img_small = img

        if len(manual_seeds) > 0:
            manual_seeds = np.asarray(manual_seeds) - (img_x, img_y)
            mask = (manual_seeds[:,0] >= 0) & (manual_seeds[:,0] < img_w) & (manual_seeds[:,1] >= 0) & (manual_seeds[:,1] < img_h)
            manual_seeds = np.round(manual_seeds[mask] / img_w * w)

        blur_size = max(0, self.options.blur_size)
        if blur_size > 0:
            blur_size = int(blur_size) // 2 * 2 + 1
            img_small = img_small.filter(ImageFilter.MedianFilter(size=blur_size))

        if w < 1 or h < 1:
            inkex.errormsg("Image is too small.")
            return

        rng = np.random.default_rng(self.options.random_seed)

        # 1. Quantize -> palette index map
        pal_indices, palette = pil_quantize_indices(img_small, self.options.n_colors)
        n_palette = len(palette)

        # 2. Build cells according to the chosen shape
        cell_shape = self.options.cell_shape
        polygons = []
        dom_idx = None
        n_seeds = min(int(self.options.n_seeds), max(1, w * h // 100))

        if cell_shape == "triangles":
            seeds = manual_seeds if len(manual_seeds) > 0 else generate_seeds(w, h, n_seeds, img_small, rng, bias_type=self.options.bias_type)
            polygons = triangle_cells(w, h, seeds)
            if not polygons:
                inkex.errormsg("Triangulation produced no cells.")
                return
        elif cell_shape == "knife_cut":
            polygons = knife_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )
            if not polygons:
                inkex.errormsg("Knife-cut produced no cells.")
                return
        elif cell_shape == "flowers":
            polygons = flowers_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )
            if not polygons:
                inkex.errormsg("Flowers-cut produced no cells.")
                return
        elif cell_shape == "flowers_soft":
            polygons = flowers_soft_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )
            if not polygons:
                inkex.errormsg("Flowers-cut produced no cells.")
                return
        else:
            # Voronoi (original path)
            seeds = manual_seeds if len(manual_seeds) > 0 else  generate_seeds(w, h, n_seeds, img_small, rng, bias_type=self.options.bias_type)
            rect = np.asarray([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])

            def _voronoi_process(args):
                i, seeds, rect = args
                poly = voronoi_cell(i, seeds, rect, None)
                if poly is None or len(poly) < 3:
                    return
                poly = simplify_poly(poly)
                if len(poly) < 3:
                    return

                lo = np.min(poly, axis=0)
                hi = np.max(poly, axis=0)

                if np.prod(hi - lo) < 4.0:
                    return
                return poly

            polygons = [x for x in thread_pool.map(_voronoi_process, [(i, seeds, rect) for i in range(len(seeds))]) if x is not None]

        dom_idx = dominant_palette_index_for_polygons(
            polygons, pal_indices, n_palette, w, h
        )

        # 5. Build SVG paths
        group = inkex.Group()
        group.set("id", "vector_stained_glass")
        group.set("inkscape:label", "Vector Stained Glass")

        scale_x = img_w / w
        scale_y = img_h / h
        offset_x = img_x
        offset_y = img_y

        paths = []
        for i, poly in enumerate(polygons):
            if len(poly) < 3:
                continue
            r, g, b = (int(v) for v in palette[dom_idx[i]])

            d_parts = [f"M {poly[0][0] * scale_x + offset_x:.3f},{poly[0][1] * scale_y + offset_y:.3f} "]
            for pt in poly[1:]:
                d_parts.append(f"L {pt[0] * scale_x + offset_x:.3f},{pt[1] * scale_y + offset_y:.3f} ")
            d_parts.append("Z")

            path = inkex.PathElement()
            path.set("d", "".join(d_parts))
            path.set("fill", f"rgb({r},{g},{b})")
            path.set("stroke", "#1a1a1a")
            sw = self.options.stroke_width * ((scale_x + scale_y) / 2.0)
            path.set("stroke-width", f"{sw:.3f}")
            path.set("stroke-linejoin", "round")
            path.set("stroke-linecap", "round")
            paths.append(path)
        group.extend(paths)

        if img_transform:
            group.set("transform", img_transform)

        self.svg.append(group)


if __name__ == "__main__":
    VectorStainedGlassExtension().run()
