#!/usr/bin/env python3
import inkex
import numpy as np
import math
import io
import base64
import os
from PIL import Image, ImageFilter

from inkex.units import convert_unit

from image_helper import pil_quantize_indices, same_color_components, component_representative_points_dt, simple_saliency, thread_pool
from geometry_helper import clip_polygon_halfplane_both, clip_polygon_rect, solve_linear_abc, voronoi_cell, simplify_poly, snap_points_to_rect,\
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
    pts = np.clip(pts, (0.0, 0.0), (w, h), out=pts)

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


def _seed_weights(img_small, rng, bias_type):
    weights = None
    if bias_type == 'edges':
        gray = np.array(img_small.convert("L"), dtype=np.float64)
        edge_map = _edge_map(gray)
        weights = edge_map + edge_map.mean() * 0.5
    elif bias_type == 'brightness':
        gray = np.array(img_small.convert("L"), dtype=np.float64)
        weights = 256.0 - gray
    elif bias_type == 'saliency':
        weights = simple_saliency(img_small.convert("L"))

    return weights

def generate_seeds(w, h, n_seeds, img_small, rng, bias_type):
    weights = _seed_weights(img_small, rng, bias_type)

    if weights is not None:
        prob = weights.ravel()
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


# ---------- A3: dominant palette index per cell (bincount + argmax) ----------

def dominant_palette_index(region_labels, pal_indices, n_cells, n_palette):
    """For each cell, the most frequent palette index (ties -> smallest index)."""
    flat_cell = region_labels.ravel().astype(np.int64)
    flat_pal = pal_indices.ravel().astype(np.int64)
    combined = flat_cell * n_palette + flat_pal
    hist = np.bincount(combined, minlength=n_cells * n_palette)
    hist = hist.reshape(n_cells, n_palette)
    return hist.argmax(axis=1).astype(np.int32)


# ---------- Metro tile cut ----------
def _metro_score_grid(img_small, rng, gw, gh, bias_type):
    """Build a (gh, gw) score map in [0, 1] used to scale tile sizes.

    - 'edges'      : gradient magnitude, strong edges -> high score -> bigger tiles
    - 'brightness' : brighter pixels -> high score -> bigger tiles
    - else         : random uniform score
    """
    weights = _seed_weights(img_small, rng, bias_type)
    if weights is None:
        return rng.uniform(size=(gh, gw))
    weights /= weights.max()
    raw = 1.0 - weights

    # Downsample to grid resolution with block mean (crop to multiple of gw/gh)
    hh, ww = raw.shape
    ys = (np.arange(gh + 1) * hh / gh).astype(np.int64); ys[-1] = hh
    xs = (np.arange(gw + 1) * ww / gw).astype(np.int64); xs[-1] = ww
    sums = np.add.reduceat(np.add.reduceat(raw, ys[:-1], axis=0), xs[:-1], axis=1)
    counts = np.outer(np.diff(ys), np.diff(xs))
    scores = sums / np.maximum(counts, 1)
    # norm
    scores -= scores.min()
    scores /= scores.max() + 1e-12
    return scores

def metro_cut_cells(img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                    bias_type):
    """Generate Metro-style rectangular tiles covering [0,w]x[0,h].

    Strategy:
      1. Estimate a low-res grid (gw, gh) from (w, h) aspect and n_seeds.
      2. Place max-size tiles at manual_seeds (pre-occupied).
      3. Fill the grid with Metro rectangles in integer grid units:
         batch candidates -> sort by area desc -> greedy vectorized placement
         -> integral-image hole fill.
      4. Scale grid rectangles back to (w, h) float coordinates.
    """
    w = float(w)
    h = float(h)
    if w < 1 or h < 1 or n_seeds < 1:
        return []

    # ---- 1. Low-res grid size from aspect ratio and n_seeds ----
    aspect = w / max(h, 1e-6)
    gw = max(2, int(round(math.sqrt(n_seeds * aspect))))
    gh = max(2, int(round(n_seeds / gw)))
    gw = max(2, min(gw, int(w)))
    gh = max(2, min(gh, int(h)))

    min_side = 1
    max_side = 3
    min_long = 1
    max_long = max_side
    min_short = 1
    max_short = max_side

    cand_tiles = np.column_stack([
        np.repeat(np.arange(min_long, max_long + 1), max_long - min_long + 1),
        np.tile(np.arange(min_short, max_short + 1), max_short - min_short + 1),
    ])
    areas = cand_tiles.prod(axis=1)
    cand_tiles = cand_tiles[np.argsort(areas)]

    occupied = np.zeros((gh, gw), dtype=bool)
    placed = []          # (x, y, rw, rh) in grid units
    occupied_area = 0

    # ---- 2. Pre-place max-size tiles at manual seeds ----
    if manual_seeds is not None and len(manual_seeds) > 0:
        seeds = np.asarray(manual_seeds, dtype=np.float64)
        # manual_seeds are in (w, h) float coordinates -> grid units
        g_seeds = np.clip(seeds * (gw / w, gh / h), (0, 0), (gw - 1, gh - 1)).astype(np.int64)

        # max-size tile: square max_side x max_side (clamped to grid)
        tw = min(max_side, gw)
        th = min(max_side, gh)

        for cx, cy in g_seeds:
            # center the tile on the seed where possible
            xi = int(cx - tw // 2)
            yi = int(cy - th // 2)
            xi = max(0, min(xi, gw - tw))
            yi = max(0, min(yi, gh - th))
            region = occupied[yi:yi + th, xi:xi + tw]
            if region.any():
                # try to slide to a free spot near the seed
                best = None
                best_d = None
                for dy in range(-th, th + 1):
                    for dx in range(-tw, tw + 1):
                        nx = xi + dx
                        ny = yi + dy
                        if nx < 0 or ny < 0 or nx + tw > gw or ny + th > gh:
                            continue
                        if occupied[ny:ny + th, nx:nx + tw].any():
                            continue
                        d = dx * dx + dy * dy
                        if best_d is None or d < best_d:
                            best_d = d
                            best = (nx, ny)
                if best is None:
                    continue
                xi, yi = best
            occupied[yi:yi + th, xi:xi + tw] = True
            occupied_area += tw * th
            placed.append((xi, yi, tw, th))

    total_area = gw * gh

    # ---- 3. Batch-generate integer-grid candidates ----
    n_cand = int(min(max(n_seeds * 12, 400), 200_000))
    horiz = rng.random(n_cand) < 0.5

    x = (rng.random(n_cand) * gw).astype(np.int64)
    y = (rng.random(n_cand) * gh).astype(np.int64)

    score_grid = _metro_score_grid(img_small, rng, gw, gh, bias_type)
    score = score_grid[y, x]
    # pick tiles by score
    tile = cand_tiles[np.floor(score * len(cand_tiles)).astype(np.int64)]
    long_side = tile[:,0]
    short_side = tile[:,1]

    rw = np.where(horiz, long_side, short_side)
    rh = np.where(horiz, short_side, long_side)

    rw = np.minimum(tile[:,0], gw)
    rh = np.minimum(tile[:,1], gh)
    
    # realign to center
    x -= rw // 2
    y -= rh // 2

    valid = (x >= 0) & (x + rw <= gw) & (y >= 0) & (y + rh <= gh)
    x, y, rw, rh = x[valid], y[valid], rw[valid], rh[valid]
    n_cand = len(rw)
    if n_cand == 0 and not placed:
        return []

    # ---- 4. Greedy placement on the low-res grid ----
    max_rects = max(n_seeds * 6, 200)

    for i in range(len(rw)):
        if len(placed) >= max_rects:
            break
        if occupied_area >= total_area * 0.985:
            break
        xi, yi, wi, hi = int(x[i]), int(y[i]), int(rw[i]), int(rh[i])
        region = occupied[yi:yi + hi, xi:xi + wi]
        if region.any():
            continue
        occupied[yi:yi + hi, xi:xi + wi] = True
        occupied_area += wi * hi
        placed.append((xi, yi, wi, hi))

    # ---- 5. Vectorized hole fill via integral image ----
    fill_size = max(1, min_side)
    if fill_size < gw and fill_size < gh:
        integral = np.zeros((gh + 1, gw + 1), dtype=np.int32)
        np.cumsum(occupied, axis=0, out=integral[1:, 1:])
        np.cumsum(integral[1:, 1:], axis=1, out=integral[1:, 1:])

        ys = np.arange(0, gh - fill_size + 1, fill_size)
        xs = np.arange(0, gw - fill_size + 1, fill_size)
        YY, XX = np.meshgrid(ys, xs, indexing="ij")
        Y2 = YY + fill_size
        X2 = XX + fill_size
        block_sum = (integral[Y2, X2] - integral[YY, X2]
                     - integral[Y2, XX] + integral[YY, XX])
        ey, ex = np.where(block_sum == 0)
        for k in range(len(ey)):
            yi = int(ys[ey[k]]); xi = int(xs[ex[k]])
            if occupied[yi:yi + fill_size, xi:xi + fill_size].any():
                continue
            occupied[yi:yi + fill_size, xi:xi + fill_size] = True
            placed.append((xi, yi, fill_size, fill_size))

    # ---- 6. Scale grid rects -> (w, h) float polygons ----
    if not placed:
        return []
    ratio = (w / gw, h / gh)
    polys = [
        np.array([
            [xi, yi],
            [xi + wi, yi],
            [xi + wi, yi + hi],
            [xi, yi + hi],
        ], dtype=np.float64) * ratio
        for (xi, yi, wi, hi) in placed
    ]
    
    return polys


# ---------- quad cut ----------
def _tri_edges(tris):
    """Yield (edge_key, tri_index, opposite_vertex) for every triangle edge."""
    for ti, (a, b, c) in enumerate(tris):
        yield (min(a, b), max(a, b)), ti, c
        yield (min(b, c), max(b, c)), ti, a
        yield (min(c, a), max(c, a)), ti, b


def merge_triangles_to_quads(points, tris, max_angle_deg=35.0, max_aspect=3.0,
                             absorb_leftovers=True):
    """Pair adjacent triangles into quads, then absorb leftover triangles
    into an adjacent quad (producing mostly quads, some pentagons) instead of
    emitting them as standalone triangles.

    points : (N,2) float array
    tris   : list of (i,j,k) index triples
    returns: list of polygons (each a list/array of 2D points)
    """
    pts = np.asarray(points, dtype=np.float64)
    n_tri = len(tris)
    if n_tri == 0:
        return []

    # edge -> list of (tri_index, opposite_vertex)
    edge_map = {}
    for key, ti, opp in _tri_edges(tris):
        edge_map.setdefault(key, []).append((ti, opp))
    edge_map = {key: lst for key, lst in edge_map.items() if len(lst) == 2}

    def _angle_at(p, q1, q2):
        v1 = q1 - p
        v2 = q2 - p
        c = (v1 * v2).sum(axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-12)
        return np.degrees(np.acos(np.clip(c, -1.0, 1.0)))

    def _quad_score(quad):
        q = pts[quad]
        if len(q.shape) > 2:
            q = q.swapaxes(0, 1)
        ang0 = _angle_at(q[1], q[0], q[2])
        ang1 = _angle_at(q[3], q[0], q[2])
        planarity = np.abs((ang0 + ang1) - 180.0)
        d1 = np.linalg.norm(q[0] - q[2], axis=-1)
        d2 = np.linalg.norm(q[1] - q[3], axis=-1)
        aspect = np.maximum(d1, d2) / (np.minimum(d1, d2) + 1e-12)
        return planarity, aspect, planarity + (aspect - 1.0) * 10.0

    # ---- Pass 1: greedy triangle pairing into quads ----
    quads = []
    key = np.asarray(list(edge_map.keys()))
    lst = np.asarray(list(edge_map.values()))
    a, b = key[:,0], key[:,1]
    o0, o1 = lst[:,0,1], lst[:,1,1]
    quads = np.column_stack([a, o0, b, o1])
    planarity, aspect, score = _quad_score(quads)
    mask = (planarity < max_angle_deg) & (aspect < max_aspect)
    quads = quads[mask]
    score = score[mask]
    key = key[mask]
    lst = lst[mask]
    order = np.argsort(score)
    quads = quads[order]
    score = score[order]
    key = key[order]
    lst = lst[order]

    used = np.repeat(False, n_tri)
    # polys: list of dicts {verts: [...], score: float, quads: bool}
    polys = []

    for k, l, s, quad in zip(key, lst, score, quads):
        (t0, o0), (t1, o1) = l
        if used[[t0, t1]].any():
            continue
        used[[t0, t1]] = True
        pi = len(polys)
        polys.append({"verts": quad, "score": s})

    # ---- Pass 2: absorb leftover triangles into adjacent quads ----
    if absorb_leftovers:
        # Build vertex-index -> poly index for adjacency lookup on shared edges.
        # For a leftover triangle, look at each of its 3 edges; the neighboring
        # triangle on that edge is either (a) absorbed into a quad -> merge with it,
        # or (b) another leftover triangle -> defer (can't merge two leftovers
        # into a quad without producing a quad; we can pair them).
        leftover_polys = []  # poly index for each leftover triangle
        for ti in range(n_tri):
            if not used[ti]:
                leftover_polys.append(ti)

        # Map (tri_index) -> poly index for already-formed quads
        # (reuse tri_to_poly)
        # Process leftovers: for each, find best-scoring adjacent quad via edges.
        # Do it iteratively so that after absorbing, the poly's edge set updates.
        # We maintain poly edge -> poly index.
        def poly_edges(poly):
            v = poly["verts"]
            return [(min(v[k], v[(k+1) % len(v)]), max(v[k], v[(k+1) % len(v)]))
                    for k in range(len(v))]

        edge_to_poly = {}
        for pi, poly in enumerate(polys):
            for e in poly_edges(poly):
                edge_to_poly[e] = pi

        # Greedy: for each leftover triangle, try its 3 edges; if any edge
        # touches an existing poly, merge with the one giving the best
        # (smallest) quad score, i.e. the most "quad-like" neighbor.
        absorbed = [False] * n_tri
        for ti in leftover_polys:
            if absorbed[ti]:
                continue
            a, b, c = tris[ti]
            best = None  # (score, pi, shared_edge, tri_opposite_vertex)
            for e, opp in [((min(a, b), max(a, b)), c),
                         ((min(b, c), max(b, c)), a),
                         ((min(c, a), max(c, a)), b)]:
                pi = edge_to_poly.get(e, None)
                if pi is None:
                    continue
                # The leftover triangle shares edge e with poly pi.
                # Merge: walk poly verts, replace edge e with the detour
                # e -> opposite vertex of the triangle.
                # Opposite vertex = the tri vertex not in e.
                merged = _merge_tri_into_poly(polys[pi]["verts"], e, opp)
                if merged is None or len(merged) < 4:
                    continue
                s = _quad_score(merged)
                if best is None or s < best[0]:
                    best = (s, pi, e, opp, merged)
            if best is None:
                # No adjacent quad yet (neighbor also a leftover). Defer:
                # pair with another leftover triangle if possible later.
                continue
            s, pi, e, opp, merged = best
            # update poly
            old_edges = poly_edges(polys[pi])
            for oe in old_edges:
                if edge_to_poly.get(oe, None) == pi:
                    del edge_to_poly[oe]
            polys[pi]["verts"] = merged
            polys[pi]["score"] = s
            for ne in poly_edges(polys[pi]):
                edge_to_poly[ne] = pi
            absorbed[ti] = True

        # Any still-unabsorbed leftovers: pair them with each other into quads
        # (two leftover triangles sharing an edge -> quad), or emit as triangles.
        remaining = [ti for ti in leftover_polys if not absorbed[ti]]
        if remaining:
            # Build edge -> list of remaining tri indices
            rem_edge_map = {}
            for ti in remaining:
                a, b, c = tris[ti]
                for e, opp in (((min(a, b), max(a, b)), c),
                               ((min(b, c), max(b, c)), a),
                               ((min(c, a), max(c, a)), b)):
                    rem_edge_map.setdefault(e, []).append((ti, opp))
            rem_used = set()
            for ti in remaining:
                if ti in rem_used:
                    continue
                a, b, c = tris[ti]
                best = None
                for e, opp_ti in (((min(a, b), max(a, b)), c),
                                  ((min(b, c), max(b, c)), a),
                                  ((min(c, a), max(c, a)), b)):
                    for (tj, opp_tj) in rem_edge_map.get(e, []):
                        if tj == ti or tj in rem_used:
                            continue
                        # quad from two triangles sharing edge e
                        ea, eb = e
                        quad = [ea, opp_ti, eb, opp_tj]
                        s = _quad_score(quad)
                        if best is None or s < best[0]:
                            best = (s, tj, quad)
                if best is not None:
                    s, tj, quad = best
                    rem_used.add(ti); rem_used.add(tj)
                    polys.append({"verts": quad, "score": s})
                else:
                    rem_used.add(ti)
                    polys.append({"verts": [a, b, c], "score": 0.0})

    else:
        # original behavior: emit leftover triangles
        for ti in range(n_tri):
            if used[ti]:
                continue
            a, b, c = tris[ti]
            polys.append({"verts": [a, b, c], "score": 0.0})

    # ---- Convert to point lists ----
    out = []
    for poly in polys:
        poly_pts = pts[poly["verts"]]
        poly_pts = simplify_poly(poly_pts)
        if len(poly_pts) >= 3:
            out.append(poly_pts)
    return out


def _merge_tri_into_poly(poly_idx, shared_edge, tri_opp):
    """Merge a triangle (sharing `shared_edge`) into a polygon.

    poly_idx : list of vertex indices (counter-clockwise)
    shared_edge : (u, v) index pair present as consecutive edge in poly_idx
    tri_opp : the triangle's third vertex (not in shared_edge)

    Returns new vertex list, or None if the edge is not found as a
    consecutive pair (either orientation).
    """
    n = len(poly_idx)
    poly_idx_next = np.roll(poly_idx, -1)
    ks = np.where(((poly_idx == shared_edge[0]) & (poly_idx_next == shared_edge[1])) | ((poly_idx == shared_edge[1]) & (poly_idx_next == shared_edge[0])))[0]
    if len(ks) > 0:
        k = ks[0]
        # Insert tri_opp between u and v (or v and u)
        new = np.insert(poly_idx, k + 1, tri_opp)
        # cancel nearby reversed edge
        mask = new != np.roll(new, 2)
        mask &= np.roll(mask, -1)
        if not mask.all():
            new = new[mask]
        return new
    return None

def quad_cut_cells(w, h, pts, snap_ratio=0.6, boundary_spacing=None):
    """Like triangle_cells, but Delaunay triangles are merged into quads."""
    n_seeds = len(pts)
    aspect = w / max(h, 1e-6)
    nx = max(2, int(round(np.sqrt(n_seeds * aspect))))
    ny = max(2, int(round(n_seeds / nx)))
    if nx * ny < n_seeds:
        ny = int(np.ceil(n_seeds / nx))

    step_x = w / nx
    step_y = h / ny

    snap_dist = snap_ratio * min(step_x, step_y)
    pts = snap_points_to_rect(pts, w, h, snap_dist)

    if boundary_spacing is None:
        boundary_spacing = min(step_x, step_y) * 0.9
    bpts = _boundary_points(w, h, boundary_spacing)

    if len(bpts) > 0 and len(pts) > 0:
        min_sep = min(step_x, step_y) * 0.5
        min_sep_sq = min_sep * min_sep
        keep_mask = np.ones(len(bpts), dtype=bool)
        chunk = 2048
        for s in range(0, len(bpts), chunk):
            e = min(s + chunk, len(bpts))
            sub = bpts[s:e]
            d2 = ((sub[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
            keep_mask[s:e] = d2.min(axis=1) >= min_sep_sq
        bpts = bpts[keep_mask]

    pts = np.vstack([pts, bpts])
    pts = dedupe_points(pts, eps=1e-6)
    pts = np.clip(pts, (0.0, 0.0), (w, h), out=pts)

    tris = delaunay_triangulate(pts)

    # Clip triangles to the rect *before* merging, so edge triangles that get
    # cut still have valid geometry; merging uses index adjacency which is
    # preserved by clipping only in position, not topology.
    # (We merge on indices, then clip the resulting quads.)
    quads = merge_triangles_to_quads(pts, tris)

    cells = []
    for poly in quads:
        clipped = clip_polygon_rect(poly, w, h)
        clipped = simplify_poly(clipped)
        if len(clipped) >= 3:
            cells.append(clipped)
    return cells


# ---------- knife cut ----------
def _pair_edge_score(pts, pairs, edge_map):
    """Average edge strength sampled at both endpoints of pair (i, j)."""
    if edge_map is None:
        return 0.0
    h, w = edge_map.shape
    ptsi = pts[pairs[:,0]]
    ptsj = pts[pairs[:,1]]
    xi = np.clip(ptsi[:,0], 0, w - 1).astype(np.int32)
    yi = np.clip(ptsi[:,1], 0, h - 1).astype(np.int32)
    xj = np.clip(ptsj[:,0], 0, w - 1).astype(np.int32)
    yj = np.clip(ptsj[:,1], 0, h - 1).astype(np.int32)
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
            scores = _pair_edge_score(pts, pairs, edge_map)
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
            left, right = clip_polygon_halfplane_both(poly, a, b, c)
            left = simplify_poly(left)
            right = simplify_poly(right)
            if len(left) >= 3:
                new_cells.append(left)
            if len(right) >= 3:
                new_cells.append(right)
        cells = new_cells
    return cells

def _select_centroids_by_area_with_quota(labels, comp_pal, comp_ids, comp_info,
                                         n_seeds, bias_scores, bias_score_bonus=0.5):
    m = len(comp_pal)
    if m <= n_seeds:
        return list(range(m))

    areas = np.array([comp_info[k][1] for k in comp_ids], dtype=np.float64)
    amax = areas.max()
    a_norm = areas / amax if amax > 0 else areas

    score = a_norm.copy()

    # ---- vectorized bias score per component ----
    bias_scores /= bias_scores.max()
    flat_labels = labels.ravel()
    flat_bias = bias_scores.ravel()

    n_labels = flat_labels.max()
    sums = np.bincount(flat_labels, weights=flat_bias, minlength=n_labels)
    counts = np.bincount(flat_labels, minlength=n_labels)

    ids = np.asarray(comp_ids)
    comp_bias = np.zeros(len(ids), dtype=np.float64)
    valid = counts[ids] > 0
    comp_bias[valid] = sums[ids[valid]] / counts[ids[valid]]

    score += comp_bias * bias_score_bonus

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
                    bias_type, bridge_ratio=0.01):
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
        bias_scores = _seed_weights(img_small, rng, bias_type)

        centroids, comp_pal, comp_ids = component_representative_points_dt(
            labels, comp_info, min_area,
            bias_scores
        )
        if not centroids:
            return []

        if len(centroids) > 1:
            keep = _select_centroids_by_area_with_quota(
                labels, comp_pal, comp_ids, comp_info, int(n_seeds),
                bias_scores
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
                    bias_type, warp_polygons=False):
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

    if warp_polygons:
        centers = generate_seeds(w, h, rng.integers(1, 3, endpoint=True), img_small, rng, bias_type)
        
        lengths = [len(poly) for poly in final_polys]
        merged = np.concatenate(final_polys)
        merged = warp_points(merged, centers, w, h, rng)
        final_polys = np.split(merged, np.cumsum(lengths)[:-1])

    return final_polys


# river
def swirl_velocity(p, w, h, centers, strengths):
    """Sum of point vortices -> tangential velocity field.
    """
    v = np.zeros_like(p, dtype=np.float64)
    rwh = np.sqrt(w * h)
    for c_n, g in zip(centers, strengths):
        c = c_n * (w, h)
        d = p - c
        r = np.hypot(d[:,0], d[:,1])
        # u = d / r[:,None]
        # v[:,0] += -g * rwh * u[:,1]
        # v[:,1] += g * rwh * u[:,0]
        theta = np.zeros_like(r, dtype=np.float64)
        mask = r > 1e-6
        theta[mask] = g * rwh / r[mask]
        c, s = np.cos(theta), np.sin(theta)
        x, y = d[:, 0], d[:, 1]
        d_next = np.column_stack([x * c - y * s, x * s + y * c])
        v += d_next - d
    return v


def warp_points(P, p_centers, w, h, rng,
               n_steps=10):
    """Warp points with  a vortex velocity field.
    """
    if len(P) == 0:
        return P

    # add some low power jittering centers
    centers_low = grid_seeds(w, h, len(p_centers))
    centers = np.vstack([p_centers, centers_low]) / (w, h)
    n_centers = len(centers)
    signs = rng.choice([1.0, -1.0], size=n_centers)
    # high powers + low powers
    strengths = signs * np.append(rng.uniform(0.1, 0.25, size=n_centers // 2), rng.uniform(0.02, 0.05, size=n_centers // 2))

    # Integrate the flow field in n_steps small Euler steps
    step_strengths = strengths / n_steps
    for _ in range(n_steps):
        v = swirl_velocity(P, w, h, centers, step_strengths)
        # damping on borders
        # weight = np.min((1 - np.cos(P / (w, h) * 2 * np.pi)) / 2, axis=1)
        # np.clip(weight, 0.0, 1.0, out=weight)
        weight = np.min(1 - np.abs(P / (w, h) * 2 - 1) ** 0.5, axis=1)

        P += v * weight[:,None]
        np.clip(P, [0.0, 0.0], [w, h], out=P)
    return P



def river_cut_cells(img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                    bias_type):
    # double n_seeds then kick out poly out of range after wrap
    rect = np.asarray([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])
    seeds = generate_seeds(w, h, n_seeds, img_small, rng, bias_type=bias_type)

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

    if len(manual_seeds) > 0:
        centers = manual_seeds
    else:
        centers = generate_seeds(w, h, rng.integers(1, 3, endpoint=True), img_small, rng, bias_type)
    
    lengths = [len(poly) for poly in polygons]
    merged = np.concatenate(polygons)
    merged = warp_points(merged, centers, w, h, rng)
    polygons = np.split(merged, np.cumsum(lengths)[:-1])

    return polygons


# ---------- line sweep cut ----------

def splinters_cut_cells(img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                         bias_type, angle_jitter=0.6):
    """Sequential splinters cut.

    Strategy:
      1. Generate seeds (manual if provided, else by bias).
      2. Start with the full rectangle.
      3. Process seeds one by one; for the polygon that currently contains the
         seed, cut it with a random-angle line passing through the seed
         (a through-the-point half-plane split), keeping both halves.
         The seed's own polygon is split, so later seeds land in smaller cells.

    angle_jitter: max deviation from a "spread" orientation, in radians
    (uniform random angle is also fine; this keeps cuts less parallel).
    """
    rect = np.asarray([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])

    if manual_seeds is not None and len(manual_seeds) > 0:
        seeds = np.asarray(manual_seeds, dtype=np.float64)
    else:
        seeds = generate_seeds(w, h, n_seeds, img_small, rng, bias_type)

    if len(seeds) == 0:
        return [rect]

    # Start with a single rectangle; cells list holds current polygons.
    cells = [rect]

    def _point_in_poly(pt, poly):
        """Even-odd point-in-polygon test (closed loop)."""
        gx, gy = pt[0], pt[1]
        x0, y0 = np.min(poly, axis=0)
        x1, y1 = np.max(poly, axis=0)
        if gx < x0 or gx > x1 or gy < y0 or gy > y1:
            return False
        
        # Edge endpoints, closed loop
        ax = poly[:, 0]                     # (N,)
        ay = poly[:, 1]
        bx = np.roll(poly[:, 0], -1)        # (N,)  b = next vertex
        by = np.roll(poly[:, 1], -1)

        # Broadcast to (N, H, W)
        AY = ay[:, None, None]
        BY = by[:, None, None]
        AX = ax[:, None, None]
        BX = bx[:, None, None]

        cond = (AY > gy) != (BY > gy)                       # (N, H, W) bool
        x_int = (BX - AX) * (gy - AY) / (BY - AY + 1e-12) + AX
        cross = cond & (gx < x_int)                         # (N, H, W)

        inside = np.bitwise_xor.reduce(cross, axis=0)       # (H, W) bool
        return inside.any()

    base_angles = np.linspace(0.0, rng.choice([-np.pi, np.pi]), max(len(seeds), 1), endpoint=False) + rng.uniform(0, 2 * np.pi)

    for k, seed in enumerate(seeds):
        # Find the cell containing this seed
        target = None
        for ci, poly in enumerate(cells):
            if _point_in_poly(seed, poly):
                target = ci
                break
        if target is None:
            continue

        poly = cells[target]

        # Random line angle: golden-angle-ish base + jitter
        theta = base_angles[k % len(base_angles)] + rng.uniform(-angle_jitter, angle_jitter)
        # Direction vector
        dx, dy = math.cos(theta), math.sin(theta)

        # Line through seed: normal (a, b), c such that a*x + b*y + c = 0
        a = -dy
        b = dx
        c = -(a * seed[0] + b * seed[1])

        left, right = clip_polygon_halfplane_both(poly, a, b, c)
        left = simplify_poly(left)
        right = simplify_poly(right)
        if len(left) < 3 or len(right) < 3:
            # Degenerate cut: skip this seed
            continue

        # Replace the target cell with the two halves
        cells[target] = left
        cells.append(right)

        if len(cells) >= n_seeds * 4:
            break

    return cells


# ---------- voronoi cut ----------

def voronoi_cut_cells(img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                         bias_type):
    seeds = manual_seeds if len(manual_seeds) > 0 else generate_seeds(w, h, n_seeds, img_small, rng, bias_type=bias_type)
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
    return polygons

def crystals_cut_cells(img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                         bias_type):
    centers = manual_seeds if len(manual_seeds) > 0 else generate_seeds(w, h, max(2, n_seeds // 100), img_small, rng, bias_type=bias_type)
    rect = np.asarray([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)])

    seeds = grid_seeds(w, h, n_seeds)
    # (N, M, 2)
    diff = seeds[:, None, :] - centers[None, :, :]
    d = np.sqrt((diff ** 2).sum(axis=-1)) + np.sqrt(w * h) * 1e-2
    # d = np.maximum(d, 1e-12)

    v = (diff / (d[..., None] ** 2)).sum(axis=1) * np.sqrt(w * h / len(centers)) * 5   # (N, 2)
    seeds += v

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
    return polygons

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
        # Edge endpoints, closed loop
        ax = poly[:, 0]                     # (N,)
        ay = poly[:, 1]
        bx = np.roll(poly[:, 0], -1)        # (N,)  b = next vertex
        by = np.roll(poly[:, 1], -1)

        # Broadcast to (N, H, W)
        AY = ay[:, None, None]
        BY = by[:, None, None]
        AX = ax[:, None, None]
        BX = bx[:, None, None]

        cond = (AY > gy) != (BY > gy)                       # (N, H, W) bool
        x_int = (BX - AX) * (gy - AY) / (BY - AY + 1e-12) + AX
        cross = cond & (gx < x_int)                         # (N, H, W)

        inside = np.bitwise_xor.reduce(cross, axis=0)       # (H, W) bool
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
            choices=["voronoi", "triangles", "quads", "knife_cut", "metro", "flowers", "flowers_soft", "river", "flowers_in_river", "splinters", "crystals"]
        )
        pars.add_argument("--n_colors", type=int, default=12)
        pars.add_argument("--n_seeds", type=int, default=600)
        pars.add_argument("--stroke_width", type=float, default=2.0)
        pars.add_argument("--bias_type", type=str, default="none",
            choices=["none", "edges", "brightness", "saliency"],
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
        elif cell_shape == "quads":
            seeds = manual_seeds if len(manual_seeds) > 0 else generate_seeds(w, h, n_seeds, img_small, rng, bias_type=self.options.bias_type)
            polygons = quad_cut_cells(w, h, seeds)
        elif cell_shape == "knife_cut":
            polygons = knife_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )
        elif cell_shape == "metro":
            polygons = metro_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )
        elif cell_shape == "flowers":
            polygons = flowers_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )
        elif cell_shape == "flowers_soft":
            polygons = flowers_soft_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )
        elif cell_shape == "river":
            polygons = river_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )
        elif cell_shape == "flowers_in_river":
            polygons = flowers_soft_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type, warp_polygons=True
            )
        elif cell_shape == "splinters":
            polygons = splinters_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )
        elif cell_shape == "crystals":
            polygons = crystals_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )
        else:
            polygons = voronoi_cut_cells(
                img_small, pal_indices, n_palette, palette, w, h, n_seeds, manual_seeds, rng,
                bias_type=self.options.bias_type
            )

        if len(polygons) == 0:
            inkex.errormsg("No cells produced.")
            return

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
