#!/usr/bin/env python3
import inkex
import numpy as np
import io
import base64
import os
from PIL import Image, ImageFilter

from inkex.units import convert_unit

from image_helper import pil_quantize_indices, same_color_components, component_representative_points_dt


# ---------- Geometry helpers ----------

def clip_polygon_halfplane(poly, a, b, c):
    """Clip a convex polygon against a*x + b*y + c <= 0 (Sutherland-Hodgman)."""
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    if n == 0:
        return np.empty((0, 2), dtype=float)

    ab = np.array([a, b], dtype=float)
    dist = poly @ ab + c
    inside = dist <= 1e-9
    # skip fully inside or outside
    inside_size = inside.sum()
    if inside_size == n:
        return poly
    elif inside_size == 0:
        return np.empty((0, 2), dtype=float)
    prev_inside = np.roll(inside, 1)

    def intersect(p1, p2):
        d1 = p1 @ ab + c
        d2 = p2 @ ab + c
        denom = d1 - d2
        if abs(denom) < 1e-12:
            return p1
        t = d1 / denom
        return p1 + t * (p2 - p1)

    out = []
    prev = poly[-1]
    for i in range(n):
        cur = poly[i]
        if inside[i]:
            if not prev_inside[i]:
                out.append(intersect(prev, cur))
            out.append(cur)
        elif prev_inside[i]:
            out.append(intersect(prev, cur))
        prev = cur
    return np.asarray(out, dtype=float) if out else np.empty((0, 2), dtype=float)


def clip_polygon_rect(poly, w, h):
    """Clip a convex polygon against the rectangle [0,w]x[0,h]."""
    poly = clip_polygon_halfplane(poly, -1.0, 0.0, 0.0)   # x >= 0
    if len(poly) < 3:
        return []
    poly = clip_polygon_halfplane(poly, 1.0, 0.0, -w)       # x <= w
    if len(poly) < 3:
        return []
    poly = clip_polygon_halfplane(poly, 0.0, -1.0, 0.0)    # y >= 0
    if len(poly) < 3:
        return []
    poly = clip_polygon_halfplane(poly, 0.0, 1.0, -h)       # y <= h
    return poly


def voronoi_cell(i, seeds, rect_w, rect_h, neighbors_idx=None):
    """Compute Voronoi cell of seed i, optionally only clipping against neighbors."""
    poly = [(0.0, 0.0), (rect_w, 0.0), (rect_w, rect_h), (0.0, rect_h)]
    xi, yi = seeds[i]

    if neighbors_idx is not None:
        iter_j = neighbors_idx[i]
    else:
        iter_j = range(len(seeds))

    for j in iter_j:
        if j == i:
            continue
        xj, yj = seeds[j]
        a = 2.0 * (xj - xi)
        b = 2.0 * (yj - yi)
        c = (xi * xi + yi * yi) - (xj * xj + yj * yj)
        poly = clip_polygon_halfplane(poly, a, b, c)
        if len(poly) < 3:
            return None
    return poly


def simplify_poly(poly, eps=0.05):
    """Remove near-duplicate consecutive vertices from a polygon."""
    if len(poly) < 3:
        return poly
    poly_diff = np.maximum(
        np.abs(np.roll(poly, -1) - poly).sum(axis=1),
        np.abs(np.roll(poly, 1) - poly).sum(axis=1)
    )
    idx = np.flatnonzero(poly_diff > eps)
    return poly[idx]


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


def _snap_points_to_rect(pts, w, h, snap_dist):
    """Snap points closer than snap_dist to a rectangle edge onto that edge.

    Returns a new snapped array (does not modify the original).
    """
    out = pts.copy()
    if len(out) == 0:
        return out

    # Distances to the four edges
    d_left = out[:, 0]
    d_right = w - out[:, 0]
    d_bottom = out[:, 1]
    d_top = h - out[:, 1]

    # Nearest edge for each point
    dists = np.stack([d_left, d_right, d_bottom, d_top], axis=1)
    nearest = np.argmin(dists, axis=1)
    min_d = dists[np.arange(len(out)), nearest]

    snap_mask = min_d < snap_dist
    # 0=left, 1=right, 2=bottom, 3=top
    out[snap_mask & (nearest == 0), 0] = 0.0
    out[snap_mask & (nearest == 1), 0] = w
    out[snap_mask & (nearest == 2), 1] = 0.0
    out[snap_mask & (nearest == 3), 1] = h

    # Clip to rectangle after snapping to guard against floating-point overshoot
    out[:, 0] = np.clip(out[:, 0], 0.0, w)
    out[:, 1] = np.clip(out[:, 1], 0.0, h)
    return out


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


def _dedupe_points(pts, eps=1e-6):
    """Remove near-duplicate points (quantize to an eps grid, then check one by one)."""
    if len(pts) == 0:
        return pts
    # Quantize to an eps grid, deduplicate via a dict, keep first occurrence
    keys = np.round(pts / eps).astype(np.int64)
    seen = {}
    keep = []
    for i, k in enumerate(map(tuple, keys)):
        if k not in seen:
            seen[k] = i
            keep.append(i)
    return pts[np.array(keep, dtype=np.int64)]


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
    pts = _snap_points_to_rect(pts, w, h, snap_dist)

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
    pts = _dedupe_points(pts, eps=1e-6)

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


def edge_weighted_seeds(w, h, n_seeds, gray, rng=None):
    """Sample seeds with probability proportional to edge strength."""
    if rng is None:
        rng = np.random.default_rng()
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edge_map = gx + gy
    prob = edge_map.ravel() + edge_map.mean() * 0.5
    prob /= prob.sum()
    idx = rng.choice(w * h, n_seeds, replace=False, p=prob)
    return np.stack([idx % w, idx // w], axis=1).astype(np.float64)


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


# ---------- Rasterization of all Voronoi regions at once ----------

def voronoi_region_labels(seeds, w, h):
    """Return an (h, w) int32 array of nearest-seed index for every pixel."""
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xs.ravel() + 0.5, ys.ravel() + 0.5])
    seeds_arr = np.asarray(seeds, dtype=np.float64)
    out = np.empty(len(pts), dtype=np.int32)
    chunk = 65536
    for s in range(0, len(pts), chunk):
        e = min(s + chunk, len(pts))
        sub = pts[s:e]
        d = ((sub[:, None, :] - seeds_arr[None, :, :]) ** 2).sum(axis=2)
        out[s:e] = np.argmin(d, axis=1)
    return out.reshape(h, w)


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

def centroid_cut_sparse(centroids, w, h, rng, edge_map=None):
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
    pairs = set()
    for i in range(n):
        j = int(nn[i])
        pairs.add((min(i, j), max(i, j)))
    pairs = np.array(list(pairs))

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
    for (i, j) in pairs:
        xi, yi = pts[i]
        xj, yj = pts[j]
        a = 2.0 * (xj - xi)
        b = 2.0 * (yj - yi)
        c = (xi * xi + yi * yi) - (xj * xj + yj * yj)
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

def knife_cut_cells(img_small, pal_indices, n_palette, w, h, n_seeds, rng,
                    seed_from_edges=False, bridge_ratio=0.01,
                    edge_weight=0.5, edge_bonus=0.5):
    # Step 1: same-color connected components
    bridge_r = int(round(max(w, h) * bridge_ratio))
    min_area = w * h / (n_seeds * n_palette)
    labels, comp_info = same_color_components(
        pal_indices, bridge_radius=bridge_r
    )

    # Optional edge map (shared by representative placement, selection, cuts)
    edge_map = None
    edge_energy = None
    if seed_from_edges:
        gray = np.array(img_small.convert("L"), dtype=np.float64)
        edge_map = _edge_map(gray)
        edge_energy = _component_edge_energy(comp_info, labels, edge_map)

    # Step 2/3: use DT to find a centered representative point for each component
    centroids, comp_pal, comp_ids = component_representative_points_dt(
        labels, comp_info, min_area,
        edge_map=edge_map,
        edge_weight=edge_weight if seed_from_edges else 0.0,
    )
    if not centroids:
        return []

    # Step 4: if there are too many components, keep the largest by area,
    #         guaranteeing at least one component per palette color.
    if len(centroids) > n_seeds:
        keep = _select_centroids_by_area_with_quota(
            comp_pal, comp_ids, comp_info, int(n_seeds),
            edge_energy=edge_energy,
            edge_bonus=edge_bonus if seed_from_edges else 0.0,
        )
        centroids = [centroids[i] for i in keep]

    # Step 5: Voronoi cut (edge-biased pair selection when edge_map is present)
    final_polys = centroid_cut_sparse(
        centroids, float(w), float(h), rng,
        edge_map=edge_map if seed_from_edges else None,
    )
    if not final_polys:
        return []
    return final_polys


# ---------- Triangle-based per-cell sampling ----------

def dominant_palette_index_for_polygons(polygons, pal_indices, n_palette, w, h):
    """For each polygon, the dominant palette index of the pixels it covers.

    We rasterize each polygon separately using a simple even-odd point-in-polygon
    test on a per-polygon bounding box. For a few hundred cells this is fine.
    """
    h_img, w_img = pal_indices.shape
    flat_pal = pal_indices.ravel()
    results = []
    for poly in polygons:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        x0 = max(0, int(np.floor(min(xs))))
        x1 = min(w_img, int(np.ceil(max(xs))) + 1)
        y0 = max(0, int(np.floor(min(ys))))
        y1 = min(h_img, int(np.ceil(max(ys))) + 1)
        if x1 <= x0 or y1 <= y0:
            results.append(0)
            continue
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
            results.append(0)
            continue
        idx = np.where(inside.ravel())[0]
        # Map flat local index back to global pixel index
        local_y, local_x = np.divmod(idx, (x1 - x0))
        glob_y = local_y + y0
        glob_x = local_x + x0
        flat = glob_y * w_img + glob_x
        vals = flat_pal[flat]
        counts = np.bincount(vals, minlength=n_palette)
        results.append(int(np.argmax(counts)))
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
            choices=["voronoi", "triangles", "knife_cut"],
            help="Cell shape: voronoi (default) or triangles (Delaunay-based) or knife_cut",
        )
        pars.add_argument("--n_colors", type=int, default=12)
        pars.add_argument("--n_seeds", type=int, default=600)
        pars.add_argument("--stroke_width", type=float, default=2.0)
        pars.add_argument("--seed_from_edges", type=inkex.Boolean, default=False)
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
        selected = [
            el for el in self.svg.selection.values()
            if isinstance(el, inkex.Image)
        ]
        if not selected:
            inkex.errormsg("Please select a bitmap image first.")
            return

        image = selected[0]
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
            if self.options.seed_from_edges:
                gray = np.array(img_small.convert("L"), dtype=np.float64)
                seeds = edge_weighted_seeds(w, h, n_seeds, gray, rng=rng)
            else:
                seeds = grid_seeds(w, h, n_seeds, rng=rng)
            polygons = triangle_cells(w, h, seeds)
            if not polygons:
                inkex.errormsg("Triangulation produced no cells.")
                return
            dom_idx = dominant_palette_index_for_polygons(
                polygons, pal_indices, n_palette, w, h
            )
        elif cell_shape == "knife_cut":
            polygons = knife_cut_cells(
                img_small, pal_indices, n_palette, w, h, n_seeds, rng,
                seed_from_edges=self.options.seed_from_edges,
            )
            if not polygons:
                inkex.errormsg("Knife-cut produced no cells.")
                return

            dom_idx = dominant_palette_index_for_polygons(
                polygons, pal_indices, n_palette, w, h
            )
        else:
            # Voronoi (original path)
            if self.options.seed_from_edges:
                gray = np.array(img_small.convert("L"), dtype=np.float64)
                seeds = edge_weighted_seeds(w, h, n_seeds, gray, rng=rng)
            else:
                seeds = grid_seeds(w, h, n_seeds, rng=rng)

            region_labels = voronoi_region_labels(seeds, w, h)
            n_cells = len(seeds)
            dom_idx = dominant_palette_index(region_labels, pal_indices, n_cells, n_palette)

            for i in range(n_cells):
                poly = voronoi_cell(i, seeds, float(w), float(h), None)
                if poly is None or len(poly) < 3:
                    continue
                poly = simplify_poly(poly)
                if len(poly) < 3:
                    continue
                xs = [p[0] for p in poly]
                ys = [p[1] for p in poly]
                if (max(xs) - min(xs)) * (max(ys) - min(ys)) < 4.0:
                    continue
                polygons.append(poly)

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
