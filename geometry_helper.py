import numpy as np

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


def solve_linear_abc(pts1, pts2):
    return np.column_stack([
        2.0 * (pts1 - pts2),
        (pts2 * pts2 - pts1 * pts1).sum(axis=1)
    ])


def voronoi_cell(i, seeds, poly, neighbors_idx=None):
    """Compute Voronoi cell of seed i, optionally only clipping against neighbors."""
    seedi = seeds[i]

    if neighbors_idx is not None:
        seeds = seeds[neighbors_idx]
    else:
        mask = np.ones(len(seeds), dtype=bool)
        mask[i] = False
        seeds = seeds[mask]

    abc = solve_linear_abc(seeds, seedi)

    for (a, b, c) in abc:
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


def snap_points_to_rect(pts, w, h, snap_dist):
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


def dedupe_points(pts, eps=1e-6):
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


def ray_polygon_intersections(origin, directions, poly, eps=1e-12):
    """
    Find the nearest forward intersection of each ray with a polygon.

    Parameters
    ----------
    origin : (2,) array_like
        Ray origin (the shared center point).
    directions : (R, 2) array_like
        Unit direction vectors, one per ray.
    poly : (N, 2) array_like
        Polygon vertices in boundary order (do NOT repeat the first vertex
        at the end; the closing edge is added automatically via np.roll).
    eps : float, optional
        Tolerance for treating a ray as parallel to an edge
        (``|cross(D, E)| < eps``) and for the parametric bounds.

    Returns
    -------
    t : (R,) ndarray
        Distance parameter of the nearest forward hit for each ray.
        ``np.inf`` if the ray does not intersect the polygon.
    hit : (R, 2) ndarray
        Intersection coordinates ``origin + t * direction``.
        ``np.nan`` for rays with no hit.

    Notes
    -----
    For each edge ``P -> Q`` the ray ``O + t*D`` is solved against
    ``P + s*E`` (with ``E = Q - P``) using cross products:

        denom = cross(D, E)
        t = cross(P - O, E) / denom
        s = cross(P - O, D) / denom

    A hit is valid when the ray is not parallel to the edge
    (``|denom| > eps``), lies in the forward direction (``t >= 0``),
    and falls on the segment (``0 <= s <= 1``). When several edges are
    hit (e.g. a concave polygon), the smallest ``t`` is kept so the
    reported point is the *visible* boundary along the ray.
    """
    O = np.asarray(origin, dtype=np.float64)
    D = np.asarray(directions, dtype=np.float64)      # (R, 2)
    P = np.asarray(poly, dtype=np.float64)            # (N, 2)
    Q = np.roll(P, -1, axis=0)                        # (N, 2) edge end points
    E = Q - P                                         # (N, 2) edge vectors

    R = len(D)
    N = len(P)

    # Broadcast to (R, N)
    PO = P[None, :, :] - O[None, None, :]             # (1, N, 2)
    Px = PO[:, :, 0]                                  # (R, N) via broadcast
    Py = PO[:, :, 1]
    Dx = D[:, 0][:, None]                             # (R, 1)
    Dy = D[:, 1][:, None]
    Ex = E[None, :, 0]                                # (1, N)
    Ey = E[None, :, 1]

    denom = Dx * Ey - Dy * Ex                         # (R, N) cross(D, E)
    safe = np.abs(denom) > eps
    inv = np.where(safe, 1.0 / np.where(safe, denom, 1.0), 0.0)

    t = (Px * Ey - Py * Ex) * inv                     # (R, N)
    s = (Px * Dy - Py * Dx) * inv                     # (R, N)

    valid = safe & (t >= -1e-9) & (s >= -1e-9) & (s <= 1.0 + 1e-9)

    t_masked = np.where(valid, t, np.inf)             # (R, N)
    best = np.argmin(t_masked, axis=1)                # (R,)
    t_best = t_masked[np.arange(R), best]             # (R,)

    has = np.isfinite(t_best)
    hit = np.full((R, 2), np.nan)
    hit[has] = O + D[has] * t_best[has, None]
    return t_best, hit


def polygon_area(poly):
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_centroid(poly):
    """Area centroid of a polygon. Returns (2,)."""
    p = np.asarray(poly, dtype=np.float64)
    if len(p) < 3:
        return p.mean(axis=0) if len(p) else np.zeros(2)
    x, y = p[:, 0], p[:, 1]
    xn = np.roll(x, -1)
    yn = np.roll(y, -1)
    cross = x * yn - xn * y                # (N,) signed cross terms
    A = 0.5 * cross.sum()
    if abs(A) < 1e-12:                     # degenerate -> vertex average
        return p.mean(axis=0)
    Cx = ((x + xn) * cross).sum() / (6.0 * A)
    Cy = ((y + yn) * cross).sum() / (6.0 * A)
    return np.array([Cx, Cy])