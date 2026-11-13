import numpy as np
from PIL import Image

# ---------- sRGB <-> Lab (D65) pure NumPy implementation ----------

def _srgb_to_linear(c):
    """c: float array in [0,1], convert sRGB nonlinear values to linear values."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

def _linear_to_srgb(c):
    """Convert linear values to sRGB nonlinear values."""
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * (c ** (1 / 2.4)) - 0.055)

def _rgb_to_lab(rgb):
    """
    rgb: (..., 3) uint8/float, values in 0..255
    Returns: (..., 3) float32 Lab (L in [0,100], a/b roughly [-128,127])
    """
    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    lin = _srgb_to_linear(rgb)

    # sRGB -> XYZ (D65)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], dtype=np.float32)
    xyz = lin @ M.T

    # D65 white point
    white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xyz = xyz / white

    # XYZ -> Lab
    eps = 216 / 24389
    kappa = 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1).astype(np.float32)

def _lab_to_rgb(lab):
    """
    lab: (..., 3) float32
    Returns: (..., 3) uint8 RGB
    """
    lab = np.asarray(lab, dtype=np.float32)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    fy = (L + 16) / 116
    fx = fy + a / 500
    fz = fy - b / 200

    eps = 216 / 24389
    kappa = 24389 / 27
    xr = np.where(fx ** 3 > eps, fx ** 3, (116 * fx - 16) / kappa)
    yr = np.where(L > kappa * eps, ((L + 16) / 116) ** 3, L / kappa)
    zr = np.where(fz ** 3 > eps, fz ** 3, (116 * fz - 16) / kappa)

    white = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
    xyz = np.stack([xr, yr, zr], axis=-1) * white

    # XYZ -> linear RGB
    M_inv = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], dtype=np.float32)
    lin = xyz @ M_inv.T
    lin = np.clip(lin, 0.0, 1.0)
    srgb = _linear_to_srgb(lin)
    return np.clip(np.round(srgb * 255), 0, 255).astype(np.uint8)

# ---------- K-Means (space-agnostic, input is already Lab) ----------

def _kmeans(points, init_centers, max_iter=20, tol=1e-3):
    centers = init_centers.astype(np.float32).copy()
    K = centers.shape[0]
    for _ in range(max_iter):
        dist = (
            np.sum(points ** 2, axis=1, keepdims=True)
            + np.sum(centers ** 2, axis=1, keepdims=True).T
            - 2 * points @ centers.T
        )
        labels = np.argmin(dist, axis=1)
        new_centers = centers.copy()
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                new_centers[k] = points[mask].mean(axis=0)
        shift = np.linalg.norm(new_centers - centers)
        centers = new_centers
        if shift < tol:
            break
    return centers

# ---------- Replacement quantize ----------

def pil_quantize_indices(img_small, k, max_iter=20, sample_size=50000):
    """
    MEDIANCUT initialization + Lab-space K-Means palette refinement.
    Returns (indices_hw, palette_Kx3), matching the original LIBIMAGEQUANT signature.
    """
    K = max(2, int(k))

    # 1. Use MEDIANCUT to get the initial palette (as initial Lab-space centers)
    q0 = img_small.quantize(
        colors=K,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.NONE,
    )
    colors0 = q0.getcolors(maxcolors=256) or []
    n_actual = (max(i for _, i in colors0) + 1) if colors0 else 1
    pal_flat0 = q0.getpalette() or []
    init_rgb = np.array(pal_flat0[: n_actual * 3], dtype=np.uint8).reshape(-1, 3)

    # 2. Extract pixels and convert to Lab (subsample for speed)
    arr = np.asarray(img_small.convert("RGB"), dtype=np.uint8).reshape(-1, 3)
    if arr.shape[0] > sample_size:
        rng = np.random.default_rng(0)
        sel = rng.choice(arr.shape[0], sample_size, replace=False)
        arr = arr[sel]

    pixels_lab = _rgb_to_lab(arr)          # (N, 3)
    init_lab = _rgb_to_lab(init_rgb)       # (K0, 3)

    # 3. Cluster in Lab space
    centers_lab = _kmeans(pixels_lab, init_lab, max_iter=max_iter)

    # 4. Convert back to RGB to build the palette
    palette = _lab_to_rgb(centers_lab)     # (K0, 3) uint8
    K_final = palette.shape[0]

    # 5. Remap the original image using the refined palette
    pal_img = Image.new("P", (1, 1))
    flat = palette.flatten().tolist()
    flat += [0] * (768 - len(flat))
    pal_img.putpalette(flat)

    q = img_small.convert("RGB").quantize(
        palette=pal_img, dither=Image.Dither.NONE
    )
    indices = np.array(q, dtype=np.int32)

    if indices.size and indices.max() >= K_final:
        indices = np.clip(indices, 0, K_final - 1)

    return indices, palette


def same_color_components(pal_indices, bridge_radius=0):
    """Label connected components of equal palette index, numpy-only.

    bridge_radius : int, in pixels (>=0). Pixels of the same palette index
        whose Chebyshev distance is <= bridge_radius are considered connected,
        i.e. gaps smaller than bridge_radius between same-color blobs are
        bridged. bridge_radius=0 reproduces strict 8-connectivity.

    Returns
    -------
    labels : (h, w) int32
    comp_info : list of (pal_idx, area)
    """
    h, w = pal_indices.shape
    flat = pal_indices.ravel().astype(np.int64)
    n = flat.size
    r = int(max(0, bridge_radius))

    edge_list = []

    # ---------- 1. Strict 8-neighborhood edges (always) ----------
    rows = np.arange(h, dtype=np.int64)
    cols = np.arange(w, dtype=np.int64)

    left = (rows[:, None] * w + cols[None, :w - 1]).ravel()
    same_h = flat[left] == flat[left + 1]
    edge_list.append(np.stack([left[same_h], left[same_h] + 1], axis=1))

    top = (rows[:h - 1, None] * w + cols[None, :]).ravel()
    same_v = flat[top] == flat[top + w]
    edge_list.append(np.stack([top[same_v], top[same_v] + w], axis=1))

    dr = (rows[:h - 1, None] * w + cols[None, :w - 1]).ravel()
    same_dr = flat[dr] == flat[dr + w + 1]
    edge_list.append(np.stack([dr[same_dr], dr[same_dr] + w + 1], axis=1))

    dl = (rows[:h - 1, None] * w + cols[None, 1:]).ravel()
    same_dl = flat[dl] == flat[dl + w - 1]
    edge_list.append(np.stack([dl[same_dl], dl[same_dl] + w - 1], axis=1))

    # ---------- 2. Bridge edges for gaps up to r pixels ----------
    if r > 1:
        flat2d = flat.reshape(h, w)
        bridge_chunks = []
        for dy in range(0, r + 1):
            for dx in range(-r, r + 1):
                if dy == 0 and dx == 0:
                    continue
                # shell of the Chebyshev ball of radius r
                if max(abs(dy), abs(dx)) != r:
                    continue
                if dy == 0 and dx < 0:
                    continue

                y0, y1 = max(0, -dy), min(h, h - dy)
                x0, x1 = max(0, -dx), min(w, w - dx)
                if y1 <= y0 or x1 <= x0:
                    continue

                a = flat2d[y0:y1, x0:x1]
                b = flat2d[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
                same = (a == b)
                if not same.any():
                    continue

                yy, xx = np.nonzero(same)
                ys = yy + y0
                xs = xx + x0
                src = ys.astype(np.int64) * w + xs
                dst = src + dy * w + dx
                bridge_chunks.append(np.stack([src, dst], axis=1))

        if bridge_chunks:
            edge_list.append(np.concatenate(bridge_chunks, axis=0))

    edges = (np.concatenate(edge_list, axis=0)
             if edge_list else np.empty((0, 2), np.int64))

    # ---------- 3. Batch union-find ----------
    parent = np.arange(n, dtype=np.int64)

    def find_batch(a):
        a = a.copy()
        while True:
            pa = parent[a]
            g = parent[pa]
            mask = pa != a
            if not mask.any():
                break
            a = np.where(mask, g, a)
        return a

    if len(edges) > 0:
        a = edges[:, 0]
        b = edges[:, 1]
        for _ in range(64):
            ra = find_batch(a)
            rb = find_batch(b)
            # (D) 只对当前尚未连通的边做 union,减少无效 minimum.at 写入
            mask = ra != rb
            if not mask.any():
                break
            lo = np.minimum(ra[mask], rb[mask])
            hi = np.maximum(ra[mask], rb[mask])
            np.minimum.at(parent, hi, lo)
            parent[a] = parent[parent[a]]
            parent[b] = parent[parent[b]]

    for _ in range(64):
        p = parent.copy()
        parent = parent[parent]
        if np.array_equal(parent, p):
            break

    # ---------- 4. Remap + stats ----------
    roots, labels_flat, area = np.unique(parent, return_inverse=True, return_counts=True)
    labels = labels_flat.reshape(h, w).astype(np.int32)
    comp_pal = flat[roots]

    comp_info = np.column_stack([comp_pal, area])
    return labels, comp_info

def _dt_1d_sq(f):
    """1D exact squared distance transform (Felzenszwalb & Huttenlocher).

    Input f: float64 array of length n, f[i] is the initial cost
        (foreground pixels f[i]=0, background pixels f[i]=+inf)
    Output: d[i] = min_j ( (i-j)^2 + f[j] )
    """
    n = len(f)
    d = np.empty(n, dtype=np.float64)
    v = np.empty(n, dtype=np.int64)     # parabola locations
    z = np.empty(n + 1, dtype=np.float64)  # parabola interval boundaries

    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf

    for q in range(1, n):
        # intersection of current parabola q with top-of-stack parabola v[k]
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = np.inf

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) * (q - v[k]) + f[v[k]]
    return d


def _dt_2d_sq(mask):
    """2D exact squared Euclidean distance transform.

    mask : (h, w) bool, True means "inside the region".
    Returns : (h, w) float64, squared distance from each region pixel to the
              nearest background pixel; background pixels are 0.
    """
    h, w = mask.shape
    INF = np.float64(1e20)

    # Column direction: first do 1D DT along each column
    # f[i] = 0 if pixel is inside the region, else +inf
    f = np.where(mask, 0.0, INF)          # (h, w)
    # Call _dt_1d_sq column by column (loop w times, h elements each, acceptable)
    col_dt = np.empty_like(f)
    for x in range(w):
        col_dt[:, x] = _dt_1d_sq(f[:, x])

    # Row direction: do 1D DT along each row (using the previous result as f)
    out = np.empty_like(col_dt)
    for y in range(h):
        out[y, :] = _dt_1d_sq(col_dt[y, :])
    return out


def component_representative_points_dt(labels, comp_info, min_area,
                                        edge_map=None, edge_weight=0.0):
    """Approach B: use distance transform to pick, for each component, the pixel
    farthest from the boundary as the representative point, ensuring it lies
    inside the region and is visually centered.

    If edge_map is provided and edge_weight > 0, the representative pixel is
    chosen to balance 'far from boundary' (DT) with 'close to a strong image
    edge' (edge_map). A small weight keeps the point inside the region while
    nudging it toward visual boundaries.

    Returns (centroids, comp_pal, comp_ids):
        centroids : list[(x, y)]  pixel center coordinates
        comp_pal  : list[int]     dominant color index of the component
        comp_ids  : list[int]     index of the component in comp_info
    """
    centroids = []
    comp_pal = []
    comp_ids = []

    # To avoid doing a full-image DT for each component (expensive), we only
    # do a local DT within each component's bounding box. When the box is much
    # smaller than the full image this gives a significant speedup.
    for k, (color, area) in enumerate(comp_info):
        if area < min_area:
            continue

        # --- pixel mask of this component + bounding box ---
        ys, xs = np.nonzero(labels == k)
        if len(ys) == 0:
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1

        sub_mask = (labels[y0:y1, x0:x1] == k)  # (h', w') bool
        dist_sq = _dt_2d_sq(sub_mask)

        if edge_map is not None and edge_weight > 0.0:
            sub_edge = edge_map[y0:y1, x0:x1]
            dmax = dist_sq[sub_mask].max()
            dt_norm = dist_sq / dmax if dmax > 0 else dist_sq
            # Higher score = better: far from boundary AND strong edge.
            score = dt_norm + edge_weight * sub_edge
            # Only consider pixels actually inside the component.
            score = np.where(sub_mask, score, -np.inf)
        else:
            score = dist_sq

        # Pick the pixel with maximum score inside the region
        # (argmax returns a flattened index)
        flat_idx = int(np.argmax(score))
        ry, rx = np.divmod(flat_idx, sub_mask.shape[1])

        # Pixel center coordinates (+0.5)
        cx = float(x0 + rx + 0.5)
        cy = float(y0 + ry + 0.5)

        centroids.append((cx, cy))
        comp_pal.append(color)
        comp_ids.append(k)

    return centroids, comp_pal, comp_ids