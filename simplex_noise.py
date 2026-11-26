import random
import math
import numpy as np


# ============ Vectorized Lightweight Simplex Noise (2D) ============
class SimplexNoise:
    def __init__(self, seed=0):
        rng = random.Random(seed)
        perm_list = list(range(256))
        rng.shuffle(perm_list)
        self.perm = np.asarray(perm_list * 2, dtype=np.int64)

        self.grad3 = np.array([
            (1, 1), (-1, 1), (1, -1), (-1, -1),
            (1, 0), (-1, 0), (0, 1), (0, -1),
        ], dtype=np.float64)

        self.F2 = 0.5 * (math.sqrt(3.0) - 1.0)
        self.G2 = (3.0 - math.sqrt(3.0)) / 6.0

    def noise_batch(self, xin, yin):
        xin = np.asarray(xin, dtype=np.float64)
        yin = np.asarray(yin, dtype=np.float64)
        perm = self.perm
        grad3 = self.grad3
        F2, G2 = self.F2, self.G2

        s = (xin + yin) * F2
        i = np.floor(xin + s).astype(np.int64)
        j = np.floor(yin + s).astype(np.int64)
        t = (i + j) * G2
        x0 = xin - (i - t)
        y0 = yin - (j - t)

        cond = x0 > y0
        i1 = np.where(cond, 1, 0).astype(np.int64)
        j1 = np.where(cond, 0, 1).astype(np.int64)

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2

        ii = i & 255
        jj = j & 255

        gi0 = perm[ii + perm[jj]] % 8
        gi1 = perm[ii + i1 + perm[jj + j1]] % 8
        gi2 = perm[ii + 1 + perm[jj + 1]] % 8

        g0x = grad3[gi0, 0]; g0y = grad3[gi0, 1]
        g1x = grad3[gi1, 0]; g1y = grad3[gi1, 1]
        g2x = grad3[gi2, 0]; g2y = grad3[gi2, 1]

        t0 = 0.5 - x0 * x0 - y0 * y0
        m0 = t0 > 0
        t0m = np.where(m0, t0 * t0, 0.0)
        n0 = np.where(m0, t0m * t0m * (g0x * x0 + g0y * y0), 0.0)

        t1 = 0.5 - x1 * x1 - y1 * y1
        m1 = t1 > 0
        t1m = np.where(m1, t1 * t1, 0.0)
        n1 = np.where(m1, t1m * t1m * (g1x * x1 + g1y * y1), 0.0)

        t2 = 0.5 - x2 * x2 - y2 * y2
        m2 = t2 > 0
        t2m = np.where(m2, t2 * t2, 0.0)
        n2 = np.where(m2, t2m * t2m * (g2x * x2 + g2y * y2), 0.0)

        return 70.0 * (n0 + n1 + n2)

    def fbm_batch(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0
        max_amp = 0.0
        for _ in range(max(1, int(octaves))):
            total += amplitude * self.noise_batch(x * frequency, y * frequency)
            max_amp += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return total / max_amp if max_amp > 0 else np.zeros_like(x)

    def turbulence_batch(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        total = np.zeros_like(x)
        amplitude = 1.0
        frequency = 1.0
        max_amp = 0.0
        for _ in range(max(1, int(octaves))):
            total += amplitude * np.abs(self.noise_batch(x * frequency, y * frequency))
            max_amp += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return (2.0 * total / max_amp - 1.0) if max_amp > 0 else np.zeros_like(x)

    def noise_gradient_batch(self, xin, yin):
        """Analytic gradient of 2D simplex noise. Returns (gx, gy)."""
        xin = np.asarray(xin, dtype=np.float64)
        yin = np.asarray(yin, dtype=np.float64)
        perm = self.perm
        grad3 = self.grad3
        F2, G2 = self.F2, self.G2

        s = (xin + yin) * F2
        i = np.floor(xin + s).astype(np.int64)
        j = np.floor(yin + s).astype(np.int64)
        t = (i + j) * G2
        x0 = xin - (i - t)
        y0 = yin - (j - t)

        cond = x0 > y0
        i1 = np.where(cond, 1, 0).astype(np.int64)
        j1 = np.where(cond, 0, 1).astype(np.int64)

        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2

        ii = i & 255
        jj = j & 255

        gi0 = perm[ii + perm[jj]] % 8
        gi1 = perm[ii + i1 + perm[jj + j1]] % 8
        gi2 = perm[ii + 1 + perm[jj + 1]] % 8

        g0x = grad3[gi0, 0]; g0y = grad3[gi0, 1]
        g1x = grad3[gi1, 0]; g1y = grad3[gi1, 1]
        g2x = grad3[gi2, 0]; g2y = grad3[gi2, 1]

        def corner(xk, yk, gx, gy):
            tk = 0.5 - xk * xk - yk * yk
            mk = tk > 0
            tkm = np.where(mk, tk, 0.0)          # t (clamped)
            sk = gx * xk + gy * yk
            t3 = tkm ** 3
            t4 = t3 * tkm
            # n = t^4 * s ; grad = t^3 * (-8 s d + t g)
            nx = np.where(mk, t3 * (-8.0 * sk * xk + tkm * gx), 0.0)
            ny = np.where(mk, t3 * (-8.0 * sk * yk + tkm * gy), 0.0)
            return nx, ny, t4 * np.where(mk, sk, 0.0)

        n0x, n0y, _ = corner(x0, y0, g0x, g0y)
        n1x, n1y, _ = corner(x1, y1, g1x, g1y)
        n2x, n2y, _ = corner(x2, y2, g2x, g2y)

        return 70.0 * (n0x + n1x + n2x), 70.0 * (n0y + n1y + n2y)

    def fbm_gradient_batch(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        gx = np.zeros_like(x); gy = np.zeros_like(y)
        amplitude = 1.0; frequency = 1.0; max_amp = 0.0
        for _ in range(max(1, int(octaves))):
            cx, cy = self.noise_gradient_batch(x * frequency, y * frequency)
            gx += amplitude * frequency * cx
            gy += amplitude * frequency * cy
            max_amp += amplitude
            amplitude *= gain
            frequency *= lacunarity
        if max_amp > 0:
            gx /= max_amp; gy /= max_amp
        return gx, gy

    def turbulence_gradient_batch(self, x, y, octaves=4, lacunarity=2.0, gain=0.5):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        gx = np.zeros_like(x); gy = np.zeros_like(y)
        amplitude = 1.0; frequency = 1.0; max_amp = 0.0
        for _ in range(max(1, int(octaves))):
            nx = x * frequency; ny = y * frequency
            v = self.noise_batch(nx, ny)
            cx, cy = self.noise_gradient_batch(nx, ny)
            sgn = np.sign(v)
            gx += amplitude * frequency * sgn * cx
            gy += amplitude * frequency * sgn * cy
            max_amp += amplitude
            amplitude *= gain
            frequency *= lacunarity
        if max_amp > 0:
            gx *= 2.0 / max_amp; gy *= 2.0 / max_amp
        return gx, gy
