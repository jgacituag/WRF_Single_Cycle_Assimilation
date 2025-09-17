import numpy as np

def full2d(nx: int, nz: int):
    """Observe every (x,z) at y=0."""
    mask = np.ones((nx, nz), dtype=bool)
    ox, oz = np.where(mask)
    oy = np.zeros_like(ox, dtype=int)
    return ox.astype(int), oy, oz.astype(int)

def every_other(nx: int, nz: int, stride_x: int = 2, stride_z: int = 2, offset_x: int = 0, offset_z: int = 0):
    """Observe a checkerboard / decimated grid at y=0."""
    xs = np.arange(offset_x, nx, max(1, stride_x))
    zs = np.arange(offset_z, nz, max(1, stride_z))
    mask = np.zeros((nx, nz), dtype=bool)
    mask[np.ix_(xs, zs)] = True
    ox, oz = np.where(mask)
    oy = np.zeros_like(ox, dtype=int)
    return ox.astype(int), oy, oz.astype(int)

def rhi(nx: int, nz: int, origin_x: int, origin_z: int,
        angles_deg: list[float], max_range: float, dr: float = 1.0):
    """
    'RHI-like' selection in the x–z plane at y=0:
      - Shoot rays from (origin_x, origin_z) at the given angles (deg, 0° = +x, 90° = +z).
      - Sample points every 'dr' (grid units) until 'max_range' or domain edge.
      - Grid indices are rounded; duplicates are removed preserving order.
    Returns integer arrays (ox, oy, oz).
    """
    ox_list, oz_list = [], []
    for th in angles_deg:
        th_rad = np.deg2rad(th)
        ux, uz = np.cos(th_rad), np.sin(th_rad)
        nsteps = int(max_range / max(dr, 1e-6))
        for k in range(1, nsteps + 1):
            x = int(round(origin_x + k * dr * ux))
            z = int(round(origin_z + k * dr * uz))
            if x < 0 or x >= nx or z < 0 or z >= nz:
                break
            if (len(ox_list) == 0) or (x != ox_list[-1] or z != oz_list[-1]):
                ox_list.append(x); oz_list.append(z)
    if not ox_list:
        return np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
    ox = np.array(ox_list, dtype=int)
    oz = np.array(oz_list, dtype=int)
    oy = np.zeros_like(ox, dtype=int)
    # Deduplicate across all rays while preserving order
    seen = set(); keep = []
    for i in range(len(ox)):
        key = (int(ox[i]), int(oz[i]))
        if key not in seen:
            seen.add(key); keep.append(i)
    keep = np.array(keep, dtype=int)
    return ox[keep], oy[keep], oz[keep]