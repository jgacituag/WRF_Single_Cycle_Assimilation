import os, sys
import numpy as np

def _import_fortran():
    try:
        from cletkf_wloc import common_da as cda
        return cda
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        third_party = os.path.normpath(os.path.join(here, "..", "third_party"))
        if third_party not in sys.path:
            sys.path.insert(0, third_party)
        from cletkf_wloc import common_da as cda
        return cda

cda = _import_fortran()

def check():
    print("Fortran LETKF loaded from:", cda.__file__)

def tempered_wloc(xf_grid, hxf, yo, obs_error, loc_scales, ox, oy, oz, steps):
    nx, ny, nz, Ne, nvar = xf_grid.shape
    nobs = len(yo)
    ntemp = len(steps)

    xf_grid = np.asfortranarray(xf_grid.astype("float32"))
    hxf     = np.asfortranarray(hxf.astype("float32"))
    yo      = np.asarray(yo, dtype="float32")
    obs_error = np.asarray(obs_error, dtype="float32")
    loc_scales = np.asarray(loc_scales, dtype="float32")
    ox = np.asarray(ox, dtype="int32")
    oy = np.asarray(oy, dtype="int32")
    oz = np.asarray(oz, dtype="int32")
    steps = np.asarray(steps, dtype="float32")

    dep = yo - hxf.mean(axis=1).astype("float32")

    xatemp = np.zeros((nx,ny,nz,Ne,nvar, ntemp+1), dtype="float32", order="F")
    xatemp[..., 0] = xf_grid
    deps = np.zeros((ntemp, nobs), dtype="float32")

    for it in range(ntemp):
        deps[it, :] = dep
        oerr_temp = obs_error * (1.0 / steps[it])

        Xa = cda.simple_letkf_wloc(
            nx=nx, ny=ny, nz=nz,
            nbv=Ne, nvar=nvar, nobs=nobs,
            hxf=hxf, xf=xatemp[..., it],
            dep=dep, ox=ox, oy=oy, oz=oz,
            locs=loc_scales, oerr=oerr_temp
        ).astype("float32")

        xatemp[..., it+1] = Xa

    return xatemp, deps