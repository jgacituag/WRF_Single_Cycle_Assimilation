"""
src/check_rerun.py

Compares the old multi-obs REF runs (moved to *_REF_v1) against the new ones, for the
five truth members both contain. Every key the old file has must be in the new file
with the same value; the new scalars are allowed and reported.

    python src/check_rerun.py

The observation noise is seeded per truth member (default_rng(42 + tm)), so the inputs
are identical. Whether the outputs are identical to the last bit depends on the
Fortran: a parallel reduction whose summation order changes between runs moves the
last digits without the result being any different. So the check reports both: files
that match exactly, and for the rest the largest relative difference, which should
sit at single-precision round-off, around 1e-6, and nowhere near a physical change.
"""

import pathlib
import sys

import numpy as np

DATA = pathlib.Path("/home/jorge.gacitua/salidas/WRF_Single_Cycle_Assimilation/data")
DATASETS = ("A", "B", "C")
HOURS = ("18", "19", "20", "21")
ROUNDOFF = 1e-5        # relative; above this a difference is reported as real


def compare(old_path, new_path):
    a, b = np.load(old_path, allow_pickle=True), np.load(new_path, allow_pickle=True)
    missing = sorted(set(a.files) - set(b.files))
    added = sorted(set(b.files) - set(a.files))
    worst, worst_key, exact = 0.0, None, True
    for k in sorted(set(a.files) & set(b.files)):
        x, y = a[k], b[k]
        if x.dtype.kind in "fc":
            if np.array_equal(x, y, equal_nan=True):
                continue
            exact = False
            ok = np.isfinite(x) & np.isfinite(y)
            if not np.array_equal(np.isfinite(x), np.isfinite(y)):
                return dict(missing=missing, added=added, exact=False,
                            worst=np.inf, worst_key=f"{k} (NaN en lugares distintos)")
            scale = np.maximum(np.abs(x[ok]), np.abs(y[ok]))
            rel = np.abs(x[ok] - y[ok]) / np.where(scale > 0, scale, 1.0)
            r = float(rel.max()) if rel.size else 0.0
            if r > worst:
                worst, worst_key = r, k
        elif not np.array_equal(x, y):
            return dict(missing=missing, added=added, exact=False,
                        worst=np.inf, worst_key=f"{k} (no numérico)")
    return dict(missing=missing, added=added, exact=exact, worst=worst,
                worst_key=worst_key)


def main():
    n_files = n_exact = 0
    bad, added_keys = [], set()
    for ds in DATASETS:
        for hh in HOURS:
            tag = f"WS_{ds}_multiobs_{hh}00_LOC2.0_REF"
            old, new = DATA / f"{tag}_v1", DATA / tag
            if not old.is_dir():
                print(f"  {tag}: sin versión vieja, se saltea")
                continue
            for op in sorted(old.glob("*.npz")):
                npth = new / op.name
                if not npth.exists():
                    bad.append((op.name, "falta en la corrida nueva"))
                    continue
                r = compare(op, npth)
                n_files += 1
                added_keys |= set(r["added"])
                if r["missing"]:
                    bad.append((op.name, f"claves que desaparecieron: {r['missing'][:5]}"))
                elif r["exact"]:
                    n_exact += 1
                elif r["worst"] > ROUNDOFF:
                    bad.append((op.name, f"{r['worst_key']} difiere en {r['worst']:.2e}"))

    print(f"\n{n_files} archivos comparados, {n_exact} idénticos bit a bit, "
          f"{n_files - n_exact - len(bad)} dentro del redondeo, {len(bad)} con problemas")
    print(f"claves nuevas por archivo: {len(added_keys)}, p.ej. {sorted(added_keys)[:4]}")
    for name, why in bad[:20]:
        print(f"  PROBLEMA  {name}: {why}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
