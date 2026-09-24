"""
Notebooks/select_point_cells.py

Chooses the four cells for point mode from the A 20:00 sweep, and writes them to
configs/point_cells_A_2000.yaml, which configs/make_campaign.py reads.

    cd Notebooks && python select_point_cells.py

The cells are defined by the PRIOR, which does not depend on the truth member. The
innovation does, so a cell has sixty innovations, one per truth; the innovation
categories appear inside each cell once the sixty members are run, rather than
defining the cell. The one exception is the fourth cell, chosen by how badly w fares
over the sixty truths, which is a property of the cell and not of any single truth.

All four are taken from points the sweep already assimilated, so point mode's last
step must reproduce the sweep's own row at the same cell.
"""

import pathlib

import numpy as np
import pandas as pd
import yaml

import nbcommon as nb

TAG = "WS_A_sweep_2000_LOC0.1_ALLMEM"
OUT = pathlib.Path(__file__).resolve().parents[1] / "configs" / "point_cells_A_2000.yaml"
REF_Z_KM = 5.0          # prefer cells near the reference level for the three prior cells

COLS = ["i", "j", "k", "z_km", "yo_clean", "dep_b", "n_active_f_point",
        "spread_f_point_obs", "method", "ntemp",
        "mean_f_point_w", "mean_a_point_w", "truth_point_w"]


def load():
    frames = []
    for f in nb.sweep_files(TAG):
        nb.assert_dataset(f, "A")
        d = nb.load_sweep(f, columns=COLS, verbose=False)
        d = d[(d.method == "TEnKF") & (d.ntemp == 1)]
        d["tm"] = int(f.rsplit("_tm", 1)[1][:2])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["skill_w"] = (np.abs(d.mean_f_point_w - d.truth_point_w)
                    - np.abs(d.mean_a_point_w - d.truth_point_w))
    return d


def main():
    d = load()
    n_tm = d.tm.nunique()
    ne = int(d.n_active_f_point.max())
    print(f"{TAG}: {n_tm} miembros verdad, {len(d):,} filas")

    # n_active is a prior property, identical across truths up to the member removed
    # as truth, so it is averaged; the cell must have been assimilated for (nearly)
    # every truth to be comparable across them.
    g = (d.groupby(["i", "j", "k"])
           .agg(z_km=("z_km", "first"), n_tm=("tm", "nunique"),
                n_active=("n_active_f_point", "mean"),
                spread=("spread_f_point_obs", "mean"),
                skill_w=("skill_w", "mean"), skill_w_sd=("skill_w", "std"),
                dep_min=("dep_b", "min"), dep_max=("dep_b", "max"),
                frac_truth_echo=("yo_clean", lambda v: float((v > 0).mean())))
           .reset_index())
    g = g[g.n_tm >= n_tm - 1]
    g["dz"] = (g.z_km - REF_Z_KM).abs()

    def pick(sel, label, why, order):
        c = sel.sort_values(order).head(5)
        print(f"\n{label}: {len(sel)} candidatas; las cinco primeras por {order}")
        print(c[["i", "j", "k", "z_km", "n_active", "spread", "skill_w",
                 "dep_min", "dep_max"]].to_string(index=False, float_format="%.3f"))
        r = c.iloc[0]
        return dict(name=label, ijk=[int(r.i), int(r.j), int(r.k)], why=why,
                    z_km=float(r.z_km), n_active=float(r.n_active),
                    spread=float(r.spread), skill_w_mean=float(r.skill_w),
                    dep_range=[float(r.dep_min), float(r.dep_max)])

    cells = [
        pick(g[g.n_active.between(1, 3)], "aire_claro",
             "uno a tres miembros con señal", ["dz", "spread"]),
        pick(g[(g.n_active - ne / 2).abs() <= 2], "borde",
             "la mitad del ensamble con señal, dispersión máxima", ["dz", "spread"]),
        pick(g[g.n_active >= ne - 4], "nucleo",
             "casi todo el ensamble con señal", ["dz", "spread"]),
        pick(g, "peor_w", "la peor skill media de w sobre los miembros verdad",
             ["skill_w"]),
    ]
    ijk = [tuple(c["ijk"]) for c in cells]
    assert len(set(ijk)) == len(ijk), "dos categorías eligieron la misma celda"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        yaml.safe_dump({"source": TAG, "n_truth_members": int(n_tm), "cells": cells},
                       f, sort_keys=False, allow_unicode=True)
    print(f"\nescrito {OUT}. Revisalo: cada categoría lista sus cinco candidatas, y "
          f"cambiar una celda es editar su ijk en ese archivo.")

    # De paso: el número que quedó sin verificar en el capítulo.
    print(f"\nobservaciones con eco en la verdad, A 20:00: "
          f"{100 * float((d.yo_clean > 0).mean()):.1f} %")


if __name__ == "__main__":
    main()
