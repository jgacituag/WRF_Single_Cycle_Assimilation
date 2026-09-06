"""
src/backfill_identity.py
========================
Write the dataset identity INTO every subset and every experiment output, so a file
says what it is without anyone having to read its name.

    dataset_id        "A" | "B" | "C" | "D"
    da_cycle_min      upstream DA cycle length [min]
    dx_km             horizontal grid spacing [km], MEASURED from pos_km
    physics           "multi" | "single"
    upstream          "GUES" | "DAFCST" -- which POST tree the prior came from
    config_index      (Ne,) int16, the physics configuration each member used
    config_index_note why config_index reads -1 for these files

Filenames are metadata that drifts. This is the part that outlasts the rename, and it
is the reason the naming discussion happened at all: nothing on disk recorded which
physics each member ran, so it could not be verified from the data.

`config_index` is written as all -1 and is NOT invented. For every dataset that existed
before the migration the mapping is unrecoverable -- it is in neither the subset npz,
the run configs, the extraction configs, nor the upstream POST trees, which carry no
namelists. A rebuilt dataset should write the real thing at extraction time.

How the write is done
---------------------
An .npz is a zip of .npy members, so the new keys are APPENDED to the archive rather
than written through np.savez. Rewriting these files to add five scalars would mean
recompressing about 250 GB; appending costs a couple of milliseconds and a kilobyte
each. Every file is opened and read back before and after, and the script stops on the
first file that does not survive the round trip.

    python src/backfill_identity.py --check    # what is present, what is missing
    python src/backfill_identity.py --apply
"""

import argparse
import collections
import csv
import io
import os
import pathlib
import re
import struct
import sys
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MANIFEST = DATA / "derived" / "rename_manifest.csv"

# Not measurable from the files: it is a property of the upstream DA configuration.
# This is the migration mapping, and it is the only hardcoded part of the identity.
DS_CYCLE = {"A": 5, "B": 60, "C": 5, "D": 5}
DS_PHYSICS = {"A": "multi", "B": "multi", "C": "multi", "D": "single"}

CONFIG_INDEX_NOTE = (
    "unrecoverable for this dataset: the member-to-physics mapping is recorded in "
    "neither the subset npz, the run config, the extraction config, nor the upstream "
    "POST tree (no namelists). -1 means 'not recorded', never 'configuration -1'. "
    "Datasets built after 2026-08-29 write the real index at extraction time."
)

IDENTITY_KEYS = ("dataset_id", "da_cycle_min", "dx_km", "physics", "upstream",
                 "source_run", "config_index", "config_index_note")

NE_RE = re.compile(r"_Ne(\d{3})_")
SUBSET_RE = re.compile(r"^subset_(?P<ds>[ABCD])_(?P<stamp>\d{14})\.npz$")


# ---------------------------------------------------------------------------
# npz helpers -- header-only reads, and appending without recompressing
# ---------------------------------------------------------------------------

def npz_keys(path):
    with zipfile.ZipFile(path) as z:
        return {n[:-4] if n.endswith(".npy") else n for n in z.namelist()}


def npy_shape(path, key):
    """Shape of one member, read from its .npy header alone -- no inflate.

    state_ensemble is 1.5 GB compressed; only its header is needed to learn Ne.
    """
    with zipfile.ZipFile(path) as z:
        with z.open(key + ".npy") as f:
            version = np.lib.format.read_magic(f)
            try:
                shape, _fortran, _dtype = np.lib.format._read_array_header(f, version)
            except AttributeError:                       # numpy >= 1.22 moved it
                shape, _fortran, _dtype = np.lib.format.read_array_header_1_0(f)
    return shape


def append_npz(path, **arrays):
    """Append new members to an existing .npz. Existing members are not touched."""
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_DEFLATED) as z:
        have = set(z.namelist())
        for k, v in arrays.items():
            name = k + ".npy"
            if name in have:
                raise KeyError(f"{path} already carries {k}")
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(v), allow_pickle=False)
            z.writestr(name, buf.getvalue())


def readable(path):
    """The archive opens and its central directory lists members.

    Deliberately NOT zipfile.testzip(): that inflates every member to verify its CRC,
    which is hours of I/O across 250 GB of run output and re-reads data this script
    never touches. Truncation -- the one failure mode actually seen here -- destroys
    the end-of-central-directory record and is caught by simply opening the archive.
    The members this script appends ARE CRC-checked, by reading them back afterwards.
    """
    try:
        with zipfile.ZipFile(path) as z:
            return bool(z.namelist())
    except Exception:                                    # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# What each file is
# ---------------------------------------------------------------------------

def run_identity_from_manifest():
    """{run directory name: (dataset, upstream)} as the migration recorded it.

    Read from the manifest rather than re-parsed from the new names: the manifest is
    what the rename was actually built from, and `upstream` in particular was taken
    from each run's own paths.prepared, not inferred from its name.
    """
    if not MANIFEST.exists():
        sys.exit(f"no manifest at {MANIFEST} -- run the migration first")
    out = {}
    with open(MANIFEST, newline="") as f:
        for r in csv.DictReader(f):
            if r["kind"] == "run_dir" and r["action"] == "move":
                out[os.path.basename(r["new"])] = (r["dataset"], r["upstream"])

    # Runs made AFTER the migration were never renamed, so the manifest has no row for
    # them -- dataset A's four sweeps, all 240 files, are in exactly that position. For
    # those the tag is the source, which is sound because the tag is not a free string:
    # src/naming.py validates it at runner startup against the config it was run with,
    # and the analysis notebooks re-check tag -> config -> subset -> dataset_id before
    # reading a single number. `upstream` is read from the subset the run names rather
    # than assumed, since that is the field the manifest took from paths.prepared.
    sys.path.insert(0, str(REPO / "src"))
    import naming
    for d in sorted(DATA.iterdir()):
        if not d.is_dir() or d.name in out or not naming.TAG_RE.match(d.name):
            continue
        ds = naming.TAG_RE.match(d.name).group("ds")
        ups = "unknown"
        cands = sorted(d.glob("*_config.yaml"))
        if cands:
            import yaml
            prep = yaml.safe_load(open(cands[0])).get("paths", {}).get("prepared", "")
            local = DATA / f"3D_subsets_{ds}" / os.path.basename(prep)
            if local.exists():
                with np.load(local, allow_pickle=False) as f_:
                    if "upstream" in f_.files:
                        ups = str(f_["upstream"])
        out[d.name] = (ds, ups)
        print(f"  {d.name}: not in the manifest (made after the migration); "
              f"dataset {ds} from the validated tag, upstream {ups} from its subset")
    return out


def source_runs(cache={}):
    """{dataset: upstream WRF experiment}, read from the extraction configs.

    Read rather than hardcoded: the config is what actually produced the subsets, so a
    table here could drift away from it silently.
    """
    if cache:
        return cache
    import yaml
    sys.path.insert(0, str(REPO / "src"))
    from extract_3d_subset import _source_run
    for f in sorted((REPO / "configs").glob("build_3D_section_Dataset_*.yaml")):
        j = yaml.safe_load(open(f)).get("cross_sections_job", {})
        ds = j.get("dataset_id")
        if ds:
            cache[ds] = _source_run(j)
    return cache


def measured_dx(ds, cache={}):
    """dx in km for a dataset, measured from a subset's pos_km. Cached per dataset."""
    if ds in cache:
        return cache[ds]
    d = DATA / f"3D_subsets_{ds}"
    # Not simply the first: dataset D's earliest subset (15Z) is a truncated write from
    # 2026-06-27 and will not open. Take the first that does.
    cands = [q for q in sorted(d.glob(f"subset_{ds}_*.npz")) if readable(q)] \
        if d.is_dir() else []
    if not cands:
        raise RuntimeError(f"no readable subset for dataset {ds}; cannot measure dx_km")
    with np.load(cands[0]) as f:
        pos = f["pos_km"]
    cache[ds] = float(np.median(np.diff(pos[:, 0, 0, 0])))
    print(f"  dx_km for {ds} measured from {cands[0].name}: {cache[ds]:.4f} km")
    return cache[ds]


def targets():
    """Every file to backfill: (path, dataset, upstream, Ne, kind), plus the damaged.

    A file that will not open as a zip is never written to -- appending to a broken
    archive would only make it harder to recover. It is reported and skipped.
    """
    runs = run_identity_from_manifest()
    out, damaged = [], []

    for ds_dir in sorted(DATA.glob("3D_subsets_[ABCD]")):
        ds = ds_dir.name[-1]
        for p in sorted(ds_dir.glob("*.npz")):
            m = SUBSET_RE.match(p.name)
            if m is None or m.group("ds") != ds:
                print(f"  skipping unrecognised subset file: {p}")
                continue
            if not readable(p):
                damaged.append((p, "archive will not open"))
                continue
            try:
                ne = npy_shape(p, "state_ensemble")[3]
            except Exception as e:                       # noqa: BLE001
                damaged.append((p, f"cannot read state_ensemble header: {e}"))
                continue
            out.append((p, ds, "GUES", int(ne), "subset"))

    for tag, (ds, ups) in sorted(runs.items()):
        d = DATA / tag
        if not d.is_dir():
            print(f"  skipping missing run directory: {d}")
            continue
        for p in sorted(d.glob("*.npz")):
            m = NE_RE.search(p.name)
            if m is None:
                print(f"  skipping run file with no _Ne###_ in its name: {p}")
                continue
            if not readable(p):
                damaged.append((p, "archive will not open"))
                continue
            out.append((p, ds, ups, int(m.group(1)), "run"))
    return out, damaged


def identity_for(ds, upstream, ne):
    return dict(
        dataset_id=np.array(ds),
        da_cycle_min=np.int16(DS_CYCLE[ds]),
        dx_km=np.float32(measured_dx(ds)),
        physics=np.array(DS_PHYSICS[ds]),
        upstream=np.array(upstream or "unknown"),
        source_run=np.array(source_runs().get(ds, "unknown")),
        config_index=np.full(int(ne), -1, np.int16),
        config_index_note=np.array(CONFIG_INDEX_NOTE),
    )


# ---------------------------------------------------------------------------

def check(items):
    have = collections.Counter()
    partial, unreadable = [], []
    for p, ds, ups, ne, kind in items:
        if not readable(p):
            unreadable.append(p)
            continue
        keys = npz_keys(p)
        present = [k for k in IDENTITY_KEYS if k in keys]
        if not present:
            have[(ds, kind, "missing")] += 1
        elif len(present) == len(IDENTITY_KEYS):
            have[(ds, kind, "complete")] += 1
        else:
            have[(ds, kind, "partial")] += 1
            partial.append((p, present))

    print(f"{'ds':4s}{'kind':10s}{'state':10s}{'files':>7s}")
    for (ds, kind, state), n in sorted(have.items()):
        print(f"{ds:4s}{kind:10s}{state:10s}{n:7d}")
    if partial:
        print(f"\n{len(partial)} file(s) carry SOME identity keys but not all:")
        for p, ks in partial[:10]:
            print(f"  {p}\n      has {ks}")
    if unreadable:
        print(f"\n{len(unreadable)} UNREADABLE archive(s):")
        for p in unreadable[:10]:
            print("   ", p)
    return not (partial or unreadable)


def apply(items, force=False):
    dxs = {}
    done = skipped = 0
    for i, (p, ds, ups, ne, kind) in enumerate(items, 1):
        if not readable(p):
            sys.exit(f"REFUSING: {p} is not a readable zip BEFORE any write.")
        keys = npz_keys(p)
        ident = identity_for(ds, ups, ne)
        dxs[ds] = float(ident["dx_km"])
        # Per key, not all-or-nothing: a file written before a key existed gets just
        # that key added. A zip member cannot be replaced without rewriting the whole
        # archive, so keys already present are left exactly as they are.
        missing = {k: v for k, v in ident.items() if k not in keys}
        if not missing:
            skipped += 1
            continue
        if force and len(missing) != len(ident):
            sys.exit(f"REFUSING: {p} carries some identity keys already "
                     f"({sorted(set(ident) - set(missing))}). Replacing a zip member "
                     f"means rewriting the archive; do that deliberately, not here.")
        append_npz(p, **missing)

        # Read back: the archive still opens, the new keys are there and correct, and
        # an existing member still inflates. Verified per file, not assumed.
        if not readable(p):
            sys.exit(f"CORRUPT after append: {p}. STOP -- do not continue.")
        with np.load(p, allow_pickle=False) as f:
            for k in missing:
                if k not in f.files:
                    sys.exit(f"read-back failed in {p}: {k} did not land")
            if str(f["dataset_id"]) != ds:
                sys.exit(f"read-back mismatch in {p}: dataset_id={f['dataset_id']!r}")
            if str(f["source_run"]) in ("", "unknown"):
                print(f"  warning: {p.name} has no source_run (no config for {ds}?)")
            if f["config_index"].shape != (ne,):
                sys.exit(f"read-back mismatch in {p}: config_index shape "
                         f"{f['config_index'].shape}, expected ({ne},)")
            probe = "pos_km" if kind == "subset" else "var_names"
            if probe in f.files:
                _ = f[probe]                      # an original member still inflates
        done += 1
        if i % 100 == 0 or i == len(items):
            print(f"  {i}/{len(items)}  written {done}, already had it {skipped}")

    print(f"\nbackfilled {done} file(s); {skipped} already carried the identity.")
    print("dx_km measured per dataset: " +
          ", ".join(f"{k}={v:.4f}" for k, v in sorted(dxs.items())))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="report coverage, write nothing")
    g.add_argument("--apply", action="store_true", help="append the identity keys")
    ap.add_argument("--force", action="store_true",
                    help="fail loudly on files that already carry an identity")
    args = ap.parse_args()

    items, damaged = targets()
    print(f"{len(items)} target file(s)")
    if damaged:
        print(f"\n{len(damaged)} DAMAGED file(s) -- skipped, never written to:")
        for q, why in damaged:
            print(f"  {q}\n      {why}  ({q.stat().st_size / 1e9:.3f} GB)")
    print()
    if args.apply:
        apply(items, force=args.force)
        print()
        check(items)
    else:
        sys.exit(0 if check(items) else 1)


if __name__ == "__main__":
    main()
