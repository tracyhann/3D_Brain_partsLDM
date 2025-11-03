#!/usr/bin/env python3
"""
batch_generate_unique.py
Generate N UNIQUE conditioning tuples for MONAI Brain-LDM and run the bundle once per tuple.

Example:
  python scripts/batch_generate.py \
    --n 1 --outdir /mnt/data/brain_synth/1029 --seed 123 \
    --config configs/inference.json --name-prefix ldm --steps 50 --load-old 0

Notes:
- Assumes your config accepts --save_dir and --save_name (as we wired earlier).
- Keeps your exact CLI style: --gender --age --ventricular_vol --brain_vol --load_old 0
"""

import argparse, os, subprocess, random, time, json, shlex
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/inference.json")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1017)
    ap.add_argument("--load-old", type=int, default=0, dest="load_old")
    ap.add_argument("--show-cmds", action="store_true")
    ap.add_argument("--round", type=int, default=4, dest="prec",
                    help="Decimal places for uniqueness on continuous params.")
    return ap.parse_args()

def lhs_unique(n, prec):
    """
    Latin Hypercube Sampling in [0,1] for 3 continuous dims (age, vcsf, brain),
    plus a 0/1 Bernoulli for gender. Ensures unique tuples when rounded to 'prec' decimals.
    """
    # LHS for 3 dims
    import math, random
    bins = [((i + random.random())/n) for i in range(n)]
    A = bins[:] ; random.shuffle(A)  # age
    V = bins[:] ; random.shuffle(V)  # ventricular_vol
    B = bins[:] ; random.shuffle(B)  # brain_vol
    # genders ~ 50/50 (no strict LHS needed)
    G = [0]* (n//2) + [1]* (n - n//2)
    random.shuffle(G)

    uniq = []
    seen = set()
    for i in range(n*3):  # generous attempts, should succeed in first pass
        idx = i % n
        g = float(G[idx])
        a = round(float(A[idx]), prec)
        v = round(float(V[idx]), prec)
        b = round(float(B[idx]), prec)
        key = (int(g), a, v, b)
        if key not in seen:
            seen.add(key)
            uniq.append((g, a, v, b))
            if len(uniq) == n:
                break
    if len(uniq) < n:
        # Fallback: random fill without duplicates
        while len(uniq) < n:
            g = float(random.random() < 0.5)
            a = round(random.random(), prec)
            v = round(random.random(), prec)
            b = round(random.random(), prec)
            key = (int(g), a, v, b)
            if key not in seen:
                seen.add(key)
                uniq.append((g, a, v, b))
    return uniq

def filename(i, g, a, v, b):
    return f"sample{i:03d}_g{int(g)}_a{a:.4f}_v{v:.4f}_b{b:.4f}.nii.gz"

def main():
    args = parse_args()
    random.seed(args.seed)

    # space-filling, unique tuples
    conds = lhs_unique(args.n, args.prec)

    manifest = []
    cmds = []

    #for i, (g,a,v,b) in enumerate(conds):
    for i, _ in enumerate(conds):
        #CONTROL
        bs = [0.]
        g = 0
        a = 0.
        v = 0.
        b = 0.
        cmd = [
            "python", "-m", "monai.bundle", "run",
            "--config_file", args.config,
            "--gender", f"{g:.1f}",
            "--age", f"{a:.4f}",
            "--ventricular_vol", f"{v:.4f}",
            "--brain_vol", f"{b:.4f}"
        ]
        # If you exposed num_inference_steps as a top-level in config, add:
        # cmd += ["--num_inference_steps", str(args.steps)]

        t0 = time.time()
        subprocess.run(cmd, capture_output=False, text=True)
        dt = time.time() - t0

        print(f"[ok] sample {i}  ({dt:.2f}s)  g={int(g)} a={a:.4f} v={v:.4f} b={b:.4f}")
        manifest.append({
            "index": i,
            "seconds": round(dt, 3),
            "gender": int(g),
            "age": float(a),
            "ventricular_vol": float(v),
            "brain_vol": float(b),
        })

    # write manifest
    with open("manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[done] Wrote {len(manifest)} entries → {'manifest.json'}")

if __name__ == "__main__":
    main()
