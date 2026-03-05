import os, glob, shutil, tarfile
import pandas as pd

# ========= config =========
SAMPLES_ROOT = "samples"   # contains many subfolders
OUT_ROOT     = "samples_worst20"
N_WORST      = 20
TAR_PATH     = OUT_ROOT + ".tar.gz"
# ==========================

def safe_copy(src, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))

def find_subject_files(folder, subject):
    # Tries common patterns. Adjust if your naming scheme is different.
    patterns = [
        os.path.join(folder, f"{subject}.nii.gz"),
        os.path.join(folder, f"{subject}*.nii.gz"),
        os.path.join(folder, f"*{subject}*.nii.gz"),
    ]
    hits = []
    for p in patterns:
        hits.extend(glob.glob(p))
    # de-dup while preserving order
    seen = set()
    uniq = []
    for h in hits:
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq

def process_folder(folder):
    qc_path  = os.path.join(folder, "qc.csv")
    vol_path = os.path.join(folder, "vol.csv")

    if not os.path.isfile(qc_path):
        return None  # skip folders without qc.csv

    df = pd.read_csv(qc_path)
    if "subject" not in df.columns:
        raise ValueError(f"{qc_path} missing 'subject' column")

    cols = [c for c in df.columns if c != "subject"]
    df["mean_qc"] = df[cols].mean(axis=1)
    df["min_qc"]  = df[cols].min(axis=1)

    worst = df.sort_values("mean_qc", ascending=True).head(N_WORST).copy()
    worst_subjects = worst["subject"].astype(str).tolist()

    # output subdir mirrors input folder name
    folder_name = os.path.basename(os.path.normpath(folder))
    out_dir = os.path.join(OUT_ROOT, folder_name)
    os.makedirs(out_dir, exist_ok=True)

    # copy qc.csv and vol.csv if present
    safe_copy(qc_path, out_dir)
    if os.path.isfile(vol_path):
        safe_copy(vol_path, out_dir)
    else:
        print(f"[warn] no vol.csv in {folder}")

    # copy worst NIfTIs
    copied = 0
    missing = []
    for s in worst_subjects:
        hits = find_subject_files(folder, s)
        if not hits:
            missing.append(s)
            continue
        for h in hits:
            safe_copy(h, out_dir)
            copied += 1

    # write a manifest (very useful later)
    manifest = worst[["subject", "mean_qc", "min_qc"]].copy()
    manifest.to_csv(os.path.join(out_dir, "worst20_manifest.csv"), index=False)

    print(f"[ok] {folder_name}: kept {len(worst_subjects)} subjects, copied {copied} nii.gz files"
          + (f", missing {len(missing)} subjects" if missing else ""))
    if missing:
        with open(os.path.join(out_dir, "missing_subjects.txt"), "w") as f:
            f.write("\n".join(missing))

    return out_dir

def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    subfolders = sorted([
        p for p in glob.glob(os.path.join(SAMPLES_ROOT, "*"))
        if os.path.isdir(p)
    ])

    if not subfolders:
        raise RuntimeError(f"No subfolders found under {SAMPLES_ROOT}")

    kept_dirs = []
    for folder in subfolders:
        out = process_folder(folder)
        if out:
            kept_dirs.append(out)

    if not kept_dirs:
        raise RuntimeError("No folders processed (did you have qc.csv files?)")

    # tar.gz the curated OUT_ROOT (contains all subdirs)
    if os.path.exists(TAR_PATH):
        os.remove(TAR_PATH)

    with tarfile.open(TAR_PATH, "w:gz") as tar:
        tar.add(OUT_ROOT, arcname=os.path.basename(OUT_ROOT))

    print(f"[done] wrote curated folders to: {OUT_ROOT}")
    print(f"[done] tar.gz saved to: {TAR_PATH}")

if __name__ == "__main__":
    main()
