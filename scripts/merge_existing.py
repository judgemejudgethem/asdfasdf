"""
merge_existing.py — Collapse already-uploaded partN files into single shards.
Reads HF_TOKEN from env. Merges per file_id, uploads merged, deletes parts.
"""
import os, re, gc, io, sys
from collections import defaultdict
from huggingface_hub import HfApi, hf_hub_download, upload_file, list_repo_files
import pyarrow as pa
import pyarrow.parquet as pq

REPO = "PERDYPTO/moe"
SUBFOLDER = "swallowcode_v2"
WORK = "/tmp/merge"
os.makedirs(WORK, exist_ok=True)
HF_TOKEN = os.environ["HF_TOKEN"]
api = HfApi()

# Optional: process only a range (batch mode)
range_start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
range_end   = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000

# Discover parts
files = list_repo_files(REPO, repo_type="dataset", token=HF_TOKEN)
prefix = SUBFOLDER + "/"
by_id = defaultdict(list)
for f in files:
    m = re.match(rf"{prefix}file(\d+)_part(\d+)\.parquet$", f)
    if m:
        fid = int(m.group(1))
        if range_start <= fid < range_end:
            by_id[fid].append(f)

print(f"Files with parts in range [{range_start}, {range_end}): {len(by_id)}")

for fid in sorted(by_id):
    parts = sorted(by_id[fid])
    merged_name = f"{prefix}file{fid:03d}.parquet"

    # Skip if merged already exists
    if merged_name in files:
        print(f"file{fid:03d}: merged already exists — deleting parts")
        for p in parts:
            try:
                api.delete_file(path_in_repo=p, repo_id=REPO,
                                repo_type="dataset", token=HF_TOKEN)
            except Exception as e:
                print(f"  delete {p} failed: {e}")
        continue

    print(f"\nfile{fid:03d}: merging {len(parts)} parts → {merged_name}")
    local_merged = os.path.join(WORK, f"file{fid:03d}.parquet")

    # Stream-merge to keep RAM low
    writer = None
    total_rows = 0
    try:
        for p in parts:
            local = hf_hub_download(REPO, p, repo_type="dataset", token=HF_TOKEN)
            tbl = pq.read_table(local)
            if writer is None:
                writer = pq.ParquetWriter(local_merged, tbl.schema,
                                          compression="zstd")
            writer.write_table(tbl)
            total_rows += tbl.num_rows
            del tbl
            gc.collect()
            os.remove(local)
        if writer is not None:
            writer.close()
    except Exception as e:
        print(f"  ❌ merge failed: {e}")
        continue

    size_mb = os.path.getsize(local_merged) / 1e6
    print(f"  merged: {total_rows:,} rows, {size_mb:.1f} MB")

    # Upload merged
    upload_file(
        path_or_fileobj=local_merged,
        path_in_repo=merged_name,
        repo_id=REPO, repo_type="dataset", token=HF_TOKEN,
        commit_message=f"Merge file{fid:03d} parts → single shard",
    )
    os.remove(local_merged)

    # Delete parts
    for p in parts:
        try:
            api.delete_file(path_in_repo=p, repo_id=REPO,
                            repo_type="dataset", token=HF_TOKEN)
        except Exception as e:
            print(f"  delete {p} failed: {e}")

    print(f"  ✅ file{fid:03d} merged")

print("\nDone.")
