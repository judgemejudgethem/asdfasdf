"""
merge_all.py — Single-job batch merge.
- Discovers all part files on HF.
- Groups by file_id, skips already-merged and incomplete.
- Processes in chunks of 10:
    * download parts for each file
    * merge locally
    * one commit per chunk = upload 10 merged + delete all their parts
- Total commits ≈ ceil(num_files / 10)
"""
import os, re, gc, time
from collections import defaultdict
from huggingface_hub import (
    HfApi, hf_hub_download,
    CommitOperationAdd, CommitOperationDelete,
    list_repo_files,
)
import pyarrow.parquet as pq

REPO = "PERDYPTO/moe"
SUBFOLDER = "swallowcode_v2"
WORK = "/tmp/merge"
CHUNK_SIZE = 10       # files per commit

os.makedirs(WORK, exist_ok=True)
HF_TOKEN = os.environ["HF_TOKEN"]
api = HfApi()


def discover():
    """Return (by_id, merged_ids). by_id: {fid: [(part_num, path), ...]}"""
    files = list_repo_files(REPO, repo_type="dataset", token=HF_TOKEN)
    prefix = SUBFOLDER + "/"
    by_id = defaultdict(list)
    merged_ids = set()
    for f in files:
        m = re.match(rf"{prefix}file(\d+)_part(\d+)\.parquet$", f)
        if m:
            by_id[int(m.group(1))].append((int(m.group(2)), f))
            continue
        m = re.match(rf"{prefix}file(\d+)\.parquet$", f)
        if m:
            merged_ids.add(int(m.group(1)))
    return by_id, merged_ids


def merge_one(fid, parts):
    """Download parts, merge into one local file. Returns (path, rows) or None."""
    out = os.path.join(WORK, f"file{fid:03d}.parquet")
    writer = None
    total_rows = 0
    try:
        for _, p in sorted(parts):
            local = hf_hub_download(REPO, p, repo_type="dataset", token=HF_TOKEN)
            tbl = pq.read_table(local)
            if writer is None:
                writer = pq.ParquetWriter(out, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            total_rows += tbl.num_rows
            del tbl
            gc.collect()
            os.remove(local)
        if writer:
            writer.close()
        return out, total_rows
    except Exception as e:
        print(f"    merge failed: {e}")
        if writer:
            writer.close()
        try:
            os.remove(out)
        except OSError:
            pass
        return None


def commit_batch(batch):
    """One commit: add N merged files + delete their part files."""
    ops = []
    for fid, path, _, _ in batch:
        ops.append(CommitOperationAdd(
            path_in_repo=f"{SUBFOLDER}/file{fid:03d}.parquet",
            path_or_fileobj=path))
    for _, _, _, parts in batch:
        for _, p in parts:
            ops.append(CommitOperationDelete(path_in_repo=p))

    msg = (f"Merge {len(batch)} shards "
           f"({batch[0][0]:03d}–{batch[-1][0]:03d})")
    api.create_commit(
        repo_id=REPO, repo_type="dataset",
        operations=ops, commit_message=msg, token=HF_TOKEN)


def main():
    t0 = time.time()
    print("Discovering parts on HF...")
    by_id, merged_ids = discover()
    print(f"  files with parts:  {len(by_id)}")
    print(f"  already merged:    {len(merged_ids)}")

    tasks = []
    incomplete = []
    already = []
    for fid in sorted(by_id):
        if fid in merged_ids:
            already.append(fid)
            continue
        nums = sorted(n for n, _ in by_id[fid])
        if nums != list(range(len(nums))):
            incomplete.append((fid, nums))
            continue
        tasks.append((fid, by_id[fid]))

    print(f"\n  to merge:          {len(tasks)}")
    if incomplete:
        print(f"  incomplete (skip): {incomplete}")
    if already:
        print(f"  already merged:    {already}")

    if not tasks:
        print("\nNothing to merge.")
        return

    # Process in chunks of CHUNK_SIZE
    total_commits = 0
    for ci in range(0, len(tasks), CHUNK_SIZE):
        chunk = tasks[ci:ci + CHUNK_SIZE]
        print(f"\n{'='*60}")
        print(f"Chunk {ci//CHUNK_SIZE + 1}: "
              f"files {chunk[0][0]:03d}–{chunk[-1][0]:03d} "
              f"({len(chunk)} files)")
        print('='*60)

        batch = []
        for fid, parts in chunk:
            print(f"  file{fid:03d}: merging {len(parts)} parts...")
            r = merge_one(fid, parts)
            if r is None:
                continue
            path, rows = r
            size_mb = os.path.getsize(path) / 1e6
            print(f"    → {rows:,} rows, {size_mb:.1f} MB")
            batch.append((fid, path, rows, parts))

        if not batch:
            print("  nothing to commit for this chunk")
            continue

        print(f"  📤 committing {len(batch)} merged + "
              f"{sum(len(b[3]) for b in batch)} deletes (1 commit)")
        try:
            commit_batch(batch)
            total_commits += 1
            print(f"  ✅ committed")
        except Exception as e:
            print(f"  ❌ commit failed: {e}")

        # Free local disk
        for _, path, _, _ in batch:
            try: os.remove(path)
            except OSError: pass
        gc.collect()

    print(f"\n{'='*60}")
    print(f"DONE — {total_commits} commits, "
          f"{(time.time()-t0)/60:.1f} min total")
    print('='*60)


if __name__ == "__main__":
    main()
