"""
process_shard.py — Round-robin slice, merge-parts, single commit per job.

Environment:
  HF_TOKEN     - HF token
  JOB_ID       - 0..TOTAL_JOBS-1 (this job's slice index)
  TOTAL_JOBS   - total number of parallel jobs (default 20)
"""
import os, json, subprocess, warnings, gc, time, glob, re
from concurrent.futures import ThreadPoolExecutor
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import list_repo_files, create_repo, upload_folder

warnings.filterwarnings("ignore", category=SyntaxWarning)

REPO = "tokyotech-llm/swallow-code-v2"
PATH_PREFIX = "stage5-auto-format/python/medium"
OUTPUT_DIR = "/tmp/cleaned"
HF_OUT = "PERDYPTO/moe"
HF_SUBFOLDER = "swallowcode_v2"
PARTS_PER_FILE = 4
TEXT_FIELDS = ["text", "path", "repo_name", "blob_id"]
CODE_FIELD = "text"
CHAR_MIN, CHAR_MAX = 50, 128_000
ROW_BATCH = 50_000
DL_RETRIES = 5


# ─────── helpers ───────
def list_jsonl_files():
    all_files = list_repo_files(REPO, repo_type="dataset",
                                token=os.environ["HF_TOKEN"])
    return sorted([f for f in all_files
                   if f.startswith(PATH_PREFIX + "/") and f.endswith(".jsonl")])


def get_file_size(path):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    r = requests.head(url, allow_redirects=True,
                      headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def download_range(path, start, end, out, max_retries=DL_RETRIES):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    total = end - start + 1
    for attempt in range(max_retries):
        already = os.path.getsize(out) if os.path.exists(out) else 0
        if already >= total:
            return
        cur_start = start + already
        headers = {"Range": f"bytes={cur_start}-{end}",
                   "Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
        mode = "ab" if already > 0 else "wb"
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=(30, 600))
            r.raise_for_status()
            with open(out, mode) as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            if os.path.getsize(out) >= total:
                return
        except Exception as e:
            print(f"    [retry {attempt+1}/{max_retries}] {type(e).__name__}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


_AST_SCRIPT = r"""
import warnings, ast, sys, json
warnings.filterwarnings("ignore")
def check(c):
    if not isinstance(c, str) or not c.strip(): return False
    try: ast.parse(c); return True
    except Exception: return False
try:
    json.dump([check(c) for c in json.load(sys.stdin)], sys.stdout)
except Exception as e:
    json.dump({"error": str(e)}, sys.stdout)
"""


def ast_ok(code, py_exes):
    import ast
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
        return True
    except Exception:
        pass
    for exe in py_exes:
        try:
            p = subprocess.run([exe, "-W", "ignore", "-c", _AST_SCRIPT],
                               input=json.dumps([code]),
                               capture_output=True, text=True, timeout=60)
            if json.loads(p.stdout)[0]:
                return True
        except Exception:
            pass
    return False


def process_part(part_idx, file_path, total_bytes, py_exes, file_id):
    """Stream one byte-range → one {file_id}_part{N}.parquet."""
    part_size = total_bytes // PARTS_PER_FILE
    start = part_idx * part_size
    end = (total_bytes - 1) if part_idx == PARTS_PER_FILE - 1 else (start + part_size - 1)

    local = f"/tmp/raw_{file_id:03d}_{part_idx}.jsonl"
    download_range(file_path, start, end, local)
    print(f"    [p{part_idx}] downloaded {os.path.getsize(local)/1e9:.2f} GB")

    out_path = os.path.join(OUTPUT_DIR, f"file{file_id:03d}_part{part_idx}.parquet")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    writer, buf = None, []
    raw = kept = disc = 0
    is_first = (part_idx == 0)
    is_last = (part_idx == PARTS_PER_FILE - 1)

    with open(local, "r", encoding="utf-8", errors="ignore") as f:
        first = True
        pending = None
        for line in f:
            if first:
                first = False
                if not is_first:
                    continue
            if pending is not None:
                cur, pending = pending, line
            else:
                pending = line
                continue
            cur = cur.strip()
            if not cur:
                continue
            try:
                ex = json.loads(cur)
            except json.JSONDecodeError:
                continue
            raw += 1
            text = " ".join(str(ex.get(f, "") or "") for f in TEXT_FIELDS)
            if not (CHAR_MIN <= len(text) <= CHAR_MAX):
                disc += 1
                continue
            code = str(ex.get(CODE_FIELD, "") or "")
            if code.strip() and not ast_ok(code, py_exes):
                disc += 1
                continue
            rec = {f: ex.get(f) for f in TEXT_FIELDS}
            rec["_source"] = REPO
            rec["_file_id"] = file_id
            buf.append(rec)
            kept += 1
            if len(buf) >= ROW_BATCH:
                if writer is None:
                    writer = pq.ParquetWriter(
                        out_path, pa.Table.from_pylist(buf).schema,
                        compression="zstd")
                writer.write_table(pa.Table.from_pylist(buf))
                buf.clear()
                gc.collect()

        if pending is not None and is_last:
            pending = pending.strip()
            if pending:
                try:
                    ex = json.loads(pending)
                    text = " ".join(str(ex.get(f, "") or "") for f in TEXT_FIELDS)
                    if CHAR_MIN <= len(text) <= CHAR_MAX:
                        code = str(ex.get(CODE_FIELD, "") or "")
                        if not code.strip() or ast_ok(code, py_exes):
                            rec = {f: ex.get(f) for f in TEXT_FIELDS}
                            rec["_source"] = REPO
                            rec["_file_id"] = file_id
                            buf.append(rec)
                            kept += 1
                        else:
                            disc += 1
                    else:
                        disc += 1
                    raw += 1
                except json.JSONDecodeError:
                    pass

    if buf:
        if writer is None:
            writer = pq.ParquetWriter(
                out_path, pa.Table.from_pylist(buf).schema,
                compression="zstd")
        writer.write_table(pa.Table.from_pylist(buf))
    if writer:
        writer.close()

    os.remove(local)
    gc.collect()
    print(f"    [p{part_idx}] raw={raw:,} kept={kept:,} disc={disc:,}")
    return {"kept": kept}


def merge_parts(file_id):
    """Combine file{ID}_part0..N → file{ID}.parquet. Delete parts."""
    parts = sorted(glob.glob(f"{OUTPUT_DIR}/file{file_id:03d}_part*.parquet"))
    if not parts:
        return None
    merged = f"{OUTPUT_DIR}/file{file_id:03d}.parquet"
    writer = None
    total_rows = 0
    for p in parts:
        tbl = pq.read_table(p)
        if writer is None:
            writer = pq.ParquetWriter(merged, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        total_rows += tbl.num_rows
        del tbl
        gc.collect()
    if writer:
        writer.close()
    for p in parts:
        try: os.remove(p)
        except OSError: pass
    return total_rows, os.path.getsize(merged) / 1e6


def completed_file_ids(existing):
    """File IDs already merged OR with part0 already on HF."""
    prefix = HF_SUBFOLDER + "/"
    done = set()
    for f in existing:
        if not f.startswith(prefix):
            continue
        m = re.match(rf"{prefix}file(\d+)\.parquet$", f)
        if m:
            done.add(int(m.group(1)))
            continue
        m = re.match(rf"{prefix}file(\d+)_part0\.parquet$", f)
        if m:
            done.add(int(m.group(1)))
    return done


def cleanup_file(fid):
    for pat in [f"{OUTPUT_DIR}/file{fid:03d}_*",
                f"/tmp/raw_{fid:03d}_*"]:
        for x in glob.glob(pat):
            try: os.remove(x)
            except OSError: pass


# ─────── main ───────
def main():
    job_id = int(os.environ["JOB_ID"])
    total_jobs = int(os.environ.get("TOTAL_JOBS", "20"))

    files = list_jsonl_files()
    print(f"Total JSONL files in source: {len(files)}")

    print("Checking HF for completed files...")
    existing = list_repo_files(HF_OUT, repo_type="dataset",
                               token=os.environ["HF_TOKEN"])
    done = completed_file_ids(existing)
    print(f"  completed: {len(done)}")

    remaining = [i for i in range(len(files)) if i not in done]
    print(f"  remaining: {len(remaining)}")

    # Deterministic slice: file `i` belongs to job `i % total_jobs`
    my_ids = [i for i in remaining if i % total_jobs == job_id]
    print(f"\nJob {job_id}/{total_jobs}: assigned {len(my_ids)} files")
    if my_ids:
        preview = my_ids[:10] + (["..."] if len(my_ids) > 10 else [])
        print(f"  files: {preview}")

    if not my_ids:
        print("Nothing to process. Exiting.")
        return

    py_exes = []
    for f in ["/tmp/py311.txt", "/tmp/py313.txt", "/tmp/py314.txt"]:
        if os.path.exists(f):
            py_exes.append(open(f).read().strip())

    merged_shards = []
    for fid in my_ids:
        target = files[fid]
        print(f"\n{'='*60}\n[{fid:03d}] {target}\n{'='*60}")
        cleanup_file(fid)
        try:
            size = get_file_size(target)
            print(f"  size: {size/1e9:.2f} GB → {PARTS_PER_FILE} parts")

            with ThreadPoolExecutor(max_workers=PARTS_PER_FILE) as ex:
                futs = [ex.submit(process_part, i, target, size, py_exes, fid)
                        for i in range(PARTS_PER_FILE)]
                stats = [f.result() for f in futs]
            kept = sum(s["kept"] for s in stats)
            print(f"  parts complete: kept={kept:,}")

            m = merge_parts(fid)
            if m:
                rows, size_mb = m
                print(f"  ✅ file{fid:03d}: {rows:,} rows, {size_mb:.1f} MB")
                merged_shards.append(fid)
            else:
                print(f"  ⚠️  nothing to merge for file{fid:03d}")
        except Exception as e:
            print(f"  ❌ failed: {e}")
            cleanup_file(fid)

    # ─── ONE commit for everything this job produced ───
    if merged_shards:
        print(f"\n{'='*60}")
        print(f"📤 Committing {len(merged_shards)} merged shards in ONE commit")
        print(f"{'='*60}")
        create_repo(HF_OUT, repo_type="dataset", private=False,
                    exist_ok=True, token=os.environ["HF_TOKEN"])
        upload_folder(
            repo_id=HF_OUT, repo_type="dataset",
            folder_path=OUTPUT_DIR, path_in_repo=HF_SUBFOLDER,
            token=os.environ["HF_TOKEN"],
            commit_message=f"Job {job_id}: {len(merged_shards)} shards "
                           f"(IDs {merged_shards[0]:03d}–{merged_shards[-1]:03d})",
        )
        print(f"✅ Committed {len(merged_shards)} shards")

        for fid in merged_shards:
            for x in glob.glob(f"{OUTPUT_DIR}/file{fid:03d}*"):
                try: os.remove(x)
                except OSError: pass
    else:
        print("\nNo shards to commit.")


if __name__ == "__main__":
    main()
