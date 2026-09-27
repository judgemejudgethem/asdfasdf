"""
process_shard.py — Batch mode with local part-merge.
Processes N files; each file is split into 4 byte-ranges, processed
concurrently, then merged into a single {file_id}.parquet before upload.
"""
import os, json, subprocess, warnings, gc, time, glob
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


# ───── Utilities ─────
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
    """Stream one byte-range part → one parquet part."""
    part_size = total_bytes // PARTS_PER_FILE
    start = part_idx * part_size
    end = (total_bytes - 1) if part_idx == PARTS_PER_FILE - 1 else (start + part_size - 1)

    local = f"/tmp/raw_{file_id:03d}_{part_idx}.jsonl"
    download_range(file_path, start, end, local)
    print(f"  [p{part_idx}] downloaded {os.path.getsize(local)/1e9:.2f} GB")

    out_path = os.path.join(OUTPUT_DIR, f"file{file_id:03d}_part{part_idx}.parquet")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    writer = None
    buf = []
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
                cur = pending
                pending = line
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
                out_path, pa.Table.from_pylist(buf).schema, compression="zstd")
        writer.write_table(pa.Table.from_pylist(buf))
        buf.clear()
    if writer is not None:
        writer.close()

    os.remove(local)
    gc.collect()
    print(f"  [p{part_idx}] raw={raw:,} kept={kept:,} disc={disc:,}")
    return {"raw": raw, "kept": kept, "disc": disc}


def merge_parts(file_id):
    """Combine file{ID}_part0..3.parquet → file{ID}.parquet. Delete parts."""
    parts = sorted(glob.glob(f"{OUTPUT_DIR}/file{file_id:03d}_part*.parquet"))
    if not parts:
        return None
    merged_path = f"{OUTPUT_DIR}/file{file_id:03d}.parquet"

    # Stream-write merged parquet to keep RAM low
    writer = None
    total_rows = 0
    for p in parts:
        tbl = pq.read_table(p)
        if writer is None:
            writer = pq.ParquetWriter(merged_path, tbl.schema, compression="zstd")
        writer.write_table(tbl)
        total_rows += tbl.num_rows
        del tbl
        gc.collect()
    if writer is not None:
        writer.close()

    # Delete parts
    for p in parts:
        try:
            os.remove(p)
        except OSError:
            pass

    return {"path": merged_path, "rows": total_rows,
            "size_mb": os.path.getsize(merged_path) / 1e6}


def file_done(fid, existing):
    """True if the merged file OR any part0 already exists on HF."""
    return (any(f.endswith(f"file{fid:03d}.parquet") for f in existing)
            or any(f.endswith(f"file{fid:03d}_part0.parquet") for f in existing))


# ───── Main ─────
def main():
    start_id = int(os.environ["START_FILE_ID"])
    batch = int(os.environ.get("BATCH_SIZE", "5"))
    files = list_jsonl_files()
    end_id = min(start_id + batch, len(files))

    print(f"Job range: files {start_id}–{end_id-1} "
          f"({end_id - start_id} of {len(files)} total)")

    print("Checking HF for existing files...")
    existing = list_repo_files(HF_OUT, repo_type="dataset",
                               token=os.environ["HF_TOKEN"])

    py_exes = []
    for f in ["/tmp/py311.txt", "/tmp/py313.txt", "/tmp/py314.txt"]:
        if os.path.exists(f):
            py_exes.append(open(f).read().strip())

    processed = []
    for fid in range(start_id, end_id):
        if file_done(fid, existing):
            print(f"[{fid:03d}] ✅ already on HF — skip")
            continue

        target = files[fid]
        print(f"\n[{fid:03d}] {target}")
        try:
            size = get_file_size(target)
            print(f"  size: {size/1e9:.2f} GB → {PARTS_PER_FILE} parts")

            with ThreadPoolExecutor(max_workers=PARTS_PER_FILE) as ex:
                futs = [ex.submit(process_part, i, target, size, py_exes, fid)
                        for i in range(PARTS_PER_FILE)]
                stats = [f.result() for f in futs]

            kept = sum(s["kept"] for s in stats)
            print(f"  parts done: kept={kept:,}. Merging...")

            merged = merge_parts(fid)
            if merged:
                print(f"  ✅ file{fid:03d}: {merged['rows']:,} rows, "
                      f"{merged['size_mb']:.1f} MB")
                processed.append(fid)
            else:
                print(f"  ⚠️ nothing to merge for {fid}")

        except Exception as e:
            print(f"  ❌ file{fid:03d} failed: {e}")
            for pat in [f"{OUTPUT_DIR}/file{fid:03d}_*",
                        f"/tmp/raw_{fid:03d}_*"]:
                for x in glob.glob(pat):
                    try: os.remove(x)
                    except OSError: pass

    if processed:
        print(f"\n📤 Uploading {len(processed)} merged shards in one commit...")
        create_repo(HF_OUT, repo_type="dataset", private=False,
                    exist_ok=True, token=os.environ["HF_TOKEN"])
        upload_folder(
            repo_id=HF_OUT, repo_type="dataset",
            folder_path=OUTPUT_DIR, path_in_repo=HF_SUBFOLDER,
            token=os.environ["HF_TOKEN"],
            commit_message=f"Batch {processed[0]:03d}–{processed[-1]:03d} "
                           f"({len(processed)} merged shards)",
        )
        print(f"✅ Uploaded {len(processed)} shards in 1 commit")

        # Cleanup local
        for fid in processed:
            for x in glob.glob(f"{OUTPUT_DIR}/file{fid:03d}*"):
                try: os.remove(x)
                except OSError: pass
    else:
        print("\nNothing new to upload")


if __name__ == "__main__":
    main()
