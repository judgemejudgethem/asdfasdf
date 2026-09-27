"""
process_shard.py — GitHub Actions worker for SwallowCode-v2
- Splits one file across 4 concurrent workers (byte ranges).
- Streams lines, buffers at most BATCH_SIZE rows, flushes to parquet.
- Peak RAM < 500 MB per worker.
"""
import os
import json
import subprocess
import warnings
import gc
from concurrent.futures import ThreadPoolExecutor
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import list_repo_files

warnings.filterwarnings("ignore", category=SyntaxWarning)

REPO = "tokyotech-llm/swallow-code-v2"
PATH_PREFIX = "stage5-auto-format/python/medium"
OUTPUT_DIR = "/tmp/cleaned"
PARTS_PER_FILE = 4          # ← 4 workers per file (matches GH runner vCPUs)
TEXT_FIELDS = ["text", "path", "repo_name", "blob_id"]
CODE_FIELD = "text"
CHAR_MIN, CHAR_MAX = 50, 128_000
BATCH_SIZE = 50_000         # rows per parquet flush


def list_jsonl_files():
    all_files = list_repo_files(REPO, repo_type="dataset",
                                token=os.environ["HF_TOKEN"])
    return sorted([f for f in all_files
                   if f.startswith(PATH_PREFIX + "/") and f.endswith(".jsonl")])


def get_file_size(file_path):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{file_path}"
    r = requests.head(url, allow_redirects=True,
                      headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
    r.raise_for_status()
    return int(r.headers["Content-Length"])


def download_byte_range(file_path, start, end, out_path):
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{file_path}"
    headers = {"Range": f"bytes={start}-{end}",
               "Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    r = requests.get(url, headers=headers, stream=True, timeout=600)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)


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
    """Fast in-process parse, then fall back to other Python versions."""
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
    """Stream one byte-range part. file_id is an INT."""
    part_size = total_bytes // PARTS_PER_FILE
    start = part_idx * part_size
    end = (total_bytes - 1) if part_idx == PARTS_PER_FILE - 1 else (start + part_size - 1)

    local_raw = f"/tmp/raw_{file_id:03d}_{part_idx}.jsonl"
    download_byte_range(file_path, start, end, local_raw)
    print(f"  [part{part_idx}] downloaded {os.path.getsize(local_raw)/1e9:.2f} GB")

    out_path = os.path.join(OUTPUT_DIR, f"file{file_id:03d}_part{part_idx}.parquet")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    writer = None
    batch = []
    raw = kept = disc = 0
    is_first = (part_idx == 0)
    is_last = (part_idx == PARTS_PER_FILE - 1)

    with open(local_raw, "r", encoding="utf-8", errors="ignore") as f:
        first_line = True
        pending = None    # lookahead buffer for last-line trimming

        for line in f:
            if first_line:
                first_line = False
                if not is_first:
                    continue   # skip partial first line on non-first parts

            if pending is not None:
                current = pending
                pending = line
            else:
                pending = line
                continue

            current = current.strip()
            if not current:
                continue
            try:
                ex = json.loads(current)
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
            batch.append(rec)
            kept += 1

            if len(batch) >= BATCH_SIZE:
                if writer is None:
                    writer = pq.ParquetWriter(
                        out_path, pa.Table.from_pylist(batch).schema,
                        compression="zstd")
                writer.write_table(pa.Table.from_pylist(batch))
                batch.clear()
                gc.collect()

        # Handle the very last line if this is the last part
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
                            batch.append(rec)
                            kept += 1
                        else:
                            disc += 1
                    else:
                        disc += 1
                    raw += 1
                except json.JSONDecodeError:
                    pass

    # Final flush
    if batch:
        if writer is None:
            writer = pq.ParquetWriter(
                out_path, pa.Table.from_pylist(batch).schema,
                compression="zstd")
        writer.write_table(pa.Table.from_pylist(batch))
        batch.clear()
    if writer is not None:
        writer.close()

    os.remove(local_raw)
    gc.collect()
    print(f"  [part{part_idx}] raw={raw:,} kept={kept:,} disc={disc:,}")
    return {"part": part_idx, "raw": raw, "kept": kept, "disc": disc}


def main():
    file_id = int(os.environ["FILE_ID"])
    files = list_jsonl_files()

    if file_id >= len(files):
        print(f"File ID {file_id} out of range (max {len(files)-1}).")
        return

    target_file = files[file_id]
    print(f"Processing file {file_id}: {target_file}")

    total_bytes = get_file_size(target_file)
    print(f"Size: {total_bytes/1e9:.2f} GB → {PARTS_PER_FILE} parts")

    py_exes = []
    for f in ["/tmp/py311.txt", "/tmp/py313.txt", "/tmp/py314.txt"]:
        if os.path.exists(f):
            py_exes.append(open(f).read().strip())

    with ThreadPoolExecutor(max_workers=PARTS_PER_FILE) as executor:
        futures = [
            executor.submit(process_part, i, target_file, total_bytes,
                            py_exes, file_id)
            for i in range(PARTS_PER_FILE)
        ]
        total_kept = 0
        for future in futures:
            r = future.result()
            total_kept += r["kept"]
        print(f"Total kept: {total_kept:,}")


if __name__ == "__main__":
    main()
