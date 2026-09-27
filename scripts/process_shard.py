import os
import json
import subprocess
import warnings
from concurrent.futures import ThreadPoolExecutor
import requests
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import list_repo_files

warnings.filterwarnings("ignore", category=SyntaxWarning)

# --- Constants ---
REPO = "tokyotech-llm/swallow-code-v2"
PATH_PREFIX = "stage5-auto-format/python/medium"
OUTPUT_DIR = "/tmp/cleaned"
PARTS_PER_FILE = 3
TEXT_FIELDS = ["text", "path", "repo_name", "blob_id"]
CODE_FIELD = "text"
CHAR_MIN, CHAR_MAX = 50, 128_000

# --- Helper Functions ---
def list_jsonl_files():
    """Lists all JSONL files in the target directory of the HF repo."""
    all_files = list_repo_files(REPO, repo_type="dataset", token=os.environ["HF_TOKEN"])
    return sorted([f for f in all_files if f.startswith(PATH_PREFIX + "/") and f.endswith(".jsonl")])

def download_byte_range(file_path, start, end, out_path):
    """Downloads a specific byte range of a remote file."""
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{file_path}"
    headers = {"Range": f"bytes={start}-{end}", "Authorization": f"Bearer {os.environ['HF_TOKEN']}"}
    r = requests.get(url, headers=headers, stream=True, timeout=300)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            f.write(chunk)

def get_file_size(file_path):
    """Gets the total size of a remote file in bytes."""
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{file_path}"
    r = requests.head(url, allow_redirects=True, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"})
    return int(r.headers["Content-Length"])

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
    """Checks if code is valid Python under 3.11, 3.13, or 3.14."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ast.parse(code)
        return True
    except Exception:
        pass
    for exe in py_exes:
        try:
            p = subprocess.run([exe, "-W", "ignore", "-c", _AST_SCRIPT], input=json.dumps([code]), capture_output=True, text=True, timeout=60)
            if json.loads(p.stdout)[0]:
                return True
        except Exception:
            pass
    return False

def process_part(part_idx, file_path, total_bytes, py_exes):
    """Processes a single byte-range part of a file."""
    part_size = total_bytes // PARTS_PER_FILE
    start = part_idx * part_size
    end = (total_bytes - 1) if part_idx == PARTS_PER_FILE - 1 else (start + part_size - 1)

    local_raw = f"/tmp/raw_part_{part_idx}.jsonl"
    download_byte_range(file_path, start, end, local_raw)
    print(f"Part {part_idx}: Downloaded {os.path.getsize(local_raw):,} bytes")

    buf = []
    raw = kept = disc = 0
    is_first = (part_idx == 0)
    is_last = (part_idx == PARTS_PER_FILE - 1)

    with open(local_raw, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Discard partial lines at the boundaries
    if not is_first and lines:
        lines = lines[1:]
    if not is_last and lines:
        lines = lines[:-1]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ex = json.loads(line)
        except json.JSONDecodeError:
            continue
        raw += 1

        text = " ".join(str(ex.get(f, "") or "") for f in TEXT_FIELDS)
        if not (CHAR_MIN <= len(text) <= CHAR_MAX):
            disc += 1
            continue

        code = ex.get(CODE_FIELD, "") or ""
        if code and not ast_ok(code, py_exes):
            disc += 1
            continue

        rec = {f: ex.get(f) for f in TEXT_FIELDS}
        rec["_source"] = REPO
        buf.append(rec)
        kept += 1

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"file{file_id:03d}_part{part_idx}.parquet")
    if buf:
        pq.write_table(pa.Table.from_pylist(buf), out_path, compression="zstd")
        print(f"Part {part_idx}: Wrote {len(buf)} rows to {out_path}")

    os.remove(local_raw)
    return {"part": part_idx, "raw": raw, "kept": kept, "disc": disc}

# --- Main Execution ---
def main():
    file_id = int(os.environ["FILE_ID"])
    files = list_jsonl_files()
    
    if file_id >= len(files):
        print(f"File ID {file_id} is out of range. Max is {len(files) - 1}.")
        return

    target_file = files[file_id]
    print(f"Processing file {file_id}: {target_file}")

    total_bytes = get_file_size(target_file)
    
    # Load Python executables
    py_exes = []
    for f in ["/tmp/py311.txt", "/tmp/py313.txt", "/tmp/py314.txt"]:
        if os.path.exists(f):
            py_exes.append(open(f).read().strip())

    # Process parts in parallel
    with ThreadPoolExecutor(max_workers=PARTS_PER_FILE) as executor:
        futures = [executor.submit(process_part, i, target_file, total_bytes, py_exes) for i in range(PARTS_PER_FILE)]
        for future in futures:
            result = future.result()
            print(f"Result: {result}")

if __name__ == "__main__":
    main()
