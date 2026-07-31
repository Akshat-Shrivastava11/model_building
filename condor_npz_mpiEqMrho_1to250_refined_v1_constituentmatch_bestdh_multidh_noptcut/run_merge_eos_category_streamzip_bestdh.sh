#!/bin/bash
set -euo pipefail

PROC_ID="${1}"

EOS_HOST="root://eosproject.cern.ch"
EOS_BASE="/eos/project/d/dragon/ashrivas/DarkHadronMassReco/training_data/NPZ_mpiEqMrho_1to250_refined_v1_constituentmatch_bestdh_multidh_noptcut"
EOS_CHUNKS="${EOS_BASE}/merged_chunks"
EOS_OUT="${EOS_BASE}/merged_categories"

if [[ "${PROC_ID}" -eq 0 ]]; then
    CATEGORY="full"
    CHUNK_GLOB="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_fully_matched_chunk"
    OUT_NAME="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_fully_matched.npz"
    CAT_CODE=2
elif [[ "${PROC_ID}" -eq 1 ]]; then
    CATEGORY="partial"
    CHUNK_GLOB="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_partially_matched_chunk"
    OUT_NAME="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_partially_matched.npz"
    CAT_CODE=1
elif [[ "${PROC_ID}" -eq 2 ]]; then
    CATEGORY="none"
    CHUNK_GLOB="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_unmatched_chunk"
    OUT_NAME="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_unmatched.npz"
    CAT_CODE=0
else
    echo "[ERROR] Bad PROC_ID=${PROC_ID}"
    exit 1
fi

EOS_OUT_FILE="${EOS_OUT}/${OUT_NAME}"

echo "=========================================================="
echo "EOS stream-zip category merge"
echo "PROC_ID  : ${PROC_ID}"
echo "CATEGORY : ${CATEGORY}"
echo "OUTPUT   : ${EOS_OUT_FILE}"
echo "HOST     : ${HOSTNAME}"
echo "PWD      : $(pwd)"
echo "DATE     : $(date)"
echo "=========================================================="

set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh
set -u

echo "[ENV] python3: $(command -v python3)"
echo "[ENV] xrdcp  : $(command -v xrdcp)"
echo "[ENV] xrdfs  : $(command -v xrdfs)"

python3 - <<'PY'
import numpy as np
print("[PYTHON] numpy:", np.__version__)
PY

if xrdfs "${EOS_HOST}" stat "${EOS_OUT_FILE}" >/dev/null 2>&1; then
    echo "[SKIP] Output already exists on EOS:"
    echo "       ${EOS_OUT_FILE}"
    exit 0
fi

mkdir -p input output

echo "[DOWNLOAD] Downloading 6 merged chunks for ${CATEGORY}"

for I in 0 1 2 3 4 5; do
    IDX=$(printf "%03d" "${I}")
    CHUNK_NAME="${CHUNK_GLOB}${IDX}.npz"
    EOS_IN="${EOS_CHUNKS}/${CHUNK_NAME}"
    LOCAL_IN="input/${CHUNK_NAME}"

    echo "  ${EOS_IN}"
    xrdcp -f "${EOS_HOST}/${EOS_IN}" "${LOCAL_IN}"

    if [[ ! -s "${LOCAL_IN}" ]]; then
        echo "[ERROR] Missing or empty downloaded chunk: ${LOCAL_IN}"
        exit 2
    fi
done

echo "[MERGE] Streaming directly into final NPZ"

python3 - <<PY
import gc
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

CAT_CODE = int("${CAT_CODE}")
OUT_NAME = "${OUT_NAME}"

INPUT_DIR = Path("input")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

files = sorted(INPUT_DIR.glob("*.npz"))

if len(files) != 6:
    raise RuntimeError(f"Expected 6 chunks, found {len(files)}")

print("[INFO] Input chunks:")
for f in files:
    print(" ", f)

def read_npy_header_from_npz(npz_path, key):
    with zipfile.ZipFile(npz_path, "r") as zf:
        name = f"{key}.npy"
        if name not in zf.namelist():
            raise KeyError(f"Missing {name} in {npz_path}")

        with zf.open(name, "r") as fp:
            version = fmt.read_magic(fp)

            if version == (1, 0):
                shape, fortran_order, dtype = fmt.read_array_header_1_0(fp)
            elif version in [(2, 0), (3, 0)]:
                shape, fortran_order, dtype = fmt.read_array_header_2_0(fp)
            else:
                raise RuntimeError(f"Unsupported npy version {version} in {npz_path}:{name}")

    if fortran_order:
        raise RuntimeError(f"Fortran-order array not supported: {npz_path}:{key}")

    return tuple(shape), np.dtype(dtype)

def inspect_files(files):
    first = np.load(files[0], allow_pickle=False)
    keys = list(first.files)
    first.close()

    info = {}
    total_n = 0

    for f in files:
        n = None

        for k in keys:
            shape, dtype = read_npy_header_from_npz(f, k)

            if n is None:
                n = shape[0]
            elif shape[0] != n:
                raise RuntimeError(f"First-dimension mismatch in {f}, key={k}")

            if k not in info:
                info[k] = {
                    "dtype": dtype,
                    "tail_shape": shape[1:],
                }
            else:
                if dtype != info[k]["dtype"]:
                    raise RuntimeError(f"Dtype mismatch for key={k} in {f}")
                if shape[1:] != info[k]["tail_shape"]:
                    raise RuntimeError(f"Shape mismatch for key={k} in {f}")

        total_n += n

    return keys, info, total_n

def write_key_direct_to_zip(zf, key, files, final_shape, dtype):
    print(f"[WRITE ZIP ENTRY] {key:32s} shape={final_shape} dtype={dtype}")

    with zf.open(f"{key}.npy", mode="w", force_zip64=True) as fp:
        fmt.write_array_header_2_0(
            fp,
            {
                "descr": np.lib.format.dtype_to_descr(np.dtype(dtype)),
                "fortran_order": False,
                "shape": tuple(final_shape),
            },
        )

        total_written = 0

        for i, f in enumerate(files):
            d = np.load(f, allow_pickle=False)
            arr = np.asarray(d[key])

            if arr.dtype != dtype:
                arr = arr.astype(dtype, copy=False)

            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)

            if arr.size:
                fp.write(memoryview(arr).cast("B"))

            total_written += arr.shape[0]
            print(f"    {i+1}/6 chunks, rows={total_written}")

            d.close()
            del arr
            gc.collect()

    if total_written != final_shape[0]:
        raise RuntimeError(f"{key}: wrote {total_written}, expected {final_shape[0]}")

keys, info, total_n = inspect_files(files)

print("[INFO] Total jets:", total_n)
print("[INFO] Keys:", keys)

outpath = OUT_DIR / OUT_NAME

with zipfile.ZipFile(
    outpath,
    mode="w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=4,
    allowZip64=True,
) as zf:
    for k in keys:
        final_shape = (total_n,) + info[k]["tail_shape"]
        dtype = info[k]["dtype"]
        write_key_direct_to_zip(zf, k, files, final_shape, dtype)

print("[VALIDATE HEADERS]", outpath)

x_shape, x_dtype = read_npy_header_from_npz(outpath, "X")
print("  X header:", x_shape, x_dtype)

d = np.load(outpath, allow_pickle=False)

cats = d["containment_category"].reshape(-1)
vals, counts = np.unique(cats, return_counts=True)
print("  containment_category:", dict(zip(vals.tolist(), counts.tolist())))

bad = int(np.sum(cats != CAT_CODE))
print("  bad category:", bad)
if bad != 0:
    raise RuntimeError(f"Bad category entries: {bad}")

kin = d["kinematics"]
pt = kin[:, 0]
if len(pt):
    print("  pT min/max:", float(pt.min()), float(pt.max()))
else:
    print("  pT min/max: EMPTY")

m = d["masses"].reshape(-1)
if len(m):
    print("  mass min/max:", float(m.min()), float(m.max()))
else:
    print("  mass min/max: EMPTY")

d.close()

print("[DONE LOCAL]", outpath)
PY

echo "[LOCAL SIZE]"
ls -lh "output/${OUT_NAME}"
du -sh .

echo "[UPLOAD] Uploading final category file to EOS"
xrdcp -f "output/${OUT_NAME}" "${EOS_HOST}/${EOS_OUT_FILE}"

echo "[VERIFY EOS]"
xrdfs "${EOS_HOST}" stat "${EOS_OUT_FILE}"

echo "[SUCCESS] ${OUT_NAME}"
