#!/bin/bash
set -euo pipefail

PROC_ID="${1}"

EOS_HOST="root://eosproject.cern.ch"
EOS_BASE="/eos/project/d/dragon/ashrivas/DarkHadronMassReco/training_data/NPZ_mpiEqMrho_1to250_refined_v1_constituentmatch_bestdh_multidh_noptcut"
EOS_SHARDS="${EOS_BASE}/shards"
EOS_OUT="${EOS_BASE}/merged_chunks"

CHUNK_SIZE=25
NCHUNKS=6

CAT_INDEX=$(( PROC_ID / NCHUNKS ))
CHUNK_ID=$(( PROC_ID % NCHUNKS ))

if [[ "${CAT_INDEX}" -eq 0 ]]; then
    CATEGORY="full"
    OUT_PREFIX="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_fully_matched"
    CAT_CODE=2
elif [[ "${CAT_INDEX}" -eq 1 ]]; then
    CATEGORY="partial"
    OUT_PREFIX="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_partially_matched"
    CAT_CODE=1
elif [[ "${CAT_INDEX}" -eq 2 ]]; then
    CATEGORY="none"
    OUT_PREFIX="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_unmatched"
    CAT_CODE=0
else
    echo "[ERROR] Bad PROC_ID=${PROC_ID}"
    exit 1
fi

OUT_NAME="${OUT_PREFIX}_chunk$(printf "%03d" "${CHUNK_ID}").npz"
EOS_OUT_FILE="${EOS_OUT}/${OUT_NAME}"

START=$(( CHUNK_ID * CHUNK_SIZE ))
STOP=$(( START + CHUNK_SIZE - 1 ))

echo "=========================================================="
echo "EOS merge chunk"
echo "PROC_ID    : ${PROC_ID}"
echo "CATEGORY   : ${CATEGORY}"
echo "CHUNK_ID   : ${CHUNK_ID}"
echo "SHARDS     : ${START} to ${STOP}"
echo "OUTPUT     : ${EOS_OUT_FILE}"
echo "HOST       : ${HOSTNAME}"
echo "PWD        : $(pwd)"
echo "DATE       : $(date)"
echo "=========================================================="

# LCG setup references unset variables, so disable nounset only while sourcing it.
set +u
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh
set -u

echo "[ENV] xrdcp: $(command -v xrdcp || echo missing)"
echo "[ENV] xrdfs: $(command -v xrdfs || echo missing)"
echo "[ENV] python3: $(command -v python3 || echo missing)"

python3 - <<'PY'
import numpy as np, zipfile
print("[PYTHON] numpy:", np.__version__)
PY

# Skip if EOS output already exists and is non-empty.
if xrdfs "${EOS_HOST}" stat "${EOS_OUT_FILE}" >/dev/null 2>&1; then
    echo "[SKIP] Output already exists on EOS:"
    echo "       ${EOS_OUT_FILE}"
    exit 0
fi

mkdir -p input tmp output

echo "[DOWNLOAD] Pulling shard NPZs from EOS"
for IDX in $(seq "${START}" "${STOP}"); do
    IDX3=$(printf "%03d" "${IDX}")
    IN_NAME="dragon_mpiEqMrho_1to250_refined_constituentmatch_bestdh_multidh_noptcut_shard_${IDX3}_${CATEGORY}.npz"
    EOS_IN_FILE="${EOS_SHARDS}/shard_${IDX3}/${IN_NAME}"
    LOCAL_FILE="input/${IN_NAME}"

    echo "  xrdcp ${EOS_IN_FILE}"
    xrdcp -f "${EOS_HOST}/${EOS_IN_FILE}" "${LOCAL_FILE}"

    if [[ ! -s "${LOCAL_FILE}" ]]; then
        echo "[ERROR] Download failed or empty: ${LOCAL_FILE}"
        exit 2
    fi
done

echo "[MERGE] Merging local downloaded shards"

python3 - <<PY
import re
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as fmt

CATEGORY = "${CATEGORY}"
CAT_CODE = int("${CAT_CODE}")
OUT_NAME = "${OUT_NAME}"

INPUT_DIR = Path("input")
TMP_DIR = Path("tmp")
OUT_DIR = Path("output")

OUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

files = sorted(INPUT_DIR.glob(f"*_{CATEGORY}.npz"))

if len(files) != ${CHUNK_SIZE}:
    raise RuntimeError(f"Expected ${CHUNK_SIZE} input files, found {len(files)}")

print("[INFO] Files:")
for f in files:
    print(" ", f)

def inspect_files(files):
    first = np.load(files[0], allow_pickle=False)
    keys = list(first.files)

    info = {}
    total_n = 0

    for f in files:
        d = np.load(f, allow_pickle=False)

        if set(d.files) != set(keys):
            raise RuntimeError(f"Key mismatch in {f}")

        n = None
        for k in keys:
            arr = d[k]
            if n is None:
                n = arr.shape[0]
            elif arr.shape[0] != n:
                raise RuntimeError(f"First-dim mismatch in {f}, key={k}")

            if k not in info:
                info[k] = {"dtype": arr.dtype, "tail_shape": arr.shape[1:]}
            else:
                if arr.dtype != info[k]["dtype"]:
                    raise RuntimeError(f"Dtype mismatch for key={k} in {f}")
                if arr.shape[1:] != info[k]["tail_shape"]:
                    raise RuntimeError(f"Shape mismatch for key={k} in {f}")

        total_n += n

    return keys, info, total_n

def write_streaming_npy(out_npy, files, key, final_shape, dtype):
    print(f"[WRITE NPY] {key:32s} shape={final_shape} dtype={dtype}")

    with open(out_npy, "wb") as fp:
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

            if (i + 1) % 5 == 0 or (i + 1) == len(files):
                print(f"    {i+1:3d}/{len(files)} files, rows={total_written}")

    if total_written != final_shape[0]:
        raise RuntimeError(f"{key}: wrote {total_written}, expected {final_shape[0]}")

def zip_npy_to_npz(npz_path, npy_paths):
    print(f"[ZIP] {npz_path}")
    with zipfile.ZipFile(
        npz_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=4,
        allowZip64=True,
    ) as zf:
        for key, npy_path in npy_paths.items():
            print(f"    adding {key}.npy")
            zf.write(npy_path, arcname=f"{key}.npy")

keys, info, total_n = inspect_files(files)
print("[INFO] total jets:", total_n)
print("[INFO] keys:", keys)

npy_paths = {}

for k in keys:
    final_shape = (total_n,) + info[k]["tail_shape"]
    dtype = info[k]["dtype"]
    npy_path = TMP_DIR / f"{k}.npy"
    write_streaming_npy(npy_path, files, k, final_shape, dtype)
    npy_paths[k] = npy_path

outpath = OUT_DIR / OUT_NAME
zip_npy_to_npz(outpath, npy_paths)

print("[VALIDATE]", outpath)
d = np.load(outpath, allow_pickle=False)
print("  X:", d["X"].shape, d["X"].dtype)
print("  N:", len(d["y"]))

cats = d["containment_category"].reshape(-1)
vals, counts = np.unique(cats, return_counts=True)
print("  containment_category:", dict(zip(vals.tolist(), counts.tolist())))

bad = int(np.sum(cats != CAT_CODE))
print("  bad category:", bad)
if bad != 0:
    raise RuntimeError(f"Bad category entries: {bad}")

if len(d["kinematics"]):
    pt = d["kinematics"][:, 0]
    print("  pT min/max:", float(pt.min()), float(pt.max()))

if len(d["masses"]):
    m = d["masses"].reshape(-1)
    print("  mass min/max:", float(m.min()), float(m.max()))

print("[DONE LOCAL]", outpath)
PY

echo "[UPLOAD] Uploading merged chunk to EOS"
xrdcp -f "output/${OUT_NAME}" "${EOS_HOST}/${EOS_OUT_FILE}"

echo "[VERIFY EOS]"
xrdfs "${EOS_HOST}" stat "${EOS_OUT_FILE}"

echo "[SUCCESS] ${OUT_NAME}"
