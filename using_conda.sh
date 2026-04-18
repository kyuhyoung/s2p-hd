#!/bin/bash

# S2P-HD + DL Stereo Matcher Conda Environment Setup Script
#
# Usage:
#   ./using_conda.sh            # create env from scratch
#   ./using_conda.sh -r         # activate existing env (fast)

set -e

# ============================================================
# Config
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_NAME=diachronicstereo
ENV_PATH=/raid/HDD/kevin_workspace/envs/${ENV_NAME}
S2P_HD_DIR=${SCRIPT_DIR}

PIP="${ENV_PATH}/bin/pip"
PYTHON="${ENV_PATH}/bin/python"

# Log file: same folder as this script, overwritten each run
LOGFILE="${SCRIPT_DIR}/using_conda.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Logging: tee to console + log file (strip ANSI for log)
exec > >(tee >(stdbuf -oL sed 's/\x1b\[[0-9;]*m//g' > "$LOGFILE")) 2>&1
echo ""
echo "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

# ============================================================
# Parse arguments
# ============================================================
REUSE=false
for arg in "$@"; do
    case $arg in
        -r) REUSE=true ;;
    esac
done

if ! command -v conda &>/dev/null; then
    echo -e "${RED}conda not found. Install Miniconda/Anaconda first.${NC}"
    exit 1
fi

eval "$(conda shell.bash hook)"

# ============================================================
# Fast reuse mode (-r)
# ============================================================
if [ "$REUSE" = true ]; then
    if [ -d "${ENV_PATH}" ]; then
        echo -e "${GREEN}Activating existing environment: ${ENV_PATH}${NC}"
        exec 1>/dev/tty 2>/dev/tty
        exec bash --rcfile <(echo "source ~/.bashrc; conda activate ${ENV_PATH}; echo -e '${GREEN}(${ENV_NAME}) environment active.${NC}'")
    else
        echo -e "${RED}Environment not found at ${ENV_PATH}. Run without -r first.${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  S2P-HD + DL Stereo Matcher Environment Setup${NC}"
echo -e "${GREEN}==================================================${NC}"

# ============================================================
# 1) System dependencies check
# ============================================================
echo -e "${YELLOW}[1/7] Checking system dependencies...${NC}"
MISSING=""
for pkg in build-essential cmake libtiff-dev libpng-dev libjpeg-dev zlib1g-dev gdal-bin libgdal-dev; do
    dpkg -s "$pkg" &>/dev/null || MISSING="$MISSING $pkg"
done

if [ -n "$MISSING" ]; then
    echo -e "${RED}Missing system packages:${MISSING}${NC}"
    echo -e "${YELLOW}Installing with apt-get...${NC}"
    sudo apt-get install -y ${MISSING}
fi

# fftw3 is needed for mgm_multi but not in the standard check list
if ! dpkg -s libfftw3-dev &>/dev/null; then
    echo -e "${YELLOW}Installing libfftw3-dev (needed for mgm_multi)...${NC}"
    sudo apt-get install -y libfftw3-dev
fi

echo -e "${GREEN}All system libraries present${NC}"

# ============================================================
# 2) Create conda environment (minimal: python + libstdcxx only)
#    Everything else via pip to avoid conda/pip conflicts.
# ============================================================
echo -e "${YELLOW}[2/7] Creating conda environment: ${ENV_PATH}${NC}"
if [ -d "${ENV_PATH}" ]; then
    echo -e "${YELLOW}Environment already exists. Removing...${NC}"
    conda env remove -p ${ENV_PATH} -y
fi

mkdir -p "$(dirname ${ENV_PATH})"
conda create -p ${ENV_PATH} python=3.11 -y

# Install libstdcxx-ng (GLIBCXX_3.4.29+) and GDAL via conda
# GDAL must come from conda to avoid libffi conflict between
# conda's libffi.so.8 (needed by python) and system's libffi.so.7 (pulled by system GDAL)
conda install -p ${ENV_PATH} -y -c conda-forge libstdcxx-ng gdal libgdal

if [ ! -f "${PYTHON}" ]; then
    echo -e "${RED}Failed to create conda environment. ${PYTHON} not found.${NC}"
    exit 1
fi
echo -e "${GREEN}Using python: ${PYTHON}${NC}"
echo -e "${GREEN}Using pip:    ${PIP}${NC}"

# ============================================================
# 3) Pin setuptools, then install PyTorch (CUDA 12.1)
# ============================================================
echo -e "${YELLOW}[3/7] Installing PyTorch (CUDA 12.1)...${NC}"
${PIP} install --no-cache-dir "setuptools<81"

${PIP} install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121

# ============================================================
# 4) Install all Python dependencies via pip only
# ============================================================
echo -e "${YELLOW}[4/7] Installing Python dependencies...${NC}"
${PIP} install --no-cache-dir \
    "numpy<2.0" \
    scipy \
    tqdm \
    matplotlib \
    scikit-image \
    scikit-learn \
    opencv-python-headless \
    numba \
    pillow \
    imageio \
    joblib

${PIP} install --no-cache-dir \
    timm==1.0.15 \
    accelerate==1.0.1 \
    xformers \
    albumentations \
    einops \
    omegaconf \
    pyyaml \
    ruamel.yaml \
    tensorboard \
    huggingface-hub \
    iio \
    kornia \
    open3d \
    hydra-core \
    rpcm

# lightglue (not on PyPI)
${PIP} install --no-cache-dir \
    "lightglue @ git+https://github.com/cvg/LightGlue.git"

# mmcv prebuilt wheel (2.2.0 has prebuilt for cu121/torch2.4/cp311; 2.1.0 does not)
echo -e "${YELLOW}Installing mmcv...${NC}"
${PIP} install --no-cache-dir \
    mmcv==2.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html

# flash-attn (optional)
echo -e "${YELLOW}Installing flash-attn (may take a few minutes to compile)...${NC}"
${PIP} install --no-cache-dir flash-attn || echo -e "${YELLOW}flash-attn install failed (non-fatal)${NC}"

# ============================================================
# 5) Install s2p-hd (editable, builds C binaries)
# ============================================================
echo -e "${YELLOW}[5/7] Installing s2p-hd...${NC}"
cd ${S2P_HD_DIR}

${PIP} install --no-cache-dir \
    "rasterio[s3]>=1.2a1" \
    utm \
    "pyproj>=3.0.0" \
    "beautifulsoup4[lxml]" \
    plyfile \
    "plyflatten @ git+https://github.com/centreborelli/plyflatten" \
    ransac \
    "rpcm>=1.4.6" \
    "srtm4>=1.1.2" \
    requests \
    cffi \
    geojson

${PIP} install --no-cache-dir -e .

# ============================================================
# 6) Verify C binaries built
# ============================================================
echo -e "${YELLOW}[6/7] Verifying s2p-hd C binaries...${NC}"
MISSING_BIN=""
for f in lib/disp_to_h.so lib/libhomography.so lib/libsift4ctypes.so bin/mgm bin/mgm_multi; do
    if [ ! -f "${S2P_HD_DIR}/$f" ]; then
        MISSING_BIN="$MISSING_BIN $f"
    fi
done

if [ -n "$MISSING_BIN" ]; then
    echo -e "${RED}Missing C binaries:${MISSING_BIN}${NC}"
    echo -e "${YELLOW}Building C binaries using conda GDAL...${NC}"
    # Use conda env's bin (for gdal-config) and lib (for linking)
    export PATH="${ENV_PATH}/bin:${PATH}"
    export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH}"
    export CFLAGS="-march=native -O3"
    export CXXFLAGS="-march=native -O3"
    make -C ${S2P_HD_DIR} clean 2>/dev/null
    make -C ${S2P_HD_DIR} || echo -e "${RED}make failed. Check system deps (libfftw3-dev, libtiff-dev, etc).${NC}"
else
    echo -e "${GREEN}All C binaries present${NC}"
fi

# ============================================================
# 7) Smoke test
# ============================================================
echo -e "${YELLOW}[7/7] Smoke test...${NC}"
${PYTHON} -c "
from s2p import config, dl_stereo
cfg = config.get_default_config()
assert cfg['dl_stereo_model'] == 'monster'
print('s2p config OK')
print('dl_stereo module OK')
import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
import numpy; print(f'NumPy {numpy.__version__}')
" && echo -e "${GREEN}Smoke test passed${NC}" || echo -e "${RED}Smoke test failed${NC}"

# ============================================================
# Done
# ============================================================
echo ""
echo -e "${GREEN}==================================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}==================================================${NC}"
echo -e "${YELLOW}To activate:  conda activate ${ENV_PATH}${NC}"
echo -e "${YELLOW}Or:           ./using_conda.sh -r${NC}"
echo -e "${GREEN}==================================================${NC}"
echo ""

# Drop into activated shell
exec 1>/dev/tty 2>/dev/tty
exec bash --rcfile <(echo "source ~/.bashrc; conda activate ${ENV_PATH}; echo -e '${GREEN}(${ENV_NAME}) environment active.${NC}'")
