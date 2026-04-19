#!/bin/bash

# S2P-HD + DL Stereo Matcher Docker Container
#
# Usage:
#   ./using_docker.sh              # build (with cache) + enter
#   ./using_docker.sh --no-cache   # rebuild from scratch
#   ./using_docker.sh --no-build   # skip build, enter existing image

DOCKER_IMAGE="s2p-hd-dl:latest"

# Directory configuration
dir_data=/data/kevin_workspace/dataset_stereo
dir_pretrained=/data/kevin_workspace/pretrained_model
# dir_diachronic no longer needed (thirdparty/ bundled in s2p-hd)

# Log file (same folder as script, overwritten each run)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="${SCRIPT_DIR}/using_docker.log"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Flush function: write to both console and log
log() {
    echo -e "$1" | tee -a "$LOGFILE"
}

# Clear log file
> "$LOGFILE"
log ""
log "========== $(date '+%Y-%m-%d %H:%M:%S') =========="

# Parse arguments
NO_CACHE=false
NO_BUILD=false
for arg in "$@"; do
    case $arg in
        --no-cache) NO_CACHE=true ;;
        --no-build) NO_BUILD=true ;;
    esac
done
if [ "$NO_CACHE" = true ] && [ "$NO_BUILD" = true ]; then
    echo "Error: --no-cache and --no-build are mutually exclusive" >&2
    exit 1
fi

log "${GREEN}==================================================${NC}"
log "${GREEN}  S2P-HD + DL Stereo Matcher Docker${NC}"
log "${GREEN}==================================================${NC}"

# ============================================================
# Check and download pretrained models if missing
# ============================================================
log "${YELLOW}Checking pretrained models...${NC}"
mkdir -p "${dir_pretrained}/diachronic-stereo"
mkdir -p "${dir_pretrained}/monster"
mkdir -p "${dir_pretrained}/foundationstereo/23-51-11"
mkdir -p "${dir_pretrained}/foundationstereo/11-33-40"
mkdir -p "${dir_pretrained}/stereoanywhere"
mkdir -p "${dir_pretrained}/Depth-Anything-V2-Large"

# download_hf: for checkpoints with a direct HTTP-fetchable URL (e.g. HuggingFace
# resolve links). Only pass URLs that are verified to return the actual file
# over wget — never a Google Drive folder / HTML gateway.
download_hf() {
    local url="$1" dst="$2" name="$3"
    if [ -f "$dst" ] && [ -s "$dst" ]; then
        log "  ${name}: already exists"
    else
        [ -f "$dst" ] && rm -f "$dst"
        log "  ${name}: downloading..."
        wget -nv --show-progress -O "$dst" "$url" 2>&1 | tee -a "$LOGFILE"
        local rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ] || [ ! -s "$dst" ]; then
            log "${RED}  ${name}: download failed (exit=$rc, size=$(stat -c%s "$dst" 2>/dev/null || echo 0))${NC}"
            log "${RED}    URL: ${url}${NC}"
            rm -f "$dst"
            return 1
        else
            log "  ${name}: done ($(du -h "$dst" | cut -f1))"
        fi
    fi
}

# ensure_gdown: install gdown on host (system pip, running under sudo) if missing.
ensure_gdown() {
    command -v gdown &>/dev/null && return 0
    log "${YELLOW}  Installing gdown (one-time)...${NC}"
    pip3 install --quiet gdown 2>&1 | tee -a "$LOGFILE"
    command -v gdown &>/dev/null
}

# check_gdrive: download a specific file from Google Drive via gdown.
# Use this for checkpoints hosted on Google Drive (virus-scan gated, not wget'able).
#   $1 dst           — final file path
#   $2 name          — display name
#   $3 file_id       — Google Drive file ID (NOT folder ID)
#   $4 fallback_url  — manual download URL shown if gdown fails
check_gdrive() {
    local dst="$1" name="$2" file_id="$3" fallback_url="$4"
    if [ -f "$dst" ] && [ -s "$dst" ]; then
        log "  ${name}: already exists"
        return 0
    fi
    if ! ensure_gdown; then
        log "${RED}  ${name}: gdown unavailable. Download manually from:${NC}"
        [ -n "$fallback_url" ] && log "    ${fallback_url}"
        return 1
    fi
    [ -f "$dst" ] && rm -f "$dst"
    mkdir -p "$(dirname "$dst")"
    log "  ${name}: downloading from Google Drive..."
    gdown --id "$file_id" -O "$dst" 2>&1 | tee -a "$LOGFILE"
    local rc=${PIPESTATUS[0]}
    if [ $rc -ne 0 ] || [ ! -s "$dst" ]; then
        log "${RED}  ${name}: gdown failed (exit=$rc). Download manually from:${NC}"
        [ -n "$fallback_url" ] && log "    ${fallback_url}"
        rm -f "$dst"
        return 1
    fi
    log "  ${name}: done ($(du -h "$dst" | cut -f1))"
}

# Diachronic MonSter (satellite fine-tuned)
download_hf \
    "https://huggingface.co/emasquil/diachronic-stereo/resolve/main/final.pth" \
    "${dir_pretrained}/diachronic-stereo/final.pth" \
    "Diachronic MonSter"

# MonSter original
download_hf \
    "https://huggingface.co/cjd24/MonSter/resolve/main/mix_all.pth" \
    "${dir_pretrained}/monster/mix_all.pth" \
    "MonSter (mix_all)"

# Depth Anything V2 Large
download_hf \
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth" \
    "${dir_pretrained}/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth" \
    "Depth Anything V2 Large"

# StereoAnywhere (CC BY-NC-SA 4.0, Univ. of Bologna)
# Folder: https://drive.google.com/drive/folders/1uQqNJo2iWoPtXlSsv2koAt2OPYHpuh1x
check_gdrive \
    "${dir_pretrained}/stereoanywhere/stereoanywhere_sceneflow.pth" \
    "StereoAnywhere (sceneflow)" \
    "11jYAFvSXNwaePwvAmrAGDJkluhqaP79J" \
    "https://drive.google.com/drive/folders/1uQqNJo2iWoPtXlSsv2koAt2OPYHpuh1x"

# FoundationStereo 23-51-11 (NVIDIA non-commercial license)
# Folder: https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf
check_gdrive \
    "${dir_pretrained}/foundationstereo/23-51-11/model_best_bp2.pth" \
    "FoundationStereo 23-51-11 model" \
    "1Yh_2o9QCUrVqZrnAXZ7RUr0zTp3JrMKe" \
    "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"

check_gdrive \
    "${dir_pretrained}/foundationstereo/23-51-11/cfg.yaml" \
    "FoundationStereo 23-51-11 cfg" \
    "1tidGICH1_kTUUqi42aboKscuMY4IK_Xr" \
    "https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf"

log "${GREEN}Pretrained models check done${NC}"

# Build (unless --no-build)
if [ "$NO_BUILD" = true ]; then
    if ! sudo docker image inspect ${DOCKER_IMAGE} &>/dev/null; then
        log "${RED}--no-build set but image ${DOCKER_IMAGE} not found. Drop --no-build or build first.${NC}"
        exit 1
    fi
    log "${YELLOW}Skipping build (--no-build), using existing image ${DOCKER_IMAGE}${NC}"
else
    if [ ! -f "${SCRIPT_DIR}/Dockerfile" ]; then
        log "${RED}Dockerfile not found. Run from s2p-hd repo root.${NC}"
        exit 1
    fi

    BUILD_FLAGS=""
    if [ "$NO_CACHE" = true ]; then
        BUILD_FLAGS="--no-cache"
        log "${YELLOW}Building (no cache)...${NC}"
    else
        log "${YELLOW}Building...${NC}"
    fi

    cd "${SCRIPT_DIR}"
    sudo docker build $BUILD_FLAGS -f Dockerfile -t ${DOCKER_IMAGE} . 2>&1 | while IFS= read -r line; do
        echo "$line"
        echo "$line" | sed 's/\x1b\[[0-9;]*m//g' >> "$LOGFILE"
    done

    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        log "${RED}Docker build failed${NC}"
        exit 1
    fi
    log "${GREEN}Build complete${NC}"
fi

log "${YELLOW}Mounts:${NC}"
log "  Data:          ${dir_data} -> /data"
log "  Pretrained:    ${dir_pretrained} -> /pretrained"
log "  s2p-hd:        ${SCRIPT_DIR} -> /workspace"
log ""
log "${GREEN}Entering container...${NC}"
log "${GREEN}==================================================${NC}"
log ""

# Run
sudo docker run --rm -it \
    --shm-size=64g \
    --gpus all \
    --user root \
    --net=host \
    -w /workspace \
    -v ${SCRIPT_DIR}:/workspace \
    -v ${dir_data}:/data \
    -v ${dir_pretrained}:/pretrained \
    ${DOCKER_IMAGE} \
    bash
