FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

WORKDIR /home/
ARG DEBIAN_FRONTEND=noninteractive

# Force apt to use IPv4 and HTTPS (port 80 blocked in some corporate networks)
RUN echo 'Acquire::ForceIPv4 "true";' > /etc/apt/apt.conf.d/99force-ipv4
RUN sed -i 's|http://archive.ubuntu.com|https://mirror.kakao.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|https://mirror.kakao.com|g' /etc/apt/sources.list

# System dependencies (s2p-hd C build + general)
RUN apt-get update && apt-get install -y \
    vim cmake build-essential gdb git \
    libfftw3-dev libgeotiff-dev libtiff-dev libgdal-dev gdal-bin \
    libopencv-dev libpng-dev libjpeg-dev zlib1g-dev \
    python3-pip python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch + xformers (CUDA 12.1, must be installed together from same index)
RUN pip3 install --no-cache-dir \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    xformers==0.0.28.post1 \
    --index-url https://download.pytorch.org/whl/cu121

# Pin torch so other packages don't upgrade it
RUN echo "torch==2.4.1" > /tmp/constraints.txt && \
    echo "torchvision==0.19.1" >> /tmp/constraints.txt && \
    echo "torchaudio==2.4.1" >> /tmp/constraints.txt && \
    echo "xformers==0.0.28.post1" >> /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# DL stereo dependencies
RUN pip3 install --no-cache-dir \
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
    joblib \
    timm==1.0.15 \
    accelerate==1.0.1 \
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
    rpcm \
    cffi \
    fire \
    packaging \
    opt_einsum \
    trimesh \
    transformations

# lightglue (not on PyPI)
RUN pip3 install --no-cache-dir \
    "lightglue @ git+https://github.com/cvg/LightGlue.git"

# mmcv prebuilt wheel
RUN pip3 install --no-cache-dir \
    mmcv==2.2.0 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html

# flash-attn (speeds up MonSter/FoundationStereo)
# Pin version compatible with torch 2.4.1 + CUDA 12.x
RUN pip3 install --no-cache-dir --no-build-isolation flash-attn==2.6.3

# s2p-hd dependencies
RUN pip3 install --no-cache-dir \
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
    geojson

# Copy and install s2p-hd
COPY ./ /home/s2p-hd/
RUN cd /home/s2p-hd/ && pip3 install -e .

# EGM2008 geoid grid
RUN python3 -c "import pyproj, urllib.request; urllib.request.urlretrieve('https://cdn.proj.org/us_nga_egm08_25.tif', pyproj.datadir.get_data_dir() + '/us_nga_egm08_25.tif')"

RUN useradd -u 1000 user
RUN chown -R 1000:1000 /home/
USER user
