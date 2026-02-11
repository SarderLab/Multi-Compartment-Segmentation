# This Dockerfile is used to generate the docker image dsarchive/histomicstk
# This docker image includes the HistomicsTK python package along with its
# dependencies.
#
# All plugins of HistomicsTK should derive from this docker image

# Modern CUDA runtime base (recommended for Python 3.12 + current PyTorch)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
LABEL com.nvidia.volumes.needed="nvidia_driver"

LABEL maintainer="Anish Tatke - Sarder Lab. <anish.tatke@ufl.edu>"

RUN echo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! STARTING THE BUILD !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Remove bad repos
RUN rm -f\
    /etc/apt/sources.list.d/cuda.list

# System deps for building Python + your usual deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ca-certificates curl wget git unzip \
      build-essential make \
      libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
      libffi-dev liblzma-dev tk-dev \
      libncursesw5-dev xz-utils \
      libcurl4-openssl-dev libexpat1-dev libhdf5-dev \
      libxml2-dev libxslt1-dev \
      ffmpeg libsm6 libxext6 \
      libtool pkg-config autoconf automake cmake \
      libmemcached-dev memcached \
    && rm -rf /var/lib/apt/lists/*

# --- Install Python 3.12 from source ---
ENV PYTHON_VERSION=3.12.8
RUN curl -fsSLO https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar -xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --with-ensurepip=install && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd / && rm -rf Python-${PYTHON_VERSION} Python-${PYTHON_VERSION}.
    
# Expose Python 3.12 as default python/python3
RUN ln -sf /usr/local/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/local/bin/python3.12 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.12 /usr/bin/pip3 && \
    ln -sf /usr/local/bin/pip3.12 /usr/bin/pip

# Make python -> python3.12
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN which  python && \
    python --version

ENV build_path=/opt/build

RUN apt-get update && \
    apt-get install -y --no-install-recommends memcached && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENV mc_path=/opt/MultiC
RUN mkdir -p $mc_path

COPY . $mc_path/
WORKDIR $mc_path

RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

RUN python -m pip install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/detectron2.git"

RUN pip install --no-cache-dir tensorboard cmake onnx
RUN pip install --no-cache-dir --no-build-isolation .
RUN pip install --no-cache-dir "setuptools<71"

RUN python --version && pip --version && pip freeze

LABEL entry_path=$mc_path/multic/cli
WORKDIR $mc_path/multic/cli

# Test our entrypoint.  If we have incompatible versions of numpy and
# openslide, one of these will fail
RUN python -m slicer_cli_web.cli_list_entrypoint --list_cli
RUN python -m slicer_cli_web.cli_list_entrypoint MultiCompartmentSegment --help

ENTRYPOINT ["/bin/bash", "docker-entrypoint.sh"]