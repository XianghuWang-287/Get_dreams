#FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
#FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04
#FROM ubuntu:22.04


# Update and install mini-conda
RUN apt-get update && apt-get install -y \
    time wget g++ libboost-all-dev git \
    libxrender1 libxext6 libsm6 libglib2.0-0 \
    && apt-get clean 

# Install Mamba
ENV CONDA_DIR /opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# copy into DreaMS
COPY ./DreaMS /DreaMS/DreaMS

RUN mamba create -n dreams_env python==3.11.0 --yes

WORKDIR /DreaMS/DreaMS
# Install DreaMS first (it will install numpy==1.25.0 and other dependencies)
RUN bash -c "source activate dreams_env && pip install -e ."

# pip install chardet
RUN bash -c "source activate dreams_env && pip install chardet"

RUN bash -c "source activate dreams_env && pip install 'setuptools<81.0'"

# Reinstall pandas/pyarrow via conda-forge to ensure binary compatibility with NumPy 1.25.0
# Explicitly pin NumPy to 1.25.0 in mamba install to prevent upgrade
RUN bash -c "source activate dreams_env && mamba install -c conda-forge 'numpy=1.25.0' pandas pyarrow --force-reinstall -y"

# Install FAISS-GPU via conda-forge (compatible with NumPy 1.25, unlike pip's faiss-gpu-cu12 which requires NumPy 2.x)
# Pin numpy=1.25.0 to prevent conda from upgrading it
RUN bash -c "source activate dreams_env && mamba install -c conda-forge faiss-gpu 'numpy=1.25.0' -y && pip install tqdm"

COPY . /DreaMS

RUN sed -i '1i import torch\nfrom functools import partial\ntorch.load = partial(torch.load, weights_only=False)' /DreaMS/DreaMS/dreams/api.py

WORKDIR /DreaMS
