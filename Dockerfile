#FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
#FROM nvidia/cuda:13.1.0-runtime-ubuntu22.04
#FROM ubuntu:22.04


# Update and install mini-conda
RUN apt-get update && apt-get install -y time wget g++ libboost-all-dev git && \
        apt-get clean

# Install Mamba
ENV CONDA_DIR /opt/conda
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && /bin/bash ~/miniforge.sh -b -p /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# copy into DreaMS
COPY . /DreaMS

RUN mamba create -n dreams_env python==3.11.0 --yes
WORKDIR /DreaMS
RUN bash -c "source activate dreams_env && cd DreaMS && pip install -e ."

# pip install chardet
RUN bash -c "source activate dreams_env && pip install chardet"

