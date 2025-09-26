FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    golang-go \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

CMD ["/bin/bash"]