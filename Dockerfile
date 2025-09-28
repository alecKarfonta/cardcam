# Trading Card Segmentation Development Environment
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# PyTorch is already installed in the base image with CUDA support
# RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Detectron2
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# Install additional ML libraries
RUN pip3 install \
    ultralytics \
    segment-anything \
    albumentations \
    opencv-python-headless \
    scikit-image \
    matplotlib \
    seaborn \
    jupyter \
    mlflow \
    dvc \
    wandb

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/{raw,processed,annotations} \
    logs \
    models \
    checkpoints \
    outputs

# Set up Jupyter
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

# Expose ports
EXPOSE 8888 5000 6006

# Default command
CMD ["bash"]
