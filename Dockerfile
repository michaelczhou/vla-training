# VLA Training Framework - Docker Image
# 基于 PyTorch 的官方镜像

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    vim \
    htop \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /workspace/vla-training/

# 安装 Python 依赖
WORKDIR /workspace/vla-training
RUN pip install --no-cache-dir -e .

# 安装可选依赖
RUN pip install --no-cache-dir \
    jupyter \
    ipywidgets \
    matplotlib \
    tensorboard \
    wandb

# 设置环境变量
ENV PYTHONPATH=/workspace/vla-training:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0

# 暴露 Jupyter 端口
EXPOSE 8888

# 暴露 TensorBoard 端口
EXPOSE 6006

# 默认命令
CMD ["/bin/bash"]
