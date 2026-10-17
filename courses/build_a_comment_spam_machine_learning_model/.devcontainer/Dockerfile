# Pulls directly from Docker Hub (Official Nvidia Repository)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Stop interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install fundamental system prerequisites and software-properties-common for PPAs
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    sudo \
    curl \
    build-essential \
    git \
    libusb-1.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Add the deadsnakes PPA repository to fetch older Python versions (like 3.9) on Ubuntu 22.04
RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update

# Install Python 3.9, its developer flags, and the clean pip module bundle
RUN apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# CORRECTED: Targets the raw script file, NOT the organizational homepage HTML
RUN curl https://bootstrap.pypa.io/pip/3.9/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    rm get-pip.py

# Explicitly link Python 3.9 as the global default commands inside this container
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3

# Force upgrade pip to your exact requested version milestone
RUN python -m pip install --no-cache-dir pip==23.3.2

# Install project dependencies system-wide (copied alone so code edits don't bust this layer)
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Non-root user matching the host user's UID/GID, so files written to the
# bind-mounted workspace stay owned by the host user instead of root
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    mkdir -p /home/$USERNAME/.claude && \
    chown $USER_UID:$USER_GID /home/$USERNAME/.claude

