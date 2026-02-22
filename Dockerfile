FROM python:3.13-slim

# uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ca-certificates \
    openssh-client \
    curl \
    wget \
    jq \
    sqlite3 \
    zip \
    unzip \
    ripgrep \
    tree \
    ffmpeg \
    vim-tiny \
    && rm -rf /var/lib/apt/lists/*

# Node.js LTS
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# yq
RUN arch="$(dpkg --print-architecture)" \
    && curl -fsSL "https://github.com/mikefarah/yq/releases/latest/download/yq_linux_${arch}" \
       -o /usr/local/bin/yq \
    && chmod +x /usr/local/bin/yq

WORKDIR /workspace
CMD ["sleep", "infinity"]
