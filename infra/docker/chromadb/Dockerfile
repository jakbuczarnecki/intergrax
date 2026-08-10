FROM chromadb/chroma:1.4.1

USER root
RUN apt-get update \
    && apt-get install --no-install-recommends --yes curl \
    && rm -rf /var/lib/apt/lists/*
