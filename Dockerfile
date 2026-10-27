# Diligence Lens — Streamlit app image
# Pair with docker-compose.yml (Qdrant service + QDRANT_URL).
# Do not bake GEMINI_API_KEY into the image.
# Embeddings: host Ollama via OLLAMA_HOST (Compose sets host.docker.internal).

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU torch only — MiniLM rerank is light; CUDA wheel (~555MB+) blows Windows builds.
# Host Ollama still does GPU embeds via host.docker.internal.
RUN pip install --no-cache-dir --timeout 120 \
    torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --timeout 120 -r requirements.txt

# App + retrieval pipeline modules (hybrid / rerank / diversity / stores)
COPY portal.py approval_portal.py approval.py risk_graph.py ./
COPY retrieval.py chunking.py hybrid.py lexical_retrieval.py ./
COPY reranking.py diversity.py qdrant_store.py semantic_cache.py ./
COPY .streamlit ./.streamlit
COPY docs ./docs
COPY NOTICE LICENSE README.md ./

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Overridden by Compose to host.docker.internal / qdrant service
ENV OLLAMA_HOST=http://127.0.0.1:11434
# Part 9 review defaults (same as local)
ENV GEMINI_MODEL=gemini-3.5-flash-lite
ENV REVIEW_RERANK_THRESHOLD=4.0
ENV REVIEW_EVIDENCE_LIMIT=8

EXPOSE 8501

CMD ["streamlit", "run", "portal.py", "--server.port=8501", "--server.address=0.0.0.0"]
