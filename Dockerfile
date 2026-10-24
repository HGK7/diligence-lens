# Diligence Lens — Streamlit app image
# Pair with docker-compose.yml (Qdrant service + QDRANT_URL).
# Do not bake GEMINI_API_KEY into the image.

FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY portal.py approval_portal.py approval.py retrieval.py risk_graph.py ./
COPY .streamlit ./.streamlit
COPY docs ./docs
COPY NOTICE LICENSE README.md ./

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Overridden by Compose to host.docker.internal / qdrant service
ENV OLLAMA_HOST=http://127.0.0.1:11434

EXPOSE 8501

CMD ["streamlit", "run", "portal.py", "--server.port=8501", "--server.address=0.0.0.0"]
