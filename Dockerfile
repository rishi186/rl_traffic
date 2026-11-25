FROM python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libatomic1 libexpat1 libgl1 libglib2.0-0 libx11-6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt eclipse-sumo==1.26.0 traci==1.26.0 \
    && sumo --version

# Copy project files
COPY . .

# Default command
CMD ["python", "scripts/train.py", "--config", "config.yaml"]
