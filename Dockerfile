FROM python:3.10-slim

# Install SUMO
RUN apt-get update && apt-get install -y --no-install-recommends \
    sumo sumo-tools \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV SUMO_HOME=/usr/share/sumo

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command
CMD ["python", "scripts/train.py", "--config", "config.yaml"]
