# -------------------------------
# Base Image
# -------------------------------
FROM python:3.10-slim

# -------------------------------
# System Dependencies
# -------------------------------
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Set Working Directory
# -------------------------------
WORKDIR /app

# -------------------------------
# Copy Files
# -------------------------------
COPY . .

# -------------------------------
# Install Python Dependencies
# -------------------------------
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -------------------------------
# Environment Variables
# -------------------------------
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/tmp/torch_cache

# -------------------------------
# Expose Port (HF uses 7860)
# -------------------------------
EXPOSE 7860

# -------------------------------
# Run FastAPI
# -------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]