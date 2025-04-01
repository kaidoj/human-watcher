FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    wget \
    gnupg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Playwright dependencies (needed for browser control if used)
RUN pip install playwright && python -m playwright install-deps && python -m playwright install

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create static directory for placeholder
RUN mkdir -p static

# Copy application code
COPY live_human_detector.py .
COPY README.md .

# Set environment variable to avoid CUDA errors if not available
ENV PYTHONUNBUFFERED=1

# Expose port for web interface
EXPOSE 5005

# Command to run when container starts
CMD ["python", "live_human_detector.py"]
