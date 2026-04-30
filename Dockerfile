FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir -e .

# Copy application code
COPY . .

# Create cache directory
RUN mkdir -p /tmp/pelgo_cache

# Expose port
EXPOSE 8000

# Default command (overridden by docker-compose)
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]