# Dockerfile

# --- Stage 1: Builder ---
# This stage installs all dependencies into a virtual environment.
FROM python:3.9-slim AS builder

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# Ensure Python output is sent straight to the terminal
ENV PYTHONUNBUFFERED 1

# Install Poetry to manage dependencies
RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt


# --- Stage 2: Final Image ---
# This stage builds the final, lean image for production.
FROM python:3.9-slim

# --- FIX: Install system dependencies required by LightGBM ---
# libgomp1 provides the libgomp.so.1 shared library for OpenMP support.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /app/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy the application code
COPY ./src ./src
COPY ./api.py .
COPY ./app.py .

# Expose ports for FastAPI and Streamlit
EXPOSE 8000
EXPOSE 8501