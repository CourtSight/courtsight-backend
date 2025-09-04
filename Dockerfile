# --------- Builder Stage ---------
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

# Set environment variables for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

WORKDIR /app

# Copy dependency files first (for better layer caching)
COPY uv.lock pyproject.toml ./

# Remove any existing virtual environment that might conflict
RUN rm -rf .venv

# Install dependencies first (for better layer caching)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project

# Copy the project source code
COPY . /app

# Install the project in non-editable mode
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-editable

# --------- Final Stage ---------
FROM python:3.11-slim-bookworm

# Install system dependencies for PostgreSQL and other libraries
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --shell /bin/bash --create-home app

# Copy the virtual environment from the builder stage
COPY --from=builder --chown=app:app /app/.venv /app/.venv

# Ensure the virtual environment is in the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Disable uvloop to avoid conflicts with nest_asyncio
ENV USE_UVLOOP=false

# Switch to the non-root user
USER app

# Set the working directory
WORKDIR /code

# Add the app directory to Python path
ENV PYTHONPATH=/code:/code/app

# -------- replace with comment to run with gunicorn --------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
# CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000"]
