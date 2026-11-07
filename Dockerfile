FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "pytest", "-q"]
