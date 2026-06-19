# Portable image for the ASTraM congestion dashboard.
# Build:  docker build -t astram-congestion .
# Run:    docker run -p 8501:8501 astram-congestion   ->  http://localhost:8501
FROM python:3.12-slim

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + data (the CSV is needed to train the model bundle)
COPY . .

# Pre-train the models into the image so the container starts fast
RUN python build_app_data.py

EXPOSE 8501
# Respect $PORT if the platform sets one (Cloud Run/Render), else default 8501
CMD streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0
