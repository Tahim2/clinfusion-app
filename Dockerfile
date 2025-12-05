# Hugging Face Spaces Docker template for Streamlit
# Uses Python image, installs requirements, and runs app.py

FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system deps needed for timm/torchvision (Pillow etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py /app/app.py

# Spaces expect the app to listen on port 7860 or 8501; Streamlit defaults to 8501
ENV PORT=8501
EXPOSE 8501

# Streamlit config to run headless inside container
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=${PORT} \
    STREAMLIT_SERVER_ENABLECORS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py"]