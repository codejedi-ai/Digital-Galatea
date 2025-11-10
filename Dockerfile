# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with retry logic
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Download NLTK data during build with error handling
RUN python -c "import nltk; import os; \
    nltk_data_dir = '/root/nltk_data'; \
    os.makedirs(nltk_data_dir, exist_ok=True); \
    nltk.data.path.append(nltk_data_dir); \
    try: \
        nltk.download('punkt', quiet=True); \
        print('Downloaded punkt'); \
    except Exception as e: \
        print(f'Failed to download punkt: {e}'); \
    try: \
        nltk.download('vader_lexicon', quiet=True); \
        print('Downloaded vader_lexicon'); \
    except Exception as e: \
        print(f'Failed to download vader_lexicon: {e}'); \
    print('NLTK download step completed')"

# Expose the port the app runs on
EXPOSE 7860

# Set environment variable for port
ENV PORT=7860

# Run the application
CMD ["python", "app.py"]
