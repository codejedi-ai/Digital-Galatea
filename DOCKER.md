# Docker Deployment Guide

This guide explains how to deploy Digital Galatea using Docker.

## Quick Start

### Using Pre-built Image from Docker Hub

```bash
docker run -d \
  --name digital-galatea \
  -p 7860:7860 \
  -e DEEPSEEK_API_KEY=your_deepseek_api_key \
  -e INFLECTION_AI_API_KEY=your_inflection_api_key \
  codejedi/digital-galatea:latest
```

### Using Docker Compose

1. Create a `.env` file with your API keys (see `.env.example`)
2. Run:
   ```bash
   docker-compose up -d
   ```

## Building the Image

### Build Locally

```bash
docker build -t digital-galatea:latest .
```

### Build and Tag for Docker Hub

```bash
docker build -t codejedi/digital-galatea:latest .
```

## Publishing to Docker Hub

1. **Login to Docker Hub**
   ```bash
   docker login
   ```

2. **Tag the Image** (if not already tagged)
   ```bash
   docker tag digital-galatea:latest codejedi/digital-galatea:latest
   ```

3. **Push to Docker Hub**
   ```bash
   docker push codejedi/digital-galatea:latest
   ```

4. **Push with Version Tag** (optional)
   ```bash
   docker tag digital-galatea:latest codejedi/digital-galatea:v1.0.0
   docker push codejedi/digital-galatea:v1.0.0
   ```

## Environment Variables

### Required Variables

- `DEEPSEEK_API_KEY` - Your DeepSeek API key
- `INFLECTION_AI_API_KEY` - Your Inflection AI API key

### Optional Variables

- `AZURE_TEXT_ANALYTICS_KEY` - Azure Text Analytics key
- `AZURE_TEXT_ANALYTICS_ENDPOINT` - Azure endpoint URL
- `ANU_QUANTUM_API_KEY` - Quantum randomness API key
- `PORT` - Server port (default: 7860)

## Running the Container

### Basic Run

```bash
docker run -d \
  --name digital-galatea \
  -p 7860:7860 \
  -e DEEPSEEK_API_KEY=your_key \
  -e INFLECTION_AI_API_KEY=your_key \
  codejedi/digital-galatea:latest
```

### With Environment File

```bash
docker run -d \
  --name digital-galatea \
  -p 7860:7860 \
  --env-file .env \
  codejedi/digital-galatea:latest
```

### With Volume for Persistence

```bash
docker run -d \
  --name digital-galatea \
  -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  codejedi/digital-galatea:latest
```

## Container Management

### View Logs

```bash
docker logs digital-galatea
docker logs -f digital-galatea  # Follow logs
```

### Stop Container

```bash
docker stop digital-galatea
```

### Start Container

```bash
docker start digital-galatea
```

### Remove Container

```bash
docker rm digital-galatea
```

### Restart Container

```bash
docker restart digital-galatea
```

## Health Checks

The container includes a health check endpoint. You can verify it's running:

```bash
curl http://localhost:7860/health
```

## Troubleshooting

### Container Won't Start

1. Check logs: `docker logs digital-galatea`
2. Verify environment variables are set correctly
3. Ensure port 7860 is not already in use

### API Errors

1. Verify API keys are correct
2. Check network connectivity
3. Review application logs for specific error messages

### NLTK Data Download

NLTK data is downloaded automatically on first container start. This may take a few minutes. Check logs to see progress.

## Production Considerations

1. **Use Environment Files**: Store sensitive keys in `.env` files (not in version control)
2. **Resource Limits**: Set appropriate memory and CPU limits
3. **Restart Policy**: Use `--restart unless-stopped` for automatic restarts
4. **Health Checks**: Monitor the `/health` endpoint
5. **Logging**: Configure log rotation and monitoring

Example production run:

```bash
docker run -d \
  --name digital-galatea \
  --restart unless-stopped \
  -p 7860:7860 \
  --memory="2g" \
  --cpus="2" \
  --env-file .env \
  codejedi/digital-galatea:latest
```

