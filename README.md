---
title: Digital Galatea AI
emoji: 🎭
colorFrom: purple
colorTo: pink
sdk: docker
pinned: false
license: mit
---

# Digital Galatea AI

Digital Galatea is a conversational AI with a dynamic emotional model. It features a web-based interface where an avatar's shape and expression change in real-time to reflect the AI's feelings, which are influenced by the conversation.

## Features

- **Conversational AI**: Powered by DeepSeek and Inflection AI for natural and engaging conversations.
- **Dynamic Emotional Model**: Simulates five core emotions: Joy, Sadness, Anger, Fear, and Curiosity.
- **Responsive Avatar**: The AI's visual avatar changes its shape and facial expression based on its dominant emotion.
- **Sentiment Analysis**: Analyzes user input to dynamically update the AI's emotional state. It uses Azure Text Analytics for high accuracy when configured, with a seamless fallback to a local NLTK VADER model.
- **Real-time Web Interface**: Built with Flask and JavaScript, the interface polls for updates to keep the avatar and emotion bars synchronized with the AI's state.

## Tech Stack

- **Backend**: Python, Flask
- **AI & Machine Learning**:
  - DeepSeek API (Reasoning/Analysis)
  - Inflection AI (Response Generation)
  - Azure Cognitive Service for Language (Text Analytics)
  - NLTK (VADER)
- **Frontend**: HTML, CSS, JavaScript
- **Environment Management**: `python-dotenv`

## Quick Start with Docker Hub

The easiest way to run Digital Galatea is using the pre-built Docker image from Docker Hub:

```bash
docker pull codejediondockerhub/digital-galatea:latest
docker run -d \
  --name digital-galatea \
  -p 7860:7860 \
  -e DEEPSEEK_API_KEY=your_deepseek_api_key \
  -e INFLECTION_AI_API_KEY=your_inflection_api_key \
  -e AZURE_TEXT_ANALYTICS_KEY=your_azure_key \
  -e AZURE_TEXT_ANALYTICS_ENDPOINT=https://your-resource.cognitiveservices.azure.com/ \
  codejediondockerhub/digital-galatea:latest
```

Or using docker-compose (see `docker-compose.yml`):

```bash
docker-compose up -d
```

Access the application at `http://localhost:7860`

## Environment Variables

### Required Environment Variables

These environment variables **must** be set for the application to function:

| Variable | Description | Where to Get It |
|----------|-------------|-----------------|
| `DEEPSEEK_API_KEY` | Your DeepSeek API key for reasoning and analysis | [DeepSeek Platform](https://platform.deepseek.com/) |
| `INFLECTION_AI_API_KEY` | Your Inflection AI API key for response generation | [Inflection AI](https://inflection.ai/) |

### Optional Environment Variables

These environment variables enhance functionality but are not required:

| Variable | Description | Default Behavior |
|----------|-------------|------------------|
| `AZURE_TEXT_ANALYTICS_KEY` | Azure Text Analytics API key for enhanced sentiment analysis | Falls back to NLTK VADER if not provided |
| `AZURE_TEXT_ANALYTICS_ENDPOINT` | Azure Text Analytics endpoint URL | Required if `AZURE_TEXT_ANALYTICS_KEY` is set |
| `ANU_QUANTUM_API_KEY` | ANU Quantum Random Numbers API key | Uses pseudo-random numbers if not provided |
| `PORT` | Port number for the Flask server | Defaults to `7860` |

### Environment Variable Examples

#### Using Docker Run

```bash
docker run -d \
  --name digital-galatea \
  -p 7860:7860 \
  -e DEEPSEEK_API_KEY=sk-your-deepseek-key \
  -e INFLECTION_AI_API_KEY=your-inflection-key \
  -e AZURE_TEXT_ANALYTICS_KEY=your-azure-key \
  -e AZURE_TEXT_ANALYTICS_ENDPOINT=https://your-resource.cognitiveservices.azure.com/ \
  codejediondockerhub/digital-galatea:latest
```

#### Using Environment File

Create a `.env` file:

```properties
# Required
DEEPSEEK_API_KEY=sk-your-deepseek-key
INFLECTION_AI_API_KEY=your-inflection-key

# Optional - Enhanced Sentiment Analysis
AZURE_TEXT_ANALYTICS_KEY=your-azure-key
AZURE_TEXT_ANALYTICS_ENDPOINT=https://your-resource.cognitiveservices.azure.com/

# Optional - Quantum Randomness
ANU_QUANTUM_API_KEY=your-quantum-key

# Optional - Port Configuration
PORT=7860
```

Then run with:

```bash
docker run -d --name digital-galatea -p 7860:7860 --env-file .env codejedi/digital-galatea:latest
```

## Docker Hub

The image is available on Docker Hub:

**Image**: `codejediondockerhub/digital-galatea:latest`

```bash
docker pull codejediondockerhub/digital-galatea:latest
```

### Building and Pushing to Docker Hub

If you want to build and push your own version:

```bash
# Build the image
docker build -t codejediondockerhub/digital-galatea:latest .

# Login to Docker Hub
docker login

# Push to Docker Hub
docker push codejediondockerhub/digital-galatea:latest
```

## Local Development

1. **Clone the Repository**
   ```bash
   git clone https://github.com/codejedi-ai/Digital-Galatea.git
   cd Digital-Galatea
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   # For Windows
   python -m venv venv
   .\venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**
   - Copy `.env.example` to `.env`
   - Add your API keys (see Environment Variables section above)

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **Access the Web Interface**
   - Open your browser and navigate to `http://127.0.0.1:7860`
   - The AI will initialize in the background. Once ready, you can start chatting.

## Deployment Options

### Heroku

The repository includes Heroku configuration files (`Procfile`, `app.json`, `runtime.txt`). 

1. Connect your GitHub repository to Heroku
2. Set environment variables in Heroku dashboard
3. Deploy automatically on push to main branch

### Hugging Face Spaces

1. Go to your Space Settings → Repository secrets
2. Add the required secrets (see Environment Variables section)
3. The app will build and deploy automatically

## API Endpoints

The application exposes the following endpoints:

- `GET /`: Serves the main chat interface.
- `POST /api/chat`: Handles chat messages and returns the AI's response.
- `GET /api/avatar`: Provides the current avatar shape, emotions, and sentiment for real-time frontend updates.
- `GET /api/is_initialized`: Check if the system is initialized and ready.
- `GET /status`: Reports the initialization status of the AI components.
- `GET /health`: A simple health check endpoint.

## How It Works

1. **User Input**: You type a message in the chat interface
2. **Sentiment Analysis**: The system analyzes the emotional tone of your message using Azure Text Analytics or NLTK VADER
3. **Emotional Processing**: Galatea's emotional state is updated based on the conversation
4. **AI Response**: DeepSeek analyzes the context, and Inflection AI generates a contextual response that reflects Galatea's emotional state
5. **Avatar Update**: The visual avatar changes shape and expression based on the dominant emotion:
   - **Circle** → High Joy (happy, positive state)
   - **Triangle** → High Sadness (melancholic state)
   - **Square** → Neutral or mixed emotions

## Credits

This project demonstrates the integration of:
- DeepSeek API for reasoning and analysis
- Inflection AI for response generation
- Azure Text Analytics and NLTK for sentiment analysis
- Real-time emotional modeling
- Dynamic visual representation of AI state

---

Made with ❤️ for exploring emotional AI
