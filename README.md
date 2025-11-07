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

- **Conversational AI**: Powered by the Google Gemini API for natural and engaging conversations.
- **Dynamic Emotional Model**: Simulates five core emotions: Joy, Sadness, Anger, Fear, and Curiosity.
- **Responsive Avatar**: The AI's visual avatar changes its shape and facial expression based on its dominant emotion.
- **Sentiment Analysis**: Analyzes user input to dynamically update the AI's emotional state. It uses Azure Text Analytics for high accuracy when configured, with a seamless fallback to a local NLTK VADER model.
- **Real-time Web Interface**: Built with Flask and JavaScript, the interface polls for updates to keep the avatar and emotion bars synchronized with the AI's state.

## Tech Stack

- **Backend**: Python, Flask
- **AI & Machine Learning**:
  - Google Gemini API
  - Azure Cognitive Service for Language (Text Analytics)
  - NLTK (VADER)
- **Frontend**: HTML, CSS, JavaScript
- **Environment Management**: `python-dotenv`

## Required Setup for Hugging Face Spaces

To run this app on Hugging Face Spaces, you need to set the following **Secret** in your Space settings:

### Required Secret
1. Go to your Space Settings → Repository secrets
2. Add the following secret:
   - **Name**: `GEMINI_API_KEY`
   - **Value**: Your Google Gemini API key from [Google AI Studio](https://ai.google.dev/)

### Optional Secrets (for enhanced sentiment analysis)
- `AZURE_TEXT_ANALYTICS_KEY`: Your Azure Text Analytics key
- `AZURE_TEXT_ANALYTICS_ENDPOINT`: Your Azure Text Analytics endpoint

If Azure credentials are not provided, the app will automatically use the built-in NLTK VADER sentiment analyzer.

## Local Development

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
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
   - Add your Google Gemini API key:
   ```properties
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

5. **Run the Application**
   ```bash
   python app.py
   ```

6. **Access the Web Interface**
   - Open your browser and navigate to `http://127.0.0.1:7860`
   - The AI will initialize in the background. Once ready, you can start chatting.

## API Endpoints

The application exposes the following endpoints:

- `GET /`: Serves the main chat interface.
- `POST /api/chat`: Handles chat messages and returns the AI's response.
- `GET /api/avatar`: Provides the current avatar shape, emotions, and sentiment for real-time frontend updates.
- `GET /status`: Reports the initialization status of the AI components.
- `GET /health`: A simple health check endpoint.

## How It Works

1. **User Input**: You type a message in the chat interface
2. **Sentiment Analysis**: The system analyzes the emotional tone of your message
3. **Emotional Processing**: Galatea's emotional state is updated based on the conversation
4. **AI Response**: Google Gemini generates a contextual response that reflects Galatea's emotional state
5. **Avatar Update**: The visual avatar changes shape and expression based on the dominant emotion:
   - **Circle** → High Joy (happy, positive state)
   - **Triangle** → High Sadness (melancholic state)
   - **Square** → Neutral or mixed emotions

## Credits

This project demonstrates the integration of:
- Google's Gemini API for conversational AI
- NLTK and Azure for sentiment analysis
- Real-time emotional modeling
- Dynamic visual representation of AI state

---

Made with ❤️ for exploring emotional AI
