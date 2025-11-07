# Galatea AI API Endpoints

This document describes all available API endpoints for the Galatea AI application.

## Base URL
- Local: `http://localhost:7860`
- Production: Configured via `PORT` environment variable (default: 7860)

## Endpoints

### 1. Home Page
**GET** `/`

Returns the main web interface HTML page.

**Response:**
- `200 OK`: HTML page rendered
- `500 Internal Server Error`: Template rendering error

---

### 2. Chat Endpoint
**POST** `/api/chat`

Sends a user message to Galatea AI and receives a response.

**Request Body:**
```json
{
  "message": "Hello, how are you?"
}
```

**Response (Success):**
```json
{
  "response": "I'm doing well, thank you for asking!",
  "avatar_shape": "Circle",
  "emotions": {
    "joy": 0.3,
    "sadness": 0.2,
    "anger": 0.1,
    "fear": 0.15,
    "curiosity": 0.25
  },
  "is_initialized": true
}
```

**Response (Not Initialized):**
```json
{
  "error": "System is not initialized yet. Please wait for initialization to complete.",
  "is_initialized": false,
  "status": "initializing"
}
```
- Status Code: `503 Service Unavailable`

**Response (Missing API Key):**
```json
{
  "error": "GEMINI_API_KEY is missing. Chat is unavailable.",
  "status": "missing_gemini_key",
  "is_initialized": false
}
```
- Status Code: `503 Service Unavailable`

**Response (No Message):**
```json
{
  "error": "No message provided"
}
```
- Status Code: `400 Bad Request`

**Notes:**
- The system must be fully initialized before chat requests are processed
- If Pi-3.1 model fails to generate a response, the application will exit immediately
- Emotional state is updated based on sentiment analysis of the user's message
- Avatar shape changes based on emotional state

---

### 3. Avatar Endpoint
**GET** `/api/avatar`

Retrieves the current avatar state and emotional information.

**Response:**
```json
{
  "avatar_shape": "Circle",
  "emotions": {
    "joy": 0.3,
    "sadness": 0.2,
    "anger": 0.1,
    "fear": 0.15,
    "curiosity": 0.25
  },
  "sentiment": {
    "sentiment": "positive",
    "positive": 0.85,
    "negative": 0.15
  },
  "is_initialized": true,
  "last_updated": "2025-11-07T18:00:00",
  "status": "ready"
}
```

**Avatar Shapes:**
- `Circle`: Default/neutral state
- `Triangle`: High energy/active emotions
- `Square`: Stable/grounded emotions

**Response (Error):**
```json
{
  "error": "Failed to get avatar information",
  "avatar_shape": "Circle",
  "status": "error"
}
```
- Status Code: `500 Internal Server Error`

---

### 4. Health Check
**GET** `/health`

Simple health check endpoint to verify the server is running.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-07T18:00:00"
}
```
- Status Code: `200 OK`

---

### 5. Availability Check
**GET** `/api/availability`

Checks if the system is available and ready to handle requests.

**Response (Available):**
```json
{
  "available": true,
  "is_initialized": true,
  "status": "ready"
}
```

**Response (Not Available):**
```json
{
  "available": false,
  "is_initialized": false,
  "status": "initializing",
  "message": "System is still initializing. Please wait."
}
```

---

### 6. Initialization Status
**GET** `/api/is_initialized`

Lightweight endpoint for polling initialization progress (used by frontend).

**Response (Initialized):**
```json
{
  "is_initialized": true,
  "initializing": false,
  "missing_gemini_key": false
}
```

**Response (Initializing):**
```json
{
  "is_initialized": false,
  "initializing": true,
  "missing_gemini_key": false,
  "status": "initializing",
  "message": "Initializing components..."
}
```

**Response (Missing API Key):**
```json
{
  "is_initialized": false,
  "initializing": false,
  "missing_gemini_key": true,
  "error_page": "/error",
  "status": "missing_api_key"
}
```

---

### 7. Status Endpoint
**GET** `/status`

Returns detailed status information about the system.

**Response:**
```json
{
  "is_initialized": true,
  "initializing": false,
  "emotions": {
    "joy": 0.3,
    "sadness": 0.2,
    "anger": 0.1,
    "fear": 0.15,
    "curiosity": 0.25
  },
  "avatar_shape": "Circle",
  "missing_gemini_key": false
}
```

---

### 8. Error Page
**GET** `/error`

Renders an informative error page when the app is unavailable.

**Response:**
- `200 OK`: HTML error page
- Displays information about missing API keys or initialization failures

---

## System Architecture

### Initialization Flow
1. **Parallel Initialization** - Runs before Flask app starts:
   - JSON Memory initialization
   - Sentiment Analyzer initialization
   - Gemini API validation
   - Inflection AI API validation
   - Quantum API validation (optional)

2. **Component Initialization** - After parallel init completes:
   - GalateaAI instance creation
   - DialogueEngine initialization
   - AvatarEngine initialization
   - Quantum Emotion Service startup (if API key available)

3. **Flask Server Start** - Only starts if all critical components are ready

### Critical Components
- **JSON Memory System**: Required
- **Sentiment Analyzer**: Required
- **Gemini API**: Required (but allows quota exceeded errors)
- **Inflection AI API**: Required

### Optional Components
- **Quantum API**: Optional (falls back to pseudo-random)

---

## Error Handling

### Rate Limits
- **Gemini API 429**: Treated as valid API key, initialization continues
- **Quantum API 429**: Falls back to pseudo-random, initialization continues

### Critical Failures
If any critical component fails to initialize, the application will:
1. Log detailed error information
2. Exit immediately with `sys.exit(1)`
3. No partial functionality is allowed

---

## Emotional State

Emotions are stored in `emotions.json` and persist across restarts. The system uses:
- **Sentiment Analysis**: Updates emotions based on user message sentiment
- **Quantum Influence**: Background service updates emotions every 10 seconds using quantum random numbers (if available)
- **Decay**: Emotions slowly decay over time (3% per interaction)

### Emotion Values
All emotions are normalized to values between 0.0 and 1.0:
- `joy`: Positive emotions
- `sadness`: Negative emotions
- `anger`: Aggressive emotions
- `fear`: Anxious emotions
- `curiosity`: Exploratory emotions

---

## Frontend Emotion Updates

The frontend updates emotions in two ways:

### 1. Immediate Update (After Chat Messages)
When a user sends a message via `POST /api/chat`, the response includes the current emotional state:
```json
{
  "response": "...",
  "emotions": {
    "joy": 0.3,
    "sadness": 0.2,
    "anger": 0.1,
    "fear": 0.15,
    "curiosity": 0.25
  }
}
```
The frontend immediately updates the emotion bars using `updateEmotionBars(data.emotions)`.

### 2. Continuous Polling (Every 1 Second)
The frontend polls `GET /api/avatar` every 1 second to get real-time emotion updates:
- **Polling Interval**: 1 second
- **Endpoint**: `GET /api/avatar`
- **Purpose**: Updates emotions even when the user is not chatting
- **Updates Include**:
  - Sentiment-based emotion changes from user messages
  - Quantum-influenced emotion changes (every 10 seconds)
  - Natural emotion decay over time

### Emotion Bar Update Function
The frontend uses the `updateEmotionBars()` function which:
- Takes emotion values (0.0 to 1.0 range)
- Converts to percentages (multiplies by 100)
- Updates the width of each emotion bar:
  - `joy-bar`
  - `sadness-bar`
  - `anger-bar`
  - `fear-bar`
  - `curiosity-bar`

### Real-Time Updates
This dual-update mechanism ensures:
- **Immediate feedback**: Emotions update right after each message
- **Continuous updates**: Emotions change in real-time even without user interaction
- **Smooth transitions**: CSS transitions (0.5s ease) provide smooth visual updates

---

## Notes

- All timestamps are in ISO 8601 format
- Emotional state values are rounded to 2 decimal places in responses
- The system requires all critical components to be ready before serving requests
- Debug mode is enabled by default (`debug=True`)
- Frontend polls `/api/avatar` every 1 second for real-time emotion updates

