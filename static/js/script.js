document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM Content Loaded');

    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const avatar = document.getElementById('avatar');
    const statusContainer = document.getElementById('status-container');

    // Base API URL detection
    const DEFAULT_API_BASE = 'http://127.0.0.1:7860';
    const origin = window.location.origin;
    const API_BASE_URL = origin && origin !== 'null' ? origin : DEFAULT_API_BASE;
    const buildApiUrl = (path) => {
        const normalizedPath = path.startsWith('/') ? path : `/${path}`;
        return `${API_BASE_URL}${normalizedPath}`;
    };

    let initializationComplete = false;
    let initializationPollingActive = false;

    // Debug: Log if elements are found
    console.log('Elements found:', {
        chatMessages: !!chatMessages,
        userInput: !!userInput,
        sendButton: !!sendButton,
        avatar: !!avatar,
        statusContainer: !!statusContainer
    });

    // Emotion bars
    const joyBar = document.getElementById('joy-bar');
    const sadnessBar = document.getElementById('sadness-bar');
    const angerBar = document.getElementById('anger-bar');
    const fearBar = document.getElementById('fear-bar');
    const curiosityBar = document.getElementById('curiosity-bar');

    // Check server health on load
    checkServerHealth();

    // Determine availability state
    checkAvailability();

    // Auto-resize textarea as user types
    if (userInput) {
        userInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';

            // Reset height if empty
            if (this.value.length === 0) {
                this.style.height = '';
            }
        });

        // Send message when Enter key is pressed (without Shift)
        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                console.log('Enter key pressed, sending message');
                sendMessage();
            }
        });
    }

    // Add keyboard shortcut for sending messages
    document.addEventListener('keydown', function(e) {
        // Command+Enter or Ctrl+Enter to send
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            e.preventDefault();
            console.log('Cmd/Ctrl+Enter pressed, sending message');
            sendMessage();
        }
    });

    if (sendButton) {
        sendButton.addEventListener('click', function() {
            console.log('Send button clicked');
            sendMessage();
        });
    } else {
        console.error('Send button not found!');
    }
    
    function checkServerHealth() {
        fetch(buildApiUrl('/health'))
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ok') {
                    if (data.missing_deepseek_key) {
                        showStatusMessage('DEEPSEEK_API_KEY missing. Chat will remain unavailable until it is configured.', true);
                        return;
                    }
                    if (data.deepseek_available) {
                        showStatusMessage('Connected to DeepSeek API', false);
                    } else {
                        showStatusMessage('Warning: DeepSeek API not available. Using fallback responses.', true);
                    }
                } else {
                    showStatusMessage('Server health check failed', true);
                }
            })
            .catch(error => {
                showStatusMessage('Failed to connect to server', true);
                console.error('Health check error:', error);
            });
    }
    
    function showStatusMessage(message, isError = false) {
        statusContainer.textContent = message;
        statusContainer.className = isError ? 
            'status-container error' : 'status-container success';
        statusContainer.style.display = 'block';
        setTimeout(() => {
            statusContainer.style.display = 'none';
        }, 5000);
    }
    
    function ensureLoadingScreenVisible() {
        const loadingScreen = document.getElementById('loading-screen');
        const chatContainer = document.getElementById('chat-container');
        if (loadingScreen) {
            loadingScreen.style.display = 'flex';
            loadingScreen.classList.remove('fade-out');
        }
        if (chatContainer) {
            chatContainer.style.display = 'none';
        }
    }

    function onAppReady() {
        if (initializationComplete) {
            return;
        }

        initializationComplete = true;
        initializationPollingActive = false;

        const loadingScreen = document.getElementById('loading-screen');
        const chatContainer = document.getElementById('chat-container');
        const loadingStatus = document.getElementById('loading-status');
        const loadingProgress = document.getElementById('loading-progress');

        if (loadingProgress) {
            loadingProgress.style.width = '100%';
        }
        if (loadingStatus) {
            loadingStatus.textContent = 'Ready! Welcome to Galatea AI';
        }

        if (loadingScreen) {
            loadingScreen.classList.add('fade-out');
            setTimeout(() => {
                loadingScreen.style.display = 'none';
                if (chatContainer) {
                    chatContainer.style.display = 'flex';
                }
            }, 500);
        } else if (chatContainer) {
            chatContainer.style.display = 'flex';
        }

        startAvatarPolling();
    }

    function checkAvailability() {
        fetch(buildApiUrl('/api/availability'))
            .then(response => response.json())
            .then(data => {
                if (!data.available) {
                    if (data.status === 'missing_deepseek_key') {
                        const errorPath = data.error_page || '/error';
                        window.location.href = `${API_BASE_URL}${errorPath}`;
                        return;
                    }

                    if (data.status === 'initializing') {
                        ensureLoadingScreenVisible();
                        if (!initializationPollingActive) {
                            initializationPollingActive = true;
                            checkInitializationStatus();
                        }
                        return;
                    }
                }

                onAppReady();
            })
            .catch(error => {
                console.error('Error checking availability:', error);
                setTimeout(checkAvailability, 3000);
            });
    }

    function sendMessage() {
        console.log('sendMessage called');

        if (!userInput) {
            console.error('userInput element not found');
            return;
        }

        const message = userInput.value.trim();
        console.log('Message to send:', message);

        if (message.length === 0) {
            console.log('Empty message, not sending');
            return;
        }

        // Add user message to chat
        addMessage(message, 'user');

        // Clear input
        userInput.value = '';
        userInput.style.height = '';

        // Show typing indicator
        addTypingIndicator();

        // Send to backend
        console.log('Sending to backend...');
        fetch(buildApiUrl('/api/chat'), {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: message })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Remove typing indicator
            removeTypingIndicator();
            
            // Add bot response
            addMessage(data.response, 'bot');
            
            // Update avatar
            updateAvatar(data.avatar_shape);
            
            // Update emotion bars
            updateEmotionBars(data.emotions);
        })
        .catch(error => {
            console.error('Error:', error);
            removeTypingIndicator();
            addMessage('Sorry, I encountered an error processing your request.', 'system');
            showStatusMessage('Error connecting to server: ' + error.message, true);
        });
    }
    
    function addMessage(text, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        contentDiv.textContent = text;
        
        messageDiv.appendChild(contentDiv);
        chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function addTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot typing-indicator';
        typingDiv.innerHTML = '<span></span><span></span><span></span>';
        typingDiv.id = 'typing-indicator';
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    
    function removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }
    
    function updateAvatar(shape) {
        // Remove all shape classes first
        avatar.classList.remove('circle', 'triangle', 'square');
        
        // Add the new shape class
        if (shape === 'Circle' || shape === 'Triangle' || shape === 'Square') {
            avatar.classList.add(shape.toLowerCase());
        }
    }
    
    function updateEmotionBars(emotions) {
        if (!emotions) return;
        
        // Update each emotion bar
        joyBar.style.width = `${emotions.joy * 100}%`;
        sadnessBar.style.width = `${emotions.sadness * 100}%`;
        angerBar.style.width = `${emotions.anger * 100}%`;
        fearBar.style.width = `${emotions.fear * 100}%`;
        curiosityBar.style.width = `${emotions.curiosity * 100}%`;
    }
    
    // Add initialization status check with loading screen
    let initCheckCount = 0;
    function checkInitializationStatus() {
        if (initializationComplete) {
            return;
        }

        const loadingScreen = document.getElementById('loading-screen');
        const chatContainer = document.getElementById('chat-container');
        const loadingStatus = document.getElementById('loading-status');
        const loadingProgress = document.getElementById('loading-progress');

        initCheckCount++;
        const progress = Math.min(initCheckCount * 10, 90); // Cap at 90% until actually ready
        if (loadingProgress) {
            loadingProgress.style.width = `${progress}%`;
        }

        fetch(buildApiUrl('/api/is_initialized'))
            .then(response => response.json())
            .then(data => {
                if (data.missing_deepseek_key) {
                    const errorPath = data.error_page || '/error';
                    window.location.href = `${API_BASE_URL}${errorPath}`;
                    return;
                }

                if (data.is_initialized) {
                    onAppReady();
                    return;
                }

                if (data.initializing) {
                    if (loadingStatus) {
                        loadingStatus.textContent = 'Initializing AI components...';
                    }
                    setTimeout(checkInitializationStatus, 2000); // Check again in 2 seconds
                    return;
                }

                if (loadingStatus) {
                    loadingStatus.textContent = 'Initialization taking longer than expected...';
                }
                setTimeout(checkInitializationStatus, 3000);
            })
            .catch(error => {
                console.error('Error checking status:', error);
                if (loadingStatus) {
                    loadingStatus.textContent = 'Error connecting to server. Retrying...';
                }
                setTimeout(checkInitializationStatus, 3000);
            });
    }

    // Add avatar polling functionality for more responsive updates
    let avatarPollInterval;
    let lastAvatarShape = '';

    function startAvatarPolling() {
        // Clear any existing interval
        if (avatarPollInterval) {
            clearInterval(avatarPollInterval);
        }

        // Poll every 1 second
        avatarPollInterval = setInterval(pollAvatarState, 1000);
        console.log("Avatar polling started");
    }

    function pollAvatarState() {
        fetch(buildApiUrl('/api/avatar'))
            .then(response => response.json())
            .then(data => {
                if (data.avatar_shape && data.is_initialized) {
                    // Only update if shape has changed
                    if (lastAvatarShape !== data.avatar_shape) {
                        console.log(`Avatar shape changed: ${lastAvatarShape} -> ${data.avatar_shape}`);
                        updateAvatar(data.avatar_shape);
                        lastAvatarShape = data.avatar_shape;
                    }
                    
                    // Update sentiment display if available
                    if (data.sentiment) {
                        updateSentiment(data.sentiment);
                    }
                    
                    // Update emotion bars if available
                    if (data.emotions) {
                        updateEmotionBars(data.emotions);
                    }
                }
            })
            .catch(error => {
                console.error('Error polling avatar:', error);
            });
    }

    // Add this new function to update sentiment visualization
    function updateSentiment(sentimentData) {
        const avatar = document.getElementById('avatar');
        
        // Remove existing sentiment classes
        avatar.classList.remove('sentiment-positive', 'sentiment-negative', 'sentiment-neutral', 'sentiment-angry');
        
        // Add the appropriate sentiment class
        if (sentimentData.sentiment === "positive") {
            avatar.classList.add('sentiment-positive');
        } else if (sentimentData.sentiment === "negative") {
            avatar.classList.add('sentiment-negative');
        } else if (sentimentData.sentiment === "angry") {
            avatar.classList.add('sentiment-angry');
        } else {
            avatar.classList.add('sentiment-neutral');
        }
        
        console.log(`Updated sentiment display: ${sentimentData.sentiment}`);
    }
});
