document.addEventListener('DOMContentLoaded', function() {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const avatar = document.getElementById('avatar');
    const statusContainer = document.getElementById('status-container');
    
    // Emotion bars
    const joyBar = document.getElementById('joy-bar');
    const sadnessBar = document.getElementById('sadness-bar');
    const angerBar = document.getElementById('anger-bar');
    const fearBar = document.getElementById('fear-bar');
    const curiosityBar = document.getElementById('curiosity-bar');
    
    // Check server health on load
    checkServerHealth();
    
    // Start checking initialization status
    checkInitializationStatus();
    
    // Auto-resize textarea as user types
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
            sendMessage();
        }
    });
    
    // Add keyboard shortcut for sending messages
    document.addEventListener('keydown', function(e) {
        // Command+Enter or Ctrl+Enter to send
        if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
            e.preventDefault();
            sendMessage();
        }
    });
    
    sendButton.addEventListener('click', sendMessage);
    
    function checkServerHealth() {
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ok') {
                    if (data.gemini_available) {
                        showStatusMessage('Connected to Gemini API', false);
                    } else {
                        showStatusMessage('Warning: Gemini API not available. Using fallback responses.', true);
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
    
    function sendMessage() {
        const message = userInput.value.trim();
        if (message.length === 0) return;
        
        // Add user message to chat
        addMessage(message, 'user');
        
        // Clear input
        userInput.value = '';
        userInput.style.height = '';
        
        // Show typing indicator
        addTypingIndicator();
        
        // Send to backend
        fetch('/api/chat', {
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
        const loadingScreen = document.getElementById('loading-screen');
        const chatContainer = document.getElementById('chat-container');
        const loadingStatus = document.getElementById('loading-status');
        const loadingProgress = document.getElementById('loading-progress');

        initCheckCount++;
        const progress = Math.min(initCheckCount * 10, 90); // Cap at 90% until actually ready
        if (loadingProgress) {
            loadingProgress.style.width = `${progress}%`;
        }

        fetch('/status')
            .then(response => response.json())
            .then(data => {
                if (data.initializing) {
                    // Still initializing
                    if (loadingStatus) {
                        loadingStatus.textContent = 'Initializing AI components...';
                    }
                    setTimeout(checkInitializationStatus, 2000); // Check again in 2 seconds
                } else if (data.is_initialized) {
                    // Initialization complete!
                    if (loadingProgress) {
                        loadingProgress.style.width = '100%';
                    }
                    if (loadingStatus) {
                        loadingStatus.textContent = 'Ready! Welcome to Galatea AI';
                    }

                    // Hide loading screen and show chat after a brief delay
                    setTimeout(() => {
                        if (loadingScreen) {
                            loadingScreen.classList.add('fade-out');
                            setTimeout(() => {
                                loadingScreen.style.display = 'none';
                                if (chatContainer) {
                                    chatContainer.style.display = 'flex';
                                }
                            }, 500);
                        }

                        // Start polling for avatar updates when initialized
                        startAvatarPolling();
                    }, 1000);
                } else {
                    // Something wrong with initialization
                    if (loadingStatus) {
                        loadingStatus.textContent = 'Initialization taking longer than expected...';
                    }
                    setTimeout(checkInitializationStatus, 3000);
                }
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
        fetch('/api/avatar')
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
