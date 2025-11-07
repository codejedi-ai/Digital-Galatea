"""Dialogue Engine - handles conversation flow and styling"""

class DialogueEngine:
    def __init__(self, ai_core):
        self.ai_core = ai_core
        self.last_user_message = ""

    def get_response(self, user_input):
        # Store the last message for sentiment analysis
        self.last_user_message = user_input
        
        ai_response = self.ai_core.process_input(user_input)
        styled_response = self.apply_style(ai_response, self.ai_core.emotional_state)
        return styled_response

    def apply_style(self, text, emotional_state):
        style = self.get_style(emotional_state)
        #selects styles based on emotions
        #add style to text
        styled_text = text # Remove the style suffix to make responses cleaner
        return styled_text

    def get_style(self, emotional_state):
        #determine style based on the state of the AI
        return "neutral"

