"""Avatar Engine - manages avatar representation based on emotional state"""
from enum import Enum

class AvatarShape(Enum): #create shape types for the avatar
    CIRCLE = "Circle"
    TRIANGLE = "Triangle"
    SQUARE = "Square"

class AvatarEngine:
    def __init__(self):
        self.avatar_model = "Circle"  # Start with a basic shape
        self.expression_parameters = {}

    def update_avatar(self, emotional_state):
        # Map emotions to avatar parameters (facial expressions, color)
        joy_level = emotional_state["joy"]
        sadness_level = emotional_state["sadness"]

        # Simple mapping (placeholder)
        self.avatar_model = self.change_avatar_shape(joy_level, sadness_level)

    def change_avatar_shape(self, joy, sad):
        #determine shape based on feelings
        if joy > 0.5:
            return AvatarShape.CIRCLE.value
        elif sad > 0.5:
            return AvatarShape.TRIANGLE.value
        else:
            return AvatarShape.SQUARE.value
            
    def render_avatar(self):
        # Simple console rendering of the avatar state
        print(f"Avatar shape: {self.avatar_model}")

