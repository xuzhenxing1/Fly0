from typing import Dict, Any
from ..ui.ui_utils import Colors


class CommandClassifier:
    
    def __init__(self, llm_interface):
        self.llm = llm_interface
    
    def classify(self, command: str) -> str:
        classification_prompt = f"""Determine if the following drone command requires visual detection (image recognition, target localization).
        Command: "{command}"
        Answer only one of the following:
        - visual: if command requires identifying targets in images (e.g., "fly to the car on the left", "fly to the red ball", "fly to the tree ahead")
        - simple: if command does not require visual detection (e.g., "fly up 10 meters", "takeoff", "land", "turn left 90 degrees")
        Answer only visual or simple, no other content."""
        
        try:
            temp_messages = [
                {"role": "system", "content": "You are a drone command classifier."},
                {"role": "user", "content": classification_prompt}
            ]
            response = self.llm.provider.chat(temp_messages, stream=False, temperature=0, max_tokens=10)
            
            if response:
                result = response.strip().lower()
                if result == 'visual':
                    return 'visual'
                elif result == 'simple':
                    return 'simple'
            
            return 'simple'
            
        except Exception as e:
            print(f"{Colors.RED}Classification failed: {e}{Colors.RESET}")
            return 'simple'
