# accessibility.py
from fastapi import FastAPI
from PIL import Image
from transformers import pipeline

class AccessibilityExtensions:
    def __init__(self):
        self.image_captioner = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        self.text_to_speech = None  # Initialize your preferred TTS engine here
        
    def generate_image_description(self, image_path: str) -> str:
        """Generate detailed description of an image for screen readers"""
        try:
            # Generate caption
            image = Image.open(image_path)
            caption = self.image_captioner(image)[0]['generated_text']
            
            # Extract any text from the image
            #text_in_image = pytesseract.image_to_string(image)
            
            return f"Image showing {caption}"
            
        except Exception as e:
            return "Error: caption couldn't be generated"
    

