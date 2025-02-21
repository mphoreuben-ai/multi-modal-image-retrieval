from fastapi import FastAPI, UploadFile, File
from transformers import CLIPProcessor, CLIPModel
from app.services import Accessibility as Accessibility
import torch
from PIL import Image
import faiss
import numpy as np
import os
from typing import List, Dict
import logging
from pathlib import Path

class ImageRetrievalSystem:
    def __init__(self, image_folder: str = os.path.abspath(os.path.join(os.getcwd(), 'app', 'static', 'images'))):
        """
        Initialize the image retrieval system with HuggingFace's CLIP model
        
        Args:
            image_folder: Directory containing images to index
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load CLIP model and processor from HuggingFace
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", from_tf =True).to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.image_folder = Path(image_folder)
        #self.image_folder.mkdir(exist_ok=True)
        
        # Initialize FAISS index
        self.index = None
        self.image_paths = []
        self.initialize_index()
        
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a single image using CLIP
        
        Args:
            image: PIL Image to encode
            
        Returns:
            numpy.ndarray: Image embedding
        """
        # Process image using CLIP processor
        inputs = self.processor(
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate image embedding
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # Normalize embedding
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query using CLIP
        
        Args:
            text: Text query to encode
            
        Returns:
            numpy.ndarray: Text embedding
        """
        # Process text using CLIP processor
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Generate text embedding
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # Normalize embedding
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()
        
    def initialize_index(self):
        """Initialize FAISS index and encode all images in the folder"""
        logging.info("Initializing image index...")
        
        # Get all image files
        #image_files = list(self.image_folder.glob("*.[jp][pn][g]"))
        logging.info(f'folder name {self.image_folder}')
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            logging.warning("No images found in the specified folder")
            return
        
        # Initialize FAISS index (512 is CLIP's embedding dimension)
        self.index = faiss.IndexFlatIP(512)
        self.image_paths = []
        
        # Process images and build index
        for img_path in image_files:
            try:
                # Open and encode image
                img_path = os.path.join(self.image_folder, img_path)
                image = Image.open(img_path).convert('RGB')
                image_features = self.encode_image(image)
                
                # Add to FAISS index
                self.index.add(image_features)
                self.image_paths.append(str(img_path))
                
            except Exception as e:
                logging.error(f"Error processing image {img_path}: {str(e)}")
        
        logging.info(f"Index built with {len(self.image_paths)} images")
    
    def search(self, query_text: str, k: int = 5) -> List[Dict]:
        """
        Search for images matching the query text
        
        Args:
            query_text: Text description to search for
            k: Number of results to return
            
        Returns:
            List[Dict]: List of results with image paths and similarity scores
        """
        if not self.index:
            raise ValueError("Index not initialized. No images available.")
        
        # Encode query text
        text_features = self.encode_text(query_text)
        
        # Search in FAISS index
        similarities, indices = self.index.search(text_features, k)
        
        # Prepare results
        
        accessibility = Accessibility.AccessibilityExtensions()
        results = []
        for similarity, idx in zip(similarities[0], indices[0]):
            results.append({
                "image_path": self.image_paths[idx],
                "similarity_score": float(similarity),
                "alt_text":accessibility.generate_image_description(self.image_paths[idx])

            })
        
        return results
    
    async def add_image(self, image_file: UploadFile) -> bool:
        """
        Add a new image to the index
        
        Args:
            image_file: Uploaded image file
            
        Returns:
            bool: Success status
        """
        try:
            # Save the image
            image_path = self.image_folder / image_file.filename
            image_content = await image_file.read()
            
            with open(image_path, "wb") as f:
                f.write(image_content)
            
            # Encode and add to index
            image = Image.open(image_path).convert('RGB')
            image_features = self.encode_image(image)
            
            self.index.add(image_features)
            self.image_paths.append(str(image_path))
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding image: {str(e)}")
            return False
        


#retrieval_system = ImageRetrievalSystem()
#results = retrieval_system.search("Fruits", 5)
#print(results)