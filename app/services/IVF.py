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

class IVFImageRetrievalSystem:
    def __init__(self, image_folder: str = os.path.abspath(os.path.join(os.getcwd(), 'app', 'static', 'images')), nlist: int = 10):
        """
        Initialize the image retrieval system with HuggingFace's CLIP model
        
        Args:
            image_folder: Directory containing images to index
            nlist: Number of clusters for the inverted file index (IVF)
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Load CLIP model and processor from HuggingFace
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", from_tf=True).to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        self.image_folder = Path(image_folder)
        self.index = None
        self.image_paths = []
        self.nlist = nlist
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
        """Initialize FAISS IVF index and encode all images in the folder"""
        logging.info("Initializing image index...")
        
        # Get all image files
        image_files = [f for f in os.listdir(self.image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        logging.info(f'folder name {self.image_folder}')
        if not image_files:
            logging.warning(f"No images found in the specified folder {self.image_folder}")
            return
        
        # Number of dimensions in CLIP's image features (512-dimensional)
        d = 512
        
        # Create a flat index for training (e.g., IndexFlatL2 or IndexFlatIP)
        quantizer = faiss.IndexFlatL2(d)  # Use L2 distance for clustering
        
        # Create the IVF index using the quantizer
        self.index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_L2)
        
        # Check if the index is trained (it is not until training is done)
        if not self.index.is_trained:
            logging.info("Training the quantizer...")
            # For training, we need to sample some image embeddings (use the first 1000 or so)
            image_embeddings = []
            image_paths_sample = []
            for img_path in image_files[:500]:  # sample first 500 images for training
                try:
                    img_path = os.path.join(self.image_folder, img_path)
                    image = Image.open(img_path).convert('RGB')
                    image_features = self.encode_image(image)
                    image_embeddings.append(image_features)
                    image_paths_sample.append(str(img_path))
                except Exception as e:
                    logging.error(f"Error processing image {img_path}: {str(e)}")
            
            image_embeddings = np.vstack(image_embeddings)
            self.index.train(image_embeddings)
        
        self.image_paths = []
        # Add images to the index
        for img_path in image_files:
            try:
                img_path = os.path.join(self.image_folder, img_path)
                image = Image.open(img_path).convert('RGB')
                image_features = self.encode_image(image)
                
                # Add image features to the FAISS index
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
        
        accessibility = Accessibility.AccessibilityExtensions()
        # Prepare results
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

