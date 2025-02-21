import streamlit as st
import requests
import os
from PIL import Image
import io
import base64
from pathlib import Path
import torch
import clip
import logging
from typing import List, Dict
import numpy as np


class ImageRetrievalUI:
    def __init__(self):
        st.set_page_config(
            page_title="Image Retrieval System",
            page_icon="🔍",
            layout="wide"
        )
        
        # Initialize session state
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        if 'results' not in st.session_state:
            st.session_state.results = []

    def render_header(self):
        """Render the application header"""
        st.title("🔍 Multi-Modal Image Retrieval System")
        st.markdown("""
        Search for images using natural language descriptions. 
        The system uses CLIP to understand both images and text.
        """)

    def render_sidebar(self):
        """Render the sidebar with settings and history"""
        with st.sidebar:
            st.header("Settings")
            
            # Number of results slider
            k_results = st.slider(
                "Number of results", 
                min_value=1, 
                max_value=20, 
                value=5
            )
            
            # Confidence threshold
            confidence_threshold = st.slider(
                "Confidence threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.2
            )
            
            # Search history
            st.header("Search History")
            for query in st.session_state.search_history[-5:]:
                st.text(query)
                
            return k_results, confidence_threshold

    def render_search_section(self):
        """Render the search input section"""
        st.header("Search")
        
        # Search input
        query = st.text_input(
            "Enter your image description",
            placeholder="e.g., 'a dog playing in the park'"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            audio_value, search_button = st.audio_input("Record a voice message"), st.button("Search", type="primary")
                
        return query, search_button, audio_value


    def render_results(self, results: List[Dict]):
        """Render search results"""
        st.header("Search Results")
        
        if not results:
            st.info("No results to display. Try searching for something!")
            return

        # Create a grid layout for results
        cols = st.columns(3)
        for idx, result in enumerate(results):
            col = cols[idx % 3]
            with col:
                # Display image
                image_path = result['image_path']
                image = Image.open(image_path)
                st.image(image, caption=f"Alt text: {result['alt_text']}. Score: {result['similarity_score']:.2f}")
                


    def main(self):
        """Main application logic"""
        self.render_header()
        k_results, confidence_threshold = self.render_sidebar()
        
        # Render main sections
        query, search_button, audio_value = self.render_search_section()
        
        
        # Handle search
        if search_button and query:
            st.session_state.search_history.append(query)
            try:
                # Make API call to backend
                response = requests.get(
                    "http://127.0.0.1:8000/search",
                    params={"query": query, "k": k_results}
                )
                
                logging.info(response.status_code)
                if response.status_code == 200:
                    results = response.json()['results']
                    # Filter by confidence threshold
                    results = [r for r in results if r['similarity_score'] >= confidence_threshold]
                    st.session_state.results = results
                else:
                    st.error("Failed to get results from the server")
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
        elif audio_value and search_button:
            st.session_state.search_history.append(query)
            try:
                # Make API call to backend
                response = requests.get(
                    "http://127.0.0.1:8000/audiosearch",
                    files={"file": audio_value}
                )
                
                logging.info(response.status_code)
                if response.status_code == 200:
                    query = response.json()['results']

                    # Make a send request to the search end point    
                    response = requests.get(
                    "http://127.0.0.1:8000/search",
                    params={"query": query['text'], "k": k_results}
                    )

                    if response.status_code == 200:
                        results = response.json()['results']
                        
                        # Filter by confidence threshold
                        results = [r for r in results if r['similarity_score'] >= confidence_threshold]
                    
                    st.write(f"Transcription: {query['text']}")                    
                    
                    st.session_state.results = results
                else:
                    st.error("Failed to get results from the server")
            except Exception as e:
                st.error(f"Error occurred: {str(e)}")
        
        
        # Display results
        self.render_results(st.session_state.results)

app = ImageRetrievalUI()
app.main()

