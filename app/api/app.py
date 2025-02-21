# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.services import ImageRetrieval as IR 
from app.services import Accessibility as Accessibility
from app.services import IVF as IVF
from app.services import Transcribe
import os

# Initialize FastAPI app
app = FastAPI()

app.mount("/app/static", StaticFiles(directory="app/static"), name="static")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize retrieval system
retrieval_system = IR.ImageRetrievalSystem()
ivf_retrieval_system = IVF.IVFImageRetrievalSystem()
accessibility = Accessibility.AccessibilityExtensions()
transcription = Transcribe

@app.get("/search")
async def search_images(query: str, k: int = 5):
    try:
        results = retrieval_system.search(query, k)
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/ivfsearch")
async def search_images_ivf(query: str, k: int = 5):
    try:
        results = ivf_retrieval_system.search(query, k)
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/audiosearch")
async def audio_search_images(file: UploadFile = File(...), k: int = 5):
    try:

        query = transcription.transcribe_audio(file)
        #results = retrieval_system.search(query, k)
        return {"status": "success", "results": query}
    except Exception as e:
        return {"status": "error", "message": str(e)}


    

