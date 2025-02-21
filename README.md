# MultiModalRetrieval

MultiModalRetrieval is a Python image retrieval system that takes a user query (in text
describing an image or audio format) and returns the top-K matching images from a provided dataset.
The project utilizes FastAPI for serving the API and Streamlit for the user interface. 

## Features
- Retrieve images from a specified directory.
- Web UI powered by **Streamlit** for easy interaction.
- REST API for accessing backend logic using **FastAPI**.


## Folder Structure

Here’s a quick overview of the folder structure:
- MultiModalRetrieval 
- MultiModalRetrieval/app 
- MultiModalRetrieval/app/api (API endpoints)
- MultiModalRetrieval/app/services (Backend logic)
- MultiModalRetrieval/app/static (Imagefolder and images)
- MultiModalRetrieval/app/ui (Streamlit UI)



### Requirements

To run the project, you need to have the following Python packages installed:

- **FastAPI**: for creating the web API.
- **Streamlit**: for creating the web UI.
- **Uvicorn**: for running the FastAPI app.

You can install all the dependencies listed in `requirements.txt` using:
bash
pip install -r requirements.txt

### Setup and Installation
git clone https://github.com/mphoreuben-ai/multi-modal-image-retrieval.git
cd MultiModalProject


### Create and activate a virtual environment
# For virtualenv (Unix/macOS)
python3 -m venv venv
source venv/bin/activate

# For virtualenv (Windows)
python -m venv venv
venv\Scripts\activate


### Install dependencies
pip install -r requirements.txt


### Running the solution
- Start by running the FastTAPI app using the below command:
uvicorn app.api.app:app --reload

- Open another cmd window and run the streamlit app using the below command:
streamlit run app/ui/UI.py

### Testing the solution
# Text Query
- NB: Clear previous recordings if any, by clicking the trash can icon that appears when hovering on the audio input.
- Input your query in the Search textfield and click the Search button.
- Scroll to see results
# Audio Query
- NB: Clear the Search textfield before clicking on the Search button
- Click on the mic icon to record and click stop when done.
- Click the Search button
- Scroll to see results


