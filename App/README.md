# Text to Face App (Suspect AI Gateway)

This project is a modular AI gateway that allows users to generate suspect descriptions and corresponding face images based on selected attributes. It consists of a **FastAPI** backend and a **Streamlit** frontend.

## Features

*   **Attribute Selection**: Select physical attributes (Gender, Age, Ethnicity, Hair, etc.) via a structured UI.
*   **Description Generation**: Generates a detailed text description (caption) of the suspect using an external Text AI model.
*   **Image Generation**: Generates a face image based on the description using an external Image AI model.
*   **Caching**: Supports caching mechanisms to retrieve previously generated results.

## Project Structure

*   `backend/`: Contains the FastAPI application, serving as a gateway to external AI models.
*   `frontend/`: Contains the Streamlit application for the user interface.
*   `data/`: Contains the data schema (`suspect_schema`) used by the frontend.
*   `assets/`: Directory for storing assets and cache files.
*   `config.py`: Centralized configuration management.

## Prerequisites

*   Python 3.8 or higher
*   pip

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd text-to-face-app
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

The application requires environment variables to connect to the external AI models.

1.  Create a `.env` file in the root directory.
2.  Add the following variables:

    ```env
    # Gateway Configuration
    GATEWAY_HOST=127.0.0.1
    GATEWAY_PORT=8000

    # External AI Model APIs (Replace with actual URLs)
    CUSTOM_TEXT_API_URL=http://<your-text-model-service>/generate-text
    CUSTOM_IMAGE_API_URL=http://<your-image-model-service>/generate-image

    # Optional: Cache File Path
    CACHE_FILE=assets/cache.json
    ```

## Running the Application

You need to run both the backend and frontend services.

### 1. Start the Backend

Run the FastAPI server from the project root:

```bash
uvicorn backend.main:app --reload
```

The server will start at `http://127.0.0.1:8000`.

### 2. Start the Frontend

Open a new terminal, navigate to the project root, and run:

```bash
streamlit run frontend/app.py
```

The application will open in your default browser at `http://localhost:8501`.

## Usage

1.  Open the Streamlit interface.
2.  Select the suspect's attributes (Gender, Age, etc.) from the sidebar or main panel.
3.  Click **"Generate Description"** to get a text prompt.
4.  Once the description is generated, click **"Generate Image"** to visualize the suspect.
