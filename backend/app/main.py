"""FastAPI application factory and middleware setup"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from app.routes import captions, images
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    app = FastAPI(
        title="Text-to-Face Generation API",
        version="1.0.0",
        description="LLaVa + RealVisXL Image Generation Pipeline",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware for Streamlit frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Allow all origins (update for production)
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Health check endpoint
    @app.get("/health", tags=["health"])
    def health_check():
        """Check if API is running"""
        return {
            "status": "healthy",
            "message": "Text-to-Face API is running"
        }
    
    # Include routers
    app.include_router(captions.router, prefix="/api", tags=["captions"])
    app.include_router(images.router, prefix="/api", tags=["images"])
    
    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {str(exc)}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(exc)}
        )
    
    logger.info("FastAPI application created successfully")
    return app


app = create_app()
