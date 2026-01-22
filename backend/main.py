"""Entry point for running the FastAPI application"""
if __name__ == "__main__":
    import uvicorn
    from app.main import app
    from app.config import settings
    
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
