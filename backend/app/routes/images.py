"""Routes for image generation"""
from fastapi import APIRouter, HTTPException
import logging
import time

from app.schemas.requests import ImageGenerationRequest, ImageGenerationResponse
from app.pipelines.image_generator import generate_image_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate-image", response_model=ImageGenerationResponse, summary="Generate Image from Prompt")
async def generate_image(request: ImageGenerationRequest):
    """
    Step 2: Generate image using RealVisXL with Compel for handling long prompts
    
    Takes a text prompt and generates an image using the RealVisXL model
    with Compel for improved prompt understanding.
    
    Parameters:
        - prompt: Text description of the face to generate
        - height: Image height (256-1024 pixels)
        - width: Image width (256-1024 pixels)
        - num_inference_steps: Denoising steps (1-100, higher = better quality but slower)
        - guidance_scale: Guidance scale (0-20, higher = more adherence to prompt)
    """
    try:
        logger.info(f"Received image generation request for prompt: {request.prompt[:100]}...")
        start_time = time.time()
        
        # Call image generation pipeline
        image, image_path = generate_image_pipeline(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale
        )
        
        generation_time = time.time() - start_time
        
        # Convert PIL image to base64 if image exists
        image_base64 = None
        if image is not None:
            import io
            import base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        logger.info(f"Image generated successfully in {generation_time:.2f}s")
        
        return ImageGenerationResponse(
            image_base64=image_base64,
            prompt=request.prompt,
            generation_time=generation_time,
            image_path=image_path,
            message="Image generated successfully (using RealVisXL placeholder - enable model to get real images)"
        )
    
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating image: {str(e)}")
