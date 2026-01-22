"""Routes for caption generation"""
from fastapi import APIRouter, HTTPException
import logging

from app.schemas.requests import CaptionRequest, CaptionResponse
from app.pipelines.caption_generator import generate_caption_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def format_user_input(attributes, description: str = "") -> str:
    """
    Convert user attribute selections to natural language prompt
    
    Example:
        Input: [{'nose', 'color', 'brown'}] → 
        Output: 'Face with brown nose. [description if provided]'
    """
    if not attributes:
        return description or "Face"
    
    lines = [f"{attr.category} with {attr.attribute} {attr.value}" 
             for attr in attributes]
    
    prompt = f"Face with {', '.join(line.split(' with ')[1] for line in lines)}"
    if description:
        prompt += f". {description}"
    
    return prompt


@router.post("/caption", response_model=CaptionResponse, summary="Generate Dense Caption from Attributes")
async def create_caption(request: CaptionRequest):
    """
    Step 1: Convert user facial attribute selections to a dense prompt using LLaVa
    
    Takes user-selected categories and attributes (e.g., nose→brown, eyes→blue)
    and generates a detailed, dense text prompt suitable for image generation.
    """
    try:
        logger.info(f"Received caption request with {len(request.attributes)} attributes")
        
        # Format user input into natural language
        formatted_prompt = format_user_input(request.attributes, request.description)
        logger.info(f"Formatted prompt: {formatted_prompt}")
        
        # Call caption generation pipeline
        dense_prompt = generate_caption_pipeline(formatted_prompt)
        
        logger.info(f"Generated dense prompt: {dense_prompt}")
        
        return CaptionResponse(
            dense_prompt=dense_prompt,
            original_input=request,
            message="Caption generated successfully (using LLaVa placeholder - enable model to get real captions)"
        )
    
    except Exception as e:
        logger.error(f"Error generating caption: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating caption: {str(e)}")
