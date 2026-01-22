"""Pipeline for generating images using RealVisXL with Compel"""
import logging
from typing import Tuple, Optional
from PIL.Image import Image

logger = logging.getLogger(__name__)


def generate_image_pipeline(
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> Tuple[Optional[Image], Optional[str]]:
    """
    Generate image using RealVisXL with Compel for handling long prompts
    
    This pipeline:
    1. Loads the RealVisXL model and LoRA weights (if enabled)
    2. Uses Compel to handle long/complex prompts
    3. Generates the image
    4. Saves the image locally
    
    Args:
        prompt: Text prompt for image generation
        height: Image height in pixels
        width: Image width in pixels
        num_inference_steps: Number of denoising steps
        guidance_scale: Guidance scale for classifier-free guidance
    
    Returns:
        tuple: (PIL Image or None, image path or None)
    """
    logger.info(f"Image generation pipeline started for prompt: {prompt[:100]}...")
    logger.info(f"Settings - Height: {height}, Width: {width}, Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    
    # ========== PLACEHOLDER CODE - UNCOMMENT WHEN MODEL IS READY ==========
    # from app.models.realviz_model import get_realviz_pipeline
    # from compel import Compel
    # import torch
    # from pathlib import Path
    # from datetime import datetime
    #
    # try:
    #     pipeline = get_realviz_pipeline()
    #     
    #     if pipeline is None:
    #         logger.warning("Pipeline is None, returning None for testing")
    #         return None, None
    #     
    #     # Initialize Compel for better prompt handling
    #     compel = Compel(
    #         tokenizer=pipeline.tokenizer,
    #         text_encoder=pipeline.text_encoder,
    #         truncate_long_prompts=False
    #     )
    #     
    #     # Build conditioning tensor from prompt
    #     conditioning = compel.build_conditioning_tensor(prompt)
    #     
    #     # Generate image
    #     with torch.no_grad():
    #         result = pipeline(
    #             prompt_embeds=conditioning,
    #             height=height,
    #             width=width,
    #             num_inference_steps=num_inference_steps,
    #             guidance_scale=guidance_scale,
    #             generator=torch.Generator().manual_seed(42)
    #         )
    #     
    #     image = result.images[0]
    #     
    #     # Save image locally
    #     output_dir = Path("outputs/generated_images")
    #     output_dir.mkdir(parents=True, exist_ok=True)
    #     
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     image_path = output_dir / f"generated_{timestamp}.png"
    #     image.save(image_path)
    #     
    #     logger.info(f"Image saved to {image_path}")
    #     return image, str(image_path)
    #
    # except Exception as e:
    #     logger.error(f"Error in image generation pipeline: {str(e)}")
    #     raise
    # ======================================================================
    
    logger.warning("⚠️  RealVisXL model is not loaded (placeholder mode)")
    logger.info("To enable real image generation, uncomment the code in image_generator.py")
    
    # For testing: Create a placeholder image
    from PIL import Image, ImageDraw, ImageFont
    from pathlib import Path
    from datetime import datetime
    
    try:
        # Create a placeholder image with text
        placeholder_image = Image.new('RGB', (width, height), color=(73, 109, 137))
        draw = ImageDraw.Draw(placeholder_image)
        
        # Add text to the placeholder
        text = f"Placeholder Image\nPrompt: {prompt[:50]}...\nSize: {width}x{height}"
        draw.text((20, height//2 - 30), text, fill=(255, 255, 255))
        
        # Save the placeholder
        output_dir = Path("outputs/generated_images")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = output_dir / f"placeholder_{timestamp}.png"
        placeholder_image.save(image_path)
        
        logger.info(f"Placeholder image saved to {image_path}")
        return placeholder_image, str(image_path)
    
    except Exception as e:
        logger.error(f"Error creating placeholder image: {str(e)}")
        return None, None
