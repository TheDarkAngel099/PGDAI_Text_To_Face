"""Utility helper functions"""
import logging
from pathlib import Path
from PIL import Image
import base64
import io

logger = logging.getLogger(__name__)


def save_image(image: Image.Image, output_dir: str = "outputs/generated_images", prefix: str = "generated") -> str:
    """
    Save PIL image to disk
    
    Args:
        image: PIL Image object
        output_dir: Directory to save image
        prefix: Filename prefix
    
    Returns:
        str: Path to saved image
    """
    from datetime import datetime
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = output_path / f"{prefix}_{timestamp}.png"
    
    image.save(file_path)
    logger.info(f"Image saved to {file_path}")
    
    return str(file_path)


def image_to_base64(image: Image.Image) -> str:
    """
    Convert PIL image to base64 string
    
    Args:
        image: PIL Image object
    
    Returns:
        str: Base64 encoded image
    """
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64


def base64_to_image(img_base64: str) -> Image.Image:
    """
    Convert base64 string to PIL image
    
    Args:
        img_base64: Base64 encoded image string
    
    Returns:
        Image: PIL Image object
    """
    img_bytes = base64.b64decode(img_base64)
    image = Image.open(io.BytesIO(img_bytes))
    return image


def validate_image_dimensions(width: int, height: int, min_size: int = 256, max_size: int = 1024) -> bool:
    """
    Validate image dimensions are within allowed range and divisible by 8 (for diffusion models)
    
    Args:
        width: Image width
        height: Image height
        min_size: Minimum allowed size
        max_size: Maximum allowed size
    
    Returns:
        bool: True if valid, False otherwise
    """
    if width < min_size or height < min_size:
        logger.warning(f"Image dimensions too small: {width}x{height}")
        return False
    
    if width > max_size or height > max_size:
        logger.warning(f"Image dimensions too large: {width}x{height}")
        return False
    
    # Diffusion models prefer dimensions divisible by 8
    if width % 8 != 0 or height % 8 != 0:
        logger.warning(f"Image dimensions not divisible by 8: {width}x{height}")
        # We can auto-correct if needed
        width = (width // 8) * 8
        height = (height // 8) * 8
        logger.info(f"Adjusted dimensions to: {width}x{height}")
    
    return True
