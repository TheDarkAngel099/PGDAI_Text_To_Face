"""LLaVa 1.5 Model Loading and Management"""
import logging

logger = logging.getLogger(__name__)

# Global model cache
_llava_model = None
_llava_processor = None


def get_llava_model():
    """
    Load and cache LLaVa 1.5 model
    
    Returns:
        tuple: (model, processor)
    
    NOTE: This function loads the actual model. 
    For development/testing, it currently returns placeholder values.
    """
    global _llava_model, _llava_processor
    
    if _llava_model is not None:
        logger.info("Returning cached LLaVa model")
        return _llava_model, _llava_processor
    
    logger.info("Loading LLaVa 1.5 model...")
    
    # ========== PLACEHOLDER CODE - UNCOMMENT WHEN MODELS ARE READY ==========
    # from transformers import AutoProcessor, LlavaForConditionalGeneration
    # import torch
    # from app.config import settings
    #
    # try:
    #     model_name = settings.llava_model_path
    #     processor = AutoProcessor.from_pretrained(model_name)
    #     model = LlavaForConditionalGeneration.from_pretrained(
    #         model_name,
    #         torch_dtype=torch.float16,
    #         device_map=settings.device
    #     )
    #     _llava_model = model
    #     _llava_processor = processor
    #     logger.info(f"LLaVa model loaded successfully from {model_name}")
    #     return model, processor
    # except Exception as e:
    #     logger.error(f"Failed to load LLaVa model: {str(e)}")
    #     raise
    # =========================================================================
    
    logger.warning("⚠️  LLaVa model loading is disabled. Using placeholder for testing.")
    logger.info("To enable real model loading, uncomment the code in llava_model.py")
    
    # Return placeholder values for testing
    return None, None


def unload_llava_model():
    """Unload the model from memory to free up VRAM"""
    global _llava_model, _llava_processor
    
    if _llava_model is not None:
        logger.info("Unloading LLaVa model...")
        _llava_model = None
        _llava_processor = None
