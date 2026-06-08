import os
import logging
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables (e.g. GEMINI_API_KEY)
load_dotenv()

def generate_image_caption(image_path: str) -> Optional[str]:
    """
    Sends an image to Google Gemini 1.5 Flash to generate a detailed caption
    suitable for RAG search context.
    
    Returns the caption string if successful, otherwise None.
    """
    if not os.path.exists(image_path):
        logger.warning(f"Image not found for captioning: {image_path}")
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not found in environment. Skipping image captioning.")
        return None

    try:
        import google.generativeai as genai
        import PIL.Image
        
        genai.configure(api_key=api_key)
        # Using Gemini 1.5 Flash for fast, cheap, and excellent vision capabilities
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        img = PIL.Image.open(image_path)
        
        prompt = (
            "You are an expert scientific data extractor for a RAG system. "
            "Please describe this image in detail. "
            "If it is a graph or chart, explain the axes, legends, trends, and any significant data points. "
            "If it is a diagram, explain the workflow or architecture it depicts. "
            "If it contains a table, summarize the key findings or reproduce the data if small. "
            "If it is a mathematical formula, write it out or describe its purpose. "
            "Keep the description factual, concise, and highly detailed to maximize searchability."
        )
        
        response = model.generate_content([prompt, img])
        
        if response and response.text:
            return response.text.strip()
        return None
        
    except ImportError:
        logger.error("google-generativeai or pillow is not installed. Skipping image captioning.")
        return None
    except Exception as e:
        logger.error(f"Failed to generate image caption for {image_path}: {e}")
        return None
