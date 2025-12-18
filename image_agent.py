import logging
import re
from g4f.client import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from prompts.prompts import IMAGE_OPTIMIZATION_SYSTEM_PROMPT
from config import (
    TELEGRAM_BOT_TOKEN,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    ai_tone_map,
    DATABASE_NAME
)
# Initialize logging
logging.basicConfig(level=logging.INFO)

# Define a single LLM for prompt optimization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
    api_key=GOOGLE_API_KEY
)
logging.info("Image Agent LLM initialized.")

# System prompt for prompt optimization
OPTIMIZATION_SYSTEM_PROMPT = """You are an expert image prompt engineer specialized in optimizing prompts for AI image generators like Midjourney, DALL-E, and Stable Diffusion.

Your tasks:
1. If the input is in Persian, translate it accurately to English
2. Enhance the prompt by adding artistic style, lighting, composition details, and other elements that will create a high-quality image
3. Format the prompt for optimal results with Midjourney/DALL-E (including proper --ar aspect ratios if mentioned)
4. DO NOT add inappropriate content or modify the core subject of the original request
5. Return ONLY the optimized prompt, without explanations or additional text

Example input: "یک گربه سیامی سفید"
Example output: "a white siamese cat with blue eyes, studio lighting, detailed fur texture, 4k, professional photography, --ar 16:9"
"""

def optimize_image_prompt(user_prompt: str) -> str:
    """
    Optimize an image generation prompt using an LLM.
    Translates Persian prompts to English and enhances them for better image generation results.
    
    Args:
        user_prompt (str): The original user prompt, can be in Persian or English
        
    Returns:
        str: Optimized English prompt for image generation
    """
    try:
        logging.info(f"Optimizing image prompt: '{user_prompt}'")
        
        # Create a specialized prompt template for optimization only
        optimization_prompt = ChatPromptTemplate.from_messages([
            ("system", IMAGE_OPTIMIZATION_SYSTEM_PROMPT),
            ("human", "{input}")
        ])
        
        # Get the optimized prompt directly from the LLM
        chain = optimization_prompt | llm
        optimized_prompt = chain.invoke({"input": user_prompt}).content.strip()
        
        logging.info(f"Original: '{user_prompt}' → Optimized: '{optimized_prompt}'")
        return optimized_prompt
    except Exception as e:
        logging.error(f"Error optimizing prompt: {e}")
        # If optimization fails, return the original prompt
        return user_prompt

def generate_image(prompt: str, model: str = "midjourney") -> str:
    """
    Generates an image based on the given prompt using the specified model.
    Direct workflow: user prompt -> optimize prompt -> image generation
    
    Args:
        prompt (str): The description of the image to generate
        model (str): The model to use (midjourney, dall-e-3, flux, etc.)
        
    Returns:
        str: The URL of the generated image
    """
    client = Client()
    try:
        # First optimize the prompt
        optimized_prompt = optimize_image_prompt(prompt)
        
        logging.info(f"Generating image with model '{model}' and optimized prompt: '{optimized_prompt}'")
        
        # Generate image with the optimized prompt - direct approach without additional agent
        response = client.images.generate(
            model=model,
            prompt=optimized_prompt,
            response_format="url"
        )
        image_url = response.data[0].url
        logging.info(f"Image generated successfully: {image_url[:60]}...")
        return image_url
        
    except Exception as e:
        logging.error(f"Error generating image with {model}: {e}")
        # Fallback to midjourney if the specified model fails
        if model != "midjourney":
            logging.info(f"Falling back to midjourney model")
            return generate_image(prompt, "midjourney")
        else:
            raise

# For backward compatibility, create a tool that directly uses generate_image
image_generation_tool = Tool(
    name="ImageGenerator",
    description="Generates an image based on a text prompt. Available models: midjourney, dall-e-3, flux-pro, flux-dev, flux.",
    func=generate_image,
    return_direct=True
)

# Create a compatibility wrapper for bot.py which expects an 'agent'
class ImageAgentWrapper:
    def __init__(self):
        logging.info("ImageAgentWrapper initialized for compatibility")
        
    def run(self, prompt):
        """
        Compatibility method that mimics the original agent.run() 
        but uses our optimized direct approach
        """
        logging.info(f"Running image generation through compatibility wrapper: {prompt}")
        return safe_agent_run(prompt)

# Create the agent instance for backward compatibility
agent = ImageAgentWrapper()

# Enhanced error handling for the agent run
def safe_agent_run(prompt):
    """Safe wrapper for agent.run with error handling"""
    try:
        # Extract model from prompt if specified
        model = "midjourney"  # default
        model_match = re.search(r'(midjourney|dall-e-3|flux-pro|flux-dev|flux)', prompt.lower())
        if model_match:
            model = model_match.group(1)
            
        # Remove model mentions from the prompt
        clean_prompt = re.sub(r'with (midjourney|dall-e-3|flux-pro|flux-dev|flux) model', '', prompt)
        clean_prompt = clean_prompt.replace("Generate an image of ", "")
        
        # Direct generation approach - skip the agent altogether
        return generate_image(clean_prompt, model)
    except Exception as e:
        logging.error(f"Image generation failed: {e}")
        # Extract URL from error message if possible
        error_str = str(e)
        url_match = re.search(r'https?://\S+', error_str)
        if url_match:
            return url_match.group(0)
        raise ValueError("Failed to generate image after multiple attempts")

# Example usage (only runs when script is executed directly)
if __name__ == "__main__":
    test_prompt = "a white siamese cat with midjourney model"
    print(f"Testing with prompt: {test_prompt}")
    try:
        image_url = safe_agent_run(f"Generate an image of {test_prompt}")
        print(f"Generated image URL: {image_url}")
    except Exception as e:
        print(f"Test failed: {e}")