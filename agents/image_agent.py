import logging
from g4f.client import Client
from langchain.agents import Tool, initialize_agent
from langchain_openai import ChatOpenAI

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Step 1: Define the image generation function
def generate_image(prompt: str, model: str = "midjourney") -> str:
    client = Client()
    response = client.images.generate(
        model=model,
        prompt=prompt,
        response_format="url"
    )
    return response.data[0].url

# Step 2: Create a LangChain Tool
image_generation_tool = Tool(
    name="ImageGenerator",
    description="Generates an image based on a text prompt. Available models , choose a model based on user input and put in model field: midjourney, dall-e-3, flux-pro, flux-dev, flux.",
    func=generate_image,
    return_direct=True
)

# Step 3: Initialize the LangChain LLM
llm = ChatOpenAI(
    base_url="http://localhost:15203/v1",
    model_name="gpt-4o",
    api_key="324",
    temperature=0.5
)
logging.info("Primary LangChain LLM initialized.")

# Step 4: Initialize the agent with the tool
tools = [image_generation_tool]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

# Example usage of the agent
prompt = "a white siamese cat with midjourney model"
image_url = agent.run(f"Generate an image of {prompt} ")
print(f"Generated image URL: {image_url}")