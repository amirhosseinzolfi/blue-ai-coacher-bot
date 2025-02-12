from langchain.tools import BaseTool
from g4f.client import Client
from langchain_community.chat_models import ChatOpenAI
import asyncio

class ImageGeneratorTool(BaseTool):
    name: str  # Type annotation added
    description: str  # Type annotation added

    def _run(self, prompt: str) -> str:
        """
        Synchronously generate an image URL given a prompt.
        """
        try:
            client = Client()
            response = client.images.generate(
                model="flux",
                prompt=prompt,
                response_format="url"
            )
            # Check that we received a valid response with data.
            if response and hasattr(response, 'data') and response.data and len(response.data) > 0:
                return response.data[0].url
            else:
                return "No image generated. The response was empty or invalid."
        except Exception as e:
            return f"Error generating image: {str(e)}"

    async def _arun(self, prompt: str) -> str:
        """
        Asynchronously generate an image URL given a prompt.
        This runs the synchronous _run method in a thread executor.
        """
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self._run, prompt)
        return result

# Example usage:
if __name__ == "__main__":
    tool = ImageGeneratorTool()
    
    # Synchronous call
    prompt_text = "a white siamese cat"
    image_url = tool._run(prompt_text)
    print(f"Generated image URL: {image_url}")

    # Alternatively, you can integrate this tool with a LangChain agent.
    # For example:
    #
    # from langchain.llms import OpenAI
    # from langchain.agents import initialize_agent, AgentType
    #
    # llm = OpenAI(temperature=0)
    # agent = initialize_agent([tool], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    # response = agent.run("Generate an image of a futuristic city skyline")
    # print(response)
