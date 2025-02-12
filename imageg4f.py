from g4f.client import Client
import g4f
import requests

client = Client()

def analyze_image(image):
    """
    Analyze the given image by providing a full explanation, description, and analysis.
    
    Parameters:
        image (BinaryIO): A binary stream (e.g., io.BytesIO) containing the image data.
                            Accepted image formats include JPEG, PNG, GIF, and BMP.
    
    Returns:
        str: A detailed analysis of the image.
    
    Example:
        import io
        with open("path/to/image.jpg", "rb") as f:
            image_stream = io.BytesIO(f.read())
        result = analyze_image(image_stream)
    """
    # Request full explanation, description, and analysis for the provided image
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": "Provide a full and complete explanation, description, and analysis of the given image."
            }
        ],
        image=image
    )
    return response.choices[0].message.content

# Example usage:
# image = requests.get("https://raw.githubusercontent.com/xtekky/gpt4free/refs/heads/main/docs/images/cat.jpeg", stream=True).raw
# result = analyze_image(image)
# print(result)

# import g4f
# import g4f.Provider

# def chat_completion(prompt):
#     client = g4f.Client(provider=g4f.Provider.Blackbox)
#     images = [
#         [open("docs/images/waterfall.jpeg", "rb"), "waterfall.jpeg"],
#         [open("docs/images/cat.webp", "rb"), "cat.webp"]
#     ]
#     response = client.chat.completions.create([{"content": prompt, "role": "user"}], "", images=images)
#     print(response.choices[0].message.content)

# prompt = "what are on this images?"
# chat_completion(prompt)