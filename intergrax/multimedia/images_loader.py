# © Artur Czarnecki. All rights reserved.
# Integrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

from __future__ import annotations
import ollama
from IPython.display import Image, display

# !ollama pull llava-llama3:latest

def transcribe_image(prompt:str, image_path: str, model:str = "llava-llama3:latest")->str:
    response = ollama.chat(
        model=model,
        messages=[
            {
                "role":"user",
                "content": prompt,
                "images": [image_path]
            }
        ]
    )
    return response['message']['content']

