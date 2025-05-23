#!/usr/bin/env python3
from openai import OpenAI
from dotenv import load_dotenv
import os

"""
A script that checks if you can connect to the AIaaS LLM inference endpoint.
"""

def main():
    load_dotenv()
    RCP_AIaaS_KEY = os.getenv("OPENAI_API_KEY")


    # OpenAI API configuration
    client = OpenAI(
        base_url="https://inference-dev.rcp.epfl.ch/v1",
        api_key=RCP_AIaaS_KEY
    )

    messages = [
        {
            "role": "user",
            "content": "What is the capital of Switzerland?"
        },
    ]

    try:
        print("sending message")
        completion = client.chat.completions.create(
            model="meta-llama/Llama-4-Scout-17B-16E-Instruct",
            messages=messages
        )
        print(completion.choices[0].message.content)
        print("-" * 50)
        print("Request token usage:")
        print(f"completion_tokens: {completion.usage.completion_tokens}")
        print(f"prompt_tokens: {completion.usage.prompt_tokens}")
        print(f"total_tokens: {completion.usage.total_tokens}")

    except Exception as e:
        print(f"Error during request: {e}")


if __name__ == "__main__":
    main()
