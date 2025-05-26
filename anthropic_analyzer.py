import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from anthropic import Anthropic
from dotenv import load_dotenv

class AnthropicAnalyzer:
    def __init__(self):
        """Initialize the Anthropic analyzer with API key from .env file."""
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
        self.client = Anthropic(api_key=api_key)

    def analyze_text(self, text: str, prompt: str = None) -> str:
        """Analyze text using Anthropic's Claude model and return the raw response text."""
        try:
            # Generate content using the model
            response = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=0.7,
                system="You are a helpful AI assistant that analyzes daily transcripts and provides structured insights.",
                messages=[
                    {"role": "user", "content": prompt if prompt else text}
                ]
            )
            
            # Get the text content
            content = response.content[0].text
            
            if not content:
                raise ValueError("Empty content in the response")

            return content
            
        except Exception as e:
            print(f"Error in Anthropic analysis: {e}")
            raise 