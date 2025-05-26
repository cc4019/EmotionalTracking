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
        print("Successfully initialized Anthropic analyzer")

    def analyze_text(self, prompt: str) -> str:
        """
        Analyze text using Anthropic's Claude model.
        Returns the raw response text.
        """
        try:
            # Create a message with the prompt
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=4000,
                temperature=0,
                system="You are Nirva, an AI journaling and life coach assistant. Analyze the user's day and provide structured insights.",
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract the response text
            response = message.content[0].text
            
            # Try to find and extract JSON content
            try:
                # Look for JSON array in Step 2
                step2_start = response.find('[{')
                step2_end = response.find('}]', step2_start) + 2
                if step2_start >= 0 and step2_end > step2_start:
                    events_json = response[step2_start:step2_end]
                    # Validate JSON
                    json.loads(events_json)
                    
                # Look for JSON objects in Step 3
                step3_start = response.find('Step 3:')
                if step3_start >= 0:
                    step3_text = response[step3_start:]
                    
                    # Try to extract all JSON objects
                    json_starts = [i for i in range(len(step3_text)) if step3_text.startswith('{', i)]
                    json_ends = [i for i in range(len(step3_text)) if step3_text.startswith('}', i)]
                    
                    for start, end in zip(json_starts, json_ends):
                        json_str = step3_text[start:end+1]
                        try:
                            # Validate JSON
                            json.loads(json_str)
                        except:
                            continue
            except:
                # If JSON validation fails, don't modify the response
                pass
            
            return response
            
        except Exception as e:
            raise Exception(f"Error in Anthropic analysis: {str(e)}") 