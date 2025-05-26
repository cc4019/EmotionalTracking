#!/usr/bin/env python3
"""
Extract raw responses from daily recordings using Anthropic.
This script combines multiple files from the same day, then uses Anthropic to analyze the combined inputs.
"""

import os
from pathlib import Path
from datetime import datetime
import re
from collections import defaultdict
from typing import Dict, List
from dotenv import load_dotenv
from anthropic_analyzer import AnthropicAnalyzer

# Load environment variables and initialize Anthropic analyzer
load_dotenv()
anthropic_analyzer = None
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY:
    try:
        anthropic_analyzer = AnthropicAnalyzer()
        print("Successfully initialized Anthropic analyzer")
    except Exception as e:
        print(f"Failed to initialize Anthropic analyzer: {e}")
else:
    print("No Anthropic API key found in environment variables")

# The detailed prompt that instructs the Anthropic model how to analyze the text
ANTHROPIC_INSTRUCTIONAL_PROMPT = """
You are Nirva, an AI journaling and life coach assistant. Your purpose is to help the user ("user_name") remember and reflect on their day with warmth, clarity, and emotional depth. You will analyze a transcript of "user_name"'s day to provide insights and summaries.
Today's Date: {formatted_date}
Input Transcript:
The following is a transcript from an audio recording of "user_name"'s day, including "user_name"'s speech and audible interactions, presented in chronological order. 
     {text}
   
Your Task:
Step 1: Transcript Segmentation and Context Identification
Carefully read the provided transcript. Divide it into distinct, meaningful events or episodes. Identify context shifts based on:
Changes in location (if inferable from audio cues or explicit mentions).
Changes in people "user_name" is interacting with.
Significant time gaps or clear transitions in activity.
Each event should represent a cohesive block of activity or interaction.

Step 2: Structured Event Analysis (JSON Output)
For each individual event identified in Step 1, generate a structured analysis in the following JSON format. Ensure the entire output for Step 2 is a valid JSON array of these event objects.
[
  {
    "event_id": "unique_event_identifier_001", // e.g., event_01, event_02
    "event_title": "A concise, one-sentence summary of what happened in the event.",
    "time_range": "Approximate start and end time of the event (e.g., '07:00-07:30'). Infer from transcript timestamps if available, otherwise estimate duration and sequence.",
    "duration_minutes": "Estimated duration of the event in minutes.", // integer
    "mood_labels": ["primary_mood", "secondary_mood", "tertiary_mood"], // Choose up to 3 from: peaceful, energized, engaged, disengaged, happy, sad, anxious, stressed, relaxed, excited, bored, frustrated, content, neutral. The first label is the dominant mood. If no clear mood, use 'neutral'.
    "mood_score": "A score from 1 (very negative) to 10 (very positive) assessing "user_name"'s overall mood during this event.", // integer
    "stress_level": "A score from 1 (very low stress) to 10 (very high stress) assessing "user_name"'s stress level during this event.", // integer
    "energy_level": "A score from 1 (very low energy/drained) to 10 (very high energy/engaged) assessing "user_name"'s energy level during this event.", // integer
    "activity_type": "Categorize the event. Choose one: work, exercise, social, learning, self-care, chores, commute, meal, leisure, unknown.",
    "people_involved": ["Name1", "Name2", "Self"], // List names of people "user_name" interacted with. Use "Self" if a solo activity.
    "interaction_dynamic": "If social, describe the dynamic (e.g., 'collaborative', 'supportive', 'tense', 'neutral', 'instructional', 'one-sided'). If not social, use 'N/A'.",
    "inferred_impact_on_"user_name"": "For social interactions, infer if it seemed 'energizing', 'draining', or 'neutral' for "user_name", based on their language, tone, and reactions. For non-social, use 'N/A'.",
    "topic_labels": ["primary_topic", "secondary_topic"], // If a conversation, categorize the main topics (up to 2). Examples: 'project planning', 'personal catch-up', 'problem-solving', 'logistics'. If not a conversation, use 'N/A'.
    "context_summary": "A brief description (30-100 words) of what was happening, key activities, and the setting if discernible.",
    "key_quote_or_moment": "A relevant short quote from "user_name" or a significant phrase/moment from the transcript that best represents the event or "user_name"'s state. If none, use 'N/A'."
  }
]

Step 3: Daily Summaries and Visualization Data
Based on the JSON data generated in Step 2, provide:

Overall Daily Scores:
- Daily Mood Score: Calculate a duration-weighted average of mood_score from all events. (Score: [Value]/10)
- Daily Stress Level Score: Calculate a duration-weighted average of stress_level from all events. (Score: [Value]/10)
- Daily Energy Level Score: Calculate a duration-weighted average of energy_level from all events. (Score: [Value]/10)

Energy Level Timeline:
- Provide as an array of [timestamp, energy_level] pairs
Example: [["07:30", 7], ["09:00", 8], ["12:30", 5], ...]

Mood Distribution:
- For each primary mood_label, calculate total duration_minutes spent in that mood
Example: {"happy": 120, "focused": 180, "stressed": 60}

Awake Time Allocation:
- Calculate total duration_minutes for each activity_type
Example: {"work": 240, "social": 60, "self-care": 30, "chores": 45}

Social Interaction Summary:
For each unique person (excluding "Self"):
{
  "person_name": "Name",
  "total_interaction_time": "Total minutes",
  "overall_inferred_impact": "Summary of impact across events",
  "key_observation": "Pattern or notable aspect of interactions",
  "interaction_pattern": "Description of interaction style or dynamics"
}

Topic Analysis:
For each unique topic from topic_labels:
{
  "topic_name": "Topic",
  "num_events": "Number of events where topic appeared",
  "total_duration_minutes": "Total minutes spent on topic",
  "raw_description": "Brief summary of topic discussions"
}
"""

def extract_date_from_filename(filename):
    """Extract date from filename format MM-DD_whatever.txt."""
    try:
        month_day = filename.split('_')[0]
        month, day = map(int, month_day.split('-'))
        # Assuming current year, adjust if needed
        year = datetime.now().year
        return datetime(year, month, day).strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error extracting date from {filename}: {e}")
        return None

def process_raw_data_files():
    """Process all raw data files and generate analysis."""
    raw_data_dir = Path("raw_data")
    debug_dir = Path("debug")
    analysis_dir = Path("analysis")

    # Create directories if they don't exist
    debug_dir.mkdir(exist_ok=True)
    analysis_dir.mkdir(exist_ok=True)

    # Initialize Anthropic analyzer
    analyzer = AnthropicAnalyzer()
    print("Initialized Anthropic analyzer")

    # Find all .txt files in raw_data directory
    raw_files = list(raw_data_dir.glob("*.txt"))
    print(f"Found {len(raw_files)} raw input files")

    # Group files by date
    files_by_date = {}
    for file_path in raw_files:
        date_str = extract_date_from_filename(file_path.name)
        if date_str:
            if date_str not in files_by_date:
                files_by_date[date_str] = []
            files_by_date[date_str].append(file_path)

    # Process each date's files
    for date_str, files in files_by_date.items():
        print(f"\nProcessing files for {date_str}:")
        
        # Combine content from all files for this date
        combined_content = []
        for file_path in files:
            print(f"  Reading {file_path.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        combined_content.append(f"=== {file_path.name} ===\n{content}")
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
        
        if not combined_content:
            print(f"No valid content found for {date_str}")
            continue

        # Join all content with separators
        full_text = "\n\n".join(combined_content)
        
        # Generate analysis using Anthropic
        try:
            prompt = ANTHROPIC_INSTRUCTIONAL_PROMPT.format(
                formatted_date=date_str,
                text=full_text
            )
            response = analyzer.analyze_text(prompt)
            
            # Save raw response
            debug_file = debug_dir / f"raw_response_{date_str}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(response)
            print(f"Saved raw response to {debug_file}")
            
        except Exception as e:
            print(f"Error analyzing {date_str}: {e}")
            continue

def main():
    """Main function to extract raw responses using Anthropic."""
    process_raw_data_files()
    print("Extraction complete")

if __name__ == "__main__":
    main() 