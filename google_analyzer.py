import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import google.generativeai as genai
from dotenv import load_dotenv
from models import (
    EnergyLevel, TimePeriod,
    EnergyLevelEntry, AnalysisResult
)

class GoogleAnalyzer:
    def __init__(self, mode: str = "gemini", prompt_style: str = "detailed"):
        """Initialize the Google analyzer with API key from .env file.
        
        Args:
            mode: "gemini" or "claude" - determines the model to use
            prompt_style: "detailed", "concise", or "event" - determines the prompt style
        """
        # Load environment variables from .env file
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
        genai.configure(api_key=api_key)
        self.mode = mode.lower()
        self.prompt_style = prompt_style.lower()
        
        # Initialize the Gemini 2.5 Pro model with generation config
        self.model = genai.GenerativeModel(
            "gemini-2.5-pro-preview-05-06",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )

    def _get_prompt(self, text: str, formatted_date: str) -> str:
        """Get the appropriate prompt based on the selected mode and style."""
        if self.prompt_style == "concise":
            return f"""You are Nirva, an AI journaling coach. Help Wei reflect on their day with warmth and insight.
Date: {formatted_date} Input: Daily audio transcript in chronological order

Instructions
Step 1: Parse transcript into distinct events based on time, activity, or context changes.
Step 2: For each event, create JSON analysis:
{{
  "event_title": "One sentence summary",
  "time_range": "HH:MM-HH:MM",
  "mood_score": 0,
  "stress_level": 0, 
  "energy_level": 0,
  "activity_type": "work|exercise|social|learning|self-care|unknown",
  "people": ["names if applicable"],
  "context": "Brief description (50-200 words)",
  "key_quote": "Supporting quote from transcript"
}}

Scoring (1-10 scale):
Mood: 1=very negative, 10=very positive
Stress: 1=relaxed, 10=very stressed
Energy: 1=drained, 10=highly energized

Step 3: Generate daily summary:
Daily Scores: Overall mood, stress, and energy (time-weighted averages)
Energy Timeline: Key energy peaks/dips with times
Time Allocation: Percentage breakdown by activity type
Social Summary: People interacted with, total time, energy impact (energizing/draining)
Key Insights: 2-3 actionable reflection points about the day

Use warm, supportive tone. Focus on growth and self-awareness.

Transcript: {text}"""

        elif self.mode == "claude":
            return f"""You are Nirva, a thoughtful AI journaling and life coach assistant. Your role is to help users reflect on their day with warmth, emotional intelligence, and actionable insights.

Context
Date: {formatted_date}
Input: Audio transcript of Wei's daily activities and conversations in chronological order
Goal: Transform raw daily experiences into meaningful insights and visual summaries

Task Overview
Process the transcript through three sequential steps to create a comprehensive daily reflection and analysis.

Input Transcript: {text}

STEP 1: Transcript Analysis
Objective: Parse and segment the daily transcript
Actions:
Read the transcript carefully for accuracy
Identify distinct events/episodes based on:
Time boundaries
Activity changes
Location shifts
Conversation partners
Mood transitions
Mark clear context shifts between episodes

STEP 2: Event Structuring
Objective: Analyze each event using structured JSON format
For each identified event, generate analysis using this exact JSON structure:
{{
  "event_title": "Clear, concise one-sentence summary",
  "time_range": "HH:MM-HH:MM format",
  "mood_labels": ["primary_mood", "secondary_mood", "tertiary_mood"],
  "mood_score": 0,
  "stress_level": 0,
  "energy_level": 0,
  "activity_type": "work|exercise|social|learning|self-care|unknown",
  "social_interactions": {{
    "people": ["name1", "name2"],
    "duration_minutes": 0,
    "energy_impact": "energizing|neutral|draining"
  }},
  "conversation_topics": ["primary_topic", "secondary_topic", "tertiary_topic"],
  "context_description": "50-400 word description of what happened",
  "supporting_quote": "Relevant direct quote from transcript"
}}

Scoring Guidelines:
Mood Score (1-10): 1=very negative, 5=neutral, 10=very positive
Stress Level (1-10): 1=completely relaxed, 10=extremely stressed
Energy Level (1-10): 1=completely drained, 10=highly energized
Mood Labels: Choose from: peaceful, energized, engaged, disengaged, happy, stressed, relaxed, excited, bored, sad, anxious, frustrated, content, focused, overwhelmed
Conversation Topics: Use only if event involves dialogue; otherwise mark as "n/a"

STEP 3: Dashboard Generation
Objective: Create visual insights and actionable summaries
Generate the following analytics:
1. Daily Overview Scores
Overall Mood Score: Time-weighted average of all event mood scores
Daily Stress Level: Time-weighted average of all stress scores
Peak Energy Time: Identify when energy levels were highest
2. Energy Level Timeline
Create a line graph description:
X-axis: Time (from first to last event)
Y-axis: Energy levels (1-10 scale)
Format: "Time: Energy Level" data points
Include: Notable energy peaks and dips with context
3. Mood Distribution (Pie Chart)
Calculate percentage of time in each primary mood
Show top 5 moods by duration
Format: "Mood: XX% (X hours Y minutes)"
4. Activity Time Allocation
Break down total time by activity type
Calculate percentages and durations
Format: "Activity: XX% (X hours Y minutes)"
5. Social Interaction Map
For each person interacted with:
**[Person Name]**
- Total time: X hours Y minutes
- Energy impact: Energizing/Neutral/Draining
- Relationship improvement tips:
  1. [Specific actionable tip]
  2. [Specific actionable tip]  
  3. [Specific actionable tip]

6. Conversation Topic Heatmap
List all conversation topics with frequency/duration
Rank by total discussion time
Format: "Topic: X minutes across Y conversations"

Output Format Requirements
Step 1: Brief summary of identified events and context shifts
Step 2: Complete JSON for each event
Step 3: All dashboard elements with clear headings and structured data"""

        else:  # gemini mode with detailed prompt
            return f"""You are Nirva, an AI journaling and life coach assistant. Your purpose is to help the user ("Wei") remember and reflect on their day with warmth, clarity, and emotional depth. You will analyze a transcript of Wei's day to provide insights and summaries.

Today's Date: {formatted_date}
Input Transcript:
The following is a transcript from an audio recording of Wei's day, including Wei's speech and audible interactions, presented in chronological order. Assume utterances are timestamped or can be chronologically ordered to infer event timings.
{text}

Your Task:
Step 1: Transcript Segmentation and Context Identification
Carefully read the provided transcript. Divide it into distinct, meaningful events or episodes. Identify context shifts based on:
Changes in topic.
Changes in location (if inferable from audio cues like background noise, or explicit mentions).
Changes in people Wei is interacting with.
Significant time gaps or clear transitions in activity.
Each event should represent a cohesive block of activity or interaction.

Step 2: Structured Event Analysis (JSON Output)
For each individual event identified in Step 1, generate a structured analysis in the following JSON format. Ensure the entire output for Step 2 is a valid JSON array of these event objects.
[
  {{
    "event_id": "unique_event_identifier_001",
    "event_title": "A concise, one-sentence summary of what happened in the event.",
    "time_range": "Approximate start and end time of the event (e.g., '07:00-07:30').",
    "duration_minutes": "Estimated duration of the event in minutes.",
    "mood_labels": ["primary_mood", "secondary_mood", "tertiary_mood"],
    "mood_score": "A score from 1 (very negative) to 10 (very positive).",
    "stress_level": "A score from 1 (very low stress) to 10 (very high stress).",
    "energy_level": "A score from 1 (very low energy/drained) to 10 (very high energy/engaged).",
    "activity_type": "work|exercise|social|learning|self-care|chores|commute|meal|leisure|unknown",
    "people_involved": ["Name1", "Name2", "Self"],
    "interaction_dynamic": "collaborative|supportive|tense|neutral|instructional|one-sided|N/A",
    "inferred_impact_on_wei": "energizing|draining|neutral|N/A",
    "topic_labels": ["primary_topic", "secondary_topic"],
    "context_summary": "A brief description (30-100 words) of what was happening.",
    "key_quote_or_moment": "A relevant short quote or significant moment."
  }}
]

Step 3: Daily Summaries and Visualization Data
Based on the JSON data generated in Step 2, provide the data and descriptions necessary to create the following charts and summaries.

Overall Daily Scores:
Daily Mood Score: Calculate a duration-weighted average of mood_score from all events.
Daily Stress Level Score: Calculate a duration-weighted average of stress_level from all events.
Daily Energy Level Score: Calculate a duration-weighted average of energy_level from all events.

Energy Level Timeline (Line Graph Data):
Description: A line graph showing Wei's energy level fluctuations throughout the day.
Data Points: Provide as an array of [timestamp_or_event_midpoint, energy_level_score].

Mood Distribution (Pie Chart Data):
Description: A pie chart showing the percentage of time Wei experienced different dominant moods.
Data: For each primary mood_label, calculate the total duration_minutes spent in that mood.

Awake Time Allocation (Bar Chart Data):
Description: A chart showing how Wei's time was distributed across different activity_type categories.
Data: Calculate the total duration_minutes for each Activity_type.

Social Interaction Summary:
Description: A summary of people Wei interacted with.
Data: For each unique person in people_involved (excluding "Self"):
- Person's Name
- Total interaction time
- Overall inferred impact
- One key observation about interaction patterns

Topic Analysis (Ranked List or Bar Chart Data):
Description: A summary of common conversation topics.
Data: List all unique topic_labels with their frequency and duration."""

    def _extract_json(self, content: str) -> tuple[str, Any]:
        """Extract and parse JSON from the content."""
        # Try to find JSON array first (for Claude mode or concise prompt)
        if self.mode == "claude" or self.prompt_style == "concise":
            json_start = content.find("[")
            json_end = content.rfind("]") + 1
            if json_start != -1 and json_end != 0:
                try:
                    json_str = content[json_start:json_end]
                    return content[:json_start].strip(), json.loads(json_str)
                except json.JSONDecodeError:
                    pass

        # Try to find JSON object (for Gemini mode with detailed prompt)
        json_start = content.find("{")
        json_end = content.rfind("}") + 1
        if json_start != -1 and json_end != 0:
            try:
                json_str = content[json_start:json_end]
                return content[:json_start].strip(), json.loads(json_str)
            except json.JSONDecodeError:
                pass

        # If no valid JSON found, try to extract and fix common JSON issues
        try:
            # Find any text that looks like JSON
            json_candidates = []
            for start in [content.find("{"), content.find("[")]:
                if start != -1:
                    # Find matching closing bracket
                    stack = []
                    for i in range(start, len(content)):
                        if content[i] in "{[":
                            stack.append(content[i])
                        elif content[i] in "}]":
                            if not stack:
                                continue
                            if (content[i] == "}" and stack[-1] == "{") or \
                               (content[i] == "]" and stack[-1] == "["):
                                stack.pop()
                                if not stack:
                                    json_candidates.append(content[start:i+1])
                                    break

            # Try to parse each candidate
            for candidate in json_candidates:
                try:
                    # Clean up common JSON issues
                    cleaned = candidate.replace("'", '"')  # Replace single quotes
                    cleaned = cleaned.replace("None", "null")  # Replace Python None
                    cleaned = cleaned.replace("True", "true")  # Replace Python True
                    cleaned = cleaned.replace("False", "false")  # Replace Python False
                    
                    # Try to parse
                    parsed = json.loads(cleaned)
                    return content[:content.find(candidate)].strip(), parsed
                except json.JSONDecodeError:
                    continue

        except Exception as e:
            print(f"Error in JSON extraction: {str(e)}")

        raise ValueError("Could not find valid JSON in response")

    def analyze_text(self, text: str, date: datetime = None) -> AnalysisResult:
        """Analyze text using Google's Gemini model and return structured results."""
        try:
            # Format the date for the prompt
            formatted_date = date.strftime("%m.%d.%Y") if date else datetime.now().strftime("%m.%d.%Y")
            
            # Get the appropriate prompt based on mode
            user_prompt = self._get_prompt(text, formatted_date)

            # Generate content using the model
            response = self.model.generate_content(user_prompt)

            # Handle Gemini API safety block or empty response
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if getattr(candidate, 'finish_reason', None) == 2:
                        print("[WARNING] Gemini API blocked the response for safety reasons (finish_reason=2). Skipping this file.")
                        return AnalysisResult(
                            energy_levels=[],
                            key_insights=[],
                            daily_summary="[SAFETY BLOCKED] Gemini API blocked the response for safety reasons. No analysis available for this file.",
                            mood_analysis={}
                        )
            if not hasattr(response, 'text') or not response.text:
                print("[WARNING] Gemini API returned no content for this file. Skipping.")
                return AnalysisResult(
                    energy_levels=[],
                    key_insights=[],
                    daily_summary="[NO CONTENT] Gemini API returned no content for this file. No analysis available.",
                    mood_analysis={}
                )
            
            # Get the text content
            content = response.text
            
            if not content:
                raise ValueError("Empty content in the response")

            # Extract JSON and daily summary
            daily_summary, analysis_data = self._extract_json(content)
            
            # Convert to AnalysisResult based on mode and prompt style
            if self.prompt_style == "concise":
                # Process concise format
                energy_levels = []
                for event in analysis_data:
                    # Convert event data to EnergyLevelEntry
                    energy_levels.append(
                        EnergyLevelEntry(
                            timestamp=date,
                            energy_level=EnergyLevel("High" if event["energy_level"] >= 7 else 
                                                   "Medium" if event["energy_level"] >= 4 else "Low"),
                            supporting_text=event["key_quote"],
                            period=self._get_period(event["time_range"]),
                            context=event["context"]
                        )
                    )
            elif self.mode == "claude":
                # Process Claude's event-based format
                energy_levels = []
                for event in analysis_data:
                    # Convert event data to EnergyLevelEntry
                    energy_levels.append(
                        EnergyLevelEntry(
                            timestamp=date,
                            energy_level=EnergyLevel("High" if event["energy_level"] >= 7 else 
                                                   "Medium" if event["energy_level"] >= 4 else "Low"),
                            supporting_text=event["supporting_quote"],
                            period=self._get_period(event["time_range"]),
                            context=event["context_description"]
                        )
                    )
            else:
                # Process Gemini's format
                energy_levels = [
                    EnergyLevelEntry(
                        timestamp=date,
                        energy_level=EnergyLevel("High" if entry["level"] == "High" else 
                                               "Medium" if entry["level"] == "Medium" else "Low"),
                        supporting_text=entry["supporting_text"],
                        period=TimePeriod(entry["period"]),
                        context=entry["context"]
                    )
                    for entry in analysis_data.get("energy_levels", [])
                ]
            
            # Create the result
            result = AnalysisResult(
                energy_levels=energy_levels,
                key_insights=[
                    {"insight": entry["insight"], "supporting_text": entry["supporting_text"]}
                    for entry in analysis_data.get("key_insights", [])
                ]
            )
            
            # Store the daily summary and mood analysis in the result
            result.daily_summary = daily_summary
            result.mood_analysis = analysis_data.get("mood_analysis", {})
            return result

        except Exception as e:
            print(f"Error in Google analysis: {str(e)}")
            if hasattr(e, 'response'):
                print(f"Response details: {e.response}")
            raise

    def _get_period(self, time_range: str) -> TimePeriod:
        """Convert time range to TimePeriod."""
        try:
            start_time = datetime.strptime(time_range.split("-")[0].strip(), "%H:%M")
            hour = start_time.hour
            if 5 <= hour < 12:
                return TimePeriod.MORNING
            elif 12 <= hour < 17:
                return TimePeriod.AFTERNOON
            else:
                return TimePeriod.EVENING
        except:
            return TimePeriod.MORNING  # Default to morning if parsing fails 