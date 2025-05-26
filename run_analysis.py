#!/usr/bin/env python3
"""
Run analysis on daily recordings and display results.
This script first uses Anthropic to analyze raw daily inputs, saves the structured responses,
then parses those responses into final JSON data.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
import json
from pydantic import BaseModel, Field, field_validator
import re
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
ANTHROPIC_INSTRUCTIONAL_PROMPT = """You are Nirva, an AI journaling and life coach assistant. Your purpose is to help the user ("Wei") remember and reflect on their day with warmth, clarity, and emotional depth. You will analyze a transcript of Wei's day to provide insights and summaries.

Today's Date: {date}
Input Transcript:
{text}

Your Task:
Step 1: Transcript Segmentation and Context Identification
Carefully read the provided transcript. Divide it into distinct, meaningful events or episodes...

Step 2: Structured Event Analysis (JSON Output)
For each individual event identified in Step 1, generate a structured analysis in JSON format...

Step 3: Daily Summaries and Visualization Data
Based on the JSON data generated in Step 2, provide the data and descriptions for charts and summaries..."""

def extract_date_from_raw_filename(filename: str) -> Optional[datetime]:
    """Extract date from filename format MM-DD_whatever.txt."""
    date_match = re.match(r"(\d{2})-(\d{2})_", filename)
    if date_match:
        month, day = date_match.groups()
        # Assuming current year for the analysis
        current_year = datetime.now().year
        return datetime(current_year, int(month), int(day))
    return None

def generate_raw_response(input_file: Path, date: datetime) -> Optional[str]:
    """Generate a structured response using Anthropic for a single input file."""
    if not anthropic_analyzer:
        print(f"Skipping Anthropic analysis for {input_file} - no analyzer available")
        return None
        
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Format the prompt with the date and content
        formatted_prompt = ANTHROPIC_INSTRUCTIONAL_PROMPT.format(
            date=date.strftime("%m.%d.%Y"),
            text=content
        )
        
        # Get the structured analysis from Anthropic
        response = anthropic_analyzer.analyze_text(content, formatted_prompt)
        
        # Format the complete response file content
        full_response = f"=== PROMPT ===\n{formatted_prompt}\n\n=== RESPONSE ===\n{response}"
        
        return full_response
    except Exception as e:
        print(f"Error generating Anthropic analysis for {input_file}: {e}")
        return None

def process_raw_data_files(raw_data_dir: Path, debug_dir: Path) -> None:
    """Process all raw data files using Anthropic and save structured responses."""
    if not raw_data_dir.exists():
        print(f"Raw data directory {raw_data_dir} does not exist")
        return
        
    # Create debug directory if it doesn't exist
    debug_dir.mkdir(exist_ok=True)
    
    # Find all text files in raw_data directory
    raw_files = list(raw_data_dir.glob("*.txt"))
    print(f"Found {len(raw_files)} raw input file(s) to process")
    
    for raw_file in raw_files:
        print(f"Processing raw input file: {raw_file}")
        
        # Extract date from filename
        date = extract_date_from_raw_filename(raw_file.name)
        if not date:
            print(f"Could not extract date from filename: {raw_file.name}. Skipping.")
            continue
            
        # Generate structured response using Anthropic
        response_text = generate_raw_response(raw_file, date)
        if not response_text:
            print(f"Failed to generate structured response for {raw_file}. Skipping.")
            continue
            
        # Save the response to a file in the debug directory
        output_file = debug_dir / f"raw_response_{date.strftime('%Y-%m-%d')}.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(response_text)
            print(f"Saved structured response to {output_file}")
        except Exception as e:
            print(f"Error saving response to {output_file}: {e}")

class MoodEntry(BaseModel):
    """Model for a single mood entry."""
    mood: str
    minutes: int

class ActivityEntry(BaseModel):
    """Model for a single activity entry."""
    activity: str
    minutes: int

class Event(BaseModel):
    """Model for a single event in the day."""
    event_id: str
    event_title: str
    time_range: str
    duration_minutes: int
    mood_labels: List[str]
    mood_score: float = Field(ge=0, le=10)
    stress_level: float = Field(ge=0, le=10)
    energy_level: float = Field(ge=0, le=10)
    activity_type: str
    people_involved: List[str]
    interaction_dynamic: str
    inferred_impact_on_wei: str
    topic_labels: List[str]
    context_summary: str
    key_quote_or_moment: str

class OverallScores(BaseModel):
    """Model for overall daily scores."""
    daily_mood_score: float = Field(ge=0, le=10)
    daily_stress_level_score: float = Field(ge=0, le=10)
    daily_energy_level_score: float = Field(ge=0, le=10)

class SocialInteractionDetail(BaseModel):
    """Model for social interaction details."""
    person_name: str
    total_interaction_time: str = "Unknown"
    overall_inferred_impact: str = "Neutral"
    key_observation: str = "N/A"

class TopicDetail(BaseModel):
    """Model for topic analysis details."""
    rank: Optional[int] = None
    topic_name: str
    num_events: Optional[int] = None
    total_duration_minutes: Optional[int] = None # Changed to int for consistency
    raw_description: str # To store the full string like "(3 events, 135 minutes)"

class RawResponse(BaseModel):
    """Model for parsing the raw response file."""
    events: List[Event]
    overall_scores: Dict[str, float]
    energy_timeline: List[List[str]]
    mood_distribution: List[MoodEntry]
    awake_time_allocation: List[ActivityEntry]
    social_interactions: List[SocialInteractionDetail]
    topic_analysis: List[TopicDetail]

    @field_validator('energy_timeline')
    @classmethod
    def validate_timeline(cls, v: List[List[str]]) -> List[List[str]]:
        # Ensure each entry has exactly 2 elements
        if not all(len(entry) == 2 for entry in v):
            raise ValueError('Each timeline entry must have exactly 2 elements')
        return v

class DailyAnalysis(BaseModel):
    """Model for the complete daily analysis."""
    events: List[Event]
    energy_timeline: List[List[str]] = Field(
        description="List of [timestamp, energy_level] pairs"
    )
    overall_scores: OverallScores
    mood_distribution: Dict[str, int] = Field(description="Mapping of mood to minutes")
    awake_time_allocation: Dict[str, int] = Field(description="Mapping of activity to minutes")
    social_interactions: List[SocialInteractionDetail]
    topic_analysis: List[TopicDetail]

def calculate_weighted_average(events, field_name):
    """Calculate duration-weighted average of a field from events."""
    total_duration = 0
    weighted_sum = 0
    for event in events:
        duration = int(event['duration_minutes']) if isinstance(event['duration_minutes'], str) else event['duration_minutes']
        value = float(event[field_name]) if isinstance(event[field_name], str) else event[field_name]
        total_duration += duration
        weighted_sum += value * duration
    return weighted_sum / total_duration if total_duration > 0 else 0

def find_json_array(text: str) -> str:
    """Find and extract the first complete JSON array from text."""
    start_match = re.search(r'\s*[\[\]]', text) # Find first [ or ] to better locate start
    if not start_match or text[start_match.start()] != '[':
        # print("Debug: No JSON array start found or first bracket is not opening.")
        return ""
    
    start = start_match.start()
    count = 0
    # Check if the text slice starting from 'start' is indeed a JSON array
    # by looking for matching brackets
    for i in range(start, len(text)):
        if text[i] == '[':
            count += 1
        elif text[i] == ']':
            count -= 1
        if count == 0 and i > start: # Found the end of the first complete array
            # print(f"Debug: JSON array found from index {start} to {i+1}")
            return text[start : i + 1]
    # print("Debug: JSON array brackets did not balance.")
    return "" # Return empty if no complete array is found

def parse_energy_timeline_from_string(timeline_string: str) -> List[List[str]]:
    """Parses a timeline string like '[[12:00, 7], [12:30, 6]]' into List[List[str]]."""
    # print(f"Debug parse_energy_timeline: Input string: '{timeline_string}'") # Debugging
    parsed_timeline = []
    if not timeline_string.startswith('[[') or not timeline_string.endswith(']]'):
        # print("Debug parse_energy_timeline: String does not start/end with [[]]") # Debugging
        return parsed_timeline

    content = timeline_string[2:-2]
    # print(f"Debug parse_energy_timeline: Content after stripping outer brackets: '{content}'") # Debugging
    
    # Split by '],[' optionally surrounded by whitespace - more robust
    pairs = re.split(r"\s*\],\s*\[\s*", content)
    # print(f"Debug parse_energy_timeline: Pairs after split: {pairs}") # Debugging
    
    for i, pair_str in enumerate(pairs):
        # print(f"Debug parse_energy_timeline: Processing pair {i}: '{pair_str}'") # Debugging
        try:
            # Split only on the first comma, strip whitespace from elements
            elements = [elem.strip() for elem in pair_str.split(',', 1)] 
            # print(f"Debug parse_energy_timeline: Elements after split: {elements}") # Debugging
            if len(elements) == 2:
                time = elements[0].replace("'", "").replace('"', '') # Remove potential quotes from time
                value = elements[1] # Value is already stripped
                parsed_timeline.append([time, value])
            # else:
                # print(f"Debug parse_energy_timeline: Pair '{pair_str}' did not have 2 elements after split.")
        except Exception as e:
            # print(f"Debug parse_energy_timeline: Error parsing pair '{pair_str}': {e}")
            continue
    # print(f"Debug parse_energy_timeline: Parsed timeline: {parsed_timeline}") # Debugging
    return parsed_timeline

def extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """Extract a section from text between start_marker and end_marker."""
    try:
        start = text.index(start_marker) + len(start_marker)
        end = text.index(end_marker, start) if end_marker else len(text)
        return text[start:end].strip()
    except ValueError:
        return ""

def clean_json_text(text: str) -> str:
    """Clean up JSON text to make it valid."""
    # Remove any trailing commas before closing brackets/braces
    text = text.replace(',]', ']').replace(',}', '}')
    
    # Fix spacing around commas and colons
    text = text.replace('",', '", ').replace('":"', '": "')
    
    # Fix spacing before arrays and objects
    text = text.replace('":[', '": [').replace('":{', '": {')
    
    # Normalize whitespace
    text = ' '.join(text.split())
    
    # Fix unescaped quotes
    text = text.replace('\\"', '"')
    text = text.replace('"[', '[').replace(']"', ']')
    
    return text

def extract_score(section, label):
    """Extract a score value from text."""
    # Try exact match first
    match = re.search(rf"{label}\s*(\d+\.?\d*)", section)
    if not match:
        # Try without the colon
        label_without_colon = label.rstrip(':')
        match = re.search(rf"{label_without_colon}\s*(\d+\.?\d*)", section)
    if not match:
        # Try just the number after the label
        label_base = label.replace('Daily ', '').replace(' Score:', '').replace(' Level Score:', '')
        match = re.search(rf"{label_base}[^\d]*(\d+\.?\d*)", section)
    if match:
        return float(match.group(1))
    return None

def safe_int(value, default=None) -> Optional[int]:
    if value is None: return default
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default

def safe_float(value, default=None) -> Optional[float]:
    if value is None: return default
    try:
        return float(str(value).strip())
    except (ValueError, TypeError):
        return default

def parse_social_interactions(text_section: str) -> List[SocialInteractionDetail]:
    interactions = []
    current_interaction_data = {}
    person_regex = r"^([A-Za-z\\u4e00-\\u9fff\\s()]+?):\\s*$"
    detail_regex = r"^\\s*-\\s*([^:]+):\\s*(.+)$"

    for line in text_section.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        person_match = re.match(person_regex, line)
        if person_match:
            if current_interaction_data and 'person_name' in current_interaction_data:
                interactions.append(SocialInteractionDetail(**current_interaction_data))
            
            matched_person_name = person_match.group(1).strip()
            current_interaction_data = {"person_name": matched_person_name}
        else:
            detail_match = re.match(detail_regex, line)
            if detail_match:
                if 'person_name' in current_interaction_data:
                    key = detail_match.group(1).strip().lower().replace(' ', '_')
                    value = detail_match.group(2).strip()
                    
                    if key == "total_interaction_time":
                        current_interaction_data['total_interaction_time'] = value
                    elif key == "overall_inferred_impact":
                        current_interaction_data['overall_inferred_impact'] = value
                    elif key == "key_observation":
                        current_interaction_data['key_observation'] = value
                        
    if current_interaction_data and 'person_name' in current_interaction_data:
        interactions.append(SocialInteractionDetail(**current_interaction_data))
            
    return interactions

def parse_topic_analysis(text_section: str) -> List[TopicDetail]:
    topics = []
    topic_regex = r"^(?:(\d+)\.\s*)?([^(:]+)(?:\s*\(([^)]+)\))?"
    lines = text_section.split('\n')
    for line in lines:
        line = line.strip()
        if not line or line.lower().startswith("description:") or line.lower().startswith("data:"):
            continue
        match = re.match(topic_regex, line)
        if match:
            rank_str, topic_name_str, details_in_parens_str = match.groups()
            topic_name = topic_name_str.strip()
            rank = safe_int(rank_str)
            num_events = None
            total_duration_minutes = None
            raw_description = details_in_parens_str.strip() if details_in_parens_str else ""
            if details_in_parens_str:
                details = details_in_parens_str.lower()
                events_match = re.search(r'(\d+)\s*event', details)
                duration_match = re.search(r'(\d+)\s*minute', details)
                if events_match:
                    num_events = safe_int(events_match.group(1))
                if duration_match:
                    total_duration_minutes = safe_int(duration_match.group(1))
            topics.append(TopicDetail(
                rank=rank, topic_name=topic_name, num_events=num_events,
                total_duration_minutes=total_duration_minutes, raw_description=raw_description
            ))
    return topics

def parse_raw_response(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text_full = f.read()
    response_marker = "=== RESPONSE ==="
    response_start_index = raw_text_full.find(response_marker)
    raw_text = raw_text_full[response_start_index + len(response_marker):] if response_start_index != -1 else raw_text_full
    if response_start_index == -1:
        print(f"Warning: Response marker '{response_marker}' not found in {file_path}.")

    events_section_text = extract_section(raw_text, "Step 2: Structured Event Analysis (JSON Output)", "Step 3: Daily Summaries and Visualization Data")
    if not events_section_text:
        raise ValueError("Could not find 'Step 2: Structured Event Analysis (JSON Output)' section.")
    
    events_json_str = find_json_array(events_section_text)
    parsed_events_as_dicts = []
    if events_json_str:
        try:
            events_list_from_json = json.loads(events_json_str)
            if isinstance(events_list_from_json, list):
                for event_dict in events_list_from_json:
                    if isinstance(event_dict, dict) and event_dict.get('event_id') != 'unique_event_identifier_001':
                        duration = safe_int(event_dict.get('duration_minutes'))
                        mood = safe_float(event_dict.get('mood_score'))
                        stress = safe_float(event_dict.get('stress_level'))
                        energy = safe_float(event_dict.get('energy_level'))
                        if duration is None or mood is None or stress is None or energy is None:
                            print(f"Warning: Skipping event {event_dict.get('event_id', 'NO_ID')} due to missing/invalid critical numeric data.")
                            continue
                        event_dict['duration_minutes'] = duration
                        event_dict['mood_score'] = mood
                        event_dict['stress_level'] = stress
                        event_dict['energy_level'] = energy
                        parsed_events_as_dicts.append(event_dict)
            else:
                print("Warning: Parsed events JSON is not a list.")
        except json.JSONDecodeError as e:
            print(f"Error parsing events JSON: {e}. Snippet: {events_json_str[:200]}...")
    else:
        print("Warning: No JSON array found in events section. Events section content snippet (first 200 chars):")
        print(events_section_text[:200] + "...")

    event_models_for_pydantic = []
    for event_d in parsed_events_as_dicts:
        try:
            event_models_for_pydantic.append(Event(**event_d))
        except Exception as p_err:
            print(f"Warning: Pydantic validation error for event {event_d.get('event_id', 'NO_ID')}: {p_err}. Skipping event.")
            continue

    scores_section_text = extract_section(raw_text, "Overall Daily Scores:", "Energy Level Timeline (Line Graph Data):")
    daily_mood_score = safe_float(extract_score(scores_section_text, "Daily Mood Score:"), default=5.0)
    daily_stress_score = safe_float(extract_score(scores_section_text, "Daily Stress Level Score:"), default=5.0)
    daily_energy_score = safe_float(extract_score(scores_section_text, "Daily Energy Level Score:"), default=5.0)

    if parsed_events_as_dicts:
        if daily_mood_score == 5.0 and extract_score(scores_section_text, "Daily Mood Score:") is None:
             daily_mood_score = calculate_weighted_average(parsed_events_as_dicts, 'mood_score')
        if daily_stress_score == 5.0 and extract_score(scores_section_text, "Daily Stress Level Score:") is None:
             daily_stress_score = calculate_weighted_average(parsed_events_as_dicts, 'stress_level')
        if daily_energy_score == 5.0 and extract_score(scores_section_text, "Daily Energy Level Score:") is None:
             daily_energy_score = calculate_weighted_average(parsed_events_as_dicts, 'energy_level')

    daily_mood_score = daily_mood_score if daily_mood_score is not None else 5.0
    daily_stress_score = daily_stress_score if daily_stress_score is not None else 5.0
    daily_energy_score = daily_energy_score if daily_energy_score is not None else 5.0

    timeline_section_text = extract_section(raw_text, "Energy Level Timeline (Line Graph Data):", "Mood Distribution (Pie Chart Data):")
    timeline_data = []
    if timeline_section_text:
        timeline_string_found = None
        for line in timeline_section_text.split('\n'):
            stripped_line = line.strip()
            if stripped_line.startswith('[[') and stripped_line.endswith(']]'):
                timeline_string_found = stripped_line
                break
        if timeline_string_found:
            timeline_data = parse_energy_timeline_from_string(timeline_string_found)
            if not timeline_data: print(f"Warning: parse_energy_timeline_from_string failed for: {timeline_string_found}")
    if not timeline_data and parsed_events_as_dicts:
        print("Info: Energy timeline data not found or failed to parse, generating from events.")
        for event_d in parsed_events_as_dicts:
            if 'time_range' in event_d and 'energy_level' in event_d:
                try:
                    start_time = str(event_d['time_range']).split('-')[0].strip()
                    energy_val = safe_float(event_d['energy_level'])
                    if energy_val is not None:
                        timeline_data.append([start_time, str(energy_val)])
                except Exception:
                    continue

    mood_section_text = extract_section(raw_text, "Mood Distribution (Pie Chart Data):", "Awake Time Allocation (Bar Chart Data):")
    mood_distribution_map = {} if not mood_section_text else parse_distribution(mood_section_text)
    awake_time_text = extract_section(raw_text, "Awake Time Allocation (Bar Chart Data):", "Social Interaction Summary:")
    if not awake_time_text: 
        awake_time_text = extract_section(raw_text, "Awake Time Allocation (Bar Chart Data):", "Topic Analysis (Ranked List):")
    time_allocation_map = {} if not awake_time_text else parse_distribution(awake_time_text)
    social_interaction_text = extract_section(raw_text, "Social Interaction Summary:", "Topic Analysis (Ranked List):")
    if not social_interaction_text: 
        social_start_marker = "Social Interaction Summary:"
        social_start_idx = raw_text.find(social_start_marker)
        if social_start_idx != -1:
            next_major_section_starts = [raw_text.find("Topic Analysis", social_start_idx), raw_text.find("\n\n\n", social_start_idx)] 
            next_major_section_starts = [idx for idx in next_major_section_starts if idx != -1]
            end_idx = min(next_major_section_starts) if next_major_section_starts else len(raw_text)
            social_interaction_text = raw_text[social_start_idx + len(social_start_marker) : end_idx].strip()
        else:
            social_interaction_text = ""
    print("DEBUG MAIN: Text passed to parse_social_interactions:")
    print("---")
    print(social_interaction_text)
    print("---")
    social_interactions_list = parse_social_interactions(social_interaction_text)
    topic_analysis_section_text = extract_section(raw_text, "Topic Analysis (Ranked List):", "\n\n\n") 
    if not topic_analysis_section_text:
        topic_analysis_start_marker = "Topic Analysis (Ranked List):"
        start_idx = raw_text.find(topic_analysis_start_marker)
        if start_idx != -1:
            topic_analysis_section_text = raw_text[start_idx + len(topic_analysis_start_marker):].strip()
        else:
            topic_analysis_section_text = ""
    else: 
        header_to_remove = "Topic Analysis (Ranked List):"
        if topic_analysis_section_text.startswith(header_to_remove):
             topic_analysis_section_text = topic_analysis_section_text[len(header_to_remove):].strip()
    topic_analysis_list = parse_topic_analysis(topic_analysis_section_text)

    raw_response_payload = {
        'events': event_models_for_pydantic,
        'overall_scores': {
            'daily_mood_score': daily_mood_score,
            'daily_stress_level_score': daily_stress_score,
            'daily_energy_level_score': daily_energy_score
        },
        'energy_timeline': timeline_data,
        'mood_distribution': [MoodEntry(mood=k, minutes=v) for k, v in mood_distribution_map.items()],
        'awake_time_allocation': [ActivityEntry(activity=k, minutes=v) for k, v in time_allocation_map.items()],
        'social_interactions': social_interactions_list,
        'topic_analysis': topic_analysis_list
    }
    try:
        raw_response_model = RawResponse(**raw_response_payload)
    except Exception as e:
        print(f"Error creating RawResponse Pydantic model: {e}")
        print(f"Payload details: Events count={len(raw_response_payload['events'])}, Timeline count={len(raw_response_payload['energy_timeline'])}, Social count={len(raw_response_payload['social_interactions'])}, Topic count={len(raw_response_payload['topic_analysis'])}")
        print(f"Overall Scores in payload: {raw_response_payload['overall_scores']}")
        if raw_response_payload['events'] and event_models_for_pydantic:
             first_event_payload = next((e_dict for e_dict in parsed_events_as_dicts if e_dict['event_id'] == event_models_for_pydantic[0].event_id), None)
             if first_event_payload:
                print(f"First successfully parsed event (ID: {event_models_for_pydantic[0].event_id}) raw numeric fields before Pydantic: D={first_event_payload['duration_minutes']}, M={first_event_payload['mood_score']}, S={first_event_payload['stress_level']}, E={first_event_payload['energy_level']}")
        raise
    return DailyAnalysis(
        events=raw_response_model.events,
        energy_timeline=raw_response_model.energy_timeline,
        overall_scores=OverallScores(**raw_response_model.overall_scores),
        mood_distribution={entry.mood: entry.minutes for entry in raw_response_model.mood_distribution},
        awake_time_allocation={entry.activity: entry.minutes for entry in raw_response_model.awake_time_allocation},
        social_interactions=raw_response_model.social_interactions,
        topic_analysis=raw_response_model.topic_analysis
    )

def parse_distribution(section):
    """Parse a distribution section into a dictionary of label -> minutes."""
    result = {}
    # Look for lines like "Label: X minutes"
    for line in section.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
        # Split on first colon
        parts = line.split(':', 1)
        if len(parts) != 2:
            continue
        label = parts[0].strip()
        # Extract the number before "minutes"
        match = re.search(r'(\d+)\s*minutes?', parts[1])
        if match:
            minutes = int(match.group(1))
            result[label] = minutes
    return result

def extract_date(text):
    """Extract the date from the raw response."""
    match = re.search(r"Today's Date:\s*(\d{2}\.\d{2}\.\d{4})", text)
    if match:
        return match.group(1)
    return None

def save_analysis(analysis_data: DailyAnalysis, date: Optional[datetime] = None) -> None:
    """Save the analysis data to a JSON file."""
    if date is None:
        date = datetime.now()
    
    filename = f"daily_analysis_{date.strftime('%Y-%m-%d')}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(analysis_data.model_dump(), f, indent=4, ensure_ascii=False)
    
    print(f"Analysis saved to {filename}")

def main():
    """Main function implementing the two-stage pipeline."""
    # Stage 1: Generate structured responses using Anthropic
    raw_data_dir = Path("raw_data")
    debug_dir = Path("debug")
    
    print("Stage 1: Generating structured responses using Anthropic")
    process_raw_data_files(raw_data_dir, debug_dir)
    print("-" * 50)
    
    # Stage 2: Parse structured responses into final JSON
    print("Stage 2: Parsing structured responses into final JSON")
    response_files = list(debug_dir.glob("raw_response_*.txt"))
    if not response_files:
        print("No raw response files found in debug directory")
        return
    
    print(f"Found {len(response_files)} raw response file(s) to process")
    
    for file_path in response_files:
        print(f"Processing raw response file: {file_path}")
        try:
            # Parse the raw response and get structured data
            analysis_data = parse_raw_response(file_path)
            
            # Extract date from filename
            date_str_match = re.search(r"raw_response_(\d{4}-\d{2}-\d{2})\.txt", file_path.name)
            if not date_str_match:
                print(f"Could not extract date from filename: {file_path.name}. Skipping.")
                continue
            
            date_str = date_str_match.group(1)
            date = datetime.strptime(date_str, '%Y-%m-%d')
            
            # Save the structured data
            save_analysis(analysis_data, date)
            print(f"Successfully processed and saved analysis for {date_str}")
            
        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
        print("-" * 30)

if __name__ == "__main__":
    main() 