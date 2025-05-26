#!/usr/bin/env python3
"""
Run analysis on daily recordings and generate structured JSON data.
This script parses the raw responses from the debug directory and generates daily analysis JSON files.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
from pydantic import BaseModel, Field, field_validator
import re
import uuid
from collections import defaultdict

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
    total_duration_minutes: Optional[int] = None
    raw_description: str

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
    start_match = re.search(r'\s*[\[\]]', text)
    if not start_match or text[start_match.start()] != '[':
        return ""
    
    start = start_match.start()
    count = 0
    for i in range(start, len(text)):
        if text[i] == '[':
            count += 1
        elif text[i] == ']':
            count -= 1
        if count == 0 and i > start:
            return text[start : i + 1]
    return ""

def parse_energy_timeline_from_string(timeline_string: str) -> List[List[str]]:
    """Parses a timeline string like '[[12:00, 7], [12:30, 6]]' into List[List[str]]."""
    parsed_timeline = []
    if not timeline_string.startswith('[[') or not timeline_string.endswith(']]'):
        return parsed_timeline

    content = timeline_string[2:-2]
    pairs = re.split(r"\s*\],\s*\[\s*", content)
    
    for pair_str in pairs:
        try:
            elements = [elem.strip() for elem in pair_str.split(',', 1)]
            if len(elements) == 2:
                time = elements[0].replace("'", "").replace('"', '')
                value = elements[1]
                parsed_timeline.append([time, value])
        except Exception:
            continue
    return parsed_timeline

def extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """Extract a section from text between start_marker and end_marker."""
    try:
        start = text.index(start_marker) + len(start_marker)
        end = text.index(end_marker, start) if end_marker else len(text)
        return text[start:end].strip()
    except ValueError:
        return ""

def extract_score(section, label):
    """Extract a score value from text."""
    match = re.search(rf"{label}\s*(\d+\.?\d*)", section)
    if not match:
        label_without_colon = label.rstrip(':')
        match = re.search(rf"{label_without_colon}\s*(\d+\.?\d*)", section)
    if not match:
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
    """Parse social interactions section into structured data."""
    interactions = []
    current_interaction = {}
    
    for line in text_section.split('\n'):
        line = line.strip()
        if not line:
            if current_interaction:
                try:
                    interactions.append(SocialInteractionDetail(**current_interaction))
                except Exception as e:
                    print(f"Warning: Could not create SocialInteractionDetail: {e}")
                current_interaction = {}
            continue
            
        if line.endswith(':'):  # New person
            if current_interaction:
                try:
                    interactions.append(SocialInteractionDetail(**current_interaction))
                except Exception as e:
                    print(f"Warning: Could not create SocialInteractionDetail: {e}")
            current_interaction = {"person_name": line[:-1].strip()}
        elif line.startswith('-') and ':' in line:
            if not current_interaction:
                continue
            key, value = line[1:].split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()
            
            if key == 'total_interaction_time':
                current_interaction['total_interaction_time'] = value
            elif key == 'overall_inferred_impact':
                current_interaction['overall_inferred_impact'] = value
            elif key == 'key_observation':
                current_interaction['key_observation'] = value
    
    if current_interaction:
        try:
            interactions.append(SocialInteractionDetail(**current_interaction))
        except Exception as e:
            print(f"Warning: Could not create final SocialInteractionDetail: {e}")
    
    return interactions

def parse_topic_analysis(text_section: str) -> List[TopicDetail]:
    """Parse topic analysis section into structured data."""
    topics = []
    for line in text_section.split('\n'):
        line = line.strip()
        if not line or line.lower().startswith(('description:', 'data:')):
            continue
            
        match = re.match(r'^(?:(\d+)\.\s*)?([^(]+)(?:\s*\(([^)]+)\))?', line)
        if match:
            rank_str, topic_name, details = match.groups()
            
            topic_data = {
                'rank': safe_int(rank_str),
                'topic_name': topic_name.strip(),
                'raw_description': details.strip() if details else "",
                'num_events': None,
                'total_duration_minutes': None
            }
            
            if details:
                events_match = re.search(r'(\d+)\s*event', details.lower())
                duration_match = re.search(r'(\d+)\s*minute', details.lower())
                
                if events_match:
                    topic_data['num_events'] = safe_int(events_match.group(1))
                if duration_match:
                    topic_data['total_duration_minutes'] = safe_int(duration_match.group(1))
            
            try:
                topics.append(TopicDetail(**topic_data))
            except Exception as e:
                print(f"Warning: Could not create TopicDetail: {e}")
                
    return topics

def parse_distribution(section: str) -> Dict[str, int]:
    """Parse a distribution section into a dictionary of label -> minutes."""
    result = {}
    for line in section.split('\n'):
        line = line.strip()
        if not line or ':' not in line:
            continue
            
        label, value = line.split(':', 1)
        label = label.strip()
        
        match = re.search(r'(\d+)\s*minutes?', value)
        if match:
            minutes = int(match.group(1))
            result[label] = minutes
            
    return result

def extract_json_sections(text: str) -> Dict[str, Any]:
    """Extract and parse JSON sections from the raw response."""
    result = {
        'events': [],
        'overall_scores': {
            'daily_mood_score': 5.0,
            'daily_stress_level_score': 5.0,
            'daily_energy_level_score': 5.0
        },
        'energy_timeline': [["00:00", "5.0"], ["23:59", "5.0"]],
        'mood_distribution': {'neutral': 1440},
        'awake_time_allocation': {'unknown': 1440},
        'social_interactions': [],
        'topic_analysis': []
    }
    
    try:
        # Extract events JSON array (Step 2)
        events_match = re.search(r'\[\s*\{.*?\}\s*\]', text, re.DOTALL)
        if events_match:
            events_text = events_match.group(0)
            try:
                events_data = json.loads(events_text)
                # Ensure each event has required fields
                for event in events_data:
                    event.setdefault('event_id', str(uuid.uuid4()))
                    event.setdefault('event_title', 'Untitled Event')
                    event.setdefault('time_range', '00:00-23:59')
                    event.setdefault('duration_minutes', 1440)
                    event.setdefault('mood_labels', ['neutral'])
                    event.setdefault('mood_score', 5.0)
                    event.setdefault('stress_level', 5.0)
                    event.setdefault('energy_level', 5.0)
                    event.setdefault('activity_type', 'unknown')
                    event.setdefault('people_involved', [])
                    event.setdefault('interaction_dynamic', 'neutral')
                    event.setdefault('inferred_impact_on_wei', 'neutral')
                    event.setdefault('topic_labels', ['daily life'])
                    event.setdefault('context_summary', '')
                    event.setdefault('key_quote_or_moment', '')
                result['events'] = events_data
                
                # Calculate overall scores from events if available
                if events_data:
                    result['overall_scores']['daily_mood_score'] = calculate_weighted_average(events_data, 'mood_score')
                    result['overall_scores']['daily_stress_level_score'] = calculate_weighted_average(events_data, 'stress_level')
                    result['overall_scores']['daily_energy_level_score'] = calculate_weighted_average(events_data, 'energy_level')
                    
                    # Generate energy timeline from events
                    timeline = []
                    for event in sorted(events_data, key=lambda x: x['time_range'].split('-')[0]):
                        try:
                            start_time = event['time_range'].split('-')[0].strip()
                            timeline.append([start_time, str(event['energy_level'])])
                        except:
                            continue
                    if timeline:
                        result['energy_timeline'] = timeline
                    
                    # Calculate mood distribution
                    mood_minutes = defaultdict(int)
                    for event in events_data:
                        primary_mood = event['mood_labels'][0] if event['mood_labels'] else 'neutral'
                        mood_minutes[primary_mood] += event['duration_minutes']
                    if mood_minutes:
                        result['mood_distribution'] = dict(mood_minutes)
                    
                    # Calculate activity allocation
                    activity_minutes = defaultdict(int)
                    for event in events_data:
                        activity_minutes[event['activity_type']] += event['duration_minutes']
                    if activity_minutes:
                        result['awake_time_allocation'] = dict(activity_minutes)
                    
                    # Extract social interactions
                    interactions = defaultdict(lambda: {
                        'person_name': '',
                        'total_interaction_time': '0 minutes',
                        'overall_inferred_impact': 'neutral',
                        'key_observation': ''
                    })
                    
                    for event in events_data:
                        for person in event['people_involved']:
                            if person != 'Self':
                                interactions[person]['person_name'] = person
                                current_minutes = int(interactions[person]['total_interaction_time'].split()[0])
                                interactions[person]['total_interaction_time'] = f"{current_minutes + event['duration_minutes']} minutes"
                                impact = event['inferred_impact_on_wei']
                                if impact == 'energizing':
                                    interactions[person]['overall_inferred_impact'] = 'energizing'
                                elif impact == 'draining' and interactions[person]['overall_inferred_impact'] != 'energizing':
                                    interactions[person]['overall_inferred_impact'] = 'draining'
                                # Add key observation from the first event with this person
                                if not interactions[person]['key_observation']:
                                    interactions[person]['key_observation'] = event['context_summary']
                    
                    if interactions:
                        result['social_interactions'] = list(interactions.values())
                    
                    # Extract topic analysis
                    topics = defaultdict(lambda: {
                        'topic_name': '',
                        'num_events': 0,
                        'total_duration_minutes': 0,
                        'raw_description': ''
                    })
                    
                    for event in events_data:
                        for topic in event['topic_labels']:
                            if topic != 'N/A':
                                topics[topic]['topic_name'] = topic
                                topics[topic]['num_events'] += 1
                                topics[topic]['total_duration_minutes'] += event['duration_minutes']
                    
                    if topics:
                        result['topic_analysis'] = list(topics.values())
                
            except json.JSONDecodeError:
                print("Warning: Could not parse events JSON")
                result['events'] = [{
                    'event_id': str(uuid.uuid4()),
                    'event_title': 'Daily Summary',
                    'time_range': '00:00-23:59',
                    'duration_minutes': 1440,
                    'mood_labels': ['neutral'],
                    'mood_score': 5.0,
                    'stress_level': 5.0,
                    'energy_level': 5.0,
                    'activity_type': 'unknown',
                    'people_involved': [],
                    'interaction_dynamic': 'neutral',
                    'inferred_impact_on_wei': 'neutral',
                    'topic_labels': ['daily life'],
                    'context_summary': 'Auto-generated daily summary',
                    'key_quote_or_moment': 'No specific moment recorded'
                }]
        
        # Extract Step 3 sections only if we don't have event-based calculations
        if not result['events']:
            step3_match = re.search(r'Step 3:.*', text, re.DOTALL)
            if step3_match:
                step3_text = step3_match.group(0)
                
                # Extract overall scores
                scores_pattern = r'Daily (\w+) Score:.*?\[(\d+\.?\d*)\]/10'
                scores = re.findall(scores_pattern, step3_text)
                for score_type, value in scores:
                    score_type = score_type.lower()
                    if score_type == 'mood':
                        result['overall_scores']['daily_mood_score'] = float(value)
                    elif score_type == 'stress':
                        result['overall_scores']['daily_stress_level_score'] = float(value)
                    elif score_type == 'energy':
                        result['overall_scores']['daily_energy_level_score'] = float(value)
                
                # Extract energy timeline
                timeline_match = re.search(r'\[\s*\[".*?\]\s*\]', step3_text)
                if timeline_match:
                    try:
                        timeline_text = timeline_match.group(0)
                        timeline_data = json.loads(timeline_text)
                        if timeline_data and len(timeline_data) > 0:
                            result['energy_timeline'] = timeline_data
                    except json.JSONDecodeError:
                        print("Warning: Could not parse energy timeline")
                
                # Extract mood distribution
                mood_match = re.search(r'\{".*?"\s*:\s*\d+.*?\}', step3_text)
                if mood_match:
                    try:
                        mood_text = mood_match.group(0)
                        mood_data = json.loads(mood_text)
                        if mood_data:
                            result['mood_distribution'] = mood_data
                    except json.JSONDecodeError:
                        print("Warning: Could not parse mood distribution")
                
                # Extract awake time allocation
                time_match = re.search(r'\{"work".*?\}', step3_text)
                if time_match:
                    try:
                        time_text = time_match.group(0)
                        time_data = json.loads(time_text)
                        if time_data:
                            result['awake_time_allocation'] = time_data
                    except json.JSONDecodeError:
                        print("Warning: Could not parse time allocation")
                
                # Extract social interactions
                social_matches = re.finditer(r'\{\s*"person_name".*?\}', step3_text)
                for match in social_matches:
                    try:
                        interaction = json.loads(match.group(0))
                        if 'person_name' in interaction:
                            result['social_interactions'].append(interaction)
                    except json.JSONDecodeError:
                        continue
                
                # Extract topic analysis
                topic_matches = re.finditer(r'\{\s*"topic_name".*?\}', step3_text)
                for match in topic_matches:
                    try:
                        topic = json.loads(match.group(0))
                        if 'topic_name' in topic:
                            result['topic_analysis'].append(topic)
                    except json.JSONDecodeError:
                        continue
    
    except Exception as e:
        print(f"Error extracting JSON sections: {e}")
    
    return result

def parse_raw_response(file_path: Path) -> DailyAnalysis:
    """Parse a raw response file into a DailyAnalysis object."""
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    # Extract the response part
    response_marker = "=== RESPONSE ==="
    response_start = raw_text.find(response_marker)
    if response_start == -1:
        raise ValueError(f"Could not find response marker in {file_path}")
    raw_text = raw_text[response_start + len(response_marker):].strip()
    
    # Extract and parse JSON sections
    structured_data = extract_json_sections(raw_text)
    
    # Create the raw response model
    raw_response = RawResponse(
        events=structured_data['events'],
        overall_scores=structured_data['overall_scores'],
        energy_timeline=structured_data['energy_timeline'],
        mood_distribution=[MoodEntry(mood=k, minutes=v) for k, v in structured_data['mood_distribution'].items()],
        awake_time_allocation=[ActivityEntry(activity=k, minutes=v) for k, v in structured_data['awake_time_allocation'].items()],
        social_interactions=structured_data['social_interactions'],
        topic_analysis=structured_data['topic_analysis']
    )
    
    # Convert to final analysis model
    return DailyAnalysis(
        events=raw_response.events,
        energy_timeline=raw_response.energy_timeline,
        overall_scores=OverallScores(**raw_response.overall_scores),
        mood_distribution={entry.mood: entry.minutes for entry in raw_response.mood_distribution},
        awake_time_allocation={entry.activity: entry.minutes for entry in raw_response.awake_time_allocation},
        social_interactions=raw_response.social_interactions,
        topic_analysis=raw_response.topic_analysis
    )

def save_analysis(analysis: DailyAnalysis, output_dir: Path, date: datetime) -> None:
    """Save the analysis data to a JSON file."""
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"daily_analysis_{date.strftime('%Y-%m-%d')}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(analysis.model_dump(), f, indent=2, ensure_ascii=False)
    
    print(f"Saved analysis to {output_file}")

def process_raw_responses():
    """Process raw responses and generate structured analysis files."""
    debug_dir = Path("debug")
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Find all raw response files
    response_files = list(debug_dir.glob("raw_response_*.txt"))
    print(f"Found {len(response_files)} raw response files")
    
    for file_path in response_files:
        try:
            # Extract date from filename
            date_str = file_path.stem.replace("raw_response_", "")
            print(f"\nProcessing response for {date_str}")
            
            # Parse and save the analysis
            analysis = parse_raw_response(file_path)
            save_analysis(analysis, analysis_dir, datetime.strptime(date_str, '%Y-%m-%d'))
            print(f"Successfully processed analysis for {date_str}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

if __name__ == "__main__":
    process_raw_responses() 