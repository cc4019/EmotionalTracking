import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from models import (
    EnergyLevel, TimePeriod,
    EnergyLevelEntry, AnalysisResult
)
from anthropic_analyzer import AnthropicAnalyzer

# Load environment variables
load_dotenv()

# Initialize Anthropic analyzer if API key is available
anthropic_analyzer = None
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY:
    try:
        anthropic_analyzer = AnthropicAnalyzer()  # No need to pass API key anymore
        print("Successfully initialized Anthropic analyzer")
    except Exception as e:
        print(f"Failed to initialize Anthropic analyzer: {e}")
else:
    print("No Anthropic API key found in environment variables")

def extract_date_from_filename(filename: str) -> datetime:
    """Extract date from filename format MM-DD_..."""
    date_match = re.match(r"(\d{2})-(\d{2})_", filename)
    if date_match:
        month, day = date_match.groups()
        # Assuming current year for the analysis
        current_year = datetime.now().year
        return datetime(current_year, int(month), int(day))
    return None

def analyze_energy_level(text: str, date: datetime = None) -> List[EnergyLevelEntry]:
    """Analyze text for energy levels."""
    entries = []
    high_energy_patterns = [
        r"excited", r"energetic", r"great!", r"amazing",
        r"fantastic", r"wonderful", r"excellent", r"enthusiastic",
        r"motivated", r"passionate", r"thrilled"
    ]
    low_energy_patterns = [
        r"tired", r"exhausted", r"drained", r"low energy",
        r"not feeling well", r"struggling", r"fatigued",
        r"worn out", r"sluggish", r"lethargic"
    ]
    medium_energy_patterns = [
        r"okay", r"fine", r"alright", r"moderate",
        r"steady", r"stable", r"balanced"
    ]
    
    # Determine time period based on context
    period = TimePeriod.MORNING  # Default to morning
    
    for pattern in high_energy_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            entries.append(EnergyLevelEntry(
                timestamp=date,
                period=period,
                energy_level=EnergyLevel.HIGH,
                context="High energy detected",
                supporting_text=text[max(0, match.start()-30):match.end()+30]
            ))
    
    for pattern in medium_energy_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            entries.append(EnergyLevelEntry(
                timestamp=date,
                period=period,
                energy_level=EnergyLevel.MEDIUM,
                context="Medium energy detected",
                supporting_text=text[max(0, match.start()-30):match.end()+30]
            ))
    
    for pattern in low_energy_patterns:
        matches = re.finditer(pattern, text.lower())
        for match in matches:
            entries.append(EnergyLevelEntry(
                timestamp=date,
                period=period,
                energy_level=EnergyLevel.LOW,
                context="Low energy detected",
                supporting_text=text[max(0, match.start()-30):match.end()+30]
            ))
    
    return entries

def analyze_text(text: str, date: Optional[datetime] = None, analyzer = None) -> AnalysisResult:
    """Analyze text using the provided analyzer and return structured results.
    
    Args:
        text: The text to analyze
        date: Optional datetime for the analysis
        analyzer: The analyzer instance to use (AnthropicAnalyzer or GoogleAnalyzer)
    
    Returns:
        AnalysisResult containing the analysis results
    """
    if analyzer is None:
        raise ValueError("An analyzer must be provided")
        
    return analyzer.analyze_text(text, date)

def analyze_file(file_path: Path) -> Tuple[datetime, AnalysisResult]:
    """Analyze a single file and return its date and results."""
    print(f"Analyzing file: {file_path}")
    date = extract_date_from_filename(file_path.name)
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return date, analyze_text(text, date, anthropic_analyzer)

def analyze_all_files(directory: Path = Path("raw_data")) -> Dict[datetime, AnalysisResult]:
    """Analyze all files in the directory and return results by date."""
    print(f"Analyzing files in directory: {directory}")
    results = {}
    if not directory.exists():
        print(f"Directory {directory} does not exist")
        return results
        
    files = list(directory.glob("*.txt"))
    print(f"Found {len(files)} text files")
    
    for file_path in files:
        try:
            print(f"Processing file: {file_path.name}")
            date, result = analyze_file(file_path)
            if date:
                results[date] = result
            else:
                print(f"Could not extract date from filename: {file_path.name}")
        except Exception as e:
            print(f"Error processing file {file_path.name}: {e}")
            
    return results 