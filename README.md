# Meeting Analysis Dashboard

This Streamlit application analyzes meeting transcripts to provide insights into energy levels, social dynamics, mood patterns, and topic distribution.

## Features

- Upload and analyze meeting transcripts
- Advanced text analysis using Anthropic's Claude API
- Visualize energy level distribution
- Track social interaction patterns
- Monitor mood trends
- Analyze topic distribution
- Download analysis results

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Anthropic API key:
   - Copy the `.env.example` file to `.env`
   - Replace `your_api_key_here` with your actual Anthropic API key
   - If no API key is provided, the system will fall back to basic pattern matching

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Usage

1. The app will automatically load any `.txt` files from the `raw_data` directory
2. The system will use Anthropic's Claude API for advanced text analysis (if API key is provided)
3. Navigate through the different tabs to view various analyses:
   - Energy Levels: Distribution of energy levels throughout the meeting
   - Social Dynamics: Analysis of positive and negative interactions
   - Mood Analysis: Distribution of different moods
   - Topic Analysis: Distribution of discussed topics
4. Use the "View Raw Analysis Data" expander to see the detailed analysis results
5. Download the analysis results using the download buttons

## Analysis Methods

The system uses two methods of analysis:

1. Anthropic Claude API (Primary)
   - Uses advanced language understanding for better context awareness
   - More accurate identification of subtle emotional indicators
   - Better handling of complex social interactions
   - Requires API key

2. Pattern Matching (Fallback)
   - Uses regular expressions to identify key patterns
   - Works without external API dependencies
   - More limited in understanding context
   - Serves as a backup when API is unavailable

## Data Structure

The application analyzes text files for:

- Energy levels (High, Medium, Low)
- Social interactions (Positive, Negative)
- Moods (Happy, Sad, Anxious, etc.)
- Topics (Work, Social, Personal, etc.)

## Contributing

Feel free to submit issues and enhancement requests! 