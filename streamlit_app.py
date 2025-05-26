import streamlit as st
import json
import pandas as pd
import altair as alt # For charts

# Move set_page_config to the very top
st.set_page_config(page_title="Daily Analysis Dashboard", layout="wide")

# Define Pydantic models (mirroring those in run_analysis.py for data validation/typing)
# This is good practice but can be omitted if direct dictionary access is preferred
# For brevity in this example, direct dictionary access will be used after loading JSON.

@st.cache_data # Cache the data loading
def load_data(file_path="daily_analysis_2025-05-10.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"Error: The file {file_path} was not found. Please run the analysis script.")
        return None
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {file_path}. The file might be corrupted.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data: {e}")
        return None

def display_overall_scores(scores):
    if not scores or not isinstance(scores, dict):
        st.warning("Overall scores are not available.")
        return
    st.subheader("Overall Daily Scores")
    cols = st.columns(3)
    cols[0].metric("Daily Mood Score", f"{scores.get('daily_mood_score', 'N/A'):.1f}/10")
    cols[1].metric("Daily Stress Level", f"{scores.get('daily_stress_level_score', 'N/A'):.1f}/10")
    cols[2].metric("Daily Energy Level", f"{scores.get('daily_energy_level_score', 'N/A'):.1f}/10")

def display_energy_timeline(timeline_data):
    st.subheader("Energy Level Timeline")
    if not timeline_data or not isinstance(timeline_data, list) or not all(isinstance(item, list) and len(item) == 2 for item in timeline_data):
        st.warning("Energy timeline data is not available or in the wrong format.")
        return

    try:
        df_timeline = pd.DataFrame(timeline_data, columns=['Time', 'Energy'])
        df_timeline['Energy'] = pd.to_numeric(df_timeline['Energy'])
        
        # Ensure 'Time' is treated as categorical for proper ordering if not already datetime
        df_timeline['Time'] = pd.Categorical(df_timeline['Time'], categories=df_timeline['Time'].unique(), ordered=True)

        chart = alt.Chart(df_timeline).mark_line(point=True).encode(
            x=alt.X('Time:N', sort=None, title='Time of Day'), # Use :N for nominal/categorical if not true datetime
            y=alt.Y('Energy:Q', scale=alt.Scale(domain=[0, 10]), title='Energy Level (0-10)'),
            tooltip=['Time', 'Energy']
        ).properties(
            title='Energy Level Throughout the Day'
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Could not display energy timeline: {e}")
        st.write("Data received for timeline:", timeline_data)


def display_mood_distribution(mood_data):
    st.subheader("Mood Distribution")
    if not mood_data or not isinstance(mood_data, dict) or not mood_data.values(): # Check if dict is not empty
        st.warning("Mood distribution data is not available.")
        return
    
    try:
        df_mood = pd.DataFrame(list(mood_data.items()), columns=['Mood', 'Minutes'])
        df_mood['Minutes'] = pd.to_numeric(df_mood['Minutes'])

        if df_mood.empty:
            st.warning("Mood distribution data is empty after processing.")
            return

        chart = alt.Chart(df_mood).mark_bar().encode(
            x='Minutes:Q',
            y=alt.Y('Mood:N', sort='-x'),
            tooltip=['Mood', 'Minutes']
        ).properties(
            title='Time Spent in Different Moods'
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Could not display mood distribution: {e}")
        st.write("Data received for mood:", mood_data)


def display_awake_time_allocation(time_data):
    st.subheader("Awake Time Allocation")
    if not time_data or not isinstance(time_data, dict) or not time_data.values():
        st.warning("Awake time allocation data is not available.")
        return

    try:
        df_time = pd.DataFrame(list(time_data.items()), columns=['Activity', 'Minutes'])
        df_time['Minutes'] = pd.to_numeric(df_time['Minutes'])
        
        if df_time.empty:
            st.warning("Awake time allocation data is empty after processing.")
            return

        chart = alt.Chart(df_time).mark_bar().encode(
            x='Minutes:Q',
            y=alt.Y('Activity:N', sort='-x'),
            tooltip=['Activity', 'Minutes']
        ).properties(
            title='Time Spent on Different Activities'
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Could not display awake time allocation: {e}")
        st.write("Data received for time allocation:", time_data)

def display_event_details(events):
    st.subheader("Event Details")
    if not events or not isinstance(events, list):
        st.warning("No event data is available.")
        return

    try:
        # Convert list of event dicts to DataFrame
        df_events = pd.DataFrame(events)
        
        if df_events.empty:
            st.warning("Event data is empty after processing.")
            return

        # Select and rename columns for display if necessary
        # For simplicity, displaying a few key fields. Add more as needed.
        display_cols = {
            "event_title": "Title",
            "time_range": "Time",
            "duration_minutes": "Duration (min)",
            "mood_score": "Mood (0-10)",
            "energy_level": "Energy (0-10)",
            "activity_type": "Activity"
        }
        
        # Filter out columns that might not exist in all event dicts to prevent KeyErrors
        cols_to_display = [col for col in display_cols.keys() if col in df_events.columns]
        df_display_events = df_events[cols_to_display].rename(columns=display_cols)

        st.dataframe(df_display_events, use_container_width=True)

        # Optionally, add expanders for full details of each event
        for i, event in enumerate(events):
            with st.expander(f"{event.get('event_title', f'Event {i+1}')} - Full Details"):
                st.json(event) # Display full JSON for the event
                
    except Exception as e:
        st.error(f"Could not display event details: {e}")
        st.write("Data received for events:", events)


def display_social_interactions(interactions):
    st.subheader("Social Interaction Summary")
    if not interactions or not isinstance(interactions, list):
        st.warning("Social interaction data is not available.")
        return

    try:
        df_interactions = pd.DataFrame(interactions)
        if df_interactions.empty:
            st.warning("Social interaction data is empty after processing.")
            return
            
        # Define desired column order and names
        interaction_cols_ordered = {
            "person_name": "Person",
            "total_interaction_time": "Time",
            "overall_inferred_impact": "Impact",
            "key_observation": "Observation"
        }
        # Filter to only columns that exist in the DataFrame
        cols_to_display = [col for col in interaction_cols_ordered.keys() if col in df_interactions.columns]
        df_display_interactions = df_interactions[cols_to_display].rename(columns=interaction_cols_ordered)
        
        st.table(df_display_interactions) # Use st.table for better formatting of this type of data
    except Exception as e:
        st.error(f"Could not display social interactions: {e}")
        st.write("Data received for social interactions:", interactions)


def display_topic_analysis(topics):
    st.subheader("Topic Analysis")
    if not topics or not isinstance(topics, list):
        st.warning("Topic analysis data is not available.")
        return

    try:
        df_topics = pd.DataFrame(topics)
        if df_topics.empty:
            st.warning("Topic analysis data is empty after processing.")
            return

        # Define desired column order and names
        topic_cols_ordered = {
            "rank": "Rank",
            "topic_name": "Topic",
            "num_events": "Events",
            "total_duration_minutes": "Duration (min)",
            "raw_description": "Details" # Retained from previous implementation
        }
        # Filter to only columns that exist in the DataFrame
        cols_to_display = [col for col in topic_cols_ordered.keys() if col in df_topics.columns]
        df_display_topics = df_topics[cols_to_display].rename(columns=topic_cols_ordered)

        st.table(df_display_topics)
    except Exception as e:
        st.error(f"Could not display topic analysis: {e}")
        st.write("Data received for topic analysis:", topics)

# Main app flow
data = load_data()

if data:
    st.title("Daily Analysis Dashboard")
    
    # Layout columns for scores and timeline
    col1, col2 = st.columns([1, 2]) # Adjust ratio as needed
    
    with col1:
        display_overall_scores(data.get('overall_scores'))
        st.markdown("---") # Divider
        if data.get('mood_distribution'):
             display_mood_distribution(data['mood_distribution'])
        else:
            st.warning("Mood distribution data missing.")

    with col2:
        if data.get('energy_timeline'):
            display_energy_timeline(data['energy_timeline'])
        else:
            st.warning("Energy timeline data missing.")
        st.markdown("---") # Divider
        if data.get('awake_time_allocation'):
            display_awake_time_allocation(data['awake_time_allocation'])
        else:
            st.warning("Awake time allocation data missing.")
            
    st.markdown("---") # Divider
    
    # Events, Social Interactions, Topic Analysis in tabs for better organization
    tab_events, tab_social, tab_topics = st.tabs(["Detailed Events", "Social Interactions", "Topic Analysis"])

    with tab_events:
        if data.get('events'):
            display_event_details(data['events'])
        else:
            st.warning("Event data missing.")
            
    with tab_social:
        if data.get('social_interactions'):
            display_social_interactions(data['social_interactions'])
        else:
            st.warning("Social interaction data missing.")

    with tab_topics:
        if data.get('topic_analysis'):
            display_topic_analysis(data['topic_analysis'])
        else:
            st.warning("Topic analysis data missing.")
else:
    st.info("Please generate or load the analysis data to view the dashboard.")

# To run the app: streamlit run streamlit_app.py 