import streamlit as st
import json
import pandas as pd
import altair as alt
from pathlib import Path
from datetime import datetime
import glob

# Ice cream color palette
ICE_CREAM_COLORS = [
    '#FFB5C2',  # Strawberry
    '#FFE5B4',  # Vanilla
    '#98FF98',  # Mint
    '#FFB347',  # Orange Sherbet
    '#E6BEE6',  # Lavender
    '#87CEEB',  # Blueberry
    '#F8B878',  # Caramel
    '#DDA0DD',  # Berry
    '#F0E68C',  # Lemon
    '#E6E6FA'   # Cream
]

# Move set_page_config to the very top
st.set_page_config(
    page_title="Nirva - Daily Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
    <style>
    .stMetric .label { font-size: 14px !important; }
    .stMetric .value { font-size: 24px !important; }
    div[data-testid="stExpander"] div[role="button"] p { font-size: 14px !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_all_analysis_files():
    """Load all analysis files from the analysis directory."""
    analysis_dir = Path("analysis")
    if not analysis_dir.exists():
        st.error("Analysis directory not found. Please run the analysis script first.")
        return {}
    
    analysis_files = glob.glob(str(analysis_dir / "daily_analysis_*.json"))
    if not analysis_files:
        st.error("No analysis files found. Please run the analysis script first.")
        return {}
    
    data_by_date = {}
    for file_path in analysis_files:
        try:
            date_str = Path(file_path).stem.replace("daily_analysis_", "")
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data_by_date[date] = data
        except Exception as e:
            st.warning(f"Error loading {file_path}: {e}")
            continue
    
    return data_by_date

def display_overall_scores(scores, comparison_scores=None):
    """Display overall scores with optional comparison."""
    if not scores or not isinstance(scores, dict):
        st.warning("Overall scores are not available.")
        return
        
    st.subheader("Overall Daily Scores")
    cols = st.columns(3)
    
    metrics = {
        "Daily Mood": "daily_mood_score",
        "Daily Stress Level": "daily_stress_level_score",
        "Daily Energy Level": "daily_energy_level_score"
    }
    
    for i, (label, key) in enumerate(metrics.items()):
        current = float(scores.get(key, 0))  # Ensure numeric value
        if comparison_scores:
            prev = float(comparison_scores.get(key, 0))
            delta = current - prev
            cols[i].metric(
                label, 
                f"{current:.1f}/10",
                f"{delta:+.1f}",
                delta_color="normal" if key == "daily_stress_level_score" else "inverse"
            )
        else:
            cols[i].metric(label, f"{current:.1f}/10")

def display_energy_timeline(timeline_data, comparison_data=None):
    """Display energy timeline with optional comparison."""
    st.subheader("Energy Level Timeline")
    if not timeline_data or not isinstance(timeline_data, list):
        st.warning("Energy timeline data is not available.")
        return

    try:
        # Convert array of [timestamp, energy_level] pairs to DataFrame
        df = pd.DataFrame(timeline_data, columns=['Time', 'Energy'])
        df['Energy'] = pd.to_numeric(df['Energy'], errors='coerce')
        df['Source'] = 'Current Day'
        
        if comparison_data:
            df_comp = pd.DataFrame(comparison_data, columns=['Time', 'Energy'])
            df_comp['Energy'] = pd.to_numeric(df_comp['Energy'], errors='coerce')
            df_comp['Source'] = 'Previous Day'
            df = pd.concat([df, df_comp])
        
        # Sort by time to ensure proper line connection
        df['Time'] = pd.Categorical(df['Time'], categories=df['Time'].unique(), ordered=True)
        
        # Add energy level labels
        energy_ranges = {
            'Very Low': (1, 2),
            'Low': (3, 4),
            'Neutral': (5, 6),
            'High': (7, 8),
            'Very High': (9, 10)
        }
        
        df['Energy_Label'] = pd.cut(
            df['Energy'],
            bins=[0] + [r[1] for r in energy_ranges.values()],
            labels=list(energy_ranges.keys()),
            include_lowest=True
        )
        
        # Create the main chart
        line = alt.Chart(df).mark_line(
            strokeWidth=3,
            point=True
        ).encode(
            x=alt.X('Time:N', 
                   sort=None, 
                   title='Time of Day',
                   axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('Energy:Q', 
                   scale=alt.Scale(domain=[0, 10]), 
                   title='Energy Level',
                   axis=alt.Axis(
                       values=[1, 3, 5, 7, 9],
                       labels=True
                   )),
            color=alt.Color(
                'Source:N',
                scale=alt.Scale(range=ICE_CREAM_COLORS[:2])
            ),
            tooltip=['Time', 'Energy', 'Energy_Label', 'Source']
        )
        
        # Add background bands for energy levels
        background = alt.Chart(pd.DataFrame({
            'y1': [0, 2, 4, 6, 8],
            'y2': [2, 4, 6, 8, 10],
            'label': ['Very Low', 'Low', 'Neutral', 'High', 'Very High']
        })).mark_rect(opacity=0.1).encode(
            y='y1:Q',
            y2='y2:Q',
            color=alt.Color(
                'label:N',
                scale=alt.Scale(range=ICE_CREAM_COLORS[2:7]),
                legend=alt.Legend(title='Energy Levels')
            )
        )
        
        # Combine charts
        chart = (background + line).properties(
            height=300,
            title='Energy Level Throughout the Day'
        )
        
        st.altair_chart(chart, use_container_width=True)
            
    except Exception as e:
        st.error(f"Could not display energy timeline: {e}")

def display_mood_distribution(mood_data, comparison_data=None):
    """Display mood distribution with optional comparison."""
    st.subheader("Mood Distribution")
    if not mood_data or not isinstance(mood_data, dict):
        st.warning("Mood distribution data is not available.")
        return

    try:
        # Convert duration-based mood data to DataFrame
        df = pd.DataFrame(list(mood_data.items()), columns=['Mood', 'Percentage'])
        df['Source'] = 'Current Day'
        
        if comparison_data:
            df_comp = pd.DataFrame(list(comparison_data.items()), columns=['Mood', 'Percentage'])
            df_comp['Source'] = 'Previous Day'
            df = pd.concat([df, df_comp])
        
        # Create pie chart for current day
        current_day = df[df['Source'] == 'Current Day'].copy()
        pie = alt.Chart(current_day).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field='Percentage', type='quantitative', stack=True),
            color=alt.Color(
                'Mood:N',
                scale=alt.Scale(range=ICE_CREAM_COLORS),
                legend=alt.Legend(title='Moods', orient='right')
            ),
            tooltip=[
                alt.Tooltip('Mood:N'),
                alt.Tooltip('Percentage:Q', format='.1f', title='% of Day')
            ]
        ).properties(
            width=400,
            height=400,
            title={
                "text": ['Mood Distribution', '(% of Day)'],
                "align": "center",
                "anchor": "middle"
            }
        )
        
        # Add text labels for percentages
        text = pie.mark_text(radius=90, size=11).encode(
            text=alt.Text('Percentage:Q', format='.1f'),
            color=alt.value('black')
        )
        
        # Combine pie chart and labels
        chart = (pie + text)
        
        if comparison_data:
            # Create comparison bar chart
            bar = alt.Chart(df).mark_bar().encode(
                x=alt.X('Source:N', title=None),
                y=alt.Y('Percentage:Q', title='% of Day'),
                color=alt.Color(
                    'Source:N',
                    scale=alt.Scale(range=ICE_CREAM_COLORS[:2])
                ),
                column=alt.Column(
                    'Mood:N',
                    sort=alt.SortField(field='Percentage', order='descending')
                ),
                tooltip=[
                    'Mood',
                    alt.Tooltip('Percentage:Q', format='.1f', title='% of Day')
                ]
            ).properties(width=100)
            
            # Display both charts
            col1, col2 = st.columns([1, 2])
            with col1:
                st.altair_chart(chart)
            with col2:
                st.altair_chart(bar, use_container_width=True)
        else:
            # Display only pie chart
            st.altair_chart(chart, use_container_width=True)
            
    except Exception as e:
        st.error(f"Could not display mood distribution: {e}")

def display_awake_time_allocation(time_data, comparison_data=None):
    """Display awake time allocation with optional comparison."""
    st.subheader("Awake Time Allocation")
    if not time_data or not isinstance(time_data, dict):
        st.warning("Awake time allocation data is not available.")
        return

    try:
        # Convert activity duration data to DataFrame
        df = pd.DataFrame(list(time_data.items()), columns=['Activity', 'Percentage'])
        df['Source'] = 'Current Day'
        
        if comparison_data:
            df_comp = pd.DataFrame(list(comparison_data.items()), columns=['Activity', 'Percentage'])
            df_comp['Source'] = 'Previous Day'
            df = pd.concat([df, df_comp])
        
        # Create stacked bar chart
        chart = alt.Chart(df).mark_bar().encode(
            y=alt.Y(
                'Activity:N',
                sort=alt.SortField(field='Percentage', order='descending'),
                title='Activity Type'
            ),
            x=alt.X(
                'Percentage:Q',
                title='% of Day',
                axis=alt.Axis(format='.1f')
            ),
            color=alt.Color(
                'Source:N',
                scale=alt.Scale(range=ICE_CREAM_COLORS[:2])
            ),
            tooltip=[
                'Activity',
                alt.Tooltip('Percentage:Q', format='.1f', title='% of Day'),
                'Source'
            ]
        ).properties(
            height=300,
            title='Time Allocation by Activity (% of Day)'
        )
        
        st.altair_chart(chart, use_container_width=True)
    except Exception as e:
        st.error(f"Could not display awake time allocation: {e}")

def display_event_details(events):
    """Display event details in an expandable table."""
    st.subheader("Event Details")
    if not events or not isinstance(events, list):
        st.warning("No event data is available.")
        return

    try:
        df = pd.DataFrame(events)
        if df.empty:
            st.warning("Event data is empty.")
            return

        # Create a summary table
        display_cols = {
            "event_title": "Title",
            "time_range": "Time",
            "duration_minutes": "Duration (min)",
            "mood_score": "Mood",
            "energy_level": "Energy",
            "activity_type": "Activity"
        }
        cols_to_display = [col for col in display_cols.keys() if col in df.columns]
        df_display = df[cols_to_display].rename(columns=display_cols)
        
        # Display summary table
        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Show detailed view in expanders
        show_details = st.checkbox("Show detailed event information")
        if show_details:
            for i, event in enumerate(events, 1):
                with st.expander(f"Event {i}: {event.get('event_title', 'Untitled')}"):
                    col1, col2 = st.columns(2)
                    
                    # Left column: Basic info
                    col1.markdown("**Basic Information**")
                    col1.write(f"Time: {event.get('time_range', 'N/A')}")
                    col1.write(f"Duration: {event.get('duration_minutes', 'N/A')} minutes")
                    col1.write(f"Activity: {event.get('activity_type', 'N/A')}")
                    
                    # Right column: Scores
                    col2.markdown("**Scores & Labels**")
                    col2.write(f"Mood: {event.get('mood_score', 'N/A')}/10")
                    col2.write(f"Energy: {event.get('energy_level', 'N/A')}/10")
                    col2.write(f"Stress: {event.get('stress_level', 'N/A')}/10")
                    
                    # Full width: Additional details
                    st.markdown("**Details**")
                    st.write(f"Context: {event.get('context_summary', 'N/A')}")
                    st.write(f"Key Moment: {event.get('key_quote_or_moment', 'N/A')}")
                    
                    if event.get('people_involved'):
                        st.write("People Involved:", ", ".join(event['people_involved']))
                    
                    if event.get('topic_labels'):
                        st.write("Topics:", ", ".join(event['topic_labels']))
    except Exception as e:
        st.error(f"Could not display event details: {e}")

def display_social_interactions(interactions):
    """Display social interactions summary."""
    st.subheader("Social Interactions")
    if not interactions or not isinstance(interactions, list):
        st.warning("Social interaction data is not available.")
        return

    try:
        for interaction in interactions:
            with st.expander(f"👤 {interaction.get('person_name', 'Unknown Person')}"):
                col1, col2 = st.columns(2)
                
                col1.markdown("**Interaction Details**")
                col1.write(f"Duration: {interaction.get('total_interaction_time', 'Unknown')}")
                col1.write(f"Impact: {interaction.get('overall_inferred_impact', 'Neutral')}")
                
                col2.markdown("**Key Observation**")
                col2.write(interaction.get('key_observation', 'No observation recorded'))
                
                # Add interaction pattern if available
                if interaction.get('interaction_pattern'):
                    st.markdown("**Interaction Pattern**")
                    st.info(interaction['interaction_pattern'])
    except Exception as e:
        st.error(f"Could not display social interactions: {e}")

def display_topic_analysis(topics):
    """Display topic analysis with visualization."""
    st.subheader("Topic Analysis")
    if not topics or not isinstance(topics, list):
        st.warning("Topic analysis data is not available.")
        return

    try:
        df = pd.DataFrame(topics)
        if df.empty:
            st.warning("Topic analysis data is empty.")
            return
            
        # Create visualization
        if 'total_duration_minutes' in df.columns and 'topic_name' in df.columns:
            # Calculate percentage for better comparison
            total_minutes = df['total_duration_minutes'].sum()
            df['percentage'] = (df['total_duration_minutes'] / total_minutes * 100).round(1)
            
            # Create bar chart
            bars = alt.Chart(df).mark_bar().encode(
                y=alt.Y(
                    'topic_name:N',
                    sort=alt.SortField(field='total_duration_minutes', order='descending'),
                    title='Topic'
                ),
                x=alt.X(
                    'total_duration_minutes:Q',
                    title='Duration (minutes)',
                    axis=alt.Axis(format='d')
                ),
                color=alt.Color(
                    'topic_name:N',
                    scale=alt.Scale(range=ICE_CREAM_COLORS),
                    legend=None
                ),
                tooltip=[
                    'topic_name',
                    alt.Tooltip('total_duration_minutes:Q', format='.0f', title='Duration (min)'),
                    alt.Tooltip('percentage:Q', format='.1f', title='% of Total Time'),
                    alt.Tooltip('num_events:Q', format='d', title='Mentions')
                ]
            )
            
            # Add text labels
            text = bars.mark_text(
                align='left',
                baseline='middle',
                dx=3
            ).encode(
                text=alt.Text('num_events:Q', format='d'),
                color=alt.value('black')
            )
            
            # Combine charts
            chart = (bars + text).properties(
                height=300,
                title='Topic Distribution'
            )
            
            st.altair_chart(chart, use_container_width=True)
        
        # Display ranked list
        st.markdown("### Ranked Topics")
        for _, row in df.sort_values('total_duration_minutes', ascending=False).iterrows():
            st.markdown(
                f"**{row['topic_name']}** ({row['num_events']} mentions / "
                f"{int(row['total_duration_minutes'])} mins)"
            )
            if 'raw_description' in row and row['raw_description']:
                st.caption(row['raw_description'])
                
    except Exception as e:
        st.error(f"Could not display topic analysis: {e}")

def main():
    st.title("Nirva - Daily Analysis Dashboard")
    
    # Load all analysis data
    data_by_date = load_all_analysis_files()
    if not data_by_date:
        return
    
    # Sidebar controls
    st.sidebar.title("Controls")
    
    # Date selection
    dates = sorted(data_by_date.keys(), reverse=True)
    selected_date = st.sidebar.selectbox(
        "Select Date",
        dates,
        format_func=lambda x: x.strftime("%B %d, %Y")
    )
    
    # Comparison options
    enable_comparison = st.sidebar.checkbox("Enable Comparison")
    comparison_date = None
    if enable_comparison:
        other_dates = [d for d in dates if d != selected_date]
        if other_dates:
            comparison_date = st.sidebar.selectbox(
                "Compare with",
                other_dates,
                format_func=lambda x: x.strftime("%B %d, %Y")
            )
    
    # Get selected data
    data = data_by_date[selected_date]
    comparison_data = data_by_date.get(comparison_date) if comparison_date else None
    
    # Display date header
    st.header(f"Analysis for {selected_date.strftime('%B %d, %Y')}")
    if comparison_date:
        st.caption(f"Comparing with {comparison_date.strftime('%B %d, %Y')}")
    
    # Display sections
    display_overall_scores(
        data.get('overall_scores'),
        comparison_data.get('overall_scores') if comparison_data else None
    )
    
    # Create two columns for the main visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        display_energy_timeline(
            data.get('energy_timeline'),
            comparison_data.get('energy_timeline') if comparison_data else None
        )
        display_mood_distribution(
            data.get('mood_distribution'),
            comparison_data.get('mood_distribution') if comparison_data else None
        )
    
    with col2:
        display_awake_time_allocation(
            data.get('awake_time_allocation'),
            comparison_data.get('awake_time_allocation') if comparison_data else None
        )
        display_topic_analysis(data.get('topic_analysis'))
    
    # Full width sections
    display_event_details(data.get('events'))
    display_social_interactions(data.get('social_interactions'))

if __name__ == "__main__":
    main()

# To run the app: streamlit run streamlit_app.py 