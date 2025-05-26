# streamlit_energy_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import networkx as nx
from pyvis.network import Network
import base64
import matplotlib.pyplot as plt
import io
from IPython.display import HTML, display
import tempfile
import os

st.set_page_config(page_title="Energy Analysis Dashboard", layout="wide")
st.title("Nirva Energy Insight Dashboard")

# Check for data file existence
if not os.path.exists('data/daily_energy_data.csv') or not os.path.exists('data/energy_exchange.csv'):
    st.error("Data files not found. Please run generate_synthetic_data.py first to create the necessary data files.")
    st.markdown("""
    ```bash
    python generate_synthetic_data.py
    ```
    """)
    st.stop()

# Load data from CSV files
df = pd.read_csv('data/daily_energy_data.csv')
energy_df = pd.read_csv('data/energy_exchange.csv')

# Extract list of people from the data
people = sorted(df['Person'].unique().tolist())

# Convert date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Energy Timeline as Line Plot
st.subheader("📈 Energy Timeline")

# Group by date and segment, calculating mean energy
timeline = df.groupby(['Date', 'Segment'])['Energy'].mean().reset_index()

# Define all possible segments for consistency
all_segments = ['Morning', 'Afternoon', 'Evening']

# Get the segments that actually exist in the data
available_segments = sorted(df['Segment'].unique().tolist())

# Ensure timeline has categorical ordering for segments
timeline['Segment'] = pd.Categorical(timeline['Segment'], categories=available_segments, ordered=True)

# Create a pivot table for easier time series creation
timeline_wide = timeline.pivot(index='Date', columns='Segment', values='Energy').reset_index()

# Fill NaN values with the average of adjacent points to ensure line continuity
for segment in available_segments:
    if segment in timeline_wide.columns:
        timeline_wide[segment] = timeline_wide[segment].interpolate(method='linear')

# Convert the wide format back to long format for Plotly
timeline_melted = timeline_wide.melt(id_vars='Date', value_vars=available_segments, 
                                     var_name='Segment', value_name='Energy')

# Create a connected line plot with explicit line and marker properties
fig1 = px.line(timeline_melted, x='Date', y='Energy', color='Segment', markers=True,
               category_orders={'Segment': available_segments},
               labels={'Date': 'Date', 'Energy': 'Energy Level', 'Segment': 'Time of Day'},
               title='Energy Level per Day by Segment',
               color_discrete_sequence=['#FEC8D8', '#FFDFD3', '#D0F4DE'], 
               template='simple_white')

# Customize the line plot
fig1.update_traces(
    mode='lines+markers',    # Ensure both lines and markers are shown
    line=dict(width=3),      # Make lines thicker
    marker=dict(size=8),     # Make markers larger
)

# Update layout for better appearance
fig1.update_layout(
    xaxis_title="Date",
    yaxis_title="Energy Level (0-2)",
    legend_title="Time of Day",
    hovermode="x unified",   # Show all values for the same x-coordinate
)

st.plotly_chart(fig1, use_container_width=True)

# Energy Correlation
st.subheader("🔍 Energy Correlation by Category")
col1, col2, col3 = st.columns(3)

with col1:
    topic_avg = df.groupby('Topic')['Energy'].mean().reset_index().sort_values(by='Energy', ascending=False)
    topic_avg['Rank'] = topic_avg['Energy'].rank(ascending=False).astype(int)
    topic_fig = px.bar(topic_avg, x='Energy', y='Topic', orientation='h', title='By Topic',
                       color='Energy', text='Rank', color_continuous_scale=px.colors.sequential.Pinkyl, template='simple_white')
    st.plotly_chart(topic_fig, use_container_width=True)

with col2:
    person_avg = df.groupby('Person')['Energy'].mean().reset_index().sort_values(by='Energy', ascending=False)
    person_avg['Rank'] = person_avg['Energy'].rank(ascending=False).astype(int)
    person_fig = px.bar(person_avg, x='Energy', y='Person', orientation='h', title='By Person',
                        color='Energy', text='Rank', color_continuous_scale=px.colors.sequential.BuGn, template='simple_white')
    st.plotly_chart(person_fig, use_container_width=True)

with col3:
    food_avg = df.groupby('Food')['Energy'].mean().reset_index().sort_values(by='Energy', ascending=False)
    food_avg['Rank'] = food_avg['Energy'].rank(ascending=False).astype(int)
    food_fig = px.bar(food_avg, x='Energy', y='Food', orientation='h', title='By Food',
                      color='Energy', text='Rank', color_continuous_scale=px.colors.sequential.Peach, template='simple_white')
    st.plotly_chart(food_fig, use_container_width=True)

# Add a spacer
st.markdown("---")

# Social Energy Graph with Bidirectional Sentiment Arrows
st.subheader("🌐 Personal Energy Exchange Network")

# Create explanation for the Self-centered graph
st.markdown("""
This graph shows how 'Self' exchanges energy with other people:
- **Teal arrows**: Energizing relationship (positive energy flow)
- **Coral arrows**: Draining relationship (negative energy flow)
- **Arrows pointing toward 'Self'**: How others affect you
- **Arrows pointing away from 'Self'**: How you affect others
""")

# Fixed positions for nodes in a star layout with Self in center
positions = {
    "Self": {"x": 0, "y": 0},
    "Mom": {"x": -200, "y": -200},
    "Friend A": {"x": 200, "y": -200},
    "Colleague": {"x": 0, "y": 200},
    "Partner": {"x": -200, "y": 150}
}

# Import required packages
from pyvis.network import Network
import networkx as nx
from streamlit.components import v1 as components
import base64
import matplotlib.pyplot as plt
import io
from IPython.display import HTML, display
import tempfile
import os

# Define more pleasing color palette
positive_color = '#20B2AA'  # Light sea green / teal
negative_color = '#FF7F50'  # Coral

# Create directed graph with NetworkX - focus on Self connections only
G = nx.DiGraph()

# Add all people as nodes
for person in people:
    G.add_node(person)

# Add directed edges with appropriate weights and sentiments
# Only include edges where Self is involved (either as source or target)
for _, row in energy_df.iterrows():
    person1 = row['Person1']
    person2 = row['Person2']
    
    # Only add connections to/from Self
    if person1 == 'Self' or person2 == 'Self':
        # Person1 -> Person2 relationship
        G.add_edge(
            person1, 
            person2, 
            weight=row['InteractionCount'],
            sentiment=row['GivingEnergy'],
            width=1 + row['InteractionCount'] * 0.5,  # Scale width based on count
            color=positive_color if row['GivingEnergy'] > 0 else negative_color
        )

# Create a PyVis network
net = Network(notebook=True, directed=True, height="600px", width="100%", bgcolor="rgba(248, 248, 255, 0.85)", font_color="#333")

# Set network options for cleaner appearance
net.set_options("""
var options = {
  "nodes": {
    "borderWidth": 2,
    "borderWidthSelected": 3,
    "color": {
      "border": "#00838F",
      "background": "#E0F7FA"
    },
    "font": {
      "size": 16,
      "bold": true
    },
    "shape": "dot",
    "size": 30
  },
  "edges": {
    "arrows": {
      "to": {
        "enabled": true,
        "scaleFactor": 0.6
      }
    },
    "color": {
      "inherit": false
    },
    "smooth": {
      "type": "straightCross",
      "forceDirection": "none"
    },
    "width": 2
  },
  "physics": {
    "enabled": false
  }
}
""")

# Copy our NetworkX graph to PyVis
net.from_nx(G)

# Style each node and edge
for node in net.nodes:
    # Make nodes a consistent size
    node['size'] = 35 if node['id'] == 'Self' else 30  # Make Self slightly larger
    node['title'] = node['id']  # Hover text
    
    # Add a special border for 'Self' node
    if node['id'] == 'Self':
        node['borderWidth'] = 3
        node['color'] = {
            'border': '#006064',
            'background': '#B2EBF2'
        }

for edge in net.edges:
    # Set edge properties based on our NetworkX attributes
    nx_edge = G.get_edge_data(edge['from'], edge['to'])
    edge['width'] = nx_edge.get('width', 2)
    edge['color'] = nx_edge.get('color', '#888888')
    
    # Use get() method to safely access keys that might not exist
    sentiment = nx_edge.get('sentiment', 0)
    interaction_count = nx_edge.get('weight', 1)  # Default to 1 if not found
    
    # Create descriptive title based on direction
    if edge['from'] == 'Self':
        relation_text = f"You → {edge['to']}"
        impact_text = "energize" if sentiment > 0 else "drain"
    else:
        relation_text = f"{edge['from']} → You"
        impact_text = "energizes" if sentiment > 0 else "drains"
    
    # Set hover text with the information
    edge['title'] = f"{relation_text}: {impact_text}<br>Strength: {abs(sentiment):.2f}<br>Interactions: {interaction_count}"

# Apply fixed positions
for node in net.nodes:
    if node['id'] in positions:
        node['x'] = positions[node['id']]['x']
        node['y'] = positions[node['id']]['y']
        node['fixed'] = True

# Use a temporary file to store the HTML
with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmpfile:
    temp_path = tmpfile.name
    net.save_graph(temp_path)
    
    # Read the HTML content
    with open(temp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Inject custom CSS to ensure arrows are properly attached to nodes
    html_content = html_content.replace('</head>', '''
    <style>
    .vis-network .vis-arrow {
        pointer-events: none !important;
    }
    </style>
    </head>
    ''')
    
    # Write the modified HTML back
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

# Display the network in Streamlit
st.components.v1.html(open(temp_path, 'r', encoding='utf-8').read(), height=600)

# Clean up the temporary file
os.unlink(temp_path)

# Add a legend manually with colored boxes
st.markdown(f"""
<style>
.legend-item {{
    display: inline-block;
    margin-right: 20px;
}}
.color-box {{
    display: inline-block;
    width: 15px;
    height: 15px;
    margin-right: 5px;
    vertical-align: middle;
}}
.teal-box {{
    background-color: {positive_color};
}}
.coral-box {{
    background-color: {negative_color};
}}
</style>
<div>
    <div class="legend-item"><span class="color-box teal-box"></span>Energizing (positive impact)</div>
    <div class="legend-item"><span class="color-box coral-box"></span>Draining (negative impact)</div>
</div>
""", unsafe_allow_html=True)

# Optional: Save or upload actual data
st.sidebar.header("Upload Your Data")
uploaded = st.sidebar.file_uploader("Upload CSV", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Data uploaded and previewed below:")
    st.write(df.head())
