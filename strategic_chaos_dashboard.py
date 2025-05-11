# Strategic Chaos F1 Dashboard - Complete Implementation

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from dash import Dash
import os
import argparse
import networkx as nx

# ======= 1. Enhanced Data Loading and Preprocessing =======

def load_all_ergast_data(base_path):
    """
    Load all CSV files from the Ergast F1 dataset
    """
    # List of all possible CSV files in the dataset
    file_list = [
        'circuits.csv',
        'constructor_results.csv',
        'constructor_standings.csv',
        'constructors.csv',
        'driver_standings.csv',
        'drivers.csv',
        'lap_times.csv',
        'pit_stops.csv',
        'qualifying.csv',
        'races.csv',
        'results.csv',
        'seasons.csv',
        'sprint_results.csv',
        'status.csv'
    ]
    
    # Initialize an empty dictionary to store dataframes
    datasets = {}
    
    # Loop through each file and load it if it exists
    for file_name in file_list:
        file_path = os.path.join(base_path, file_name)
        if os.path.exists(file_path):
            # Extract the name without extension to use as a key
            key = file_name.replace('.csv', '')
            try:
                # Load the CSV file
                datasets[key] = pd.read_csv(file_path)
                print(f"Successfully loaded {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {str(e)}")
        else:
            print(f"Warning: {file_name} not found at {file_path}")
    
    return datasets

def create_focused_dataframes(datasets):
    """
    Create focused dataframes for each visualization without joining everything
    """
    focused_dfs = {}
    
    # 1. Race Flow / Track Position Dataframe
    # Needs: lap_times, drivers, constructors, races, results
    if all(key in datasets for key in ['lap_times', 'drivers', 'constructors', 'races', 'results']):
        # Start with lap times
        df_race_flow = datasets['lap_times'].copy()
        
        # Add driver information
        driver_info = datasets['drivers'][['driverId', 'code', 'forename', 'surname']].copy()
        driver_info['driver_name'] = driver_info['forename'] + ' ' + driver_info['surname']
        df_race_flow = pd.merge(df_race_flow, driver_info, on='driverId', how='left')
        
        # Add constructor (team) information via results
        team_info = pd.merge(
            datasets['results'][['raceId', 'driverId', 'constructorId']],
            datasets['constructors'][['constructorId', 'name', 'nationality']],
            on='constructorId',
            how='left'
        )
        team_info = team_info.rename(columns={'name': 'team_name'})
        
        # Add to race flow dataframe
        df_race_flow = pd.merge(
            df_race_flow,
            team_info,
            on=['raceId', 'driverId'],
            how='left'
        )
        
        # Add race information
        race_info = datasets['races'][['raceId', 'year', 'name', 'round']].copy()
        race_info = race_info.rename(columns={'name': 'race_name'})
        df_race_flow = pd.merge(df_race_flow, race_info, on='raceId', how='left')
        
        focused_dfs['race_flow'] = df_race_flow
    
    # 2. Pit Stop Analysis Dataframe
    if all(key in datasets for key in ['pit_stops', 'drivers', 'races']):
        df_pit_stops = datasets['pit_stops'].copy()
        
        # Add driver information
        driver_info = datasets['drivers'][['driverId', 'code', 'forename', 'surname']].copy()
        driver_info['driver_name'] = driver_info['forename'] + ' ' + driver_info['surname']
        df_pit_stops = pd.merge(df_pit_stops, driver_info, on='driverId', how='left')
        
        # Add race information
        race_info = datasets['races'][['raceId', 'year', 'name', 'round']].copy()
        df_pit_stops = pd.merge(df_pit_stops, race_info, on='raceId', how='left')
        
        focused_dfs['pit_stops'] = df_pit_stops
    
    # 3. Results and DNF Analysis Dataframe
    if all(key in datasets for key in ['results', 'drivers', 'constructors', 'status', 'races']):
        df_results = datasets['results'].copy()
        
        # Add driver information
        driver_info = datasets['drivers'][['driverId', 'code', 'forename', 'surname']].copy()
        driver_info['driver_name'] = driver_info['forename'] + ' ' + driver_info['surname']
        df_results = pd.merge(df_results, driver_info, on='driverId', how='left')
        
        # Add constructor information
        constructor_info = datasets['constructors'][['constructorId', 'name']].copy()
        constructor_info = constructor_info.rename(columns={'name': 'team_name'})
        df_results = pd.merge(df_results, constructor_info, on='constructorId', how='left')
        
        # Add status information (for DNFs)
        df_results = pd.merge(df_results, datasets['status'], on='statusId', how='left')
        
        # Add race information
        race_info = datasets['races'][['raceId', 'year', 'name', 'round']].copy()
        df_results = pd.merge(df_results, race_info, on='raceId', how='left')
        
        focused_dfs['results'] = df_results
    
    # 4. Qualifying vs Results (Comeback Analysis)
    if all(key in datasets for key in ['qualifying', 'results', 'drivers', 'races']):
        # Start with qualifying data
        df_qualifying = datasets['qualifying'][['raceId', 'driverId', 'position']].copy()
        df_qualifying = df_qualifying.rename(columns={'position': 'grid_position'})
        
        # Add results data
        results_info = datasets['results'][['raceId', 'driverId', 'position', 'statusId']].copy()
        results_info = results_info.rename(columns={'position': 'final_position'})
        
        # Merge qualifying and results
        df_comeback = pd.merge(
            df_qualifying,
            results_info,
            on=['raceId', 'driverId'],
            how='inner'
        )
        
        # Add driver information
        driver_info = datasets['drivers'][['driverId', 'code', 'forename', 'surname']].copy()
        driver_info['driver_name'] = driver_info['forename'] + ' ' + driver_info['surname']
        df_comeback = pd.merge(df_comeback, driver_info, on='driverId', how='left')
        
        # Calculate position change
        # First, replace '\N' values with NaN
        if 'grid_position' in df_comeback.columns and 'final_position' in df_comeback.columns:
            df_comeback['grid_position'] = df_comeback['grid_position'].replace('\\N', np.nan)
            df_comeback['final_position'] = df_comeback['final_position'].replace('\\N', np.nan)
        
            # Now convert to float and calculate position change
            df_comeback['grid_position'] = pd.to_numeric(df_comeback['grid_position'], errors='coerce')
            df_comeback['final_position'] = pd.to_numeric(df_comeback['final_position'], errors='coerce')
            df_comeback['position_change'] = df_comeback['grid_position'] - df_comeback['final_position']
        
        focused_dfs['comeback'] = df_comeback
        
    # 5. Race List for Dropdown
    if 'races' in datasets:
        df_races = datasets['races'][['raceId', 'year', 'name', 'round']].copy()
        df_races['race_name'] = df_races['name'] + ' ' + df_races['year'].astype(str) + ' (Round ' + df_races['round'].astype(str) + ')'
        focused_dfs['races'] = df_races
    
    return focused_dfs

def calculate_chaos_metrics(focused_dfs, race_id):
    """
    Calculate custom chaos metrics for a specific race
    
    chaos_score = (DNFs × 5) + mean(position_change) + pit_stop_variance
    """
    # Initialize metrics dictionary
    metrics = {
        'dnf_count': 0,
        'mean_position_change': 0,
        'pit_variance': 0,
        'chaos_score': 0,
        'fastest_lap': 0,
        'avg_pit_time': 0
    }
    
    # Calculate DNF count from results
    if 'results' in focused_dfs:
        race_results = focused_dfs['results'][focused_dfs['results']['raceId'] == race_id]
        metrics['dnf_count'] = len(race_results[race_results['status'] != 'Finished'])
    
    # Calculate mean position change from comeback
    if 'comeback' in focused_dfs:
        race_comeback = focused_dfs['comeback'][focused_dfs['comeback']['raceId'] == race_id]
        metrics['mean_position_change'] = abs(race_comeback['position_change']).mean()
    
    # Calculate pit stop variance
    if 'pit_stops' in focused_dfs:
        race_pits = focused_dfs['pit_stops'][focused_dfs['pit_stops']['raceId'] == race_id]
        if len(race_pits) > 0:
            metrics['pit_variance'] = race_pits['milliseconds'].var() / 1000000  # Scale down for better number
            metrics['avg_pit_time'] = race_pits['milliseconds'].mean() / 1000  # Convert to seconds
    
    # Calculate fastest lap
    if 'race_flow' in focused_dfs:
        race_laps = focused_dfs['race_flow'][focused_dfs['race_flow']['raceId'] == race_id]
        if len(race_laps) > 0:
            metrics['fastest_lap'] = race_laps['milliseconds'].min() / 1000  # Convert to seconds
    
    # Calculate chaos score
    metrics['chaos_score'] = (metrics['dnf_count'] * 5) + metrics['mean_position_change'] + metrics['pit_variance']
    
    return metrics

def get_driver_color_mapping(focused_dfs, race_id):
    """
    Create consistent color mapping for drivers across all visualizations
    """
    # Get unique drivers in this race from lap times
    if 'race_flow' in focused_dfs:
        race_laps = focused_dfs['race_flow'][focused_dfs['race_flow']['raceId'] == race_id]
        drivers = race_laps['code'].unique()
        
        # Create color mapping
        color_map = {}
        for i, driver in enumerate(drivers):
            color_map[driver] = px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
        
        return color_map
    
    # Fallback to results if lap times not available
    elif 'results' in focused_dfs:
        race_results = focused_dfs['results'][focused_dfs['results']['raceId'] == race_id]
        drivers = race_results['code'].unique()
        
        # Create color mapping
        color_map = {}
        for i, driver in enumerate(drivers):
            color_map[driver] = px.colors.qualitative.Bold[i % len(px.colors.qualitative.Bold)]
        
        return color_map
    
    # Empty mapping if no driver data available
    return {}

# ======= 2. Visualization Functions =======

def create_race_flow_chart(focused_dfs, race_id, driver_colors=None):
    """
    Create multi-line position-vs-lap chart with P1 at top
    """
    # Filter for specific race
    if 'race_flow' not in focused_dfs:
        # Create empty figure if data not available
        fig = go.Figure()
        fig.add_annotation(
            text="Race flow data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    race_laps = focused_dfs['race_flow'][focused_dfs['race_flow']['raceId'] == race_id]
    
    if race_laps.empty:
        # Create empty figure if no data for this race
        fig = go.Figure()
        fig.add_annotation(
            text="No lap data available for this race",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Create color mapping if not provided
    if driver_colors is None:
        driver_colors = get_driver_color_mapping(focused_dfs, race_id)
    
    # Create line chart
    fig = px.line(
        race_laps, 
        x='lap', 
        y='position',
        color='code',
        labels={'position': 'Position', 'lap': 'Lap'},
        title='Race Position Flow',
        color_discrete_map=driver_colors,
        hover_data=['driver_name', 'team_name']
    )
    
    # Customize layout (P1 at top)
    fig.update_layout(
        yaxis=dict(
            autorange='reversed',
            title='Position'
        ),
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white'),
        legend_title_text='Driver',
        hovermode='closest',
        title_font_size=20,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    
    # Add custom hover template
    fig.update_traces(
        hovertemplate='<b>%{customdata[0]}</b><br>' +
                      'Team: %{customdata[1]}<br>' +
                      'Lap: %{x}<br>' +
                      'Position: %{y}<br>' +
                      '<extra></extra>'
    )
    
    return fig

def create_pit_stop_gantt(focused_dfs, race_id, driver_colors=None):
    """
    Create Gantt chart showing pit stops
    """
    # Check if data is available
    if 'pit_stops' not in focused_dfs:
        # Create empty figure if data not available
        fig = go.Figure()
        fig.add_annotation(
            text="Pit stop data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Filter for specific race
    race_pits = focused_dfs['pit_stops'][focused_dfs['pit_stops']['raceId'] == race_id]
    
    if race_pits.empty:
        # Create empty figure with message if no pit stops
        fig = go.Figure()
        fig.add_annotation(
            text="No pit stop data available for this race",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Print debug info
    print(f"Pit stop data for race {race_id}:")
    print(f"Records: {len(race_pits)}")
    print(f"Columns: {race_pits.columns}")
    print(race_pits.head())
    
    # Calculate start and end lap positions for each pit stop
    race_pits['stop_end'] = race_pits['lap'] + (race_pits['milliseconds'] / (60 * 1000 * 1.5))  # Scale for visibility
    
    # Create color mapping if not provided
    if driver_colors is None:
        driver_colors = get_driver_color_mapping(focused_dfs, race_id)
    
    # Create custom Gantt chart using go.Bar
    fig = go.Figure()
    
    # Add bars for each pit stop
    for driver in race_pits['driver_name'].unique():
        driver_pits = race_pits[race_pits['driver_name'] == driver]
        driver_code = driver_pits['code'].iloc[0]
        color = driver_colors.get(driver_code, 'gray')
        
        for _, pit in driver_pits.iterrows():
            fig.add_trace(go.Bar(
                x=[pit['stop_end'] - pit['lap']],  # Width of bar is pit stop duration
                y=[driver],
                orientation='h',
                base=pit['lap'],  # Starting point of bar
                marker_color=color,
                name=driver_code,
                showlegend=False,
                hovertemplate='<b>%{customdata[0]}</b><br>' +
                              'Pit Stop #: %{customdata[1]}<br>' +
                              'Duration: %{customdata[2]:.2f} seconds<br>' +
                              'Lap: %{customdata[3]}<br>' +
                              '<extra></extra>',
                customdata=[[driver, pit['stop'], pit['milliseconds']/1000, pit['lap']]]
            ))
    
    # Add driver codes as a single legend entry
    for driver_code, color in driver_colors.items():
        fig.add_trace(go.Bar(
            x=[0],
            y=[driver_code],
            orientation='h',
            marker_color=color,
            name=driver_code,
            legendgroup=driver_code,
        ))
    
    # Update layout
    fig.update_layout(
        title='Pit Stop Timeline',
        xaxis_title='Lap',
        yaxis_title='Driver',
        barmode='overlay',
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white'),
        margin=dict(l=30, r=30, t=60, b=30),
        legend=dict(
            title="Driver",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1
        )
    )
    
    # Force x-axis to be numeric laps, not dates
    max_lap = race_pits['lap'].max() + 5  # Add some padding
    fig.update_xaxes(range=[0, max_lap], tickmode='linear', dtick=5)
    
    return fig

def create_dnf_donut_chart(focused_dfs, race_id):
    """
    Create donut chart showing DNF reasons
    """
    # Check if data is available
    if 'results' not in focused_dfs:
        # Create empty figure if data not available
        fig = go.Figure()
        fig.add_annotation(
            text="Results data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Filter for specific race
    race_results = focused_dfs['results'][focused_dfs['results']['raceId'] == race_id]
    
    if race_results.empty:
        # Create empty figure if no results
        fig = go.Figure()
        fig.add_annotation(
            text="No results data available for this race",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Filter for DNF entries
    dnf_results = race_results[race_results['status'] != 'Finished']
    
    if dnf_results.empty:
        # Create empty figure with message if no DNFs
        fig = go.Figure()
        fig.add_annotation(
            text="No DNFs in this race",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Count by status
    dnf_counts = dnf_results['status'].value_counts().reset_index()
    dnf_counts.columns = ['status', 'count']
    
    # Create donut chart
    fig = px.pie(
        dnf_counts,
        values='count',
        names='status',
        title='DNF Reasons',
        hole=0.5,
        color_discrete_sequence=px.colors.qualitative.Bold
    )
    
    # Update traces
    fig.update_traces(
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>' +
                      'Count: %{value}<br>' +
                      'Percentage: %{percent}<br>' +
                      '<extra></extra>'
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white'),
        legend=dict(orientation='h', yanchor='bottom', y=-0.3, xanchor='center', x=0.5),
        margin=dict(l=20, r=20, t=50, b=30),
    )
    
    return fig

# 2. Add new visualization functions

def create_tire_strategy_visualization(focused_dfs, race_id, driver_colors=None):
    """
    Create a visualization showing tire compound performance degradation.
    This tracks lap times through each stint and shows how different
    tire compounds degrade over time.
    """
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Check if necessary data is available
    required_dfs = ['race_flow', 'pit_stops', 'results']
    missing_dfs = [df for df in required_dfs if df not in focused_dfs]
    
    if missing_dfs:
        # Create empty figure if data not available
        fig = go.Figure()
        fig.add_annotation(
            text=f"Missing required data: {', '.join(missing_dfs)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Filter for specific race
    race_laps = focused_dfs['race_flow'][focused_dfs['race_flow']['raceId'] == race_id]
    race_pits = focused_dfs['pit_stops'][focused_dfs['pit_stops']['raceId'] == race_id]
    
    if race_laps.empty or race_pits.empty:
        # Create empty figure if no data for this race
        fig = go.Figure()
        fig.add_annotation(
            text="No lap or pit stop data available for this race",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Create color mapping if not provided
    if driver_colors is None:
        driver_colors = get_driver_color_mapping(focused_dfs, race_id)
    
    # Get top drivers for analysis (to avoid too much clutter)
    top_drivers = race_laps[race_laps['lap'] == 1]['code'].unique()[:8]
    
    # Convert milliseconds to seconds for better readability
    race_laps['lap_time_seconds'] = race_laps['milliseconds'] / 1000
    
    # Create a simulated tire compound assignment (since we don't have real data)
    # In a real implementation, this would come from the dataset
    compound_colors = {
        'Soft': 'red',
        'Medium': 'yellow',
        'Hard': 'white',
        'Intermediate': 'green',
        'Wet': 'blue'
    }
    
    # Create subplots - one row per driver
    fig = make_subplots(
        rows=len(top_drivers),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[f"Driver: {driver}" for driver in top_drivers]
    )
    
    for i, driver in enumerate(top_drivers):
        driver_row = i + 1
        driver_laps = race_laps[race_laps['code'] == driver].sort_values('lap')
        
        # Get driver's pit stops to identify stints
        driver_pits = race_pits[race_pits['code'] == driver].sort_values('lap')
        
        # Add the start of the race as lap 0
        pit_laps = [0] + driver_pits['lap'].tolist() + [driver_laps['lap'].max()]
        
        # Assign a simulated tire compound to each stint
        # In real implementation, this would come from the dataset
        compounds = []
        for j in range(len(pit_laps) - 1):
            # Simulate tire strategy - in reality this would come from data
            if j == 0:
                compound = 'Soft' if np.random.random() > 0.5 else 'Medium'
            elif j == 1:
                compound = 'Medium' if np.random.random() > 0.3 else 'Hard'
            else:
                compound = np.random.choice(['Soft', 'Medium', 'Hard'], p=[0.3, 0.4, 0.3])
            compounds.append(compound)
        
        # Plot each stint with a different color based on tire compound
        for j in range(len(pit_laps) - 1):
            stint_start = pit_laps[j]
            stint_end = pit_laps[j + 1]
            compound = compounds[j]
            
            # Get laps in this stint
            stint_laps = driver_laps[(driver_laps['lap'] > stint_start) & (driver_laps['lap'] <= stint_end)]
            
            if not stint_laps.empty:
                # Calculate lap number relative to stint start for x-axis
                stint_laps['stint_lap'] = stint_laps['lap'] - stint_start
                
                # Calculate baseline lap time (first lap of stint) for normalization
                baseline = stint_laps['lap_time_seconds'].iloc[0] if not stint_laps.empty else 0
                
                # Normalize lap times to show percentage degradation
                stint_laps['normalized_time'] = (stint_laps['lap_time_seconds'] / baseline) * 100 - 100
                
                # Plot the stint
                fig.add_trace(
                    go.Scatter(
                        x=stint_laps['stint_lap'],
                        y=stint_laps['normalized_time'],
                        mode='lines+markers',
                        name=f"{driver} - {compound}",
                        line=dict(color=compound_colors[compound], width=2, dash='solid'),
                        marker=dict(size=5, color=driver_colors.get(driver, 'gray')),
                        legendgroup=driver,
                        hovertemplate='<b>%{text}</b><br>' +
                                    'Lap in stint: %{x}<br>' +
                                    'Degradation: %{y:.2f}%<br>' +
                                    f'Compound: {compound}<br>' +
                                    '<extra></extra>',
                        text=[driver] * len(stint_laps),
                        showlegend=(j == 0)  # Only show in legend for first stint
                    ),
                    row=driver_row, col=1
                )
                
                # Add compound indicator at the start of each stint
                fig.add_annotation(
                    x=0,
                    y=0,
                    text=compound[0],  # First letter of compound
                    showarrow=False,
                    font=dict(color="black", size=10),
                    bgcolor=compound_colors[compound],
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=3,
                    opacity=0.8,
                    row=driver_row, col=1
                )
    
    # Update layout
    fig.update_layout(
        title='Tire Degradation by Stint',
        xaxis_title='Lap in Stint',
        yaxis_title='Performance Degradation (%)',
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white'),
        legend_title="Driver + Compound",
        hovermode='closest',
        height=100 * len(top_drivers) + 150,  # Adjust height based on number of drivers
        margin=dict(l=30, r=30, t=60, b=30),
    )
    
    # Update all y-axes
    for i in range(1, len(top_drivers) + 1):
        fig.update_yaxes(
            title_text="Degradation (%)" if i == len(top_drivers) // 2 + 1 else "",
            range=[-5, 20],  # Range from -5% to 20% performance drop
            row=i, col=1
        )
    
    # Update last x-axis
    fig.update_xaxes(
        title_text="Lap in Stint",
        row=len(top_drivers), col=1
    )
    
    # Add a legend for tire compounds
    for compound, color in compound_colors.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='lines',
                line=dict(color=color, width=4),
                name=compound,
                legendgroup="compounds",
                showlegend=True
            )
        )
    
    return fig

def create_overtake_network(focused_dfs, race_id, driver_colors=None):
    """
    Create a network graph showing overtaking relationships between drivers
    """
    # Check if data is available
    if 'race_flow' not in focused_dfs:
        # Create empty figure if data not available
        fig = go.Figure()
        fig.add_annotation(
            text="Race flow data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Filter for specific race
    race_laps = focused_dfs['race_flow'][focused_dfs['race_flow']['raceId'] == race_id]
    
    if race_laps.empty:
        # Create empty figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No lap data available for this race",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Create color mapping if not provided
    if driver_colors is None:
        driver_colors = get_driver_color_mapping(focused_dfs, race_id)
    
    # Track overtakes between drivers
    overtakes = {}  # {(overtaker, overtaken): count}
    driver_overtake_counts = {}  # {driver: total_overtakes}
    
    # For each lap, check for position changes
    for lap in sorted(race_laps['lap'].unique())[1:]:  # Skip first lap
        prev_lap = lap - 1
        
        # Get positions for current and previous lap
        prev_lap_pos = race_laps[race_laps['lap'] == prev_lap][['code', 'position']].set_index('code')['position']
        curr_lap_pos = race_laps[race_laps['lap'] == lap][['code', 'position']].set_index('code')['position']
        
        # Check for position changes
        for driver in prev_lap_pos.index:
            if driver in curr_lap_pos.index:
                if prev_lap_pos[driver] > curr_lap_pos[driver]:  # Driver moved up (overtook someone)
                    # Find who was overtaken
                    for other_driver in prev_lap_pos.index:
                        if other_driver != driver and other_driver in curr_lap_pos.index:
                            if prev_lap_pos[other_driver] < prev_lap_pos[driver] and curr_lap_pos[other_driver] > curr_lap_pos[driver]:
                                # This is an overtake
                                pair = (driver, other_driver)
                                overtakes[pair] = overtakes.get(pair, 0) + 1
                                
                                # Update driver counts
                                driver_overtake_counts[driver] = driver_overtake_counts.get(driver, 0) + 1
                                driver_overtake_counts[other_driver] = driver_overtake_counts.get(other_driver, 0) + 1
    
    # If no overtakes detected
    if not overtakes:
        fig = go.Figure()
        fig.add_annotation(
            text="No overtakes detected in this race",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Create network layout
    # Create graph
    G = nx.Graph()
    
    # Add nodes
    for driver in driver_overtake_counts:
        # Get size proportional to number of overtakes
        size = 10 + (driver_overtake_counts[driver] * 5)
        G.add_node(driver, size=size)
    
    # Add edges
    for (driver1, driver2), count in overtakes.items():
        G.add_edge(driver1, driver2, weight=count)
    
    # Create positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Create network visualization
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G.edges[edge]['weight']
        
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.extend([weight, weight, None])
    
    # Create edges trace
    edges_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Create nodes trace
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        size = G.nodes[node]['size']
        node_size.append(size)
        node_text.append(f"{node}: {driver_overtake_counts[node]} overtakes")
        node_color.append(driver_colors.get(node, 'gray'))
    
    nodes_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=False,
            color=node_color,
            size=node_size,
            line=dict(width=1, color='#333333')
        ),
        name=''
    )
    
    # Create separate legend traces
    legend_traces = []
    for driver, count in driver_overtake_counts.items():
        legend_traces.append(
            go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=driver_colors.get(driver, 'gray')),
                name=driver,
                showlegend=True
            )
        )
    
    # Create figure
    fig = go.Figure(data=[edges_trace, nodes_trace] + legend_traces)
    
    # Update layout
    fig.update_layout(
        title='Driver Overtaking Network',
        titlefont_size=16,
        showlegend=True,
        hovermode='closest',
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white'),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        legend=dict(
            title="Driver",
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_chaos_spiral(focused_dfs, race_id, driver_colors=None):
    """
    Create polar plot showing position across laps in a circular form
    """
    # Check if data is available
    if 'race_flow' not in focused_dfs:
        # Create empty figure if data not available
        fig = go.Figure()
        fig.add_annotation(
            text="Race flow data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Filter for specific race
    race_laps = focused_dfs['race_flow'][focused_dfs['race_flow']['raceId'] == race_id]
    
    if race_laps.empty:
        # Create empty figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No lap data available for this race",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Create color mapping if not provided
    if driver_colors is None:
        driver_colors = get_driver_color_mapping(focused_dfs, race_id)
    
    # Limit to top 10 drivers for clarity
    top_drivers = race_laps[race_laps['lap'] == race_laps['lap'].max()].sort_values('position')['code'].unique()[:10]
    race_laps_filtered = race_laps[race_laps['code'].isin(top_drivers)]
    
    # Create spiral plot
    fig = go.Figure()
    
    # For each driver, add a trace
    for driver in top_drivers:
        driver_data = race_laps_filtered[race_laps_filtered['code'] == driver]
        
        fig.add_trace(go.Scatterpolar(
            r=driver_data['lap'],
            theta=driver_data['position'] * (360/20),  # Scale to 360 degrees (assuming max 20 positions)
            mode='lines+markers',
            name=driver,
            line=dict(color=driver_colors.get(driver, 'gray'), width=2),
            marker=dict(size=6, color=driver_colors.get(driver, 'gray')),
            hovertext=driver_data['driver_name'],
            hoverinfo='text+r+theta'
        ))
    
    # Update layout
    fig.update_layout(
        title='Race Chaos Spiral',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, race_laps['lap'].max() + 1],
                title='Lap',
                color='white'
            ),
            angularaxis=dict(
                visible=True,
                direction='clockwise',
                period=360,
                color='white'
            ),
            bgcolor='#111111'
        ),
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white'),
        showlegend=True,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    
    return fig

def create_chaos_heatmap(focused_dfs, race_id):
    """
    Create heatmap showing Chaos Index across drivers
    """
    # Check if data is available
    if 'race_flow' not in focused_dfs:
        # Create empty figure if data not available
        fig = go.Figure()
        fig.add_annotation(
            text="Race flow data not available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Filter for specific race
    race_laps = focused_dfs['race_flow'][focused_dfs['race_flow']['raceId'] == race_id]
    
    if race_laps.empty:
        # Create empty figure if no data
        fig = go.Figure()
        fig.add_annotation(
            text="No lap data available for this race",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Calculate lap-by-lap position changes for each driver
    chaos_data = []
    
    for driver_code in race_laps['code'].unique():
        driver_laps = race_laps[race_laps['code'] == driver_code].sort_values('lap')
        
        for i in range(1, len(driver_laps)):
            prev_pos = driver_laps.iloc[i-1]['position']
            curr_pos = driver_laps.iloc[i]['position']
            pos_change = abs(prev_pos - curr_pos)
            
            if pos_change > 0:
                chaos_data.append({
                    'code': driver_code,
                    'lap': driver_laps.iloc[i]['lap'],
                    'position_change': pos_change,
                    'driver_name': driver_laps.iloc[i]['driver_name']
                })
    
    if not chaos_data:
        # Create empty figure with message if no chaos data
        fig = go.Figure()
        fig.add_annotation(
            text="No position changes in this race",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(color="white", size=16)
        )
        fig.update_layout(
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            font=dict(color='white'),
            margin=dict(l=30, r=30, t=60, b=30),
        )
        return fig
    
    # Convert to DataFrame
    df_chaos = pd.DataFrame(chaos_data)
    
    # Create pivot table for heatmap
    pivot_chaos = df_chaos.pivot_table(
        index='code',
        columns='lap',
        values='position_change',
        aggfunc='sum',
        fill_value=0
    )
    
    # Create heatmap
    fig = px.imshow(
        pivot_chaos,
        labels=dict(x="Lap", y="Driver", color="Position Changes"),
        title="Race Chaos Heatmap",
        color_continuous_scale='Plasma',
        aspect='auto'
    )
    
    # Update layout
    fig.update_layout(
        plot_bgcolor='#111111',
        paper_bgcolor='#111111',
        font=dict(color='white'),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    
    return fig

def create_summary_cards(chaos_metrics, race_info=None):
    """
    Create summary cards with race statistics
    """
    cards = [
        dbc.Card(
            dbc.CardBody([
                html.H5("Chaos Score", className="card-title"),
                html.H3(f"{chaos_metrics['chaos_score']:.2f}", className="card-text text-center"),
                html.P("Higher score indicates more chaotic race", className="card-text text-muted small"),
            ]),
            className="bg-dark text-white m-2"
        ),
        dbc.Card(
            dbc.CardBody([
                html.H5("DNFs", className="card-title"),
                html.H3(f"{chaos_metrics['dnf_count']}", className="card-text text-center"),
                html.P("Number of cars that did not finish", className="card-text text-muted small"),
            ]),
            className="bg-dark text-white m-2"
        ),
        dbc.Card(
            dbc.CardBody([
                html.H5("Avg Position Change", className="card-title"),
                html.H3(f"{chaos_metrics['mean_position_change']:.2f}", className="card-text text-center"),
                html.P("Average positions gained/lost", className="card-text text-muted small"),
            ]),
            className="bg-dark text-white m-2"
        ),
        dbc.Card(
            dbc.CardBody([
                html.H5("Avg Pit Time", className="card-title"),
                html.H3(f"{chaos_metrics['avg_pit_time']:.2f}s", className="card-text text-center"),
                html.P("Average pit stop duration", className="card-text text-muted small"),
            ]),
            className="bg-dark text-white m-2"
        ),
    ]
    
    return cards

# ======= 3. Dashboard Layout and Callbacks =======

def create_dashboard(data_path):
    """
    Create Strategic Chaos Dashboard
    """
    # Load and process data
    datasets = load_all_ergast_data(data_path)
    focused_dfs = create_focused_dataframes(datasets)
    
    # Get list of races for dropdown
    if 'races' in focused_dfs:
        races = focused_dfs['races'].copy()
        races['race_display'] = races['name'] + ' ' + races['year'].astype(str) + ' (Round ' + races['round'].astype(str) + ')'
        race_options = [{'label': race, 'value': id} for race, id in zip(races['race_display'], races['raceId'])]
    else:
        # Fallback if race data not available
        race_options = [{'label': 'No race data available', 'value': -1}]
    
    # Initialize app
    app = Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
    
    # Define the layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Strategic Chaos", className="display-4 text-center my-3"),
                html.H5("F1 Race Intelligence Dashboard", className="text-center text-muted mb-4"),
            ], width=12)
        ]),
        
        dbc.Row([
            dbc.Col([
                html.Label("Select Race:"),
                dcc.Dropdown(
                    id='race-selector',
                    options=race_options,
                    value=race_options[0]['value'] if race_options else None,  # Default to first race if available
                    className="mb-4"
                ),
            ], width=6),
            
            dbc.Col([
                html.Div(id='race-info', className="text-right mt-2")
            ], width=6),
        ]),
        
        dbc.Row([
            dbc.Col([
                dbc.Row(id='summary-cards', className="mb-3"),
            ], width=12),
        ]),
        
        # First row of visualizations
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='race-flow-chart', style={"height": "400px"}),
            ], width=12),
        ], className="mb-3"),
        
        # Second row of visualizations
        # Second row of visualizations
dbc.Row([
    dbc.Col([
        dcc.Graph(id='tire-strategy-viz', style={"height": "400px"}),  # Changed from lap-time-evolution
    ], width=6),
    
    dbc.Col([
        dcc.Graph(id='pit-stop-gantt', style={"height": "400px"}),
    ], width=6),
], className="mb-3"),
        
        # Third row of visualizations
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='chaos-spiral', style={"height": "400px"}),
            ], width=6),
            
            dbc.Col([
                dcc.Graph(id='dnf-donut', style={"height": "400px"}),
            ], width=6),
        ], className="mb-3"),
        
        # Fourth row of visualizations
        dbc.Row([
            dbc.Col([
                dcc.Graph(id='chaos-heatmap', style={"height": "400px"}),
            ], width=6),
            
            dbc.Col([
                dcc.Graph(id='overtake-network', style={"height": "400px"}),
            ], width=6),
        ], className="mb-3"),
        
    ], fluid=True, style={"backgroundColor": "#111111", "color": "white", "minHeight": "100vh"})
    
    @app.callback(
        [Output('race-flow-chart', 'figure'),
         Output('tire-strategy-viz', 'figure'),
         Output('pit-stop-gantt', 'figure'),
         Output('dnf-donut', 'figure'),
         Output('chaos-spiral', 'figure'),
         Output('chaos-heatmap', 'figure'),
         Output('overtake-network', 'figure'),
         Output('summary-cards', 'children'),
         Output('race-info', 'children')],
        [Input('race-selector', 'value')]
    )
    def update_dashboard(race_id):
        """
        Update all dashboard components when race is selected
        """
        # Handle case when no race is selected
        if race_id is None or race_id == -1:
            # Create empty figures
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                text="No race data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(color="white", size=16)
            )
            empty_fig.update_layout(
                plot_bgcolor='#111111',
                paper_bgcolor='#111111',
                font=dict(color='white'),
                margin=dict(l=30, r=30, t=60, b=30),
            )
            
            # Create empty cards
            empty_metrics = {
                'chaos_score': 0,
                'dnf_count': 0,
                'mean_position_change': 0,
                'avg_pit_time': 0
            }
            empty_cards = create_summary_cards(empty_metrics)
            
            # Create empty race info
            empty_race_info = html.Div([
                html.H4("No Race Selected", className="text-right"),
                html.P("Please select a race from the dropdown", className="text-right text-muted"),
            ])
            
            return (empty_fig, empty_fig, empty_fig, empty_fig, 
                  empty_fig, empty_fig, empty_fig, 
                  empty_cards, empty_race_info)
        
        # Get consistent color mapping for all visualizations
        driver_colors = get_driver_color_mapping(focused_dfs, race_id)
        
        # Calculate chaos metrics
        chaos_metrics = calculate_chaos_metrics(focused_dfs, race_id)
        
        # Get race info
        if 'races' in focused_dfs:
            races = focused_dfs['races']
            race_name = races[races['raceId'] == race_id]['race_name'].values[0] if not races[races['raceId'] == race_id].empty else "Unknown Race"
            
            race_info_html = html.Div([
                html.H4(race_name, className="text-right"),
                html.P(f"Year: {races[races['raceId'] == race_id]['year'].values[0] if not races[races['raceId'] == race_id].empty else 'Unknown'}, " + 
                      f"Round: {races[races['raceId'] == race_id]['round'].values[0] if not races[races['raceId'] == race_id].empty else 'Unknown'}", 
                      className="text-right text-muted"),
            ])
        else:
            race_info_html = html.Div([
                html.H4("Race Information Not Available", className="text-right"),
                html.P("Race data could not be loaded", className="text-right text-muted"),
            ])
        
        # Create all visualizations
        race_flow_fig = create_race_flow_chart(focused_dfs, race_id, driver_colors)
        tire_strategy_fig = create_tire_strategy_visualization(focused_dfs, race_id, driver_colors)
        pit_stop_fig = create_pit_stop_gantt(focused_dfs, race_id, driver_colors)
        dnf_fig = create_dnf_donut_chart(focused_dfs, race_id)
        chaos_spiral_fig = create_chaos_spiral(focused_dfs, race_id, driver_colors)
        chaos_heatmap_fig = create_chaos_heatmap(focused_dfs, race_id)
        overtake_network_fig = create_overtake_network(focused_dfs, race_id, driver_colors)
        
        # Create summary cards
        cards = create_summary_cards(chaos_metrics)
        
        return (race_flow_fig, tire_strategy_fig, pit_stop_fig, dnf_fig, 
               chaos_spiral_fig, chaos_heatmap_fig, overtake_network_fig, 
               cards, race_info_html)
    
    return app

# ======= 4. Custom CSS for Dark Mode Styling =======

custom_css = """
body {
    background-color: #111111;
    color: white;
    font-family: 'Roboto', sans-serif;
}

.dash-dropdown .Select-control {
    background-color: #333333;
    color: white;
    border-color: #444444;
}

.dash-dropdown .Select-menu-outer {
    background-color: #333333;
    color: white;
    border-color: #444444;
}

.dash-dropdown .Select-value-label {
    color: white !important;
}

.dash-dropdown .Select-placeholder {
    color: #aaaaaa !important;
}

.Select--single > .Select-control .Select-value, .Select-placeholder {
    color: white !important;
}

.Select-menu-outer .Select-option {
    background-color: #333333;
    color: white;
}

.Select-menu-outer .Select-option:hover {
    background-color: #444444;
}

.card {
    background-color: #222222 !important;
    border-color: #333333 !important;
}

.text-muted {
    color: #aaaaaa !important;
}
"""

# ======= 5. Main Function to Run Dashboard =======

def main():
    """
    Main function to run the Strategic Chaos Dashboard
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description='Strategic Chaos F1 Dashboard')
    parser.add_argument('--data-path', type=str, default="/Users/kartikvadhawana/Desktop/project vis final/archive (3)", help='Path to directory containing F1 CSV data files')
    parser.add_argument('--port', type=int, default=8060, 
                        help='Port to run the dashboard server on')
    parser.add_argument('--debug', action='store_true', 
                        help='Run in debug mode')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create and run dashboard
    app = create_dashboard(args.data_path)
    
    # Add custom CSS
    app.index_string = """
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>Strategic Chaos - F1 Race Intelligence Dashboard</title>
            {%favicon%}
            {%css%}
            <style>
                """ + custom_css + """
            </style>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    """
    
    # Run the dashboard
    app.run_server(debug=args.debug, port=args.port)

if __name__ == "__main__":
    main()