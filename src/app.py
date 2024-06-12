# Imports 
from pathlib import Path
import json
import numpy as np
import os
import zstandard as zstd

from collections import defaultdict
import matplotlib.pyplot as plt
import ecg_plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import dash
from dash import dcc, html, Input, Output

# Load model specs

# Define the directory path
model_dir = Path('vae')

# Load Model Information
MODEL_INFO = json.load(open(model_dir / 'MODEL_INFO.json'))
i_factors = json.load(open(model_dir / 'val_i_factors.json'))
i_factor_ids = [f'{i + 1}' for i in i_factors['indices']]

# Loading ecg data
directory = 'ecg_data'
parent_dir = os.getcwd()
path = os.path.join(parent_dir, directory)

arrays = []

for filename in sorted(os.listdir(path)):
    if filename.endswith(".zst"):
        with open(os.path.join(path, filename), 'rb') as f:
            compressed_data = f.read()
            decompressed_data = zstd.decompress(compressed_data)

            if filename.endswith("13.zst"):
                array = np.frombuffer(decompressed_data, dtype=np.float32).reshape((538, 512, 8))
            else:
                array = np.frombuffer(decompressed_data, dtype=np.float32).reshape((3500, 512, 8))

            arrays.append(array)

decoded_ecgs_array = np.concatenate(arrays, axis=0)
# print(decoded_ecgs_array.shape)

# code to make 12 leads from 8
def leads8to12(data):
    lead3 = np.subtract(data[:,:,1], data[:,:,0])
    leadavr = np.add(data[:,:,0], data[:,:,1])*(-0.5)
    leadavl = np.subtract(data[:,:,0], 0.5*data[:,:,1])
    leadavf = np.subtract(data[:,:,1], 0.5*data[:,:,0])
    new_leads = np.stack((data[:,:,0], data[:,:,1], lead3, leadavr, leadavl, leadavf, data[:,:,2], data[:,:,3], data[:,:,4], data[:,:,5], data[:,:,6], data[:,:,7]), axis=2)
    return new_leads
    
reshaped_decoded_ecgs = [array[np.newaxis, ...] for array in decoded_ecgs_array]
# reshaped_decoded_ecgs[0].shape

decoded_ecgs_array_12L = [leads8to12(x) for x in reshaped_decoded_ecgs]
# print(len(decoded_ecgs_array_12L))

# Code to visualise ecgs

ECG_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# COLORMAP TEMPLATES FOR HORIZONTAL AND VERTICAL LEADS
LEAD_COLORS_plotly = {
    'I': 'indigo',    # Viridis (2)
    'II': 'slateblue',     # Viridis (4)
    'III': 'teal',          # Viridis (6)
    'aVR': 'limegreen',      # Viridis (9)
    'aVL': 'midnightblue', # Viridis (1)
    'aVF': 'mediumturquoise',          # Viridis (5)
    'V1': 'darkslateblue',   # Plasma (2)
    'V2': 'mediumslateblue',         # Plasma (3)
    'V3': 'mediumslateblue',         # Plasma (4)
    'V4': 'mediumslateblue',         # Plasma (5)
    'V5': 'mediumorchid',  # Plasma (6)
    'V6': 'fuchsia'  # Plasma (7)
}
LEAD_COLORS = defaultdict(lambda: 'black', LEAD_COLORS_plotly)


def plot_ecg_plotly(ecg_signal, sampling_rate, lead_names=None, subplot_shape=None, ylim=None, share_ylim=True,
                    title=None, std=None, percentiles=None, figsize=None, show_axes=True, show_legend=False, **kwargs):
    
    """
    Plots ECG signal(s) in the time domain using Plotly.

    Arguments:
        ecg_signal (ndarray): ECG signal(s) of shape (num_samples, num_leads).
        sampling_rate (int): Sampling rate of the ECG signal(s).
        lead_names (list): List of lead names. If None, the leads will be named as Lead 1, Lead 2, etc.
        subplot_shape (tuple): Shape of the subplot grid. If None, the shape will be automatically determined.
        ylim (tuple): Y-axis limits of the plot.
        share_ylim (bool): If True, the y-axis limits of the subplots will be shared.
        title (str): Title of the plot.
        std (ndarray): Standard deviation of the ECG signal(s) of shape (num_samples, num_leads).
        percentiles (tuple): Percentiles of the ECG signal(s) of shape (2, num_samples, num_leads).
        figsize (tuple): Figure size in pixels (width, height).
        show_axes (bool): If True, the axes of the plot will be plotted.
        show_legend (bool): If True, the legend will be shown.
        **kwargs: Additional arguments to be passed to the Plotly go.Scatter function.

    Returns:
        fig (plotly.graph_objects.Figure): Figure object.
    """
    # Check ECG signal shape
    if len(ecg_signal.shape) != 2:
        raise ValueError('ECG signal must have shape: (num_samples, num_leads)')

    # Get number of ECG leads and time_index vector
    time_index = np.arange(ecg_signal.shape[0]) / sampling_rate
    num_leads = ecg_signal.shape[1]

    # If share_ylim, find ECG max and min values
    ylim_ = None
    if ylim is not None:
        ylim_ = ylim
    if ylim is None and share_ylim is True:
        ylim_ = (np.min(ecg_signal), np.max(ecg_signal))

    # Check for Lead Names
    if lead_names is not None:
        # Check number of leads
        if len(lead_names) != num_leads:
            raise ValueError('Number of lead names must match the number of leads in the ECG data.')
        lead_colors = LEAD_COLORS_plotly
    else:
        lead_names = [f'Lead {i + 1}' for i in range(num_leads)]  # Lead x
        lead_colors = dict(zip(lead_names, LEAD_COLORS_plotly))

    # Check subplot shape
    if subplot_shape is None:
        subplot_shape = (num_leads, 1)

    # Create subplots
    fig = make_subplots(rows=subplot_shape[0], cols=subplot_shape[1], shared_yaxes=share_ylim, x_title="Time (seconds)", y_title="Amplitude (mV)")

    # Plotting
    for i in range(num_leads):
        row = i // subplot_shape[1] + 1
        col = i % subplot_shape[1] + 1

        fig.add_trace(go.Scatter(
            x=time_index,
            y=ecg_signal[:, i],
            mode='lines',
            name=lead_names[i],
            line=dict(color=lead_colors[lead_names[i]]),
            showlegend=show_legend
        ), row=row, col=col)

        if std is not None:
            fig.add_trace(go.Scatter(
                x=time_index,
                y=ecg_signal[:, i] - std[:, i],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=time_index,
                y=ecg_signal[:, i] + std[:, i],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=lead_colors[lead_names[i]],
                opacity=0.2,
                showlegend=False
            ), row=row, col=col)

        if percentiles is not None:
            fig.add_trace(go.Scatter(
                x=time_index,
                y=percentiles[0][:, i],
                mode='lines',
                line=dict(width=0),
                showlegend=False
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=time_index,
                y=percentiles[1][:, i],
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                fillcolor=lead_colors[lead_names[i]],
                opacity=0.2,
                showlegend=False
            ), row=row, col=col)

    # Update layout
    fig.update_layout(
        title=title,
        # xaxis=dict(title='Time (seconds)', domain = [0,1]),
        # yaxis = dict(title='Amplitude (mV)'),
        # xaxis_title='Time (seconds)',
        # yaxis_title='Amplitude (mV)',
        showlegend=show_legend,
        height=figsize[1] if figsize else None,
        width=figsize[0] if figsize else None
    )
    # fig.update_layout(xaxis=dict(title='Common X-Axis Title', domain=[0, 1]),
    #               yaxis=dict(title='Common Y-Axis Title', domain=[0, 1]))

    if ylim_ is not None:
        fig.update_yaxes(range=ylim_)

    if not show_axes:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

    # fig.show()

    return fig

# Make the dash app

broadqrs_ddrtree = pd.read_csv("broadqrs_relevant_plotting_use.csv")

# Map branch coordinates to colors
color_map = {
    'Branch 1': 'firebrick',
    'Branch 2': 'gold',
    'Branch 3': 'forestgreen',
    'Branch 4': 'lightseagreen',
    'Branch 5': 'royalblue',
    'Branch 6': 'orchid'
}

branch_map = {
    'Branch 1': 'Higher risk LBBB (Phenogroup 1)',
    'Branch 2': 'Higher risk LBBB/NSIVCD (Phenogroup 2)',
    'Branch 3': 'Higher risk IVCD (Phenogroup 3)',
    'Branch 4': 'Average branch RBBB (Phenogroup 4)',
    'Branch 5': 'Lower risk RBBB (Phenogroup 5)',
    'Branch 6': 'Higher risk RBBB (Phenogroup 6)'
}

color_map1 = {
    'firebrick': (178, 34, 34),
    'gold': (255, 215, 0),
    'forestgreen': (34, 139, 34),
    'lightseagreen': (32, 178, 170),
    'royalblue': (65, 105, 225),
    'orchid': (218, 112, 214)
}

broadqrs_ddrtree['phenogroup'] = broadqrs_ddrtree['merged_branchcoords'].map(branch_map)

# Add a new column 'color' with mapped colors
broadqrs_ddrtree['color'] = broadqrs_ddrtree['merged_branchcoords'].map(color_map).fillna('gray')

# Create initial Plotly figure
fig = go.Figure()

# Add the scattergl trace
fig.add_trace(go.Scattergl(
    x=broadqrs_ddrtree['Z1'],
    y=broadqrs_ddrtree['Z2'],
    mode='markers',
    marker=dict(
        size=7,
        color=broadqrs_ddrtree['color'],
        opacity=0.7,
        line=dict(width=1, color='black')
    ),
    name='Scatter Points',
    hoverinfo='x+y+text',
    text=broadqrs_ddrtree['phenogroup']
))

# Customize the layout
fig.update_layout(
    title='Broad QRS DDRTree',
    xaxis_title='Dimension 1',
    yaxis_title='Dimension 2',
    width=700,
    height=700,
    font=dict(
        size=15
    )
)

# Create Dash app
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1(children='Visualising ECGs from the broad QRS DDRTree', style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(
            id='scatter-plot',
            figure=fig
        ),
        dcc.Graph(
            id='hover-data-plot',
            style={'margin-top': '1px'}  # Adjust the margin-top value as needed
        )
    ], style={'display': 'flex'}),
])

@app.callback(
    Output('hover-data-plot', 'figure'),
    [Input('scatter-plot', 'hoverData')]
)
def update_hover_plot(hoverData):
    if hoverData is None:
        # If there is no hover data, return a blank plot with just the text annotation
        return {
            'data': [],
            'layout': {
                'annotations': [
                    {
                        'x': 3.2,
                        'y': 1.5,
                        'text': "Hover over a point on the tree to see the reconstructed ECG.",
                        'showarrow': False,
                        'font': {'size': 15, 'color': 'black'},
                    }
                ],
                'xaxis': {'visible': False},
                'yaxis': {'visible': False},
                'width': 600,
                'height': 500,
            }
        }
    
    # If hover data is available, proceed to retrieve and plot the ECG data
    point_index = hoverData['points'][0]['pointIndex']
    
    # Retrieve the ECG data for the hovered point
    ecg_data = decoded_ecgs_array_12L[point_index][0]
    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    # Generate the ECG plot using viz.plot_ecg
    fig_plotly = plot_ecg_plotly(ecg_data, 400, lead_names=lead_names, subplot_shape=(3, 4), ylim=(-2, 2), subplots=True, figsize=(1000, 700))

    # Customize the layout to show the axes and title based on broadqrs_ddrtree color
    phenogroup = broadqrs_ddrtree['phenogroup'].iloc[point_index]
    point_color_name = hoverData['points'][0]['marker.color']
    rgb_color = color_map1.get(point_color_name)
    plotly_color = f'rgb{rgb_color}'
    fig_plotly['layout']['title'].update(text=f"Reconstructed ECG - {phenogroup}", font=dict(color=plotly_color, size=15))
    
    return fig_plotly

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8052, debug=True)


