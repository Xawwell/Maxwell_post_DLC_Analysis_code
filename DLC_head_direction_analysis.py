import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Sample head direction data (replace this with your actual data)
head_directions = np.load('/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/DLC_analysis/head_directions.npy')

def plot_head_direction(frame):
    # Clear the existing plot
    plt.clf()
    
    # Set the limits and title of the plot
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.title(f"Head Direction at Frame {frame}")

    # Compute the arrow's direction
    dx = np.cos(head_directions[frame])
    dy = np.sin(head_directions[frame])

    # Create an arrow representing the head direction
    plt.arrow(0, 0, dx, dy, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    # Show the plot
    plt.show()

# Create a slider widget to select the frame
frame_slider = widgets.IntSlider(value=0, min=0, max=len(head_directions)-1, step=1, description='Frame:')

# Create an interactive widget
widgets.interactive(plot_head_direction, frame=frame_slider)
