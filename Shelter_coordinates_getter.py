import matplotlib.pyplot as plt
import numpy as np
import os

# Initialize lists to store the coordinates and centroids
shelter_coords = []
shelter_centroids = []

def onclick(event):
    ix, iy = event.xdata, event.ydata
    coord_order = ['top-left', 'top-right', 'bottom-right', 'bottom-left']
    order_index = len(shelter_coords) % 4

    print(f'Point {len(shelter_coords) + 1}: {coord_order[order_index]} corner (x = {ix}, y = {iy})')
    shelter_coords.append((ix, iy))
    ax.plot(ix, iy, 'ro')
    plt.draw()

    if order_index == 3:
        calculate_and_plot_centroid()

    if len(shelter_coords) == 12:
        disconnect_and_save_data()

def calculate_and_plot_centroid():
    current_shelter_coords = shelter_coords[-4:]
    centroid_x = sum(point[0] for point in current_shelter_coords) / 4
    centroid_y = sum(point[1] for point in current_shelter_coords) / 4
    shelter_centroids.append((centroid_x, centroid_y))
    ax.plot(centroid_x, centroid_y, 'x', color='yellow')
    plt.draw()
    print(f"Centroid for Shelter {len(shelter_centroids)}: (x = {centroid_x}, y = {centroid_y})")

def disconnect_and_save_data():
    plt.gcf().canvas.mpl_disconnect(cid)
    plt.close()

    shelters_array = np.array(shelter_coords)
    centroids_array = np.array(shelter_centroids)

    np.save(os.path.join(base_path, 'shelter_coords.npy'), shelters_array)
    np.save(os.path.join(base_path, 'shelter_centroids.npy'), centroids_array)
    print("Shelter coordinates and centroids saved.")

# Set the base path for saving the files
base_path = '/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/DLC_analysis'

# Load the image
image_path = '/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/DLC_analysis/shelter_frame_mouseid_1119976.jpg'
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image file not found at {image_path}")

image = plt.imread(image_path)
fig, ax = plt.subplots()
ax.imshow(image)
cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)

plt.title('Click the corners of each shelter in the following order:\n'
          'Top-left, top-right, bottom-right, bottom-left.\n'
          'Repeat this for Shelter 1, Shelter 2, and then Shelter 3.')
plt.show()
