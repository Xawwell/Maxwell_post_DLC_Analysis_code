# #right now we have the x,y positions from the DLC and we want to bascially first apply a Kalman filter which ultimately increases the accuracy of the trackingimport h5py
# import h5py

# class DLCDataProcessorH5:
#     def __init__(self, h5_file_path):
#         self.h5_file_path = h5_file_path
#         self.dlc_output = None
#         self.tracking_data_body_parts = {
#             'bodyparts': [
#                 'nose', 'headcenter', 'neck', 'rightear', 'leftear',
#                 'lefthip', 'righthip', 'leftside', 'rightside',
#                 'bodycenter', 'tailbase', 't1', 't2'
#             ]
#         }

#     def extract_data_from_dlc_file(self) -> None:
#         """
#         Ingests an H5 file outputted from DLC analysis and extracts the specified body parts.
#         """
#         # Load DLC Tracking Data from H5 file
#         with h5py.File(self.h5_file_path, 'r') as h5_file:
#             # Extracting data for each body part
#             for body_part in self.tracking_data_body_parts['bodyparts']:
#                 # Extract coordinates and likelihood for each body part
#                 coords = h5_file['df_with_missing/table'][body_part][:]
#                 self.dlc_output = self.dlc_output or {}
#                 self.dlc_output[body_part] = {
#                     'x': coords[:, 0],
#                     'y': coords[:, 1],
#                     'likelihood': coords[:, 2]
#                 }

#         print(f"Extracted data for body parts: {list(self.dlc_output.keys())}")
#         return None

# # Example usage
# h5_file_path = '/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/DLC_analysis/camDLC_resnet50_Maxwelll_HaranOct20shuffle1_100000.h5'
# processor = DLCDataProcessorH5(h5_file_path)
# processor.extract_data_from_dlc_file()
# # Now processor.dlc_output contains the extracted data


import h5py
import pandas as pd 
import os

csv_path ='/Users/chenx/Desktop/Branco Lab/Shelter Choice Project/DLC_analysis/camDLC_resnet50_Maxwelll_HaranOct20shuffle1_100000.csv'



dataname = str(csv_path)

#loading output of DLC
Dataframe = pd.read_csv(os.path.join(dataname))
Dataframe.head()