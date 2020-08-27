import os
import shutil
from utils.utils import *

# Define filenames and directories
save_dir = "output/" # save directory
input_dir = "captured/" # input directory
'''if os.path.exists(input_dir):
    shutil.rmtree(input_dir)  # delete input folder to create new input data
    os.makedirs(input_dir)  '''
cap_input = input_dir + getDT() # sets the current date and time as the input filename

# Capturing input options
picture  =  True           # Change to False for capturing video input 
view_output = False        # Change to True to display output detections
save = True                # Change to True to save inferences to output folder
seconds = 5                # Length in seconds for capturing video

# Inference options
conf_thresh = 0.4          # Object confidence threshold
iou_thresh = 0.5           # IOU threshold for NMS
imgsz = 640                # Image size
device = 'cpu'             # Torch device to use
weights = "models/weights/model.pt" # weights file

