"""
    --Optional/Experimental Feature-- 
    Allows for automatic sending of inferenced images to Bluetooth-enabled smartphones with OBEX Push Service.
    
    Features to add:
    1. Allow control from Buoywatch app
"""
import bluetooth
import os
import time
import cv2
from PyOBEX import client,responses
from utils.settings import *

remove = False
trusted_devices = ['B4:BF:F6:EE:B0:12']


# Adds a new trusted device 
def add_devices(device_address):
    global devices
    devices.append(device_address)
    
# Gets all inferenced images from output folder and converts to bytes
def get_images():
    global remove, save_dir
    images = os.listdir(save_dir)
    
    if len(images) == 0:
        raise Exception("No inferenced images found")
    files = {} # dictionary of filenames and their bytes-converted images
    
    for image in images:
        img = cv2.imread(save_dir+str(image))
        success, im_buf_arr = cv2.imencode(".jpg", img)
        byte_im = im_buf_arr.tobytes() # converts numpy array into bytes
        
        files[image] = byte_im
        if remove:
            os.remove(image) #removes current image from output folder

    return files

# Sends the inferenced images to all devices
def send():
    for device in devices:
        services = bluetooth.find_service(address=device)
        for trusted in trusted_devices:
            if trusted == device: # checks if device is a trusted device
                for service in services:
                    if service.get("name") == "OBEX Object Push":
                        port = service.get("port")
                        c = client.Client(device, port)
                        r = c.connect()
                        
                        start = time.monotonic()
                        while not isinstance(r, responses.ConnectSuccess):
                            if start - time.monotonic() == 3:
                                c.disconnect() # disconnects after 3 seconds
                                break
                        
                        files = get_images()
                        for key in files.keys():
                            c.put(key,files[key])
                        c.disconnect()
                        
while True:
    devices = bluetooth.discover_devices(duration=2)
    print("Discovered devices:\n", devices)
    if len(devices) > 0:
        send()
