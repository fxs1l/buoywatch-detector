'''
    Pseudocode for testing without Raspberry Pi
'''

#import picamera
from time import sleep
from utils.settings import *
from subprocess import call

# Rotates the device
def rotate():
    #insert rotate codes here
    print("Rotating device")

# Sends to RaspberryPi server
def notify(detections=0):
    #insert notify code here
    if detections != 0:
        print("Sent to server: " + str(detections) + " boats")

# Picamera takes a picture
def takePic(loop=10):
    '''with picamera.PiCamera() as camera:
        camera.vflip = True
        global cap_input
        for i in range(loop):
            same = 1
            if os.path.exists(cap_input + img_formats[1]): # check if current file exists and adds (1)
                cap_input = cap_input + "(same)" + img_formats[1]
                same = same + 1
            else:
                cap_input = cap_input + img_formats[1]
            camera.capture(cap_input)'''
    print(loop, " pic/s taken!")

# Picamera takes a video
def takeVideo():
    '''with picamera.PiCamera() as camera:
        camera.vflip = True
    global cap_input, seconds
    camera.start_recording(cap_input + ".h264")
    sleep(seconds)
    camera.stop_recording()
    command = "MP4BOX -add " + cap_input + ".h264 " + cap_input + vid_formats[2]
    call([command], shell=True)
    print("Converting video...")
    os.remove(cap_input + ".h264")'''
    print("Video taken!")

