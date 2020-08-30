import picamera
import busio
from digitalio import DigitalInOut, Direction, Pull
import board
import RPi.GPIO as GPIO
import adafruit_rfm9x
from time import sleep
from utils.settings import *
from subprocess import call

# Configure LoRa Radio
CS = DigitalInOut(board.CE1)
RESET = DigitalInOut(board.D25)
spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
rfm9x = adafruit_rfm9x.RFM9x(spi, CS, RESET, 915.0)
rfm9x.tx_power = 23
prev_packet = None

# Configure servo motor
servoPIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN, GPIO.OUT)
p = GPIO.PWM(servoPIN, 50) 
p.start(2.5) 

# Rotates the device
def rotate():
    print("Rotating device")
    p.ChangeDutyCycle(5) # left -90 deg position
    sleep(1.5)
    
# Sends to Raspberry Pi receiver
def notify(detections=0, light_sensor=False):
    if detections != 0:
        # check for packet rx
        packet = None
        packet = rfm9x.receive()
        prev_packet = packet
        packet_text = str(prev_packet, "utf-8")
        if light_sensor:
            if detections < 30:
                string = bytes("Illegal Fishing detected near Buoy 1\nCoordinates:(10.101193,123.450346)")
                rfm9x.send(string)
                display.show()
            print("Sent to server: Light detected")
        else:
            detections_str = "Illegal Fishing detected near Buoy 1\nCoordinates:(10.101193,123.450346)\n" + str(detections) + "/s detected"
            string = bytes(detections_str)
            display.show()
            print("Sent to server: " + detections + " boats")
        

# Picamera takes a picture
def takePic(loop=10):
    with picamera.PiCamera() as camera:
        camera.vflip = True
        global cap_input
        for i in range(loop):
            same = 1
            if os.path.exists(cap_input + img_formats[1]): # check if current file exists and adds (1)
                cap_input = cap_input + "(same)" + img_formats[1]
                same = same + 1
            else:
                cap_input = cap_input + img_formats[1]
            camera.capture(cap_input)
    print(loop, " pic/s taken!")

# Picamera takes a video
def takeVideo():
    with picamera.PiCamera() as camera:
        camera.vflip = True
    global cap_input, seconds
    camera.start_recording(cap_input + ".h264")
    sleep(seconds)
    camera.stop_recording()
    command = "MP4BOX -add " + cap_input + ".h264 " + cap_input + vid_formats[2]
    call([command], shell=True)
    print("Converting video...")
    os.remove(cap_input + ".h264")
    print("Video taken!")
