import board
import RPi.GPIO as GPIO
from utils.check_night import *
from rpi_modules.commands import *
GPIO.setmode(GPIO.BOARD)

#define the pin that goes to the circuit
pin_to_circuit = 7
# os.system("sudo rfcomm listen hci0&")

def rc_time (pin_to_circuit):
    count = 0
  
    #Output on the pin for 
    GPIO.setup(pin_to_circuit, GPIO.OUT)
    GPIO.output(pin_to_circuit, GPIO.LOW)
    time.sleep(0.1)

    #Change the pin back to input
    GPIO.setup(pin_to_circuit, GPIO.IN)
  
    #Count until the pin goes high
    while (GPIO.input(pin_to_circuit) == GPIO.LOW):
        count += 1

    return count
    
try:
    # Main loop
    while True:
        start, now, end = get_times()
        if now > start or now < end:
            val = rc_time(pin_to_circuit)
            # alert the receiver
            notify(detections=val,light_sensor=True)
            display.show()
            time.sleep(0.1)
    
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
