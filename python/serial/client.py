import serial

ser = serial.Serial()
ser.port = "/dev/ttyUSB0"
ser.baudrate = 115200
ser.baudrate = 9600
ser.bytesize = serial.EIGHTBITS
ser.parity = serial.PARITY_NONE
ser.timeout = 1.

ser.open()
print (ser.is_open)

# ser.write("hello".encode('utf-8'))
ser.write(b"hello")

ser.close()
print (ser.is_open)
