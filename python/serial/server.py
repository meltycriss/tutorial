import serial

ser = serial.Serial()
ser.port = "/dev/ttyUSB1"
ser.baudrate = 115200
ser.bytesize = serial.EIGHTBITS
ser.parity = serial.PARITY_NONE
ser.timeout = 1.

ser.open()
print (ser.is_open)

while True:
    line = ser.read()
    print (line.hex())
    # print (type(line))
    # if line!='':
    #     print (chr(int(line, 16)))
    # print (line)

ser.close()
print (ser.is_open)
