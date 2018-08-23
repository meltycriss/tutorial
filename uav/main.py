import queue
from NatNetClient import NatNetClient
import time
import serial
import struct

###########################################
# hyper-parameter
###########################################

hp_n_bot = 3 # number of robots
hp_n_obs = 0 # number of obstacles
hp_queue_size = 10 # queue buffer size

###########################################
# optitrack stuff
###########################################

qs_bot = [queue.Queue(hp_queue_size) for _ in range(hp_n_bot)]
qs_obs = [queue.Queue(hp_queue_size) for _ in range(hp_n_obs)]

def receiveRigidBodyFrame(id, position, rotation):
    global qs_bot
    global qs_obs

    # ROBOTS SHOULD PRECEDE OBSTACLES IN MOTIVE
    if id <= hp_n_bot:
        q = qs_bot[id-1]
    else:
        q = qs_obs[id-hp_n_bot-1]

    q = qs[id-1]
    if q.full():
        q.get()
    q.put(position)

###########################################
# serial port stuff
###########################################

# command reference
CMD_TAKEOFF = 1
CMD_LAND = 2
CMD_RESET = 3
CMD_CTRL = 4

ser = serial.Serial()
ser.port = "/dev/ttyUSB0"
ser.baudrate = 115200
ser.bytesize = serial.EIGHTBITS
ser.parity = serial.PARITY_NONE
ser.timeout = 1.

# auxiliary function encoding float to unsigned int
def float_to_uint(f):
    # >I refers to big endian unsigned integer
    # >f refers to big endian float32
    return struct.unpack('>I', struct.pack('>f', f))[0]

#####################################################################
#   info        #   size    #   remark
#####################################################################
# header        #   1B      #   0xfe
# robot index   #   1B      #
# command       #   1B      #
# v_x           #   4B      #   big endian(significant first) float32
# v_y           #   4B      #
# v_z           #   4B      #
# w             #   4B      #
# checksum      #   1B      #   byte-wise sum of v_x, v_y, v_z and w
#####################################################################
def sendCommand(id, cmd, x, y, z, w):
    assert isinstance(id, int)
    assert isinstance(cmd, int)
    assert isinstance(x, float)
    assert isinstance(y, float)
    assert isinstance(z, float)
    assert isinstance(w, float) # rotation
    # restriction of the receiver
    assert (x<3.)
    assert (y<3.)
    assert (z<3.)
    assert (w<3.)

    header = bytearray.fromhex('fe')
    index  = bytearray.fromhex(format(id, '02x')) # robot are 1-idx
    command  = bytearray.fromhex(format(cmd, '02x'))

    ctrl_vars = [x, y, z, w]
    ctrl_vars_uint = list(map(float_to_uint, ctrl_vars))
    ctrl_vars_ba = bytearray()
    for ctrl_var in ctrl_vars_uint:
        ctrl_vars_ba += bytearray.fromhex(format(ctrl_var, '08x'))

    bytewise_sum = sum([b for b in ctrl_vars_ba])
    checksum = bytearray.fromhex(format(bytewise_sum % 100, '02x'))

    # for b in ctrl_vars_ba:
    #     print (hex(b))
    # print (bytewise_sum)
    # print (int.from_bytes(index, byteorder='big'), )

    frame = header + index + command + ctrl_vars_ba + checksum
    num_of_bytes = ser.write(frame)
    # print (num_of_bytes)

###########################################
# main
###########################################
if __name__=='__main__':
    # open motive client
    streamingClient = NatNetClient()
    streamingClient.rigidBodyListener = receiveRigidBodyFrame
    streamingClient.run()
    # open serial port communication
    ser.open()
    # main loop
    while True:
        time.sleep(1)
        for i, q in enumerate(qs):
            p = q.get()
            print ("{}: {}".format(i, p))
            print ((type(p)))
        sendCommand(3, CMD_CTRL, 1., -1.522, -2.333, -2.5666)
        break
    ser.close() # what if the process is killed?
