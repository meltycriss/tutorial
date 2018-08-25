import rpyc
import numpy as np
import time

# for numpy
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True

c = rpyc.connect("172.18.197.179", 18861)
x = np.ones(42)

s = c.root.get_init_state()
t = []
for _ in range(10):
    begin = time.time()
    o = np.ones(42)
    a, s = c.root.get_velocity(o, s)
    # for numpy
    a = rpyc.utils.classic.obtain(a)
    t.append(time.time()-begin)
print (np.average(np.array(t)))
