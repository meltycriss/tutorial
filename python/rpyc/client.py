import rpyc
import numpy as np
import time

c = rpyc.connect("localhost", 18861)
x = np.ones(42)

def get_answer(x):
    return x+1

t = []
for _ in range(1):
    begin = time.time()
    k = c.root.get_answer(x)
    # k = get_answer(x)
    t.append(time.time()-begin)
print (np.average(np.array(t)))
