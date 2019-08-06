import time
import sys

sys.stdout.write('Loading: \n0')
for i in range(100):
    sys.stdout.write(str(i) + '\r')
    sys.stdout.flush()
    time.sleep(0.5)