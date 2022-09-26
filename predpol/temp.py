import numpy as np
import matplotlib.pyplot as plt
from time import sleep

x = np.array([0])
y = np.array([0])

plt.ion()
fig = plt.figure()
ax=fig.add_subplot(111)
ax.set_xlim([0,50])
ax.set_ylim([0,2500])
line,  = ax.plot(x,y)
plt.show()
for i in range(51):
    x = np.append(x,[x[-1]+1])
    y = np.append(y,[x[-1]**2])
    line.set_data(x,y)
    plt.pause(0.01)