import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torchvision.models as models
import pickle
from tqdm import tqdm


with open('./frames.pickle', 'rb') as f:
    frames = pickle.load(f)

with open('./angles.pickle', 'rb') as f:
    angles = pickle.load(f)

    
Writer = animation.writers['ffmpeg']
writer = Writer(fps=60, metadata=dict(artist='Me'), bitrate=1800)

x = []
y = []
max_fr = 1000
frames = frames[:max_fr]
for i in tqdm(range(max_fr)):

    y.append(angles[i])
    x.append(i)

fig = plt.figure()
p1 = fig.add_subplot(121)


p2 = fig.add_subplot(122)
p2.grid()

ys = []
def function(x):
    y = x
    ys.append(y)
    return y

# set up empty lines to be updates later on
l1  = p1.imshow(frames[0])
l2, = p2.plot(x, y,'b')
l3, = p2.plot(x, y, '*')

def run1(c):
    l1.set_array(frames[c])

def run2(c, x, y, line):
    
    if c >= 10:
        if ((c+10) < len(x)):
            l2.set_data(x[(c-10):(c+10)], y[(c-10):(c+10)])
            l2.axes.axis([x[c-10], x[c+10], -180, 180])
        else:
            l2.set_data(x[(c-10):], y[(c-10):])
            l2.axes.axis([x[c-10], x[-1], -180, 180]) 
    else:
        l2.set_data(x[:(c+10)], y[0:(c+10)])
        l2.axes.axis([x[0], x[c+10], -180, 180])

    l3.set_data(x[c], y[c])
    asp = np.diff(l2.axes.get_xlim())[0] / np.diff(l2.axes.get_ylim())[0]
    l2.axes.set_aspect(asp)
    
    l1.set_array(frames[c])
    
    return l2, l1, l3, 

#ani1 = animation.FuncAnimation(fig,run1 ,interval=25, frames =200)
ani2 = animation.FuncAnimation(fig,run2 ,interval=25, frames=max_fr, fargs=[x, y, l2], blit=True)

ani2.save('im.mp4', writer=writer)
plt.show()