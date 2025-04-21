from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import tifffile as tf
import numpy as np
from tomllib import load
import os



## Configure:

path = 'data/simple_frap/'

config = load(open(path + 'config.toml', 'rb'))

filenames = next(os.walk(path), (None, None, []))[2] 
filenames = set(filenames) - set(config['exclude_files']) - set(['config.toml'])
filenames = sorted(list(filenames))

times = config['times']

base_image_number = config['base_image_number']
size = { 'px': 512, 'um': 425.1 }



## Load files:

images = []
for filename in filenames:
    images.append(tf.imread(path + filename)[:, :, 0])


    
## Process data:

scale = config['size']['um']/config['size']['px']

base_image = (images[0] + images[1])/2
images = images[base_image_number:]
times = times[base_image_number:]
N = len(images)

# Normalise:
images = [ image / base_image for image in images ]

# Calculate mean profile for every image:
mean_intensity = np.zeros((base_image.shape[0], N))
for i in range(N):
    mean_intensity[:, i] = np.sum(images[i], axis=1) / images[i].shape[0]

# Fit a Gaussian to every profile:
def f(x, a, b, c, d):
    return a * np.exp(-(x-b)**2/(2*c**2)) + d

parameters = np.zeros((N, 4))
errors = np.zeros((N, 4))

xs = np.arange(base_image.shape[0])
for i in range(N):
    guess = [-0.05, xs[xs.shape[0] // 2], 44, 1.04]
    popt, pcov = curve_fit(f, xs, mean_intensity[:, i], p0=guess)
    parameters[i, :] = popt
    errors[i, :] = np.sqrt(np.diag(pcov))

    

## Postprocess results:

# Save mean series:
mean_intensity_image = np.stack((mean_intensity, mean_intensity, mean_intensity), axis=-1)
tf.imwrite('result/mean-intensity.tif', mean_intensity_image, photometric='rgb')

# Plot mean profiles with approximations:
xs = np.arange(base_image.shape[0]) * scale
for i in range(N):
    ys = f(xs, *(parameters[i, :]))
    plt.plot(xs, mean_intensity[:, i], label='Measured')
    plt.plot(xs, ys, label='Approximation')
    plt.xlabel('Coordinate, um')
    plt.ylabel('Intensity, a.u.')
    plt.legend()
    plt.savefig(f'result/profile-{i}.png', dpi=300)
    plt.cla()

# Plot approximation parameters over time:
fig, twin0 = plt.subplots()
fig.subplots_adjust(right=0.75)

twin1 = twin0.twinx()
twin2 = twin0.twinx()
twin3 = twin0.twinx()

twin2.spines.right.set_position(('axes', 1.1))
twin3.spines.right.set_position(('axes', 1.2))

p0, = twin0.plot(times, parameters[:, 2], color='tab:orange', label='c')
p1, = twin1.plot(times, parameters[:, 0], color='tab:grey', linewidth=0.5, label='a')
p2, = twin2.plot(times, parameters[:, 1], color='tab:pink', linewidth=0.5, label='b')
p3, = twin3.plot(times, parameters[:, 3], color='tab:olive', linewidth=0.5, label='d')

twin0.set_xlabel('Time')

twin0.yaxis.label.set_color(p0.get_color())
twin1.yaxis.label.set_color(p1.get_color())
twin2.yaxis.label.set_color(p2.get_color())
twin3.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
twin0.tick_params(axis='x', **tkw)
twin0.tick_params(axis='y', colors=p0.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p1.get_color(), labelrotation=90, **tkw)
for tick in twin1.get_yticklabels():
    tick.set_verticalalignment('center')
    twin2.tick_params(axis='y', colors=p2.get_color(), labelrotation=90, **tkw)
for tick in twin2.get_yticklabels():
    tick.set_verticalalignment('center')
    twin3.tick_params(axis='y', colors=p3.get_color(), labelrotation=90, **tkw)
for tick in twin3.get_yticklabels():
    tick.set_verticalalignment('center')


twin0.legend(handles=[p0, p1, p2, p3])

plt.savefig(f'result/parameters.png', dpi=300)
plt.cla()

# Plot approximation errors over time
fig, twin0 = plt.subplots()
fig.subplots_adjust(right=0.75)

twin1 = twin0.twinx()
twin2 = twin0.twinx()
twin3 = twin0.twinx()

twin2.spines.right.set_position(('axes', 1.1))
twin3.spines.right.set_position(('axes', 1.2))

p0, = twin0.plot(times, errors[:, 2], color='tab:orange', label='c')
p1, = twin1.plot(times, errors[:, 0], color='tab:grey', linewidth=0.5, label='a')
p2, = twin2.plot(times, errors[:, 1], color='tab:pink', linewidth=0.5, label='b')
p3, = twin3.plot(times, errors[:, 3], color='tab:olive', linewidth=0.5, label='d')

twin0.set_xlabel('Time')

twin0.yaxis.label.set_color(p0.get_color())
twin1.yaxis.label.set_color(p1.get_color())
twin2.yaxis.label.set_color(p2.get_color())
twin3.yaxis.label.set_color(p3.get_color())

tkw = dict(size=4, width=1.5)
twin0.tick_params(axis='x', **tkw)
twin0.tick_params(axis='y', colors=p0.get_color(), **tkw)
twin1.tick_params(axis='y', colors=p1.get_color(), labelrotation=90, **tkw)
for tick in twin1.get_yticklabels():
    tick.set_verticalalignment('center')
    twin2.tick_params(axis='y', colors=p2.get_color(), labelrotation=90, **tkw)
for tick in twin2.get_yticklabels():
    tick.set_verticalalignment('center')
    twin3.tick_params(axis='y', colors=p3.get_color(), labelrotation=90, **tkw)
for tick in twin3.get_yticklabels():
    tick.set_verticalalignment('center')

twin0.legend(handles=[p0, p1, p2, p3])

plt.savefig(f'result/errors.png', dpi=300)
