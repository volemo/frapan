from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import tifffile as tf
import numpy as np



## Configure:

filenames = [
    'data/simple_frap/t00.tif', 
    'data/simple_frap/t01.tif', 
    'data/simple_frap/t02.tif', 
    'data/simple_frap/t03.tif', 
    'data/simple_frap/t04.tif', 
    'data/simple_frap/t05.tif', 
    'data/simple_frap/t06.tif', 
    'data/simple_frap/t07.tif', 
    'data/simple_frap/t08.tif', 
    'data/simple_frap/t09.tif', 
    'data/simple_frap/t10.tif', 
    'data/simple_frap/t11.tif', 
    'data/simple_frap/t12.tif', 
    'data/simple_frap/t13.tif', 
    'data/simple_frap/t14.tif', 
    'data/simple_frap/t15.tif', 
    'data/simple_frap/t16.tif', 
    'data/simple_frap/t17.tif', 
    'data/simple_frap/t18.tif', 
    'data/simple_frap/t19.tif', 
]

times = [
    0.000, 0.393, 1.397, 1.790, 2.182, 2.575, 2.967, 3.360, 3.753, 4.145,
    4.538, 4.930, 5.323, 5.715, 6.108, 6.500, 6.898, 7.285, 7.678, 8.070
]

base_image_number = 3
size = { 'px': 512, 'um': 425.1 }



## Load files:

images = []
for filename in filenames:
    images.append(tf.imread(filename)[:, :, 0])


    
## Process data:

scale = size['um']/size['px']

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

# Offset the right spine of twin2.  The ticks and label have already been
# placed on the right by twinx above.
twin2.spines.right.set_position(('axes', 1.1))
twin3.spines.right.set_position(('axes', 1.2))

p0, = twin0.plot(times, parameters[:, 2], color='tab:orange', label='c')
p1, = twin1.plot(times, parameters[:, 0], color='tab:grey', linewidth=0.5, label='a')
p2, = twin2.plot(times, parameters[:, 1], color='tab:pink', linewidth=0.5, label='b')
p3, = twin3.plot(times, parameters[:, 3], color='tab:brown', linewidth=0.5, label='d')

twin0.set_xlabel('Time')
# twin0.set_ylabel('c')
# twin1.set_ylabel('a')
# twin2.set_ylabel('b')
# twin2.set_ylabel('d')

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

