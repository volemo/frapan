"""A utility for plotting a TIFF image as a 3D distribution of points.
"""

import numpy as np
import tifffile as tf
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

parser = ArgumentParser(prog='Visualise TIFF',
                        description=('A utility for plotting a TIFF image'
                                     'as a 3D distribution of points.'))
parser.add_argument('input_filename')
parser.add_argument('-o', dest='output_filename')
args = parser.parse_args()
inf = args.input_filename
outf = args.output_filename
outf = outf if outf else inf + '.stl'


image = tf.imread(inf)

width = image.shape[0]
height = image.shape[1]
shape = (width * height, 3)

vectors = np.zeros((width*height, 3))

for i in range(width):
  for j in range(height):
    vectors[j*width + i] = [i, j, image[i, j, 0]]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

print(len(vectors))
vectors = vectors[::30]

x = vectors[:, 0]
y = vectors[:, 1]
z = vectors[:, 2]

ax.scatter(x, y, z, c=z)

plt.show()
