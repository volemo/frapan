from matplotlib import pyplot as plt
import tifffile as tf
import numpy as np
import os

def gaussian_data():
  # ==== Setup ====
  path = './tests/gaussian/'
  width = 512
  height = 512
  count = 32
  shape = count, height, width # (t, y, x)

  # Additive noise:
  ana = 0 # amplitude
  anm = 0 # mean
  anv = 0 # variance

  # Parameter noise:
  pna = 0 # amplitude
  pnm = 0 # mean
  pnv = 0 # variance

  # Distribution function:
  def gaussian(x, t, y):
    rnd = lambda: pna * np.random.normal(pnm, pnv)
    a = 1 + rnd()
    b = height / 2 + rnd()
    c = c if (c := 1 * t**2 + rnd()) != 0 else 1e-4
    d = 0
    return a * np.exp(-(x-b)**2/(2*c**2)) + d


  # ==== Create ====
  series = np.zeros(shape)

  # Target distribution:
  ys = np.arange(height)
  for k in range(count):
    for i in range(width):
      series[k, :, i] = gaussian(ys, k, i)

  # Additive noise:
  series += ana * np.random.normal(anm, anv, shape)


  # ==== Save ====
  if not os.path.exists(path):
    os.makedirs(path)

  # Series:
  for k in range(count):
    image = series[k, :, :]
    image = image.T
    image = np.stack((image, image, image), axis=-1)
    tf.imwrite(path + f't{k:02d}.tif', image, photometric='rgb')

  # Config:
  config = (f'title = "test series: gaussian distribution"\n'
            f'times = {list(range(count))}\n'
            f'base_images_number = 1\n'
            f'[size]\n'
            f'px.width = {width}\n'
            f'px.height = {height}\n'
            f'um.height = 1\n')

  with open(path + 'config.toml', "w") as config_file:
    config_file.write(config)


  # ==== Plot for comparison ====
  profiles = []
  for k in range(count):
    profile = np.sum(series[k, :, :], axis=-1) / width
    plt.plot(ys, profile)
    plt.savefig(path + f'compare/profile-{k}.png')
    plt.close()
    profiles.append(profile)

  profiles = np.concatenate(profiles, axis=0)
  tf.imwrite(path + 'compare/mean-profiles.tif',
             np.stack([profiles]*3, axis=-1),
             photometric='rgb')

if __name__ == '__main__':
  gaussian_data()
