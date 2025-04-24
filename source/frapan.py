from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import optimize
from inspect import signature
from mergedeep import merge
from tomllib import load
from pprint import pp
import tifffile as tf
import numpy as np
import sys
import os

class Frapan:
    # WARNING: This implementation assumes profile size is constant across the series.

    default_config = {
        'results_directory': 'results/',
        'temporary_directory': 'temporary/',
        'approximation': {
            'func': 'gaussian',
        }
    }
    
    def __init__(self, path, adhoc_config={}):
        path += '/' if path[-1] != '/' else ''
        file_config = load(open(path + 'config.toml', 'rb'))
        self.config = merge({}, self.default_config, file_config, adhoc_config)

        filenames = next(os.walk(path), (None, None, []))[2] 
        filenames = set(filenames) - set(self.config['exclude_files']) - set(['config.toml'])
        self.filenames = sorted(list(filenames))
        print(self.filenames)

        self.images = []
        for filename in filenames:
            self.images.append(tf.imread(path + filename)[:, :, 0])

        self.expand_config()
            
        self.bi_num = self.config['base_images_number']
        self.shape = (self.config['size']['px']['width'], self.config['size']['px']['height'])
        self.rslt_dir = self.config['results_directory']
        self.tmp_dir = self.config['temporary_directory']


    def expand_config(self):
        real_size = self.config['size']['um']['height']
        pixel_size = self.config['size']['px']['height']
        # TODO: Check which properties are set (i.e. is size.scale set? is size.px set? etc).
        self.config['size']['scale'] = real_size / pixel_size 


    def divide_by_base(self):
        self.base_image = sum(self.images[:self.bi_num]) / self.bi_num 
        self.images = self.images[self.bi_num:]

        self.config['times'] = self.config['times'][self.bi_num:]

        self.images = [ image / self.base_image for image in self.images ]

        
    def subtract_base(self):
        self.base_image = sum(self.images[:self.bi_num]) / self.bi_num 
        self.images = self.images[self.bi_num:]

        self.config['times'] = self.config['times'][self.bi_num:]

        self.images = [ image - self.base_image for image in self.images ]


    def normalise_images(self):
        self.divide_by_base()
        
        
    def compute_mean_profiles(self):
        N = len(self.images)
        self.mean_profiles = np.zeros((self.shape[0], N))
        for i in range(N):
            self.mean_profiles[:, i] = np.sum(self.images[i], axis=1) / self.shape[0] 
            

    def smooth_out(self):
        N = len(self.mean_profiles[0, :])
        w = len(self.mean_profiles[:, 0]) // 25 
        profiles = np.zeros((len(self.mean_profiles[:, 0]) - w + 1, N))
        for i in range(N):
            profiles[:, i] = np.convolve(self.mean_profiles[:, i], np.ones(w), 'valid') / w 

            for j in range(w//2):
                self.mean_profiles[j, i] = profiles[0, i]
            for j in range(len(profiles)):
                self.mean_profiles[j + w//2, i] = profiles[j, i]
            for j in range(1, w//2):
                self.mean_profiles[-j, i] = profiles[-1, i]

    funcs = {
        'gaussian': lambda x, a, b, c, d: a * np.exp(-(x-b)**2/(2*c**2)) + d,
        'double_gaussian': lambda x, a1, a2, b1, b2, c1, c2, d: \
        a1 * np.exp(-(x-b1)**2/(2*c1**2)) + a2 * np.exp(-(x-b2)**2/(2*c2**2)) + d,
    }

    
    def guess_gaussian_parameters(self, ys, i):
        plt.plot(ys)
        w = len(ys) // 25 
        ys = np.convolve(ys, np.ones(w), 'valid') / w
        plt.plot(ys)
        plt.savefig(self.tmp_dir + f'guessing-{i}.png', dpi=300)
        plt.close()
        return [-0.05, ys.shape[0] // 2, 44, 1.04]
        
    
    def approximate(self, func=None):
        N = len(self.images)
        self.func = func if func else self.funcs[self.config['approximation']['func']]
        sig = signature(self.func)
        par_num = len(sig.parameters) - 1
        self.parameters = np.zeros((N, par_num))
        self.errors = np.zeros((N, par_num))

        xs = np.arange(self.shape[0])
        for i in range(N):
            try:
                guess = None
                if 'guess' not in self.config['approximation']:
                    # NB! This is a guess for the parameters of a *gaussian*.
                    guess = self.guess_gaussian_parameters(self.mean_profiles[:, i], i)
                else:
                    guess = self.config['approximation']['guess'] 

                popt, pcov = curve_fit(self.func, xs, self.mean_profiles[:, i], p0=guess)
                self.parameters[i, :] = popt
                self.errors[i, :] = np.sqrt(np.diag(pcov))
            except optimize.OptimizeWarning:
                print('Warning: Approximation: For image ' + str(i) + 'covariance of the ' + \
                      'parameters could not be estimated. (This mean the approximation is ' + \
                      'probably out of wack.')
            except RuntimeError:
                print(f'Error: Approximation: Could not find optimal parameters for image {i}.')


    def save_mean_profiles(self):
        mp_image = self.mean_profiles
        mp_image = np.stack((mp_image, mp_image, mp_image), axis=-1)
        path = self.rslt_dir + 'mean-profiles.tif'
        tf.imwrite(path, mp_image, photometric='rgb')


    def plot_mean_profiles(self):
        N = len(self.images)
        xs = np.arange(self.shape[0]) * self.config['size']['scale']
        for i in range(N):
            plt.plot(xs, self.mean_profiles[:, i], label='Measured')
            plt.xlabel('Coordinate, um')
            plt.ylabel('Intensity, a.u.')
            plt.legend()
            plt.savefig(self.rslt_dir + f'profile-{i + self.bi_num}.png', dpi=300)
            plt.close()
                

    def plot_mean_profiles_with_approximations(self):
        N = len(self.images)
        xs = np.arange(self.shape[0]) * self.config['size']['scale']
        for i in range(N):
            ys = self.func(xs, *(self.parameters[i, :]))
            plt.plot(xs, self.mean_profiles[:, i], label='Measured')
            plt.plot(xs, ys, label='Approximation')
            plt.xlabel('Coordinate, um')
            plt.ylabel('Intensity, a.u.')
            plt.legend()
            plt.savefig(self.rslt_dir + f'profile-{i + self.bi_num}.png', dpi=300)
            plt.close()


    def plot_approximation_parameters_over_time(self):
        fig, twin0 = plt.subplots()
        fig.subplots_adjust(right=0.75)

        twin1 = twin0.twinx()
        twin2 = twin0.twinx()
        twin3 = twin0.twinx()

        twin2.spines.right.set_position(('axes', 1.1))
        twin3.spines.right.set_position(('axes', 1.2))

        ts = self.config['times']
        ps = self.parameters 
        p0, = twin0.plot(ts, ps[:, 2], color='tab:orange', label='c')
        p1, = twin1.plot(ts, ps[:, 0], color='tab:grey', linewidth=0.5, label='a')
        p2, = twin2.plot(ts, ps[:, 1], color='tab:pink', linewidth=0.5, label='b')
        p3, = twin3.plot(ts, ps[:, 3], color='tab:olive', linewidth=0.5, label='d')

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

        plt.savefig(self.rslt_dir + f'parameters.png', dpi=300)
        plt.close()


    def plot_approximation_errors_over_time(self):
        fig, twin0 = plt.subplots()
        fig.subplots_adjust(right=0.75)

        twin1 = twin0.twinx()
        twin2 = twin0.twinx()
        twin3 = twin0.twinx()

        twin2.spines.right.set_position(('axes', 1.1))
        twin3.spines.right.set_position(('axes', 1.2))

        ts = self.config['times']
        es = self.errors
        p0, = twin0.plot(ts, es[:, 2], color='tab:orange', label='c')
        p1, = twin1.plot(ts, es[:, 0], color='tab:grey', linewidth=0.5, label='a')
        p2, = twin2.plot(ts, es[:, 1], color='tab:pink', linewidth=0.5, label='b')
        p3, = twin3.plot(ts, es[:, 3], color='tab:olive', linewidth=0.5, label='d')

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

        plt.savefig(self.rslt_dir + f'errors.png', dpi=300)
        plt.close()



if __name__ == '__main__':
    frapan = Frapan(sys.argv[-1])

    frapan.normalise_images()
    # frapan.normalise_by_subtracting()
    frapan.compute_mean_profiles()
    frapan.smooth_out()
    frapan.approximate()

    frapan.save_mean_profiles()
    # frapan.plot_mean_profiles()
    frapan.plot_mean_profiles_with_approximations()
    # frapan.plot_approximation_parameters_over_time()
    # frapan.plot_approximation_errors_over_time()
