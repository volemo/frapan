from frapan import Frapan
import sys

frapan = Frapan(sys.argv[-1])

frapan.normalise_images()
frapan.compute_mean_profiles()
frapan.approximate()

frapan.save_mean_profiles()
frapan.plot_mean_profiles_with_approximations()
frapan.plot_approximation_parameters_over_time()
frapan.plot_approximation_errors_over_time()
