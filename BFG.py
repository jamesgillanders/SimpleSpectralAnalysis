# importing all the necessary modules and packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import matplotlib
from astropy.convolution import convolve, Box1DKernel
import scipy as sp
from scipy import signal
import random
from scipy.stats import binned_statistic
import os

############### User input section - nice and prominent at the top here ###############

# setting font for output plots/figures
matplotlib.rc("font", family="Arial", size=14)

# provide filename for the spectrum we are fitting
spectrum_filename = ""

# if spectrum is in flux units, then provide distance (in cm) to convert to luminosity
distance_to_transient = 1.234e26 # cm

# object name - this is used for saved filenames
object_name = ""

# scaling factor for plot e.g. 1e37
scaling_factor = 1e36

# these are the ranges for the random number generator
# places constraints on how far we can modify current best-fit parameters
lower_range = 0.5  # lowest we want is half current value
upper_range = 2.0  # highest we want is double current value

# these are the  ranges for the random number generator for when we randomly jump away
# from our best fits
mc_lower_range = 0.4  # lowest we want is a fifth of current value
mc_upper_range = 2.5  # highest we want is 5 times current value

# what y-axis value (flux/luminosity) does the continuum for the spectrum lie at?
continuum_value = 2.8e36

# limits the number of iterations; i.e. will try 1000 times to beat the best-fit
# parameters - if it can't then those are saved as the best-fit
max_iteration_counter = 1000

# sets the number of times you want the code to find a best fit
#  the higher this is, the more fits are obtained - better averages
max_number_of_iterations = 11

# this is the upper limit for the probability of jumping away from our best-fit value
# numbers generated are between 0 and 1, so chances of jumping are 1 in 200 (for 0.005)
#  too high a limit, and the code won't complete - max_iteration_counter will never get
# to 1000 - can tweak per model, but 0.0005 (1 in 2000) is good
mc_walk_value_upper_limit = 0.0005  #  1 in 2000

# rest wavelength(s) of the line(s) under investigation
# can fit more than 1 Gaussian to the data, if needed (think CaII NIR triplet)
# rest_wavelengths = [8498.02, 8542.09, 8662.14]
# are the rest wavelengths almost coincident with the feature?
# If so the code struggles to properly get offsets
# due to the fact it bases offset guesses on a factor system - so if offset is
# very small, then a 50% increase may only be ~10 Angstroms - need to
# artificially increase this- add few 100 Angstroms to the rest wavelengths
offset_constant = 500
rest_wavelengths = [10037 + offset_constant, 10327 + offset_constant, 10915 + offset_constant]

# these are the values we start on for the fitting code
# change to best first guesses by eye
height = 1.1e36  # in whatever units the y-axis has
sigma = 1000  # in angstroms
offset = -400 # offset_constant  # in angstroms

# list of the relative strength(s) of the line(s) we are interested in - if only doing
# 1 line then put in whatever number you want BUT make sure the list is not empty
relative_strengths = [1.0e6 * 4, 8.7e6 * 4, 7.46e6 * 2]
# these are the Einstein A values * (2*J + 1) of CaII transitions

# specify the range of the model Gaussian you want to fit to the observed feature
# e.g. for 1-sigma, set = 1, 5-sigma, set =5 etc. Quality of fit will only be assessed
# in this wavleength range
chi_sq_factor = 2

# do you want to fit the input spectrum or a smoothed version of it? If you wish to use
# the input spectrum then set flag to False. if the input spectrum is noisy you will
# want to smooth. do you want to use median filter to smooth spectrum? (this keeps same
# number of data points). if so set flag to median_filter. or do you want to rebin and
# reduce no. of data points (recommended to improve runtime if spectrum has high
# resolution) set flag to bin_means if you want to
# use_smoothed_spectrum = False # no smoothing
# use_smoothed_spectrum = 1 # median_filter
use_smoothed_spectrum = 2  # bin_means

# this is needed if using median_filter
# set this to an odd number and increase for more smoothing
# check spectrum has been smoothed properly in output plot (do not over or under smooth)
smoothing_number = 21

#  this is used for the bin_means method
# sets how heavily the spectrum will be rebinned
# i.e. 20 means there will be 1 point every 20 Angstroms
rebinning_number = 10

# do you want to see the test plots showing smoothed/rebinned spectra? True/False
see_test_smooth = False
# do you want the 1-5 sigma points for mean best fit plotted?
plot_sigma_points = True
# display rest wavelength locations on the plot?
plot_rest_waves = True

####################### end of the user-input section #################################
####################### let the code do the rest of the business ######################


def _mkdir(newdir):
    """this function is for creating the required directory for the output files"""
    """works the way a good mkdir should :)
        - already exists, silently complete
        - regular file in the way, raise an exception
        - parent directory(ies) does not exist, make them as well
    """
    if os.path.isdir(newdir):
        pass
    elif os.path.isfile(newdir):
        raise OSError("a file with the same name as the desired " "dir, '%s', already exists." % newdir)
    else:
        head, tail = os.path.split(newdir)
        if head and not os.path.isdir(head):
            _mkdir(head)
        # print "_mkdir %s" % repr(newdir)
        if tail:
            os.mkdir(newdir)
    return newdir


def plot_demo_smoothed_spectrum_fn(spectrum, smooth_spectrum, scaling_factor):
    """The rebinning package gives us back the edges of the bins it creates but not the
    midpoint of the bin. However, if the bin size is large then
    using the lower or upper bound of a bin won't give the right wavelength. For the
    flux value in that bin - we actually want the wavelength averaged across that bin
    so - we need to average the upper and lower bin edge for each flux value to get
    actual wavelength value - super minor effect but proper practise"""
    fig, ax = plt.subplots()
    plt.plot(
        spectrum[:, 0], spectrum[:, 1] / scaling_factor, color="gray", label="Spectrum", alpha=0.4, linewidth=1.0,
    )
    plt.plot(
        smooth_spectrum[:, 0], smooth_spectrum[:, 1] / scaling_factor, color="blue", label="Smooth Spectrum", alpha=0.7, linewidth=1.0,
    )
    plt.title(f"Plot to demonstrate smoothing \n Went from {len(spectrum)} to {len(smooth_spectrum)} points in spectrum")
    # setting titles for x and y axes
    plt.xlabel("Rest Wavelength ($\mathrm{\AA}$)")
    plt.ylabel("Luminosity ($10^{%d}$ erg $\mathrm{s^{-1}}$ $\mathrm{\AA^{-1}}$)" % math.log10(scaling_factor))
    plt.show()

    return


def smooth_spectrum_fn(spectrum, use_smoothed_spectrum, smooth_number):
    """This generates the smooth_spectrum that we will fit our models to - depending on
    use_smoothed_spectrum flag, we will either rebin, smooth, or not modify the
    observed spectrum"""
    if use_smoothed_spectrum == 1:
        # Smooth the spectrum with convolve
        # kernel size needs to be odd and controls level of smoothing
        smooth_flux = sp.signal.medfilt(spectrum[:, 1], kernel_size=smooth_number)
        smooth_spectrum = np.c_[spectrum[:, 0], smooth_flux]

    elif use_smoothed_spectrum == 2:
        sum_of_wavelength_differences1 = 0
        counter1 = 0
        for ii in range(1, len(spectrum)):
            wavelength_separation1 = spectrum[ii, 0] - spectrum[ii - 1, 0]
            sum_of_wavelength_differences1 += wavelength_separation1
            counter1 += 1
        initial_binning1 = sum_of_wavelength_differences1 / counter1

        desired_binning1 = smooth_number

        # this rebins the spectra
        # kernel size needs to be odd and controls level of smoothing
        bin_means = binned_statistic(spectrum[:, 0], spectrum[:, 1], bins=int(len(spectrum) / desired_binning1 * initial_binning1),)

        sum_of_wavelength_differences2 = 0
        counter2 = 0
        for ii in range(1, len(bin_means.bin_edges)):
            wavelength_separation2 = bin_means.bin_edges[ii] - bin_means.bin_edges[ii - 1]
            sum_of_wavelength_differences2 += wavelength_separation2
            counter2 += 1
        new_binning1 = sum_of_wavelength_differences2 / counter2

        binned_spectrum_wavelengths = []
        for jj in range(1, len(bin_means.bin_edges)):
            binned_spectrum_wavelengths.append((bin_means.bin_edges[jj] + bin_means.bin_edges[jj - 1]) / 2)

        smooth_spectrum = np.c_[binned_spectrum_wavelengths[:], bin_means.statistic[:]]

    elif use_smoothed_spectrum is False:
        smooth_spectrum = spectrum

    else:
        print("Something wrong! See rebin_spectrum function!")

    return smooth_spectrum


def calculate_chi_sq_fn(rest_wavelengths, offset, smooth_spectrum, chi_sq_factor):

    chi_sq = 0  # setting chi_sq back to 0

    # here we are calculating the stepsize of the spectrum and where the centre
    # of the generated gaussian will lie
    centre_of_gaussian = np.mean(rest_wavelengths) + offset
    stepsize_of_spectrum = (max(smooth_spectrum[:, 0]) - min(smooth_spectrum[:, 0])) / (len(smooth_spectrum) - 1)
    # this creates a list with exact range and no. points as in the spectrum
    # and extends it so that the gaussian can have wavelengths outside the
    # range of the spectrum
    x = np.arange(min(smooth_spectrum[:, 0]), max(smooth_spectrum[:, 0]) * 2.0, stepsize_of_spectrum,)

    # this is the lower and upper limits for the gaussian fit that we are going
    # to calculate the chi-squared fit across. We calculate the stepsize and
    # figure out how many correspond to soem amount of sigma and set that as
    # lower and upper ranges relative to centre
    lower_point_of_gaussian = int((centre_of_gaussian - chi_sq_factor * sigma - min(smooth_spectrum[:, 0])) / (stepsize_of_spectrum))
    upper_point_of_gaussian = int((centre_of_gaussian + chi_sq_factor * sigma - min(smooth_spectrum[:, 0])) / (stepsize_of_spectrum))

    while lower_point_of_gaussian <= 0:
        lower_point_of_gaussian = int(upper_point_of_gaussian + 10)

    while upper_point_of_gaussian >= len(smooth_spectrum[:, 0]):
        upper_point_of_gaussian = int(upper_point_of_gaussian - 10)

    # here we want to set the range over which we want to compare the formed
    # gaussian and the observed spectrum
    for j in range(len(smooth_spectrum[lower_point_of_gaussian:upper_point_of_gaussian, 0])):
        # the offset applied here shifts the index of the gaussian
        # corresponding to the size of the offset calculated by the program
        # this brings the spectrum and gaussian into the same wavelength range
        # DISCLAIMER - we don't *technically* calculate the chi-squared value
        # for our gaussian fit as we don't consider the errors in the spectrum
        # we do the observed-expected bit and can calculate a sort of relative
        # chi-squared value which we use to determine the quality of the fit
        # These values are not the actual chi-squared values for the fit
        chi_sq += ((smooth_spectrum[lower_point_of_gaussian + j, 1] - sum_gaussian[lower_point_of_gaussian + j - int(offset / stepsize_of_spectrum)]) / 1) ** 2

    return chi_sq


def generate_sum_gaussian_fn(
    smooth_spectrum, rest_wavelengths, height, relative_strengths, sigma, continuum_value,
):
    # this creates a list with exact range and no. points as in the
    # spectrum and extends it so that the gaussian can have wavelengths
    # outside the range of the spectrum
    stepsize_of_spectrum = (max(smooth_spectrum[:, 0]) - min(smooth_spectrum[:, 0])) / (len(smooth_spectrum) - 1)
    x = np.arange(min(smooth_spectrum[:, 0]), max(smooth_spectrum[:, 0]) * 2.0, stepsize_of_spectrum,)

    # these lists and arrays will store the parameters of the gaussians
    # produced from the current values
    gaussian_height = []
    gaussian = np.zeros((len(x), len(rest_wavelengths)))
    sum_gaussian = np.zeros((len(x), 1))

    # this for loop generates all of the gaussians we are working with
    # can be any amount depending on input values
    for l in range(len(rest_wavelengths)):
        #  this calculates the height of the gaussian but scales it
        # depending on the strength of the line; i.e. stronger lines will
        # dominate and therefore aren't scaled down as much as weaker
        # lines
        gaussian_height.append(height / max(relative_strengths) * (relative_strengths[l]))
        # calculates the gaussian based on previous height calculation
        # need to add continuum value for cases where spectrum is not at 0
        gaussian[:, l] = gaussian_height[l] * np.exp((-1 * (x - rest_wavelengths[l]) ** 2) / (2 * sigma ** 2))
    # this then sums up all the gaussians (if there is more than 1)
    sum_gaussian = np.sum(gaussian, axis=1) + continuum_value

    return sum_gaussian


def plot_sigma_points_fn(sum_gaussian, scaling_factor):
    max_of_gaussian = max(sum_gaussian)
    one_sigma = max_of_gaussian - 1 * np.std(sum_gaussian)
    two_sigma = max_of_gaussian - 2 * np.std(sum_gaussian)
    three_sigma = max_of_gaussian - 3 * np.std(sum_gaussian)
    four_sigma = max_of_gaussian - 4 * np.std(sum_gaussian)
    five_sigma = max_of_gaussian - 5 * np.std(sum_gaussian)

    max_of_gaussian_index = np.where(sum_gaussian == max_of_gaussian)[0]

    blue_sigma_values = np.zeros((5,2))
    for aa in range(len(sum_gaussian)):
        if sum_gaussian[aa] < one_sigma and aa < max_of_gaussian_index:
            blue_sigma_values[0,0] = x[aa]
            blue_sigma_values[0,1] = sum_gaussian[aa]
        if sum_gaussian[aa] < two_sigma and aa < max_of_gaussian_index:
            blue_sigma_values[1,0] = x[aa]
            blue_sigma_values[1,1] = sum_gaussian[aa]
        if sum_gaussian[aa] < three_sigma and aa < max_of_gaussian_index:
            blue_sigma_values[2,0] = x[aa]
            blue_sigma_values[2,1] = sum_gaussian[aa]
        if sum_gaussian[aa] < four_sigma and aa < max_of_gaussian_index:
            blue_sigma_values[3,0] = x[aa]
            blue_sigma_values[3,1] = sum_gaussian[aa]
        if sum_gaussian[aa] < five_sigma and aa < max_of_gaussian_index:
            blue_sigma_values[4,0] = x[aa]
            blue_sigma_values[4,1] = sum_gaussian[aa]

    red_sigma_values = np.zeros((5,2))
    for bb in range(len(sum_gaussian)):
        if sum_gaussian[bb] > one_sigma and bb > max_of_gaussian_index:
            red_sigma_values[0,0] = x[bb]
            red_sigma_values[0,1] = sum_gaussian[bb]
        if sum_gaussian[bb] > two_sigma and bb > max_of_gaussian_index:
            red_sigma_values[1,0] = x[bb]
            red_sigma_values[1,1] = sum_gaussian[bb]
        if sum_gaussian[bb] > three_sigma and bb > max_of_gaussian_index:
            red_sigma_values[2,0] = x[bb]
            red_sigma_values[2,1] = sum_gaussian[bb]
        if sum_gaussian[bb] > four_sigma and bb > max_of_gaussian_index:
            red_sigma_values[3,0] = x[bb]
            red_sigma_values[3,1] = sum_gaussian[bb]
        if sum_gaussian[bb] > five_sigma and bb > max_of_gaussian_index:
            red_sigma_values[4,0] = x[bb]
            red_sigma_values[4,1] = sum_gaussian[bb]

    color_counter = 0
    colors = plt.cm.seismic(np.linspace(0,1,10))

    sigma_values = np.concatenate([blue_sigma_values[::-1], red_sigma_values])
    for aa in range(len(sigma_values)):
        plt.scatter(sigma_values[aa,0], sigma_values[aa,1] / scaling_factor, marker='x', s=40, alpha = 1.0, zorder=10000, color = colors[aa])
        # TODO: add in functionality to label the sigma points
        # plt.text(sigma_values[aa,0], sigma_values[aa,1] / scaling_factor, r"$1\,\sigma$")

    return



############################## Start of main body of code #####################

# loading in spectrum for analysis
spectrum = np.loadtxt(spectrum_filename)
spectrum[:, 1] = spectrum[:, 1] * (4 * math.pi * (distance_to_transient ** 2))

# determine whether the code is smoothing or rebinng code and set smooth factor
if use_smoothed_spectrum == 1:
    smooth_number = smoothing_number
elif use_smoothed_spectrum == 2:
    smooth_number = rebinning_number
elif use_smoothed_spectrum is False:
    smooth_number = 0
else:
    print(
        "Error! Not sure how you want the spectrum smoothed/rebinned.", "Please enter valid entry for use_smoothed_spectrum",
    )

# run the functions to smooth/rebin spectrum
smooth_spectrum = smooth_spectrum_fn(spectrum, use_smoothed_spectrum, smooth_number)

# optionally run the function to visually check on spectral smoothing/rebinning
if see_test_smooth is True:
    plot_demo_smoothed_spectrum_fn(spectrum, smooth_spectrum, scaling_factor)

height_old = height
sigma_old = sigma
offset_old = offset

# initial chi-squared value - set high so first step is always better
chi_sq = 10e1000
chi_sq_old = chi_sq

# defining the counters
height_counter = 0
sigma_counter = 0
offset_counter = 0

# these are lists that will store the best fit values the code obtains right
# before it randomly jumps away from the current best fit value
offset_best_fit_values = []
height_best_fit_values = []
sigma_best_fit_values = []

# these are the lists that store all the best values obtained for the gaussians
# different to above as they only store the final best-fits for analysis
# whereas these lists contain all those values and also any best-fit values
# it obtains as it converges towards those values
offset_values = []
height_values = []
sigma_values = []

# these store every guess made by the code for each of the parameters
# should be scattered around the best-fit values
offset_guesses = []
offset_chi_squared = []
height_guesses = []
height_chi_squared = []
sigma_guesses = []
sigma_chi_squared = []

# start of the huge for loop to iterate over the best-fit values

# this is the massive for loop that runs a convergence fitting thing over
# and over again - user chooses how many iterations
for iteration_number in range(max_number_of_iterations):
    print(f"\nWe are on iteration number {iteration_number + 1} out of {max_number_of_iterations}\n")

    # redefining the counters to be 0 here for the next run of the while statement
    height_counter = 0
    sigma_counter = 0
    offset_counter = 0

    # this while loop contains all the stuff on fitting the height
    while height_counter < max_iteration_counter:
        height = height * random.uniform(lower_range, upper_range)
        height_guesses.append(height)

        sum_gaussian = generate_sum_gaussian_fn(smooth_spectrum, rest_wavelengths, height, relative_strengths, sigma, continuum_value)

        # launch the function to calculate quality of agreement
        chi_sq = calculate_chi_sq_fn(rest_wavelengths, offset, smooth_spectrum, chi_sq_factor)

        # if the new gaussian is a better fit than the previous best-fit then
        # the best-fit parameters are updated and stored for future runs
        # otherwise we reset our values for values and continue
        if chi_sq < chi_sq_old * 0.999:
            height_old = height
            chi_sq_old = chi_sq
            height_values.append(height)
            height_chi_squared.append(chi_sq)
            print(f"New height estimate:\t{height}")
        else:
            # if we don't get a new best-fit there is a small chance that the
            # code diverges anyway - this allows the code multiple
            # opportunities to converge down to a minimum and hopefully we
            # always converge to the same value - however this does allow us to
            # get out of some local minima that the code potentially gets
            # trapped in that doesn't represent the absolute best fit
            mc_walk_value = random.random()
            if mc_walk_value < mc_walk_value_upper_limit:
                print(f"I have randomly jumped away from our best fit position (step {height_counter}/{max_iteration_counter})")
                # saving best-fit parameters
                offset_best_fit_values.append(offset_old)
                height_best_fit_values.append(height_old)
                sigma_best_fit_values.append(sigma_old)

                height = height * random.uniform(mc_lower_range, mc_upper_range)
                chi_sq = 10e1000
                chi_sq_old = 10e1000
                # resetting counter to 0 as we want it to have enough steps to
                # converge to a best-fit value again
                height_counter = 0

            # otherwise we reset our values for height and continue
            height = height_old
            height_counter += 1

    # this while loop contains all the stuff on fitting the width
    while sigma_counter < max_iteration_counter:

        random_number = random.uniform(lower_range, upper_range)
        sigma = sigma * random_number
        sigma_guesses.append(sigma)

        sum_gaussian = generate_sum_gaussian_fn(smooth_spectrum, rest_wavelengths, height, relative_strengths, sigma, continuum_value)

        chi_sq = calculate_chi_sq_fn(rest_wavelengths, offset, smooth_spectrum, chi_sq_factor)

        # if the new gaussian is a better fit than the previous best-fit then
        # the best-fit parameters are updated and stored for future runs
        # otherwise we reset our values for values and continue
        if chi_sq < chi_sq_old * 0.999:
            sigma_old = sigma
            chi_sq_old = chi_sq
            sigma_values.append(sigma)
            sigma_chi_squared.append(chi_sq)
            print(f"New sigma estimate:\t{sigma}")
        else:
            # if we don't get a new best-fit there is a small chance that the
            # code diverges anyway - this allows the code multiple
            # opportunities to converge down to a minimum and hopefully we
            # always converge to the same value - however this does allow us to
            # get out of some local minima that the code potentially gets
            # trapped in that doesn't represent the absolute best fit
            mc_walk_value = random.random()
            if mc_walk_value < mc_walk_value_upper_limit:
                print(f"I have randomly jumped away from our best fit position (step {sigma_counter}/{max_iteration_counter})")

                offset_best_fit_values.append(offset_old)
                height_best_fit_values.append(height_old)
                sigma_best_fit_values.append(sigma_old)

                sigma = sigma * random.uniform(mc_lower_range, mc_upper_range)
                chi_sq = 10e1000
                chi_sq_old = 10e1000
                sigma_counter = 0

            # otherwise we reset our values for values and continue
            sigma = sigma_old
            sigma_counter += 1

    # this while statement contains all the convergence stuff on the offset value
    while offset_counter < max_iteration_counter:

        # generates random number and scales the offset value based off that
        random_number = random.uniform(lower_range, upper_range)
        offset = offset * random_number
        offset_guesses.append(offset)

        sum_gaussian = generate_sum_gaussian_fn(smooth_spectrum, rest_wavelengths, height, relative_strengths, sigma, continuum_value)

        chi_sq = calculate_chi_sq_fn(rest_wavelengths, offset, smooth_spectrum, chi_sq_factor)

        # this allows us to save the offset and chi-squared values as our new
        # best-fit values if they are better than the previous values
        # if they give a worse fit then they are discarded and we
        # revert back to the previous values
        if chi_sq <= chi_sq_old * 0.99:
            offset_old = offset
            chi_sq_old = chi_sq
            offset_values.append(offset)
            offset_chi_squared.append(chi_sq)
            print(f"New offset estimate:\t{offset}")
        else:
            # if we don't get a new best-fit there is a small chance that the
            # code diverges anyway - this allows the code multiple
            # opportunities to converge down to a minimum and hopefully we
            # always converge to the same value - however this does allow us to
            # get out of some local minima that the code potentially gets
            # trapped in that doesn't represent the absolute best fit
            mc_walk_value = random.random()
            if mc_walk_value < mc_walk_value_upper_limit:
                print(f"I have randomly jumped away from our best fit position (step {offset_counter}/{max_iteration_counter})")

                offset_best_fit_values.append(offset_old)
                height_best_fit_values.append(height_old)
                sigma_best_fit_values.append(sigma_old)

                offset = offset * random.uniform(mc_lower_range, mc_upper_range)
                chi_sq = 10e1000
                chi_sq_old = 10e1000
                offset_counter = 0

            # otherwise we reset our values for values and continue
            offset = offset_old
            offset_counter += 1


print(f"\nWe have MEAN estimates for height, sigma and offset:\t{np.mean(height_best_fit_values):.3E} erg/s,\t{np.mean(sigma_best_fit_values):.5g} Angstroms,\t{np.mean(offset_best_fit_values):.5g} Angstroms")
print(f"We have MEDIAN estimates for height, sigma and offset:\t{np.median(height_best_fit_values):.3E} erg/s,\t{np.median(sigma_best_fit_values):.5g} Angstroms,\t{np.median(offset_best_fit_values):.5g} Angstroms")
print(f"We have STANDARD DEVIATION estimates for height, sigma and offset:\t{np.std(height_best_fit_values):.3E} erg/s,\t{np.std(sigma_best_fit_values):.5g} Angstroms,\t{np.std(offset_best_fit_values):.5g} Angstroms")

mean_velocity = 0
median_velocity = 0
# printing our velocity estimates for the 3 ca triplet gaussians
for l in range(len(rest_wavelengths)):
    mean_velocity += np.mean(sigma) * 2.355 * 3e5 / rest_wavelengths[l] / len(rest_wavelengths)
    median_velocity += np.median(sigma) * 2.355 * 3e5 / rest_wavelengths[l] / len(rest_wavelengths)
print(f"\nMEAN velocity estimate:\t{mean_velocity:.2f} km/s")
print(f"MEDIAN velocity estimate:\t{median_velocity:.2f} km/s")

# now we can make the best fit plot with all our lovely gaussians on it
fig, ax = plt.subplots()

if use_smoothed_spectrum == 2:
    plt.plot(
        spectrum[:, 0], spectrum[:, 1] / scaling_factor, color="gray", label="Spectrum", alpha=1.0, linewidth=1.0,
    )
    plt.plot(
        smooth_spectrum[:, 0], smooth_spectrum[:, 1] / scaling_factor, color="black", label="Rebinned Spectrum", alpha=1.0, linewidth=1.0,
    )
elif use_smoothed_spectrum == 1:
    plt.plot(
        spectrum[:, 0], spectrum[:, 1] / scaling_factor, color="gray", label="Spectrum", alpha=1.0, linewidth=1.0,
    )
    plt.plot(
        smooth_spectrum[:, 0], smooth_spectrum[:, 1] / scaling_factor, color="black", label="Smoothed Spectrum", alpha=1.0, linewidth=1.0,
    )
elif use_smoothed_spectrum is False:
    plt.plot(
        spectrum[:, 0], spectrum[:, 1] / scaling_factor, color="black", label="Spectrum", alpha=1.0, linewidth=1.0,
    )
else:
    print("Error! Plotting issue")

plt.axhline(continuum_value / scaling_factor, color="orange", linestyle="--")

if plot_rest_waves is True:
    for aa in range(len(rest_wavelengths)):
        plt.axvline(rest_wavelengths[aa] - offset_constant, color='black', linestyle='--', linewidth=2.5 * relative_strengths[aa] / max(relative_strengths))

# this plots the gaussians that correspond to all the best-fit values we saved
for b in range(len(offset_best_fit_values)):
    # we redefine this every loop as we offset it later to calculate the
    # chi-squared values so this resets it
    # this defines a range of wavelengths with the same stepsize as the input
    # spectrum - this is extended beyond the range of the spectrum to
    # accommodate edge cases (currently only works for upper limit cases)
    stepsize_of_spectrum = (max(smooth_spectrum[:, 0]) - min(smooth_spectrum[:, 0])) / (len(smooth_spectrum) - 1)
    # this creates a list with exact range and no. points as in the spectrum
    # and extends it so that the gaussian can have wavelengths outside the
    # range of the spectrum
    x = np.arange(min(smooth_spectrum[:, 0]), max(smooth_spectrum[:, 0]) * 2.0, stepsize_of_spectrum,)

    # these lists and arrays will store the parameters of the gaussians
    # produced from the current values
    gaussian_height = []
    gaussian = np.zeros((len(x), len(rest_wavelengths)))
    sum_gaussian = np.zeros(len(x))

    # this loop sums up the individual chi-squared values for the gaussian
    # fit to the spectrum within the wavelength range we are concerned with
    for l in range(len(rest_wavelengths)):
        gaussian_height.append(height_best_fit_values[b] / max(relative_strengths) * (relative_strengths[l]))
        gaussian[:, l] = gaussian_height[l] * np.exp((-1 * (x - rest_wavelengths[l]) ** 2) / (2 * sigma_best_fit_values[b] ** 2))

    # this then sums up all the gaussians (if there is more than 1)
    sum_gaussian = np.sum(gaussian, axis=1) + continuum_value

    # here we are calculating the stepsize of the spectrum and where the centre
    # of the generated gaussian will lie
    stepsize_of_spectrum = (max(smooth_spectrum[:, 0]) - min(smooth_spectrum[:, 0])) / len(smooth_spectrum[:, 0])
    centre_of_gaussian = np.mean(rest_wavelengths) + offset

    # this is the lower and upper limits for the gaussian fit that we are going
    # to calculate the chi-squared fit across - we calculate the stepsize and
    # figure out how many correspond to some amount of sigma and set that as
    # lower andd upper ranges relative to centre
    lower_point_of_gaussian = int((centre_of_gaussian - 2 * sigma - min(smooth_spectrum[:, 0])) / (stepsize_of_spectrum))
    upper_point_of_gaussian = int((centre_of_gaussian + 2 * sigma - min(smooth_spectrum[:, 0])) / (stepsize_of_spectrum))

    while lower_point_of_gaussian <= 0:
        lower_point_of_gaussian = int(upper_point_of_gaussian + 10)

    while upper_point_of_gaussian >= len(smooth_spectrum):
        upper_point_of_gaussian = int(upper_point_of_gaussian - 10)

    # have to shift the x values after the gaussians are generated as they
    # depend on x
    x = x + offset_best_fit_values[b]
    # plots faint gaussians on the plot to illustrate where all our best-fits
    # appear - should lie mostly around the best-fit we generate
    plt.plot(
        x[lower_point_of_gaussian:], sum_gaussian[lower_point_of_gaussian:] / scaling_factor, linewidth=0.5, color="blue", alpha=0.3, label="",
    )

    # this plots the gaussian that we obtain from the median from all our
    # best-fit values for the gaussian models
    # we redefine this every loop as we offset it later to calculate the
    # chi-squared values so this resets it
    # this defines a range of wavelengths with the same stepsize as the input
    # spectrum - this is extended beyond the range of the spectrum to
    # accommodate edge cases (currently only works for upper limit cases)
    stepsize_of_spectrum = (max(smooth_spectrum[:, 0]) - min(smooth_spectrum[:, 0])) / (len(smooth_spectrum) - 1)
    # this creates a list with exact range and no. points as in the spectrum
    # and extends it so that the gaussian can have wavelengths outside the
    # range of the spectrum
    x = np.arange(
        min(smooth_spectrum[:, 0]), max(smooth_spectrum[:, 0]) * 2.0, stepsize_of_spectrum,
    )

# these lists and arrays will store the parameters of the gaussians produced
# from the current values
gaussian_height = []
gaussian = np.zeros((len(x), len(rest_wavelengths)))
sum_gaussian = np.zeros(len(x))


# this loop sums up the individual chi-squared values for the gaussian fit
# to the spectrum within the wavelength range we are concerned with
for l in range(len(rest_wavelengths)):
    gaussian_height.append(np.median(height_best_fit_values) / max(relative_strengths) * (relative_strengths[l]))
    gaussian[:, l] = gaussian_height[l] * np.exp((-1 * (x - rest_wavelengths[l]) ** 2) / (2 * np.median(sigma_best_fit_values) ** 2))

sum_gaussian = np.sum(gaussian, axis=1) + continuum_value

# here we are calculating the stepsize of the spectrum and where the centre
# of the generated gaussian will lie
stepsize_of_spectrum = (max(smooth_spectrum[:, 0]) - min(smooth_spectrum[:, 0])) / len(smooth_spectrum[:, 0])
centre_of_gaussian = np.mean(rest_wavelengths) + offset

# this is the lower and upper limits for the gaussian fit that we are going to
# calculate the chi-squared fit across - we calculate the stepsize and figure
# out how many correspond to soem amount of sigma and set that as lower and
# upper ranges relative to centre
lower_point_of_gaussian = int((centre_of_gaussian - 2 * sigma - min(smooth_spectrum[:, 0])) / (stepsize_of_spectrum))
upper_point_of_gaussian = int((centre_of_gaussian + 2 * sigma - min(smooth_spectrum[:, 0])) / (stepsize_of_spectrum))

# have to shift the x values after the gaussians are generated as they depend on x
x = x + np.median(offset_best_fit_values)
plt.plot(
    x[lower_point_of_gaussian:], sum_gaussian[lower_point_of_gaussian:] / scaling_factor, linewidth=2.0, color="green", linestyle=":", alpha=0.7, label="Median Best-Fit Gaussian",
)

stepsize_of_spectrum = (max(smooth_spectrum[:, 0]) - min(smooth_spectrum[:, 0])) / (len(smooth_spectrum) - 1)
x = np.arange(
    min(smooth_spectrum[:, 0]), max(smooth_spectrum[:, 0]) * 2.0, stepsize_of_spectrum,
)

gaussian_height = []
gaussian = np.zeros((len(x), len(rest_wavelengths)))
sum_gaussian = np.zeros(len(x))

# this loop sums up the individual chi-squared values for the gaussian fit to the spectrum within the wavelength range we are concerned with
for l in range(len(rest_wavelengths)):
    gaussian_height.append(np.mean(height_best_fit_values) / max(relative_strengths) * (relative_strengths[l]))
    gaussian[:, l] = gaussian_height[l] * np.exp((-1 * (x - rest_wavelengths[l]) ** 2) / (2 * np.mean(sigma_best_fit_values) ** 2))

sum_gaussian = np.sum(gaussian, axis=1) + continuum_value  # this then sums up all the gaussians (if there is more than 1)

# here we are calculating the stepsize of the spectrum and where the centre of the generated gaussian will lie
stepsize_of_spectrum = (max(smooth_spectrum[:, 0]) - min(smooth_spectrum[:, 0])) / len(smooth_spectrum[:, 0])
centre_of_gaussian = np.mean(rest_wavelengths) + offset

# this is the lower and upper limits for the gaussian fit that we are going to calculate the chi-squared fit across
# we calculate the stepsize and figure out how many correspond to soem amount of sigma and set that as lower andd upper ranges relative to centre
lower_point_of_gaussian = int((centre_of_gaussian - 2 * sigma - min(smooth_spectrum[:, 0])) / (stepsize_of_spectrum))
upper_point_of_gaussian = int((centre_of_gaussian + 2 * sigma - min(smooth_spectrum[:, 0])) / (stepsize_of_spectrum))


# have to shift the x values after the gaussians are generated as they depend on x
x = x + np.mean(offset_best_fit_values)
# plots faint gaussians on the plot to illustrate where all our best-fits appear - should lie mostly around the best-fit we generate
plt.plot(
    x[lower_point_of_gaussian - 30 :], sum_gaussian[lower_point_of_gaussian - 30 :] / scaling_factor, linewidth=2.0, color="red", linestyle=":", alpha=0.7, label="Mean Best-Fit Gaussian",
)

# this saves the gaussian fit
np.savetxt(
    f"{_mkdir('gaussian_fitter_outputs')}/best-fit_gaussian.dat", np.c_[x[lower_point_of_gaussian - 30 :], sum_gaussian[lower_point_of_gaussian - 30 :],],
)

if plot_sigma_points is True:
    plot_sigma_points_fn(sum_gaussian, scaling_factor)

plt.legend(loc="best", edgecolor="white", facecolor="white", framealpha=0.7, fontsize=12)
plt.minorticks_on()
plt.xlabel("Rest Wavelength ($\mathrm{\AA}$)")
plt.ylabel("Luminosity ($10^{%d}$ erg $\mathrm{s^{-1}}$ $\mathrm{\AA^{-1}}$)" % math.log10(scaling_factor))

# adjusting x and y axis limits
if centre_of_gaussian - 7*np.mean(sigma_best_fit_values) <= min(smooth_spectrum[:,0]):
    # sets lower limit based on edge of spectrum
    lower_xlim = min(smooth_spectrum[:,0] - 200)
elif centre_of_gaussian - 7*np.mean(sigma_best_fit_values) > min(smooth_spectrum[:,0]):
    # sets lower limit based on gaussian range of interest
    lower_xlim = centre_of_gaussian - 7*np.mean(sigma_best_fit_values)
else:
    print("Something wrong with determing x-range of figure")
if centre_of_gaussian + 7*np.mean(sigma_best_fit_values) >= max(smooth_spectrum[:,0]):
    # sets lower limit based on edge of spectrum
    upper_xlim = max(smooth_spectrum[:,0] + 200)
elif centre_of_gaussian + 7*np.mean(sigma_best_fit_values) < max(smooth_spectrum[:,0]):
    # sets lower limit based on gaussian range of interest
    upper_xlim = centre_of_gaussian + 7*np.mean(sigma_best_fit_values)
else:
    print("Something wrong with determing x-range of figure")
plt.xlim([lower_xlim, upper_xlim])
# this will set the upper and lower limits relative to the height of the mean best-fit gaussian
plt.ylim([(min(sum_gaussian) - 0.5*max(sum_gaussian))/scaling_factor, (max(sum_gaussian)*1.5)/scaling_factor])

ax.tick_params(
    axis="y", direction="in", top=True, bottom=True, left=True, right=True, which="both",
)
ax.tick_params(
    axis="x", direction="in", top=True, bottom=True, left=True, right=True, which="both",
)

plt.savefig(
    f"{_mkdir('gaussian_fitter_outputs')}/" + object_name + "_gaussian_fits.png", dpi=500, bbox_inches="tight",
)
plt.close()

