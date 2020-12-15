# SimpleSpectralAnalysis
James Gillanders (12/2020)
Collection of tools either written or modified by me to model spectra

Here I have stored some tools that I have written or modified to
perform some spectral modelling.

BFG.py
------

This program can be used to fit Gaussian emission features.
It starts with best guesses for the fitting parameters
(peak height, sigma, and offset). From this, a Gaussian is
generated and compared with the observed feature. A 'goodness'
of fit test is performed (analogous to chi-squared fitting).
Then each of the parameters are randomly varied, and new Gaussians
are generated based on these. If the new model is a better fit, then
its parameters are stored.

The code possesses the ability to fit multiple composite Gaussians to
a simgle observed feature (think CaII NIR triplet). All it needs is the
individual rest wavelengths and values for their relative strengths.
This is as of yet cannot be treated as a free parameter.

To prevent the code from potentially converging to a solution it thinks
is good, but perhaps is not the best, there is a non-zero chance that at
any given step, the code will randomly vary the parameter under
investigation, basically 'kicking' it away from the current best-fit.

All best-fits, pre-'kick' are stored and then these are averaged and
used to determine the mean and median best fits.

TODO:
  Upload the fitter for PCygni profiles.
  Modify BFG.py to treat relative strengths as a free parameter
  
