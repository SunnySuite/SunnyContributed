
using Sunny, GLMakie, LinearAlgebra, Optim

# Define the chemical cell for YVO₃ following the SpinW tutorial 14.

units = Units(:meV, :angstrom)
a = 5.2821 / sqrt(2)
b = 5.6144 / sqrt(2)
c = 7.5283
latvecs = lattice_vectors(a, b, c, 90, 90, 90)
positions = [[0, 0, 0], [0, 0, 1/2]]
types = ["V", "V"]
cryst = Crystal(latvecs, positions, 1; types)

# Create an empty system for YVO₃. We will be modifying it throughout the
# fitting process.

sys = System(cryst, [1 => Moment(s=1/2, g=2), 2 => Moment(s=1/2, g=2)], :dipole_large_s; dims=(2, 2, 1))

# Write a function that fills parameters (Jab, Jc, δ, K1, K2, d) extracted from
# the vector x.

function fill_interactions!(x)
    (Jab, Jc, δ, K1, K2, d) = x
    Jc1 = [-Jc*(1+δ)+K2 0 -d; 0 -Jc*(1+δ) 0; +d 0 -Jc*(1+δ)]
    Jc2 = [-Jc*(1-δ)+K2 0 +d; 0 -Jc*(1-δ) 0; -d 0 -Jc*(1-δ)]
    set_exchange!(sys, Jab, Bond(1, 1, [1, 0, 0]))
    set_exchange!(sys, Jab, Bond(2, 2, [1, 0, 0]))
    set_exchange!(sys, Jc1, Bond(1, 2, [0, 0, 0]))
    set_exchange!(sys, Jc2, Bond(2, 1, [0, 0, 1]))
    set_exchange!(sys, Jab, Bond(1, 1, [0, 1, 0]))
    set_exchange!(sys, Jab, Bond(2, 2, [0, 1, 0]))
    set_onsite_coupling!(sys, S -> -K1*S[1]^2, 1)
    set_onsite_coupling!(sys, S -> -K1*S[1]^2, 2)
end

# Write a function that calculates and returns intensities along a path in
# ``𝐪``-space.

qs = [[0.75, 0.75, 0], [0.5, 0.5, 0], [0.5, 0.5, 1]]
path = q_space_path(cryst, qs, 200)
energies = range(0, 20, 200)
measure = ssf_perp(sys)
kernel = lorentzian(fwhm=0.5)

function calculate_intensities!(x)
    fill_interactions!(x)
    randomize_spins!(sys)
    minimize_energy!(sys)
    swt = SpinWaveTheory(sys; measure)
    return intensities(swt, path; energies, kernel)
end

# Define reference parameters based on previous work. These are used to
# calculate a reference result, `res_ref`, which will be used as the fitting
# target.

x_ref = [2.6, 3.1, 0.35, 0.90, 0.97, 1.15]
res_ref = calculate_intensities!(x_ref)
plot_intensities(res_ref; units)

# Now suppose we don't know `x_ref`, but we have some guess for the parameters
# `x`. Construct a function to evaluate "goodness of fit" χ²  according to the
# L2 distance between intensity data. For this synthetic problem, if we could
# find parameters `x` that match `x_ref`, then χ² = 0.

function evaluate_chi2(x)
    try
        res = calculate_intensities!(x)
        return norm(res.data - res_ref.data)
    catch e
        return Inf
    end
end


# Select an initial guess parameter guess that is not too far from `x_ref`.

# x_ref = [2.6, 3.1, 0.35, 0.90, 0.97, 1.15]
x_guess_1 = [2.0, 4.0, 0.0, 0.5, 1.5, 0.5]
evaluate_chi2(x_guess_1)

# Use the NelderMead() algorithm to find a local minimum. The resulting χ² is
# much smaller.

options = Optim.Options(; iterations=1000, show_trace=true, show_every=50)
fit_1 = Optim.optimize(evaluate_chi2, x_guess_1, NelderMead(), options)
evaluate_chi2(fit_1.minimizer)

# The fitted parameters are in reasonable agreement with `x_ref`

fit_1.minimizer

# Use the ParticleSwarm() algorithm to overcome barriers in the fitting
# landscape, χ². Begin with a random perturbation of the previous local minimum.
# Iterate this several times by hand until the local minimum is escaped.

x_guess_2 = fit_1.minimizer + 0.25 * randn(6)
options = Optim.Options(; iterations=200, show_trace=true, show_every=50)
fit_2 = Optim.optimize(evaluate_chi2, x_guess_2, Optim.ParticleSwarm(), options)

# Use the NelderMead() algorithm to find a local minimum starting from the
# previous fit.

options = Optim.Options(; iterations=500, show_trace=true, show_every=50)
fit_3 = Optim.optimize(evaluate_chi2, fit_2.minimizer, Optim.NelderMead(), options)

# Check agreement with targets: [2.6, 3.1, 0.35, 0.90, 0.97, 1.15]

fit_3.minimizer

# Plot a panel of fits

fig = Figure()

res = calculate_intensities!(x_ref)
plot_intensities!(fig[1, 1], res; units)

res = calculate_intensities!(x_guess_1)
plot_intensities!(fig[2, 1], res; units)

res = calculate_intensities!(fit_1.minimizer)
plot_intensities!(fig[1, 2], res; units)

res = calculate_intensities!(fit_3.minimizer)
plot_intensities!(fig[2, 2], res; units)
