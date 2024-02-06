# This script computes the susceptibility χ using Sunny's LSWT tools
# Run one of the example_* functions at the bottom to see it in action!
# 
# Requires Sunny.jl PR#174, so this might not work on main Sunny as of this writing (Oct 9)


# To use the complex plots, you need to install this unregistered package:
#
#   Pkg.add(PackageSpec(url="https://github.com/luchr/ComplexPortraits.jl", rev="master"))
#
#using ComplexPortraits
using Sunny, GLMakie, StaticArrays, LinearAlgebra

# Steal this internal type
import Sunny: BandStructure

function plot_susceptibility(band_structure;kwargs...)
    f = Figure()
    ax = Axis(f[1,1]; xlabel = "Energy (meV)", ylabel = "χ(iω)")
    plot_susceptibility!(ax,band_structure;kwargs...)
    f
end

function plot_susceptibility!(ax,band_structure::BandStructure{NBands,ComplexF64};part = abs, energies = nothing,decay = 0.1) where NBands
    energies = if isnothing(energies)
        peak_bounds = extrema(band_structure.dispersion)

        autocutoff = 1e-1
        max_peak = maximum(abs.(band_structure.intensity))
        # A/x = cutoff
        # x = A/cutoff
        x_cut = max_peak / autocutoff

        num_steps_per_HWHM = 16
        range(peak_bounds[1] - x_cut, peak_bounds[2] + x_cut; step = decay / num_steps_per_HWHM)
    else
        energies
    end
    χ = map(s -> susceptibility_spectral_function(band_structure,s),decay .+ im .* energies)
    lines!(ax,energies,part.(χ))
end


function correlator_spectral_function(band_structure::BandStructure{NBands,ComplexF64},s) where NBands
    # Theorem:
    # ========
    # If the fourier transform of <B(t) A> has a term
    #   2π * Aᵢ * δ(ω - ωᵢ)
    # then the laplace transform has a corresponding term:
    #   Aᵢ / (s - iωᵢ)
    val = 0. + 0im
    for i = 1:NBands
        val += band_structure.intensity[i] / (s - im * band_structure.dispersion[i])
    end
    return val
end

# Usually, χ(s) is related to the Laplace transform of the correlator <B(t) A>
# by symmetrizing over [ωC-†] (with no factor 1/2), which is the simultaneous:
# - replacing ω → -ω
# - complex conjugation
# - multiplication by (-1)
# - replacing A → A†
#
# But if we assume A = A†, then we just need [ωC-], which is what this function implements.
function susceptibility_spectral_function(band_structure::BandStructure{NBands,ComplexF64},s) where NBands
    val = 0. + 0im
    for i = 1:NBands
        val += + band_structure.intensity[i] / (s - im * band_structure.dispersion[i])
        val += - conj(band_structure.intensity[i] / (s + im * band_structure.dispersion[i]))
    end
    return val
end

function kramers_kronig_matrix(ωs)
  dω = ωs[2] - ωs[1]
  n = length(ωs)
  [dω ./ (i == j ? Inf : (ωs[i] - ω[j])) for i = 1:n, j = 1:n] ./ (im * π)
end

function intensities_spectral_function(swt::SpinWaveTheory, ks, ωvals, formula; susceptibility = true, decay = 0.1, part = identity)
    if !isnothing(formula.kernel)
        error("intensities_correlator_spectral_function: Can't compute spectral function if a broadening kernel is applied.\nTry intensity_formula(...; kernel = delta_function_kernel)")
    end

    if susceptibility && any(norm.(ks) .> 0)
        println("Warning, operating at q nonzero, but in this case Aq is not Aq† (it's A-q†) so result may be incorrect.")
    end

    # Get the type parameter from the BandStructure
    return_type = typeof(formula).parameters[1].parameters[2]
    if return_type != ComplexF64
        error("Can't compute spectral function when the intensity isn't complex-valued")
    end

    spectral_f = susceptibility ? susceptibility_spectral_function : correlator_spectral_function

    is = zeros(typeof(part(0. +1im)), size(ks)..., length(ωvals))
    for kidx in CartesianIndices(ks)
        band_structure = formula.calc_intensity(swt, SVector{3,Float64}(ks[kidx]))
        χ = map(s -> spectral_f(band_structure,s),decay .+ im .* ωvals)
        is[kidx,:] .= part.(χ)
    end
    return is
end


function four_panel_plot(ks,energies,data,name)
  f = Figure()
  ax_r = Axis(f[1,2],title = "Real part of $name")
  ax_i = Axis(f[1,3],title = "Imag part of $name")
  ax_abs = Axis(f[2,2],title = "|$name|")
  ax_arg = Axis(f[2,3],title = "arg $name")

  hm = heatmap!(ax_r,ks,energies,real.(data),colormap=:redgreensplit)
  cr = hm.calculated_colors[].colorrange[]
  sym_range = maximum(abs.(cr))
  hm.colorrange[] = [-sym_range,sym_range]
  Colorbar(f[1,1],hm)

  hm = heatmap!(ax_i,ks,energies,imag.(data))
  Colorbar(f[1,4],hm)

  hm = heatmap!(ax_abs,ks,energies,abs.(data))
  Colorbar(f[2,1],hm)

  csp = ComplexPortraits.cs_p()
  coloring = z -> csp(0.,z)
  hm = image!(ax_arg,ks,energies,coloring.(data))
  #Colorbar(f[2,4],hm)
  f
end

function example_simple()
  sys = System(Sunny.cubic_crystal(), (1,2,1), [SpinInfo(1;S=1/2,g=1)], :SUN, units = Units.theory)
  set_external_field!(sys,[0,0,0.5])
  set_exchange!(sys,-1.,Bond(1,1,[0,1,0]))
  randomize_spins!(sys)
  minimize_energy!(sys)
  minimize_energy!(sys)
  minimize_energy!(sys)

  swt = SpinWaveTheory(sys)
  ks = range(0,1,length=200)
  qs = [[0,k,0] for k in ks]
  formula = intensity_formula(swt, [(:Sx,:Sy)], kernel = delta_function_kernel) do k,ω,S
    S[1]
  end
  energies = range(-5,5,length = 200)

  data = intensities_spectral_function(swt, qs, energies, formula, susceptibility = true)
  four_panel_plot(ks,energies,data,"χxy")
end

function example_fei2()
  a = b = 4.05012  # Lattice constants for triangular lattice
  c = 6.75214      # Spacing in the z-direction
  latvecs = lattice_vectors(a, b, c, 90, 90, 120) # A 3x3 matrix of lattice vectors that
                                                  # define the conventional unit cell
  positions = [[0, 0, 0], [1/3, 2/3, 1/4], [2/3, 1/3, 3/4]]  # Positions of atoms in fractions
                                                             # of lattice vectors
  types = ["Fe", "I", "I"]
  FeI2 = Crystal(latvecs, positions; types)
  cryst = subcrystal(FeI2, "Fe")
  sys = System(cryst, (4,4,4), [SpinInfo(1, S=1, g=2)], :SUN, seed=2)
  J1pm   = -0.236
  J1pmpm = -0.161
  J1zpm  = -0.261
  J2pm   = 0.026
  J3pm   = 0.166
  J′0pm  = 0.037
  J′1pm  = 0.013
  J′2apm = 0.068

  J1zz   = -0.236
  J2zz   = 0.113
  J3zz   = 0.211
  J′0zz  = -0.036
  J′1zz  = 0.051
  J′2azz = 0.073

  J1xx = J1pm + J1pmpm
  J1yy = J1pm - J1pmpm
  J1yz = J1zpm

  set_exchange!(sys, [J1xx   0.0    0.0;
                      0.0    J1yy   J1yz;
                      0.0    J1yz   J1zz], Bond(1,1,[1,0,0]))
  set_exchange!(sys, [J2pm   0.0    0.0;
                      0.0    J2pm   0.0;
                      0.0    0.0    J2zz], Bond(1,1,[1,2,0]))
  set_exchange!(sys, [J3pm   0.0    0.0;
                      0.0    J3pm   0.0;
                      0.0    0.0    J3zz], Bond(1,1,[2,0,0]))
  set_exchange!(sys, [J′0pm  0.0    0.0;
                      0.0    J′0pm  0.0;
                      0.0    0.0    J′0zz], Bond(1,1,[0,0,1]))
  set_exchange!(sys, [J′1pm  0.0    0.0;
                      0.0    J′1pm  0.0;
                      0.0    0.0    J′1zz], Bond(1,1,[1,0,1]))
  set_exchange!(sys, [J′2apm 0.0    0.0;
                      0.0    J′2apm 0.0;
                      0.0    0.0    J′2azz], Bond(1,1,[1,2,1]))

  D = 2.165
  S = spin_operators(sys, 1)
  set_onsite_coupling!(sys, -D*S[3]^2, 1)

  randomize_spins!(sys)
  minimize_energy!(sys);

  sys_min = reshape_supercell(sys, [1 0 0; 0 1 -2; 0 1 2])
  randomize_spins!(sys_min)
  minimize_energy!(sys_min)

  swt = SpinWaveTheory(sys_min)

  q_points = [[0,0,0], [1,0,0], [0,1,0], [1/2,0,0], [0,1,0], [0,0,0]];
  density = 50
  path, xticks = reciprocal_space_path(cryst, q_points, density);

  formula = intensity_formula(swt, [(:Sx,:Sy)], kernel = delta_function_kernel, return_type = ComplexF64) do k,ω,S
    S[1]
  end
  energies = range(-8,8,length = 400)

  #data = intensities_spectral_function(swt, path, energies, formula, decay = 0.2, susceptibility = true)
  #display(four_panel_plot(1:length(path),energies,data,"χxy"))


  Bzs = range(0,20,length = 300)
  dat = zeros(ComplexF64,length(Bzs),length(energies))
  for (i,Bz) in enumerate(Bzs)
    set_external_field!(sys_min,[0,0,Bz])
    randomize_spins!(sys_min)
    minimize_energy!(sys_min)
    swt = SpinWaveTheory(sys_min)
    formula = intensity_formula(swt, [(:Sx,:Sx),(:Sx,:Sy)], kernel = delta_function_kernel, return_type = ComplexF64) do k,ω,S
      S[1]
    end
    dat[i,:] = intensities_spectral_function(swt, [[0.,0,0]], energies, formula, decay = 0.2, susceptibility = true)
  end
  display(four_panel_plot(Bzs,energies,dat,"χxx(B)"))
end
  
# Extra support instance
function Sunny.intensity_formula(f::Function,s,required_correlations; kwargs...)
    corr_ix = Sunny.lookup_correlations(s.observables,required_correlations)
    intensity_formula(f,s,corr_ix;kwargs...)
end

