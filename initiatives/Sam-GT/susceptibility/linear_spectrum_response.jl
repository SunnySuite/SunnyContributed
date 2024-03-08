using Sunny, GLMakie, LinearAlgebra, StaticArrays, FFTW
# add https://github.com/JuliaMath/Bessels.jl.git
# using Bessels

cryst = Crystal(I(3), [[0,0,0]], 1)
sys = System(cryst,(1,1,1),[SpinInfo(1,S=1,g=2)],:dipole)
Bz = -1
set_external_field!(sys,[0,0,Bz])

function go_to_theta(th,sys)
  set_dipole!(sys,[0,sin(th),cos(th)],(1,1,1,1))
end
go_to_theta(3pi/4,sys)
theta = acos(sys.dipoles[1][3])
println("θ = $theta")
omega0 = sys.gs[1][1] * sys.units.μB * Bz
println("ω0 = $omega0")

s_dot_obs(i) = Sunny.NonLocalObservableOperator(sys -> map(x -> x[i],sys.dipoles .× Sunny.energy_grad_dipoles(sys)))

dsc = dynamical_correlations(sys;Δt = 0.05, nω = 600, ωmax = 1.0, observables = [:Sx => [1. 0 0], :Sy => [0. 1 0], :Sz => [0. 0 1], :A => Sunny.energy_gradient_dipole_observable(1), :B => Sunny.energy_gradient_dipole_observable(2), :C => Sunny.energy_gradient_dipole_observable(3), :field => Sunny.NonLocalObservableOperator(sys -> map(x -> x[1],sys.extfield)), :one => Sunny.NonLocalObservableOperator(sys -> map(x -> 1.0,sys.extfield)), :Sxdot => s_dot_obs(1), :Sydot => s_dot_obs(2), :Szdot => s_dot_obs(3)])

beta = 0.1
langevin = Langevin(0.05;λ = 0.1, kT = 1/beta)
println("x = βω₀ = $(beta * omega0)")

viewed_thetas = Float64[]

glob_s = 0.008
dt = dsc.Δt * dsc.measperiod
filt = reshape(exp.(- glob_s .* range(0,step = dt,length = size(dsc.samplebuf,6))),1,1,1,1,1,size(dsc.samplebuf,6))

function put_samples!(dsc,sys; set_theta = nothing)
  langevin.λ = isnothing(set_theta) ? 0.1 : 0
  if !isnothing(set_theta)
    go_to_theta(set_theta,sys)
  end
  for k = 1:200
    for j = 1:1000
      step!(sys,langevin)
    end
    theta = acos(sys.dipoles[1][3])
    #println("θ = $theta")
    push!(viewed_thetas,theta)

    Sunny.new_sample!(dsc,sys,() -> nothing)
    #filt = reshape(exp.(- glob_s .* range(0,step = dt,length = size(dsc.samplebuf,6))),1,1,1,1,1,size(dsc.samplebuf,6))
    #dsc.samplebuf .*= filt
    Sunny.accum_sample!(dsc)
  end
  dsc
end



function show_correlation_spectrum(dsc,sys,i,j;set_theta = nothing)
  N = size(dsc.data,7)
  params = unit_resolution_binning_parameters(dsc;negative_energies = true)
  is_full = intensities_binned(dsc,params,intensity_formula(dsc,:full))[1][1,1,1,:]
  is = map(x -> x[i,j],is_full)

  # Normalize to unitary FFT
  is *= sqrt(N)

  ωs = axes_bincenters(params)[4]#2π * fftfreq(N,1/dt)
  dw = ωs[2] - ωs[1]
  I = sortperm(ωs)
  f = Figure()
  ax = Axis(f[1,1])
  vlines!(ax,omega0,color = :blue,alpha = 0.1)
  lines!(ax,ωs[I],real.(is[I]),color = :black)
  lines!(ax,ωs[I],imag.(is[I]),color = :black,linestyle = :dash)

  rms_val = sqrt(sum(abs2.(is[I]))/N)
  println("rms_val = $rms_val")
  hlines!(ax,rms_val,color = :red,alpha = 0.3)

  if isnothing(set_theta)
    x = beta * omega0
    S = 1
    f1 = 2 * (-1 + S * x * coth(S * x))/(S^2 * x^2)
    #f2(x) = (x/2) * pi * besseli(2,S * x) * csch(S * x)
    f2 = (6 + 2 * S^2 * x^2 - 6 * S * x * coth(S * x)) / (S^3 * x^2)
    #f1(x) = 2 * S * csch(S*x) * (x * cosh(x) - sinh(x))/x^2
    #f2(x) = 2 * S * csch(S*x) * (-3 * x * cosh(x) + (3 + x^2) * sinh(x))/x^2

    if (i,j) == (1,1) || (i,j) == (2,2) || (i,j) == (1,2) || (i,j) == (2,1)
      # RMS value, so sqrt(1/2) for sin(t)
      theory_amplitude = S * S * f1/2 /sqrt(2)
      scatter!(ax,omega0,abs(theory_amplitude),color = :blue)
      scatter!(ax,-omega0,abs(theory_amplitude),color = :blue)
    end

    if (i,j) == (3,3)
      theory_amplitude = S * S * (1 - f1)
      scatter!(ax,0,theory_amplitude)
    end

    if (i,j) == (1,4) || (i,j) == (2,5)
      # RMS value, so sqrt(1/2) for sin(t)
      # Response amplitude:
      #theory_amplitude = (S * f2(x) / 2 + f1) / sqrt(2)
      # Correlation related to response amplitude:
      theory_amplitude = (f2 / 2) / sqrt(2) / beta
      scatter!(ax,omega0,abs(theory_amplitude),color = :blue)
      scatter!(ax,-omega0,abs(theory_amplitude),color = :blue)
    end

    if (i,j) == (3,6)
      theory_amplitude = - f2/beta
      scatter!(ax,0,abs(theory_amplitude))
    end
  else
    x = beta * omega0
    S = 1
    f1 = sin(set_theta)^2
    f2 = x * S * cos(set_theta) * sin(set_theta)^2

    if (i,j) == (1,1) || (i,j) == (2,2) || (i,j) == (1,2) || (i,j) == (2,1)
      # RMS value, so sqrt(1/2) for sin(t)
      theory_amplitude = S * S * f1/2 /sqrt(2)
      scatter!(ax,omega0,abs(theory_amplitude),color = :blue)
      scatter!(ax,-omega0,abs(theory_amplitude),color = :blue)
    end

    if (i,j) == (3,3)
      theory_amplitude = S * S * (1 - f1)
      scatter!(ax,0,theory_amplitude)
    end

    if (i,j) == (1,4) || (i,j) == (2,5)
      # RMS value, so sqrt(1/2) for sin(t)
      # Response amplitude:
      #theory_amplitude = (S * f2(x) / 2 + f1(x)) / sqrt(2)
      # Correlation related to response amplitude:
      theory_amplitude = (f2 / 2) / sqrt(2) / beta
      scatter!(ax,omega0,abs(theory_amplitude),color = :blue)
      scatter!(ax,-omega0,abs(theory_amplitude),color = :blue)
    end

    if (i,j) == (3,6)
      theory_amplitude = - f2/beta
      scatter!(ax,0,abs(theory_amplitude))
    end
  end

  xlims!(ax,(-2 * omega0,2 * omega0))
  f
end

function show_chi_spectrum(dsc,sys,i,j)
  N = size(dsc.data,7)
  params = unit_resolution_binning_parameters(dsc;negative_energies = true)
  is_full = intensities_binned(dsc,params,intensity_formula(dsc,:full))[1][1,1,1,:]

  # Normalize to unitary FFT
  #is_full *= sqrt(N)

  is_at(i,j) = map(x -> x[i,j],is_full)

  response_variable_map = [4,5,6]
  S = 1
  chi = beta * is_at(i,response_variable_map[j]) .+ 2 * is_at(i,j) / S^2
  ifft!(chi)
  chi[fftfreq(length(chi)) .< 0] .= 0
  fft!(chi)

  ωs = axes_bincenters(params)[4]#2π * fftfreq(N,1/dt)
  dw = ωs[2] - ωs[1]
  I = sortperm(ωs)
  f = Figure()
  ax = Axis(f[1,1])
  vlines!(ax,omega0,color = :blue,alpha = 0.1)
  lines!(ax,ωs[I],real.(chi[I]),color = :black)
  lines!(ax,ωs[I],imag.(chi[I]),color = :black,linestyle = :dash)

  rms_val = sqrt(sum(abs2.(chi[I]))/N)
  println("rms_val = $rms_val")
  hlines!(ax,rms_val,color = :red,alpha = 0.3)

  xlims!(ax,(-2 * omega0,2 * omega0))
  display(f)
  chi
end

function show_applied_field_chi_spectrum(dsc,sys,i,j)
  N = size(dsc.data,7)
  params = unit_resolution_binning_parameters(dsc;negative_energies = true)
  is_full = intensities_binned(dsc,params,intensity_formula(dsc,:full))[1][1,1,1,:]

  # Normalize to unitary FFT
  is_full *= sqrt(N)

  is_at(i,j) = map(x -> x[i,j],is_full)

  response_variable_map = [9,10,11]
  S = 1
  chi = -beta * sys.units.μB * sys.gs[1][1] * is_at(i,response_variable_map[j])
  ifft!(chi)
  chi[fftfreq(length(chi)) .< 0] .= 0
  fft!(chi)

  ωs = axes_bincenters(params)[4]#2π * fftfreq(N,1/dt)
  dw = ωs[2] - ωs[1]
  I = sortperm(ωs)
  f = Figure()
  ax = Axis(f[1,1])
  vlines!(ax,omega0,color = :blue,alpha = 0.1)
  lines!(ax,ωs[I],real.(chi[I]),color = :black)
  lines!(ax,ωs[I],imag.(chi[I]),color = :black,linestyle = :dash)

  rms_val = sqrt(sum(abs2.(chi[I]))/N)
  println("rms_val = $rms_val")
  hlines!(ax,rms_val,color = :red,alpha = 0.3)

  xlims!(ax,(-2 * omega0,2 * omega0))
  display(f)
  chi
end



function show_spectrum(dsc,sys,i;epsilon = nothing,set_theta = 2.1, field_mode = true)
  theta = set_theta
  go_to_theta(theta,sys)
  Sunny.new_sample!(dsc,sys,() -> nothing)
  dt = dsc.Δt * dsc.measperiod

  # Fourier resample!
  zbuf = zeros(ComplexF64,size(dsc.samplebuf)[1:5]...,size(dsc.data,7))
  zbuf_mask = zeros(Float64,size(dsc.data,7))
  filt = exp.(- glob_s .* range(0,step = dt,length = size(dsc.data,7)))
  zbuf[:,:,:,:,:,1:size(dsc.samplebuf,6)] .= dsc.samplebuf #.* filt
  zbuf_mask[1:size(dsc.samplebuf,6)] .= 1


  N = size(zbuf,6)
  Nnz = size(dsc.samplebuf,6)
  obs_spectra = fft(zbuf[:,1,1,1,1,:],2) / sqrt(Nnz) #/ sqrt(size(dsc.samplebuf,6))
  ωs = 2π * fftfreq(N,1/dt)
  dw = ωs[2]
  I = sortperm(ωs)
  f = Figure()
  ax = Axis(f[1,1])

    vlines!(ax,omega0,color = :blue,alpha = 0.1)
  if isnothing(epsilon)

    lines!(ax,ωs[I],real.(obs_spectra[i,I]),color = :black,linewidth = 3)
    lines!(ax,ωs[I],imag.(obs_spectra[i,I]),color = :black,linestyle = :dash, linewidth = 3)
    #lines!(ax,ωs[I],abs2.(obs_spectra[i,I]),color = :red)

    # RMS value because unitary FFT
    #rms_val = sqrt(sum(abs2.(obs_spectra[i,I]))/N)
    rms_val = sqrt(sum(abs2.(obs_spectra[i,I]))/N)
    println("rms_val = $rms_val")
    hlines!(ax,rms_val,color = :red,alpha = 0.3)

    if i == 1 || i == 2
      # RMS value, so sqrt(1/2) for sin(t)
      theory_amplitude = sin(theta)/sqrt(2)
      scatter!(ax,omega0,abs(theory_amplitude),color = :blue)
      scatter!(ax,-omega0,abs(theory_amplitude),color = :blue)
    end

    if i == 3
      theory_amplitude = abs(cos(theta))
      scatter!(ax,0,theory_amplitude)
    end

    if i == 4 || i == 5
      # RMS value, so sqrt(1/2) for sin(t)
      theory_amplitude = omega0 * sin(theta) * cos(theta) / sqrt(2)
      scatter!(ax,omega0,abs(theory_amplitude),color = :blue)
      scatter!(ax,-omega0,abs(theory_amplitude),color = :blue)
    end

    if i == 6
      theory_amplitude = -omega0 * sin(theta) * sin(theta)
      scatter!(ax,0,abs(theory_amplitude))
    end

    obs_spectra_pert = obs_spectra
  else
    phase = 0.0
    cb = function()
      phase = phase + dsc.Δt
      if field_mode
        set_external_field!(sys,[0,0,-1] .+ exp(- glob_s * phase) * cos(phase * omega0) * epsilon)
      else
        sys.dipoles[1] = Sunny.normalize_dipole(sys.dipoles[1] + dsc.Δt * exp(- glob_s * phase) * cos(phase * omega0) * epsilon,1)
      end
    end
    go_to_theta(theta,sys)

    Sunny.new_sample!(dsc,sys,cb)
    zbuf .= 0
    zbuf[:,:,:,:,:,1:size(dsc.samplebuf,6)] .= dsc.samplebuf #.* filt
    obs_spectra_pert = fft(zbuf[:,1,1,1,1,:],2) / sqrt(Nnz)

    #lines!(ax,ωs[I],real.(obs_spectra_pert[i,I]),color = :green)
    #lines!(ax,ωs[I],imag.(obs_spectra_pert[i,I]),color = :green,linestyle = :dash)

    function apply_filt(x)
      fftshift(fft(filt .* ifft(ifftshift(x))))
    end
    function pretty_show(x,color)
      x = apply_filt(x)
      lines!(ax,ωs[I],real.(x),color = color)
      lines!(ax,ωs[I],imag.(x),color = color,linestyle = :dash)
    end

    obs_spectra_fluct = obs_spectra_pert .- obs_spectra
    pretty_show(obs_spectra_fluct[i,I],:green)
    pretty_show(obs_spectra_pert[7,I],:red)
    #pretty_show(chixx,:blue)
    fudge = 500 * sqrt(Nnz) #1e1 * sqrt(N)
    #fudge = sqrt(Nnz) #1e1 * sqrt(N)
    conv_result = fftshift(fft(zbuf_mask .* ifft(ifftshift(chi_force_xx .* obs_spectra_pert[7,I]))))
    pretty_show(fudge * conv_result,:orange)

    println("Without filter:")
    println(sqrt(sum(abs2.(obs_spectra[i,I]))/N))
    println(sqrt(sum(abs2.(obs_spectra_pert[7,I]))/N))
    println(sqrt(sum(abs2.(chixx))/N))
    println(sqrt(sum(abs2.(conv_result))/N))
    println(sqrt(sum(abs2.(obs_spectra_fluct[i,I]))/N)) # Scales like nω for some reason???

    println("With filter:")
    println(sqrt(sum(abs2.(apply_filt(obs_spectra[i,I])))/N))
    println(sqrt(sum(abs2.(apply_filt(obs_spectra_pert[7,I])))/N))
    println(sqrt(sum(abs2.(apply_filt(chixx)))/N))
    println(sqrt(sum(abs2.(apply_filt(conv_result)))/N))
    println(sqrt(sum(abs2.(apply_filt(obs_spectra_fluct[i,I])))/N)) # Scales like nω for some reason???

    #display((abs.(apply_filt(obs_spectra_pert[7,I] .* chixx)) ./ abs.(apply_filt(obs_spectra_fluct[i,I])))[600:850])

    #lines!(ax,ωs[I],real.(obs_spectra_fluct[i,I]),color = :green)
    #lines!(ax,ωs[I],imag.(obs_spectra_fluct[i,I]),color = :green,linestyle = :dash)

    #lines!(ax,ωs[I],real.(obs_spectra_pert[7,I]),color = :red)
    #lines!(ax,ωs[I],imag.(obs_spectra_pert[7,I]),color = :red,linestyle = :dash)
#
    #lines!(ax,ωs[I],real.(chixx),color = :blue)
    #lines!(ax,ωs[I],imag.(chixx),color = :blue,linestyle = :dash)

    #lines!(ax,ωs[I],real.(fudge * obs_spectra_pert[7,I] .* chixx),color = :orange)
    #lines!(ax,ωs[I],imag.(fudge * obs_spectra_pert[7,I] .* chixx),color = :orange,linestyle = :dash)

    #lines!(ax,ωs[I],real.(obs_spectra[i,I]),color = :black,linewidth = 3)
    #lines!(ax,ωs[I],imag.(obs_spectra[i,I]),color = :black,linestyle = :dash, linewidth = 3)

    #lines!(ax,ωs[I],real.(obs_spectra_pert[i,I]),color = :green)
    #lines!(ax,ωs[I],imag.(obs_spectra_pert[i,I]),color = :green,linestyle = :dash)

    set_external_field!(sys,[0,0,-1])
  end

  go_to_theta(theta,sys)
  xlims!(ax,(-2 * omega0,2 * omega0))
  display(f)
  obs_spectra_pert[:,I]
end


