
function f_target(v)
  x,y,z = v
  x^2 + 33*(y-8)^2 + 200*((y+z) - 4)^2
end

function f_target_2d(v)
  x,y = v
  x^2 + 33*(y-8)^2
end

function plot_2d()
  cent, cov = thermal_basin(f_target_2d,[0.,8],10.0;j_max = 20)
  λ, V = eigen(cov)
  display(eigen(cov))
  xs = range(-100,100;length = 30)
  ys = range(-10 + 8,10 + 8;length = 30)
  L = zeros(Float64,length(xs),length(ys))
  for (ix,x) in enumerate(xs), (iy,y) in enumerate(ys)
    L[ix,iy] = f_target_2d([x,y])
  end
  f = Figure(); ax = Axis(f[1,1])
  contourf!(ax,xs,ys,L)
  scatter!(ax,cent[1],cent[2])
  σ = sqrt.(λ)
  scatter!(ax,cent[1] + 10 * σ[1] * V[1,1],cent[2] + 10 * σ[1] * V[2,1])
  scatter!(ax,cent[1] + 10 * σ[2] * V[1,2],cent[2] + 10 * σ[2] * V[2,2])
  f
end

function find_basin(f,x0,f_max,v;eps_init = 1e-4,log_step = 0.1,log_max = 5)
  js = zeros(Float64,size(v,2))
  for s = [-1,1]
    for k = 1:size(v,2)
      v0 = s * v[:,k]
      fx = f(x0)
      #println("x = $x0, f(x) = $fx")
      j = 1
      while fx < f_max
        x = x0 + v0 * 10^(log10(eps_init) + j * log_step)
        fx = f(x)
        #println("x = $x, f(x) = $fx")
        j = j + 1
        if j * log_step > log_max
          println("Not finding boundary!")
          break
        end
      end
      println("v0 = $v0, j = $j")
      js[k] = max(js[k],j)
    end
  end
  
  widths = 10 .^ (log10(eps_init) .+ js * log_step)
  display(widths)
  # Scale to ellipse coordinates
  v * diagm(widths)
end

function rot_quarter_radian(M)
  c = cos(1/4)
  s = sin(1/4)
  R = [c s 0; -s c 0; 0 0 1]
  R' * M * R
end



function thermal_basin(f,x0,kT; j_max = 2000, noise_scale = 1.0, verbose = false)
  x = copy(x0)

  # Statistics we want of the probability distribution
  cov = zeros(Float64,length(x),length(x)) # Covariance
  cent = zeros(Float64,length(x)) # Mean
  xs = typeof(x0)[]

  if verbose
    ff = Figure()
    ax1 = Axis(ff[1,1])
    if length(x0) >= 4
      ax2 = Axis(ff[2,1])
    end
    display(ff)
  end

  j_incl = 0 # Number of terms in mean
  j_accept = 0 # Number of accepted steps
  f0 = f(x0) # Initialize
  for j = 1:j_max

    # Proposed step
    # (random walk with axis-aligned ellipsoid covariance)
    dx = noise_scale .* randn(Float64,size(x0))

    # Metropolis acceptance probability
    x .= x0 + dx

    if verbose
      println("$x0 → $f0")
      scatter!(ax1,x0[1],x0[2],color = :blue)
      if length(x0) >= 4
        scatter!(ax2,x0[3],x0[4],color = :red)
      end
    end

    fx = f(x)
    accept_prob = min(1,exp(-(fx-f0)/kT))

    if rand() < accept_prob
      # Accept
      j_accept = j_accept + 1
      # Make step
      x0 .= x
      f0 = f(x0)
    end

    # Update statistics (regardless of accepting the step)
    j_incl = j_incl + 1
    cent .= cent .+ (x0 .- cent) ./ j_incl
    cov .= cov .+ ((x0 * x0') .- cov) ./ j_incl
    push!(xs,copy(x0))
  end
  println("Success rate = $(j_accept / j_max)")
  # ⟨x⟩, ⟨x^2⟩ - ⟨x⟩^2
  cent, (cov .- (cent * cent')), xs
end


