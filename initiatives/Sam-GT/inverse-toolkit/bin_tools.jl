using LinearAlgebra, Statistics, GLMakie

"""
    chunk_params(params,data;chunking = (1,1,1,1), nan = true)

Exactly re-bin `data` described by `params::HistogramParameters` into new larger bins
which are integer multiples of the orginal bin size. If the `chunking` is incommensurate
with the available data, some data (at most `chunking[i] - 1`) will be truncated at the
far edge along each axis `i`. Returns the new parameters and new data array.

If `nan = false`, any `NaN` in the original data is replaced by zero.
"""
function chunk_params(params_orig,data_orig;chunking = (1,1,1,1), nan = true)
  params = copy(params_orig)
  data = zeros(Float64,size(data_orig).÷chunking)
  data_nonan = if nan
    data_orig
  else
    data_nonan = copy(data_orig)
    data_nonan[isnan.(data_orig)] .= 0.
    data_nonan
  end
  for i = 1:4
    params.binwidth[i] *= chunking[i]
    for ci = CartesianIndices(size(data))
      i1 = (1:chunking[1]) .+ (ci[1] - 1) * chunking[1]
      i2 = (1:chunking[2]) .+ (ci[2] - 1) * chunking[2]
      i3 = (1:chunking[3]) .+ (ci[3] - 1) * chunking[3]
      i4 = (1:chunking[4]) .+ (ci[4] - 1) * chunking[4]
      data[ci] = sum(data_nonan[i1,i2,i3,i4])
    end
  end
  params, data
end

"""
    approximate_bintegrate(ax,desire_start,desire_end,params,data;restrict = false,nan=true)

*Exactly* integrate `data` described by `params::HistogramParameters` according to *approximately*
the limits of integration `[desire_start, desire_end]` along axis `ax` (1, 2, 3, or 4).
Returns the exact new parameters and new data array.

If `restrict = true`, the original data is truncated to the specified interval, but not summed over.

If `nan = false`, any `NaN` in the original data is replaced by zero.
"""
function approximate_bintegrate(ax,desire_start,desire_end,params,data;restrict = false,nan=true)
  x0 = params.binstart[ax]
  dx = params.binwidth[ax]
  x1 = (desire_start - x0)/dx
  x2 = (desire_end - x0)/dx

  x1 = max(0,x1)
  x2 = min(params.numbins[ax],x2)

  # Edge case
  x2 = max(1,x2)
  x1 = min(params.numbins[ax]-1,x1)

  x1 = round(Int64,x1)
  x2 = round(Int64,x2)
  bes = Sunny.axes_binedges(params)[ax]
  ix = (x1+1):x2
  data_nonan = if nan
    data
  else
    data_nonan = copy(data)
    data_nonan[isnan.(data)] .= 0.
    data_nonan
  end
  new_data = if restrict
    if ax == 1
      data_nonan[ix,:,:,:]
    elseif ax == 2
      data_nonan[:,ix,:,:]
    elseif ax == 3
      data_nonan[:,:,ix,:]
    elseif ax == 4
      data_nonan[:,:,:,ix]
    end
  else
    if ax == 1
      sum(data_nonan[ix,:,:,:],dims=1)
    elseif ax == 2
      sum(data_nonan[:,ix,:,:],dims=2)
    elseif ax == 3
      sum(data_nonan[:,:,ix,:],dims=3)
    elseif ax == 4
      sum(data_nonan[:,:,:,ix],dims=4)
    end
  end
  new_params = copy(params)
  new_params.binstart[ax] = bes[x1+1]
  new_params.binend[ax] = bes[x2+1]
  new_params.binwidth[ax] = (restrict ? 1 : (0.1 + x2 - x1)) * params.binwidth[ax]
  new_params, new_data
end


"""
    log_sweep(x0,v,f::Function; count = 30, dlog = 0.2, verbose = true)

Sweep the expensive real-valued function `f(x)` over a logarithmically
spaced range of vectors `xⱼ = x0 + v * 10^(j * dlog)` for `j = 1…count`.
Plots and returns the sequence of scalars `10^(j * dlog)` together with the
function values `f(xⱼ)`.
"""
function log_sweep(x0,v,f;count = 30, dlog = 0.2, verbose = true)
  sc = zeros(Float64,count)
  vals = zeros(Float64,count)
  for j = 1:count
    sc[j] = 10^(j * dlog)
    x = x0 .+ sc[j] * v
    vals[j] = f(x)
  end
  if verbose
    display(plot(sc,vals))
  end
  sc, vals
end


