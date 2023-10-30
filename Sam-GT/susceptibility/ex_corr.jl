
using FFTW, GLMakie

#L = 10
L = 402

xx = randn(L)

for j = 1:40; xx .= (xx + [0;xx[1:end-1]] + 0.01 .* [yy[2:end];0]) ./ 2; end

yy = randn(L)

for j = 1:40; yy .= (yy + [0;yy[1:end-1]] + 0.01 .* [xx[2:end];0]) ./ 2; end

time_2T = length(xx)
longest_correlation_delay = div(time_2T,4,RoundDown)
left_zero_ix = 1:longest_correlation_delay
right_zero_ix = (time_2T - longest_correlation_delay + 1):time_2T
central_ix = (longest_correlation_delay + 1):(time_2T - longest_correlation_delay)
#rearrange_ix = [right_zero_ix ; left_zero_ix]
rearrange_ix = [(1:longest_correlation_delay+1) ; right_zero_ix]

time_T = length(xx)÷2
li = 1:time_T
ri = (time_T+1):time_2T

lzb_x = copy(xx); lzb_x[li] .= 0;
lzb_y = copy(yy); lzb_y[li] .= 0;
rzb_x = copy(xx); rzb_x[ri] .= 0;
rzb_y = copy(yy); rzb_y[ri] .= 0;

c_xy_rz = real.(ifft(fft(yy) .* conj.(fft(rzb_x))))[li]
c_xy_lz = real.(ifft(fft(lzb_y) .* conj.(fft(xx))))[li]
c_xy = (c_xy_rz + c_xy_lz) ./ 2

c_yx_rz = real.(ifft(fft(xx) .* conj.(fft(rzb_y))))[li]
c_yx_lz = real.(ifft(fft(lzb_x) .* conj.(fft(yy))))[li]
c_yx = (c_yx_rz + c_yx_lz) ./ 2

c_xx_rz = real.(ifft(fft(xx) .* conj.(fft(rzb_x))))[li]
#c_xx_lz = real.(ifft(conj.(fft(lzb_x)) .* fft(xx)))[ri]
c_xx_lz = real.(ifft(fft(lzb_x) .* conj.(fft(xx))))[li]
c_xx = (c_xx_rz + c_xx_lz) ./ 2

c_yy_rz = real.(ifft(fft(yy) .* conj.(fft(rzb_y))))[li]
c_yy_lz = real.(ifft(fft(lzb_y) .* conj.(fft(yy))))[li]
c_yy = (c_yy_rz + c_yy_lz) ./ 2

appended_xy = [c_xy;reverse(c_yx)[1:end-1]]
fft_appended_xy = fft(appended_xy)

cc_xy = copy(c_xy)
cc_yx = copy(c_yx)
cc_xy[1] /= 2
cc_yx[1] /= 2

pasted_xy = cc_xy .+ circshift(reverse(cc_yx),1)
fft_pasted_xy = fft(pasted_xy)

conjsym_fft_xy = fft(cc_xy) .+ conj.(fft(cc_yx))

source_xx = copy(xx); source_xx[left_zero_ix] .= 0; source_xx[right_zero_ix] .= 0;
source_yy = copy(yy); source_yy[left_zero_ix] .= 0; source_yy[right_zero_ix] .= 0;


window = cos.(range(0,π,length = time_2T÷2)).^2
# non-conj is delayed
corr_xx = real.(FFTW.ifft(fft(xx) .* conj.(fft(source_xx))))[rearrange_ix]
#corr_xx_sym = real.(FFTW.ifft(fft(source_xx) .* conj.(fft(xx))))[rearrange_ix]
corr_xx_conj = real.(ifft(conj.(fft(corr_xx))))

corr_xy = real.(FFTW.ifft(fft(xx) .* conj.(fft(source_yy))))[rearrange_ix]
corr_xy_conj = real.(ifft(conj.(fft(corr_xy))))
#corr_xy_sym = real.(FFTW.ifft(fft(source_xx) .* conj.(fft(yy))))[rearrange_ix]

corr_yx = real.(FFTW.ifft(fft(yy) .* conj.(fft(source_xx))))[rearrange_ix]
#corr_yx_sym = real.(FFTW.ifft(fft(source_yy) .* conj.(fft(xx))))[rearrange_ix]
corr_yx_conj = real.(ifft(conj.(fft(corr_yx)))) # recovers corr_xy_sym

corr_yy = real.(FFTW.ifft(fft(yy) .* conj.(fft(source_yy))))[rearrange_ix]
corr_yy_conj = real.(ifft(conj.(fft(corr_yy))))
#corr_yy_sym = real.(FFTW.ifft(fft(source_yy) .* conj.(fft(yy))))[rearrange_ix]
nothing
