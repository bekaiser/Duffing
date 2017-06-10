# A Duffing oscillator 
# Bryan Kaiser
# 6/18/16

using DataArrays
using PyPlot
using PyCall
@pyimport numpy as np
@pyimport pylab as py
# using Base.FFTW
# using HDF5


# =============================================================================
# function declaration for input parameter section

function meshgrid{T}(vx::AbstractVector{T}, vy::AbstractVector{T})
    m, n = length(vy), length(vx)
    vx = reshape(vx, 1, n)
    vy = reshape(vy, m, 1)
    (repmat(vx, m, 1), repmat(vy, 1, n))
end

function kf_RK4(G)
	kF = G
	return kF
end

function kg_RK4(F,G,alpha,beta,delta,gamma,omega,t)
	kG = -delta*G-alpha*F-beta*F^3.0+gamma*cos(omega*t) # Duffing
	#kG = delta*G-alpha*F-delta*(F^2.0)*G+gamma*cos(omega*t) # van der Pol
	return kG
end


# =============================================================================
# simulation parameters

# time series parameters
const N = Int32(1E7)
const dt = 2.0*pi/100.0 # s, time step intervals (4th-order Runge-Kutta)
const dt_2 = dt/2.0 # s, time step intervals (4th-order Runge-Kutta) 
const dt_6 = dt/6.0 # s, time step intervals (4th-order Runge-Kutta)  
const n0 = 0
F = zeros(Int32(N),1)
G = zeros(Int32(N),1)
t = zeros(Int32(N),1)

# forcing parameters
const alpha = 0.0 # 1/s^2
const beta = 1.0 # 1/(K^2s^2)
const gamma = 7.5 # K/s^2
const delta = 0.05 # 1/s
const omega = 1.0 #2.0*pi/(dt*Nperiod)
period = 2.0*pi
@show(period/dt)
Ncycle = Int32(round(period/dt))  
@show(Ncycle)


#=
# output plots
const plotflag = 0 # enter 1 for .png output plots or 0 for no output plots
const nplot_interval = 10 # number of time steps between output plots
const nplot_start = 0 # number of time steps after n0 before starting output plots

# output .h5 files
const writeflag = 1 # enter 1 for .h5 file output or anything else for no output files
const nwrite_interval = 100 # number of time steps between output files
const nwrite_start = 100 # number of time steps after n0 before starting output plots
=#


## ============================================================================
# simulation initialization 

if n0 == 0 # specify initial conditions
F[1] = 0.0
G[1] = 1.0

else # input q field from .h5 file	
if n0 <10 # filename
readname = "./output/duffing_00000$(n0).h5"
elseif n0 <100 
readname = "./output/duffing_0000$(n0).h5"
elseif n0 <1000
readname = "./output/duffing_000$(n0).h5"
elseif n0 <10000
readname = "./output/duffing_00$(n0).h5"
elseif n0 <100000
readname = "./output/duffing_0$(n0).h5"
elseif n0 <1000000
readname = "./output/duffing_$(n0).h5"
end
Fn = h5read(readname,"F") # 1/s^2, p.v. initial field
Gn = h5read(readname,"G") # 1/s^2, p.v. initial field
readStr = "\ninput .h5 file "readname
println(readStr) 

end # initialization set up


## ============================================================================
# time advancement

nprintout = 0 # output plot initialize

nplot = 0 # counter for output plots
#nwrite = 0 # counter for output files
#n10step = 0 # counter for time steps
t[1] = 0.0 # initial time

tic()
for n = 1:N-1

	if n/10 == nprintout # output plots	
	@show(n)
	nprintout = nprintout+1
	end

	#=
	py.figure() # psi layer 1 plot
	CP1 = py.plot(F[1:Int32(n)],G[1:Int32(n)]) 
	#py.clim(-4,4) # only applies to colors on the plot, not colorbar
	xlabel("T")
	ylabel("dT")
	if n < 100
	plotname = "./output/0000$(nprintout).png"
	elseif n <1000
	plotname = "./output/000$(nprintout).png"
	elseif n <10000
	plotname = "./output/00$(nprintout).png"
	elseif n <100000
	plotname = "./output/0$(nprintout).png"
	end
	py.savefig(plotname,format="png")
	py.close()
	nprintout = nprintout+1
	end # output plots
	=#

	# first RK4 coefficient
	F1 = F[n]
	G1 = G[n]
	kf1 = G1 #kf_RK4(G1)
	kg1 = kg_RK4(F1,G1,alpha,beta,delta,gamma,omega,t[n])

	# second RK4 coefficient
	F2 = F1+kf1.*dt_2
	G2 = G1+kg1.*dt_2
	kf2 = G2 #kf_RK4(G2)
	kg2 = kg_RK4(F2,G2,alpha,beta,delta,gamma,omega,t[n]+dt_2)

	# third RK4 coefficient
	F3 = F1+kf2.*dt_2
	G3 = G1+kg2.*dt_2
	kf3 = G3 #kf_RK4(G3)
	kg3 = kg_RK4(F3,G3,alpha,beta,delta,gamma,omega,t[n]+dt_2)

	# fourth RK4 coefficient
	F4 = F1+kf3.*dt
	G4 = G1+kg3.*dt
	kf4 = G4 #kf_RK4(G4)
	kg4 = kg_RK4(F4,G4,alpha,beta,delta,gamma,omega,t[n]+dt)

	# time advancement
	F[n+1] = F1+(kf1+kf2.*2.0+kf3.*2.0+kf4).*dt_6
	G[n+1] = G1+(kg1+kg2.*2.0+kg3.*2.0+kg4).*dt_6
	t[n+1] = t[n]+dt

end # time advancement loop
toc()

# phase trajectory
py.figure()
CP1 = py.plot(F[1000:8000,1],G[1000:8000,1]) 
py.xlabel("T")
py.ylabel("dT/dt")
#py.xlim([-2,2])
#py.ylim([-2,2])
#py.savefig("vdP_d001_a1.jpg",dpi=200)
py.savefig("duff_a0_b1_d005_g75.jpg",dpi=200)
py.show()


# Poincare section (find points at each forcing period)
locs = collect(5*Ncycle:Ncycle:N)
#locs = find(F -> F == 0.0,F)
#locs = find(F -> -0.01 < F < 0.01,F)
F0 = F[locs]
G0 = G[locs]
py.scatter(F0,G0) #s=area,c=colors,alpha=0.5)
py.xlabel("T")
py.ylabel("dT/dt")
py.title("t=[0,2\pi,4\pi,...]")
py.savefig("duff_a0_b1_d005_g75_PS.jpg",dpi=200)
py.show()

# forcing parameters
const alpha = 0.0 # 1/s^2
const beta = 1.0 # 1/(K^2s^2)
const gamma = 7.5 # K/s^2
const delta = 0.05 # 1/s
const omega = 1.0 #2.0*pi/(dt*Nperiod)

