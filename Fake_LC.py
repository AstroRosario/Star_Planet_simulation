#Generating a fake light curve of a star

import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import batman
from pylab import *
from astropy.timeseries import BoxLeastSquares
from transitleastsquares import period_grid

#function magnitude to flux
def mag_to_flux(t, mag, exptime=30.0 / 60.0 / 24.0, zeropoint=20.44): 
    #zeropoint is based on TESS instrument handbook,correction factor/filter
    flux = 10 ** (-0.4 * (mag - zeropoint))
    flux = flux*np.ones_like(t) / 1e6
    return flux

# setting a realistic signal for sine wave
def quasiperiodic_signal(A, T, t): #amplitude, period, time 
    from astropy.modeling.functional_models import Sine1D
     #A amplitude of sine wave
    s1 = Sine1D(amplitude = A, frequency = 1/T)    
    signal = s1(t)
    return signal


#function to simulate random light curve
def lightcurve(tmin, tmax, cadence, mag, A, T, zeropoint):
    t = np.arange(tmin, tmax, cadence)
    flux = mag_to_flux(t,mag, exptime=cadence, zeropoint=zeropoint)
    flux = 1.0 + np.random.randn(t.size) * flux
    
    periodic_signal = quasiperiodic_signal(A,T,t)+np.median(flux)
    injected_signal = (flux + periodic_signal - np.median(flux)) / np.median(flux)
    
    return t, injected_signal, periodic_signal


#scenario parameters
duration=27
cadence= (2/60)/24 #unit of days 
cadence = (2*u.min).to(u.day).value
T=6
start=0
mag=10
zeropoint=20.44
A = 20/1e3 #amplitude of injected sine wave
time, injected_signal, periodic_signal = lightcurve(start,duration,cadence,mag,A,T,zeropoint)

plt.plot(time, injected_signal, color='black')
plt.title('Fake Light Curve')
plt.xlabel('Days')
plt.ylabel('Flux')
plt.plot(time, periodic_signal, color='orange')
plt.show()

#Lomb-Scargle
def LS(t, f, minT, maxT):
    from astropy.timeseries import LombScargle
    min_freq = 1/maxT
    max_freq = 1/minT

    model=LombScargle(t, f)
    freq, power = model.autopower(method = 'fast', normalization = 'standard', minimum_frequency = min_freq, maximum_frequency = max_freq) #power-spectral distribution
    period = 1/freq
    return period,power

period, power = LS(time, injected_signal, 1, 30)
plt.plot(period, power, color='r')
plt.title("Periodogram of Fake Light Curve")
plt.xlabel('Period(days)')
plt.ylabel('Power')

plt.show()

#Lightcurve with injected transit 

fs = 10 #fontsize for text in figures

ndays = 27 #days of simulated observation
cadence= 2 * 60 #measurements taken twice an hour
pts_per_day = 24 * cadence #data points per day
scatter_frac = 0.00025  # add scatter to data to make it realistic

start=0.0
stop=ndays
stepsize=ndays*pts_per_day
t = 30 + np.linspace(start, stop, stepsize) 

flux = 1.0 + np.random.randn(t.size) * scatter_frac
fluxerr = np.ones_like(flux) * scatter_frac

plt.figure(figsize=(14,6)) #width, height
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False) #turns off scientific notation
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.plot(t, flux, marker='.', color='k', linestyle='none')
plt.xlabel("Time (Days)")
plt.ylabel("Flux")
plt.title("Fake Light Curve")
plt.ylim(0.9975,1.0025)
plt.show()



from astropy import constants as const
from astropy import units as u

Rad_sun = const.R_sun.cgs #cm
#print(Rad_sun)
Rad_earth = const.R_earth.cgs
Mass_sun = 1.989 * 10.0**33.0 #grams

time_start = t[0] #[0] is the first data point in array

#I'll use batman package to create transit
ma = batman.TransitParams()
ma.t0 = time_start  # time of inferior conjunction; first transit is X days after start
ma.per = 4.8910  # orbital period
ma.rp = 2.0 * Rad_earth / Rad_sun  # planet radius (in units of stellar radii) 
ma.a = 5  # semi-major axis (in units of stellar radii)
ma.inc = 90  # orbital inclination (in degrees)
ma.ecc = 0  # eccentricity
ma.w = 90  # longitude of periastron (in degrees)
ma.u = [0.4, 0.4]  # limb darkening coefficients
ma.limb_dark = "quadratic"  # limb darkening model
m = batman.TransitModel(ma, t)  # initializes model
synthetic_signal = m.light_curve(ma)  # calculates light curve

injected_flux = synthetic_signal - flux + 1

plt.figure(figsize=(14,6)) #width, height
plt.gca().get_xaxis().get_major_formatter().set_useOffset(False) #turns off scientific notation
plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)
plt.plot(t,injected_flux,marker='.',color='black',linestyle='none')
plt.plot(t,synthetic_signal,marker='.',color='purple',linestyle='none')
plt.xlabel("Time (Days)")
plt.ylabel("Flux")
plt.title("Transit Injected Fake Light Curve")
plt.ylim(0.9975,1.0025)
plt.show()


#Box Least Square Periodogram
LCduration = t[-1] - t[0] #duration of light curve (last minus first data point)

#min/max of orbital period grid
minP = 1.01
maxP = LCduration #orbital period for grid

minT = 1.0/24.0
maxT = 7.0 / 24.0 #transit durations hours

#values for the star
R_star = 0.20 #Solar radii
M_star = 0.18 #solar masses

durations = np.linspace(minT, maxT, 50)

periods = period_grid(R_star=R_star, M_star=M_star, time_span=LCduration, period_min=minP, period_max=maxP) #oversampling factor

bls = BoxLeastSquares(t, injected_flux) 
bls_power = bls.power(periods, durations)

bls_SDE = (bls_power.power - np.mean(bls_power.power)) / np.std(bls_power.power)

index = np.argmax(bls_power.power) #finds strongest peak in BLS power spectrum
BLS_periods=bls_power.period[index]
BLS_t0s=bls_power.transit_time[index]
BLS_depths=bls_power.depth[index]
dur = minT #0.5

bls_model=bls.model(t,bls_power.period[index], bls_power.duration[index], bls_power.transit_time[index])



# #custom binning function
def bin_func(time, flux, error, binsize):
    good = np.where(np.isfinite(time))
    time_fit = time[good]
    flux_fit = flux[good]
    error_fit = error[good]
    time_max = np.max(time_fit)
    time_min = np.min(time_fit)
    npoints = len(time_fit)
    nbins = int(np.math.ceil((time_max - time_min)/binsize)) #binsize in days
    bin_time = np.full((nbins,), np.nan)
    bin_flux = np.full((nbins,), np.nan)
    bin_err = np.full((nbins,), np.nan)
    for i in range(0, nbins-1):
        tobin = [np.where( (time_fit >= (time_min + i *binsize)) & (time_fit < (time_min + (i + 1) * binsize)) )]
        if tobin[0] != -1:
            #inverse variance weighted means
            bin_flux[i] = ((flux_fit[tobin] / (error_fit[tobin]**2.0)).sum()) /((1.0 / error_fit[tobin]**2.0).sum())
            bin_time[i] = ((flux_fit[tobin] / (error_fit[tobin]**2.0)).sum()) /((1.0 / error_fit[tobin]**2.0).sum())
            bin_err[i] = 1.0 / (np.sqrt (1.0 / (error_fit[tobin]**2.0)).sum() )

    good2 = np.where(np.isfinite(bin_time))
    bin_time = bin_time[good2]
    bin_flux = bin_flux[good2]
    bin_err = bin_err[good2]

    return bin_time, bin_flux, bin_err

fs=10
plt.figure(figsize=(8, 4))
plt.plot(np.log10(bls_power.period), bls_SDE, "k")
plt.axvline(np.log10(ma.per), color="grey", lw=8, alpha=0.5) #injected transit period
plt.annotate("period = {0:.4f} days".format(BLS_periods),(0, 1), xycoords="axes fraction",
                        xytext=(50, -5), textcoords="offset points",va="top", ha="left", fontsize=12)
plt.ylabel("SDE",fontsize=fs)
plt.xlim(np.log10(periods.min())-0.1, np.log10(periods.max())+0.1)
# plt.ylim(bls_SDE.min()-0.25, bls_SDE.max()+0.25)
plt.xlabel("log(Period ( days))",fontsize=fs)
plt.show()

# Plot the folded transit
p = BLS_periods
x_fold = (t - BLS_t0s + 0.5 * p) % p - 0.5 * p

 
binsize= 60.0 / (60.0 * 24.0) 
bint,binf,binfe = bin_func(x_fold, injected_flux,fluxerr,binsize)

plt.plot(24.0 * x_fold, injected_flux, color='grey',marker=".",linestyle='none',label='injected flux')
#plt. plot(24.0*bint, binf, color="r",marker='o',label=str(binsize*24*60)+' Min. bins') #hours
plt.plot(24.0 * x_fold, bls_model, ".b",label='BLS Model') #hours

plt.xlim(-0.5 * 24.0, 0.5 * 24.0)
plt.ylim(np.min(injected_flux)-0.001, np.max(injected_flux) + 0.001)  #just for specific target      
plt.ylabel("Normalized Relative Flux [ppt]",fontsize=fs)
plt.xlabel("Time since transit (hours)",fontsize=fs)
plt.legend(loc='upper right',ncol=3)
plt.show()