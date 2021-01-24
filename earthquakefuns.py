import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def plotseismic(st,stns=[],chn='vertical',reftime=None,tlm=[-30,1000],
                flm=[0.005,8]):
    """
    plot the seismograms
    :param         st: the seismic traces 
    :param       stns: a list of the stations to plot
    :param        chn: which channel to plot 
                           ('east','north',or 'vertical')
    :param    reftime: the zero time---earthquake time
    :param        tlm: time range to plot in s 
                           (default: [-30,1000])
    :param        flm: frequency limit to plot
    """

    # make sure we have a list of stations
    stns=np.atleast_1d(stns)
    if stns.size==0:
        stns=np.unique([tr.stats.station for tr in st])[0:5]
    Ns=len(stns)

    # potential channels
    if  chn in ['vertical','Z']:
        cget='?HZ'
    elif chn in ['east','1','E']:
        cget='?H[1E]'
    elif chn in ['north','2','N']:
        cget='?H[2N]'

    # reference time if not given
    if reftime is None:
        reftime=st[0].stats.starttime

    # make a figure
    f=plt.figure(figsize=(10,8))
    gs,p=gridspec.GridSpec(Ns,1),[]
    for gsi in gs:
        p.append(plt.subplot(gsi))
    pm=np.array(p).reshape([Ns,1])
    gs.update(left=0.13,right=0.85)
    gs.update(bottom=0.08,top=0.97)
    gs.update(hspace=0.,wspace=0.03)
    p=np.array(p)

    k=-1
    amp,shf=1,1.1
    for stni in stns:
        k=k+1
        # find the relevant seismic data
        sti=st.select(station=stni,channel=cget)
        
        if len(sti):
            tr=sti[0].copy()
            tr.detrend()
            tr.filter('bandpass',freqmin=flm[0],freqmax=flm[1])
            tr.trim(starttime=reftime+tlm[0],endtime=reftime+tlm[1])

            # timing
            tms=tr.times()+(tr.stats.starttime-reftime)

            # plot and label
            p[k].plot(tms,tr.data)
            p[k].set_ylabel('displacement at\n'+\
                            '.'.join([tr.stats.network,tr.stats.station,tr.stats.channel])+\
                            '\n(m)')
            p[k].set_ylim(np.array([-1,1])*1.1*np.max(np.abs(tr.data)))
            p[k].set_xlim(tlm)
            

        
        
    
def spheredist(loc1,loc2):
    """
    :param     loc1: [lon1,lat1] or Nx2 array of locations
    :param     loc2: [lon2,lat2] or Nx2 array of locations
    :return    dsts: distances in degrees [# of loc1 by # of loc2]
    :return      az: azimuths from location 1 to location 2
    """

    # make them a 2-d grid
    loc1=np.atleast_2d(loc1)
    loc2=np.atleast_2d(loc2)

    # latitude in radians
    phi1=loc1[:,1]*(np.pi/180)
    phi2=loc2[:,1]*(np.pi/180)

    # longitude 
    thet1=loc1[:,0]
    thet2=loc2[:,0]

    # to correct dimensions
    phi1=phi1.reshape([phi1.size,1])
    thet1=thet1.reshape([thet1.size,1])
    phi2=phi2.reshape([1,phi2.size])
    thet2=thet2.reshape([1,thet2.size])

    # longitude difference in radians
    thet2=(thet2-thet1)*(np.pi/180)
    
    # to distances
    dsts=np.multiply(np.cos(phi1),np.cos(phi2))
    dsts=np.multiply(dsts,np.cos(thet2))
    dsts=dsts+np.multiply(np.sin(phi1),np.sin(phi2))
    dsts=np.arccos(np.minimum(dsts,1.))

    # to azimuths
    az=np.multiply(np.cos(phi1),np.tan(phi2))
    az=az-np.multiply(np.sin(phi1),np.cos(thet2))
    az=np.divide(np.sin(thet2),az)
    az=np.arctan(az)

    # there's an azimuthal ambiguity of 180 degrees, 
    # so let's check the law of sines for each azimuth
    df1=np.multiply(np.sin(az),np.sin(dsts)) - \
        np.multiply(np.sin(thet2),np.sin(np.pi/2-phi2))
    df2=np.multiply(np.sin(az+np.pi),np.sin(dsts)) - \
        np.multiply(np.sin(thet2),np.sin(np.pi/2-phi2))
    sw=np.abs(df2)<np.abs(df1)
    az[sw]=az[sw]+np.pi

    # and still one ambiguity---when they're along a line of longitude
    sw=np.logical_or(az==0.,az==np.pi)
    if len(sw):
        phi1=phi1-np.zeros(phi2.shape)
        phi2=phi2-np.zeros(phi1.shape)
        sw0=(np.pi/2.-phi2[sw].flatten())+(np.pi/2.-phi1[sw].flatten())<=np.pi

        shp=az.shape
        az=az.flatten()
        sw=np.where(sw.flatten())[0]
        az[sw[sw0]]=0.
        az[sw[~sw0]]=np.pi
        az=az.reshape(shp)

    # back to degrees
    az=az*(180/np.pi) % 360
    dsts=dsts*(180/np.pi)
    
    return dsts,az
