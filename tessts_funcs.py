import numpy as np
from scipy.ndimage.filters import convolve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def Event_ID(Sigmask, Significance, Minlength):
    """
    Identifies events in a datacube, with a primary input of a boolean array for where pixels are 3std above background.
    Event duration is calculated by differencing the positions of False values in the boolean array.
    The event mask is saved as a tuple.
    """
    binary = Sigmask >= Significance


    tarr = np.copy(binary)
    summed_binary = np.nansum(binary,axis=0)
    leng = 3
    X = np.where(summed_binary >= Minlength)[0]
    Y = np.where(summed_binary >= Minlength)[1]

    if False:
        for i in range(leng-2):
            kern = np.zeros((leng, 1, 1))
            kern[[0, -1]] = 1
            tarr[convolve(tarr*1, kern) > 1] = True
            leng -= 1

    events = []
    eventtime = []
    eventtime = []
    eventmask = []

    for i in range(len(X)):
        temp = np.insert(tarr[:,X[i],Y[i]],0,False) # add a false value to the start of the array
        testf = np.diff(np.where(~temp)[0])
        indf = np.where(~temp)[0]
        testf[testf == 1] = 0
        testf = np.append(testf,0)


        if len(indf[testf>Minlength]) > 0:
            for j in range(len(indf[testf>Minlength])):
                if abs((indf[testf>Minlength][j] + testf[testf>Minlength][j]-1) - indf[testf>Minlength][j]) < 48: # Condition on events shorter than a day 
                    start = indf[testf>Minlength][j]
                    end = (indf[testf>Minlength][j] + testf[testf>Minlength][j]-1)
                    #if np.nansum(Eventmask[start:end,X[i],Y[i]]) / abs(end-start) > 0.5:
                    events.append(indf[testf>Minlength][j])
                    eventtime.append([indf[testf>Minlength][j], (indf[testf>Minlength][j] + testf[testf>Minlength][j]-1)])
                    masky = [np.array(X[i]), np.array(Y[i])]
                    eventmask.append(masky)    
                else:
                    events.append(indf[testf>Minlength][j])
                    eventtime.append([indf[testf>Minlength][j], (indf[testf>Minlength][j] + testf[testf>Minlength][j]-1)])
                    masky = [np.array(X[i]), np.array(Y[i])]
                    eventmask.append(masky)

    events = np.array(events)
    eventtime = np.array(eventtime)
    return events, eventtime, eventmask

def Match_events(Events, Eventtime, Eventmask, Seperation = 5):
    """
    Matches flagged pixels that have coincident event times of +-5 cadences and are closer than 4 pix
    seperation.
    """
    i = 0
    eventmask2 = []
    while len(Events) > i:
        coincident = (np.isclose(Eventtime[i, 0], Eventtime[i:, 0], atol = Seperation) + np.isclose(
            Eventtime[i, 1], Eventtime[i:, 1], atol = Seperation))
        dist = np.sqrt((np.array(Eventmask)[i, 0]-np.array(Eventmask)[i:, 0])**2 + (
            np.array(Eventmask)[i, 1]-np.array(Eventmask)[i:, 1])**2)
        dist = dist < 5

        coincident = coincident * dist
        if sum(coincident*1) > 1:
            newmask = Eventmask[i].copy()

            for j in (np.where(coincident)[0][1:] + i):
                newmask[0] = np.append(newmask[0], Eventmask[j][0])
                newmask[1] = np.append(newmask[1], Eventmask[j][1])
            eventmask2.append(newmask)
            Events = np.delete(Events, np.where(coincident)[0][1:]+i)
            Eventtime = np.delete(Eventtime, np.where(
                coincident)[0][1:]+i, axis=(0))
            killer = sorted(
                (np.where(coincident)[0][1:]+i), key=int, reverse=True)
            for kill in killer:
                del Eventmask[kill]
        else:
            eventmask2.append(Eventmask[i])
        i += 1
    return Events, Eventtime, eventmask2



def TESS_Fig(Events,Eventtime,Eventmask,Data,Time,wcs):
    """
    Makes the main K2:BS pipeline figure. Contains light curve with diagnostics, alongside event info.
    """
    print('Number of events: ', len(Events))
    for i in range(len(Events)):
        #print(i)
        mask = np.zeros((Data.shape[1],Data.shape[2]))
        mask[Eventmask[i][0],Eventmask[i][1]] = 1
        
        if np.isnan(Time[Eventtime[i][1]]):
            Eventtime[i][1] = Eventtime[i][1] -1
        
        # Find Coords of transient
        position = np.where(mask)
        if len(position[0]) == 0:
            print(Broken)
        Mid = ([position[0][0]],[position[1][0]])
        maxcolor = -1000 # Set a bad value for error identification
        for j in range(len(position[0])):
            lcpos = np.copy(Data[Eventtime[i][0]:Eventtime[i][-1],position[0][j],position[1][j]])
            nonanind = np.isfinite(lcpos)
            temp = sorted(lcpos[nonanind].flatten())
            temp = np.array(temp)
            if len(temp) > 0:
                temp  = temp[-1] # get brightest point
                if temp > maxcolor:
                    maxcolor = temp
                    Mid = ([position[0][j]],[position[1][j]])

        if len(Mid[0]) == 1:
            Coord = pix2coord(Mid[1],Mid[0],wcs)
        elif len(Mid[0]) > 1:
            Coord = pix2coord(Mid[1][0],Mid[0][0],wcs)
        #print('position')
        # Generate a light curve from the transient masks
        LC = Lightcurve(Data, mask)
        #print('lightcurve')
        fig = plt.figure(figsize=(10,6))
        # set up subplot grid
        gridspec.GridSpec(2,3)
        #plt.suptitle('TIC: ' + TIC(File) + '\nSource: '+ Source[i] + ' (' + SourceType[i] + ')')
        # large subplot
        plt.subplot2grid((2,3), (0,0), colspan=2, rowspan=2)
        plt.title('Event light curve ('+str(round(Coord[0],3))+', '+str(round(Coord[1],3))+')')
        plt.xlabel('Time (+'+str(np.floor(Time[0]))+' BJD)')
        plt.ylabel('Counts')
        #print('labels')
        if Eventtime[i][-1] < len(Time):
            #print('axspan')
            plt.axvspan(Time[Eventtime[i][0]]-np.floor(Time[0]),Time[Eventtime[i][-1]]-np.floor(Time[0]), color = 'orange',alpha=0.5, label = 'Event duration', rasterized=True)
            #print('axspan done')
        else:
            plt.axvspan(Time[Eventtime[i][0]]-np.floor(Time[0]),Time[-1]-np.floor(Time[0]), color = 'orange',alpha=0.5, label = 'Event duration', rasterized=True)
        #if (Eventtime[i][-1] - Eventtime[i][0]) < 48:
            #print('quality time')
            #plt.axvline(Time[Quality[0]]-np.floor(Time[0]),color = 'red', linestyle='dashed',label = 'Quality', alpha = 0.5, rasterized=True)
            #print('iterate')
            #for j in range(Quality.shape[0]-1):
            #    j += 1 
            #    if j < len(Quality):
            #        plt.axvline(Time[Quality[j]]-np.floor(Time[0]), linestyle='dashed', color = 'red', alpha = 0.5, rasterized=True)
            #        if i == 1:
            #            print(j)
            
        
        #print('plotting light curve')
        plt.plot(Time - np.floor(Time[0]), LC,'.', label = 'Event LC',alpha=0.5, rasterized=True)
        #plt.axhline(np.nanmedian(LC) + 3*np.nanstd(LC), linestyle = '--', color = 'red')
        #print('plot lcs')
        # func_time = interp1d(np.where(np.isfinite(Time))[0],Time[np.isfinite(Time)],kind='cubic')
        # x = np.arange(0,len(Time),1)
        # i_time = func_time(x)

        xmin = Time[Eventtime[i][0]]-np.floor(Time[0])-(Eventtime[i][-1]-Eventtime[i][0])*(Time[1]-Time[0])*2
        if Eventtime[i][-1] < len(Time):
            xmax = Time[Eventtime[i][-1]]-np.floor(Time[0])+(Eventtime[i][-1]-Eventtime[i][0])*(Time[1]-Time[0])*2
        else:
            xmax = Time[-1]-np.floor(Time[0])
        if xmin < 0:
            xmin = 0
        if xmax > Time[-1] - np.floor(Time[0]):
            xmax = Time[-1] - np.floor(Time[0])
        if np.isfinite(xmin) & np.isfinite(xmax):
            plt.xlim(xmin,xmax) 

        lclim = np.copy(LC[Eventtime[i,0]:Eventtime[i,1]])

        temp = sorted(lclim[np.isfinite(lclim)].flatten())
        temp = np.array(temp)
        maxy  = temp[-1] # get 8th brightest point

        temp = sorted(LC[np.isfinite(LC)].flatten())
        temp = np.array(temp)
        miny  = temp[10] # get 10th faintest point

        ymin = miny - 0.1*miny
        ymax = maxy + 0.1*maxy

        plt.ylim(ymin,ymax)
        plt.legend(loc = 1)
        plt.minorticks_on()
        ylims, xlims = Cutout(Data,Mid)
        #print('main')
        # small subplot 1 Reference image plot
        ax = plt.subplot2grid((2,3), (0,2))
        plt.title('Reference')
        plt.imshow(np.nanmedian(Data,axis=(0)), origin='lower',vmin=0,vmax = maxcolor)
        plt.xlim(xlims[0],xlims[1])
        plt.ylim(ylims[0],ylims[1])
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='black')
        plt.colorbar(fraction=0.046, pad=0.04,extend='max')
        plt.plot(position[1],position[0],'r.',ms = 15, rasterized=True)
        plt.minorticks_on()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        # small subplot 2 Image of event
        ax = plt.subplot2grid((2,3), (1,2))
        plt.title('Event')
        plt.imshow(Data[np.where(Data*mask==np.nanmax(Data[Eventtime[i][0]:Eventtime[i][-1]]*mask))[0][0],:,:], origin='lower',vmin=0,vmax = maxcolor)
        plt.xlim(xlims[0],xlims[1])
        plt.ylim(ylims[0],ylims[1])
        current_cmap = plt.cm.get_cmap()
        current_cmap.set_bad(color='black')
        plt.colorbar(fraction=0.046, pad=0.04,extend='max')
        plt.plot(position[1],position[0],'r.',ms = 15, rasterized=True)
        plt.minorticks_on()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        
        
        #directory = Save_environment(Eventtime[i],brightnesses[i],Source[i],SourceType[i],Save)
            

        #plt.savefig(directory + 'tess' + TIC(File)+'_'+str(i)+'.pdf', bbox_inches = 'tight')
        
    return  

def Lightcurve(Data, Mask, Normalise = False):
    if type(Mask) == list:
        mask = np.zeros((Data.shape[1],Data.shape[2]))
        mask[Mask[0],Mask[1]] = 1
        Mask = mask*1.0
    Mask[Mask == 0.0] = np.nan
    LC = np.nansum(Data*Mask, axis = (1,2))
    LC[LC == 0] = np.nan
    for k in range(len(LC)):
        if np.isnan(Data[k]*Mask).all(): # np.isnan(np.sum(Data[k]*Mask)) & (np.nansum(Data[k]*Mask) == 0):
            LC[k] = np.nan
    if Normalise:
        LC = LC / np.nanmedian(LC)
    return LC

def Cutout(Data,Position):
    """
    Limit the imshow dimensions to 20 square.
    Inputs:
    -------
    Data        - 3d array
    Position    - list

    Output:
    -------
    cutout_dims - 2x2 array 
    """
    cutout_dims = np.array([[0, Data.shape[1]],[0, Data.shape[2]]])
    for i in range(2):
        if (Data.shape[i] > 19):
            dim0 = [Position[i][0] - 6, Position[i][0] + 6]

            bound = [(dim0[0] < 0), (dim0[1] > Data.shape[1])]

            if any(bound):
                if bound[0]:
                    diff = abs(dim0[0])
                    dim0[0] = 0
                    dim0[1] += diff

                if bound[1]:
                    diff = abs(dim0[1] - Data.shape[1])
                    dim0[1] = Data.shape[1]
                    dim0[0] -= diff

            cutout_dims[i,0] = dim0[0]
            cutout_dims[i,1] = dim0[1]
    return cutout_dims - 0.5

def pix2coord(x, y, mywcs):
    """
    Calculates RA and DEC from the pixel coordinates
    """
    wx, wy = mywcs.wcs_pix2world(x, y, 0)
    return np.array([float(wx), float(wy)])