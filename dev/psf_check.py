import matplotlib.pyplot as plt
import numpy as np

from PRF import TESS_PRF
from scipy.ndimage import center_of_mass

from photutils.detection import StarFinder
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from copy import deepcopy

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))

from scipy.stats import pearsonr
def psflike_check(res, data,prf,corlim=0.8,psfdifflim=0.5):
    cors = []
    maxloc = []
    diff = []
    for i in range(len(res)):
        try:
            x = res[i]['xcentroid']
            y = res[i]['ycentroid']

            cut = deepcopy(data)[int(y)-2:int(y)+3,int(x)-2:int(x)+3]
            cut /= np.nansum(cut)
            cm = center_of_mass(cut)
            localpsf = prf.locate(cm[0],cm[1],(5,5))
            localpsf /= np.nansum(localpsf)
            if cut.shape == localpsf.shape:
                #ind = localpsf >= np.sort(localpsf.flatten())[-4]
                ind = localpsf >= np.sort(localpsf.flatten())[len(localpsf.flatten())//2]
                cor = pearsonr(cut[ind].flatten(),localpsf[ind].flatten()).correlation
                cors += [cor]#[np.nanmean(c)]
                locpsf = np.where(np.nanmax(localpsf) == localpsf)
                locdata = np.where(np.nanmax(cut) == cut)
                maxloc += [locpsf == locdata]
                diff += [np.nansum(abs(cut[ind]-localpsf[ind]))]
            else:
                cors += [0]
                maxloc += [False]
                diff += [2]
        except:
            cors += [0]
            maxloc += [False]
            diff += [2]
                
    cors = np.array(cors)
    cors = np.round(cors,2)
    maxloc = np.array(maxloc)
    diff = np.array(diff)
    ind = (cors >= corlim) & (diff < psfdifflim)
    return ind, cors,diff

def spatial_group(result):
    coords  = np.array([result.xint.values,result.yint.values]).T
    d = np.sqrt((result.xcentroid.values[:,np.newaxis] - result.xcentroid.values[np.newaxis,:])**2+ 
               (result.ycentroid.values[:,np.newaxis] - result.ycentroid.values[np.newaxis,:])**2)
    d2 = np.sqrt((result.xcentroid.values-45)**2+ 
               (result.ycentroid.values-45)**2)

    indo = d < 1
    detecs = np.nansum(indo,axis=1)
    positions = np.unique(indo,axis=1)
    counter = 0

    obj = np.zeros(result.shape[0],dtype=int)
    for i in range(positions.shape[1]):
        obj[positions[:,i]] = counter 
        counter += 1
    result['objid'] = obj
    return result
    
def source_detect(tess,corlim=0.8,psfdifflim=0.5):
    
    prf = TESS_PRF(tess.tpf.camera,tess.tpf.ccd,tess.tpf.sector,
                   tess.tpf.column+tess.flux.shape[2]/2,tess.tpf.row+tess.flux.shape[1]/2)
    psf = prf.locate(5,5,(11,11))
    result = None
    for i in range(len(tess.flux)):
        data = tess.flux[i]
        mean, med, std = sigma_clipped_stats(data, sigma=3.0)
        finder = StarFinder(med + 5*std,kernel=psf)
        res = finder.find_stars(deepcopy(data))
        psf_res = finder.find_stars(psf)
        if res is not None:
            res['frame'] = i
            ind, cors,diff = psflike_check(res,data,prf,corlim=corlim,psfdifflim=psfdifflim)
            res['psflike'] = cors
            res['psfdiff'] = diff
            res = res[ind]
            res = res.to_pandas()
            if result is not None:
                result = result.append(res)
            else:
                result = res
    result['xint'] = deepcopy(result['xcentroid'].values).astype(int)
    result['yint'] = deepcopy(result['ycentroid'].values).astype(int)
    ind = (result['xint'].values >5) & (result['xint'].values < data.shape[1]-5) & (result['yint'].values >5) & (result['yint'].values < data.shape[0]-5)
    result = result[ind]
    result = spatial_group(result)
    return result


