import os
import cupy as cp
import pandas as pd
from rich.progress import track
from .preproccessor import *


def GBCC(BGR_data):
    GBCC = (BGR_data[:, 1] - BGR_data[:, 0]) / cp.sum(BGR_data, axis=1)
    GCC = BGR_data[:, 1] / cp.sum(BGR_data, axis=1)
    GBCC_r = cp.count_nonzero(GBCC<0)/len(GBCC)


    return cp.nanmean(GBCC),GBCC_r,cp.nanmean(GCC)




def snow_sts(site,path,outpath,mask_inf):
    
    file_list = os.listdir(path)
    data = pd.DataFrame()
    for i in track(range(len(file_list)),description='Calculating GBCC, Site:'+site):
        filename = file_list[i]
        file_inf = filename.split('_')

        RGB_data = load(path,filename,mask_inf,blur=True).img2BGRarray()
        
        data.loc[i,'file'] = filename
        data.loc[i,'site'] = file_inf[0]
        data.loc[i,'year'] = file_inf[1]
        data.loc[i,'month'] = file_inf[2]
        data.loc[i,'day'] = file_inf[3]
        data.loc[i,'time'] = file_inf[4].split('.')[0]
        data.loc[i,'GBCC'] = GBCC(RGB_data)[0].item()
        data.loc[i,'GBCC_r'] = GBCC(RGB_data)[1].item()
        data.loc[i,'GCC'] = GBCC(RGB_data)[2].item()
        data.loc[i,'fog'] = fog(path,filename,mask_inf).label()
        pass


    
    data.to_csv(outpath+site+'_GBCC.csv',index=False)
        
    
    return data
    