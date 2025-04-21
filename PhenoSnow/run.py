import os
from PhenoSnow import cal_GBCC,snowdetection,trainingset,selfadp
import pandas as pd


def snow_detection(site,veg_type,path_site,out_path,mask_site):

    if os.path.exists(out_path+site+'_GBCC.csv'):
        GBCC_data = pd.read_csv(out_path+site+'_GBCC.csv')
    else:
        GBCC_data = cal_GBCC.snow_sts(site,path_site,out_path,mask_site)

    if os.path.exists(out_path+site+'_training.csv'):
        training_data = pd.read_csv(out_path+site+'_training.csv')
    else:
        training_data = trainingset.CreateTrainingSet(site,GBCC_data,path_site,out_path,mask_site).create_training()
    if os.path.exists(out_path+site+'_nu1.csv'):
        nu1 = pd.read_csv(out_path+site+'_nu1.csv').opt_nu1[0]

    else:
        nu1 = selfadp.ParaAdp(site,path_site,out_path,mask_site,GBCC_data,training_data).Search_nu1()[0]

    if os.path.exists(out_path+site+'_nu2.csv'):
        nu2 = pd.read_csv(out_path+site+'_nu2.csv').opt_nu2[0]
    else:
        nu2 = selfadp.ParaAdp(site,path_site,out_path,mask_site,GBCC_data,training_data).Search_nu2(nu1=nu1)[0]


 
    file_list = os.listdir(path_site)
    
    results = snowdetection.SnowDetection(site,file_list,path_site,out_path,mask_site,training_data,nu1,nu2,veg_type).classify()
    
    return results


