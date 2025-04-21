import os
import pandas as pd
from PhenoSnow import run
from PhenoSnow import loadmodel_classify


outp1 = ' ' #output folder1
outp2 = ' ' #output folder2


traning_path = ' '
test_path = ' '

site_list = pd.read_csv('site.csv')#sitelist
for i in range(len(site_list)):
    
    site = site_list.iloc[i,0]
    veg_type = site_list.loc[i,'veg_type']
    mask_site = site+'_mask.tif' #mask path
    out_path = outp1+site+'/'
    path_site = traning_path+site+'/'
    os.makedirs(out_path, exist_ok=True)
    res = run.snow_detection(site,veg_type,path_site,out_path,mask_site)

    m_path = outp1 + site + '/'
    path_site = test_path + site + '/'
    model1 = m_path + site + '_sgdvm1.pkl'
    model2 = m_path + site + '_sgdvm2.pkl'
    file_list = os.listdir(path_site)
    out_path = outp2 + site + '/'
    os.makedirs(out_path, exist_ok=True)
    res = loadmodel_classify.LoadModel(site, file_list, path_site, out_path, mask_site, model1, model2, model2,res[1],
                                       veg_type=veg_type).classify()
    


