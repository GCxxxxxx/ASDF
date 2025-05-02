import os
from PhenoSnow import loadmodel_classify

output = './output/'
site = 'burnssagebrush'
veg_type = 'SH'
inpu_folder = './input/'
mask_site = './burnssagebrush_mask.tif'
model1 = './burnssagebrush_sgdvm1.pkl'
model2 = './burnssagebrush_sgdvm2.pkl'

file_list = os.listdir(inpu_folder)

GBCC90 = 0.07571148692685614    # An output value during model construction
res = loadmodel_classify.LoadModel(site, file_list, inpu_folder, output, mask_site, model1, model2, GBCC90,veg_type=veg_type).classify()





