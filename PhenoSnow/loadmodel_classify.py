import cupy as cp
import pandas as pd
import joblib
from rich.progress import track
from .preproccessor import *


class LoadModel():
    
    def __init__(self,site,file_list,img_path,out_path,mask_inf,model1,model2,GBCC_snow90,veg_type='SH',snow_baseline=0,random_state=11,write=True):
        
        super(LoadModel, self).__init__()
        self.site = site
        self.file_list = file_list
        self.img_path = img_path
        self.out_path = out_path
        self.mask_inf = mask_inf
        self.model1 = model1
        self.model2 = model2
        self.snow_baseline = snow_baseline
        self.random_state = random_state
        self.color1 = [255, 255, 255]
        self.color2 = [0, 0, 0]
        self.write = write
        self.veg_type = veg_type
        self.GBCC_snow90 = GBCC_snow90
        
    def classify(self):
        sgdocsvm_1 = joblib.load(self.model1)
        sgdocsvm_2 = joblib.load(self.model2)
        
        
        new_data = pd.DataFrame()
        for i in track(range(len(self.file_list)),description='Calculating Snow Ratio, Site:'+self.site):
            BGR_data = load(self.img_path, self.file_list[i], self.mask_inf).img2BGRarray()

            data_m = (BGR_data[:, 0] ** 2 + BGR_data[:, 1] ** 2 + BGR_data[:, 2] ** 2) ** (1 / 2)
            BGR_data0 = BGR_data / cp.tile(data_m.reshape(len(data_m), 1), 3)
            BGR_data0 = cp.nan_to_num(BGR_data0)
            L_data = load(self.img_path, self.file_list[i], self.mask_inf).img2Larray()


            BGR_data1 = BGR_data / cp.max(BGR_data).item()
            data = cp.column_stack((BGR_data0, L_data, BGR_data1))
            data_frame = pd.DataFrame(columns=['B0', 'G0', 'R0', 'L', 'B1', 'G1', 'R1'], data=data.get())

            res1 = sgdocsvm_1.predict(data_frame[sgdocsvm_1.feature_names_in_])
            res1[res1 < 1] = 0
            res2 = sgdocsvm_2.predict(data_frame[sgdocsvm_2.feature_names_in_])
            res2[res2 < 1] = 0
            res3 = res1 + res2
            res3[res3 > 1] = 1

            GBCC = cp.nanmean((BGR_data[:, 1] - BGR_data[:, 0]) / cp.sum(BGR_data, axis=1))
            if self.write == True:
                img_cv = cv.imread(self.img_path + self.file_list[i])
                mask_cv = cv.imread(self.mask_inf, 0)
                mask_cv = mask_cv + 1
                mask_cv[mask_cv > 0] = 255

                if img_cv.shape[:2] == mask_cv.shape:
                    mask = mask_cv
                else:
                    mask = cv.resize(mask_cv, dsize=(img_cv.shape[1], img_cv.shape[0]), interpolation=cv.INTER_NEAREST)

                img_svm1 = cp.zeros_like(img_cv)
                img_svm1[mask == 255] = cp.asarray([self.color2, self.color1])[res1]

                img_svm2 = cp.zeros_like(img_cv)
                img_svm2[mask == 255] = cp.asarray([self.color2, self.color1])[res2]

                img_svm1 = cp.uint8(img_svm1.get())
                img_svm2 = cp.uint8(img_svm2.get())
                dst = cv.bitwise_and(img_cv, img_cv, mask=mask)

                panorama = cv.hconcat([dst, img_svm1, img_svm2])
                os.makedirs(self.out_path + self.site + '/', exist_ok=True)
                cv.imwrite(self.out_path + self.site + '/' + self.file_list[i], panorama)
                
            file_inf = self.file_list[i].split('_')
            site = file_inf[0]
            year = file_inf[1]
            month = file_inf[2]
            day = file_inf[3]
            time = file_inf[4].split('.')[0]
            new_data.loc[i, 'site'] = site
            new_data.loc[i, 'year'] = year
            new_data.loc[i, 'month'] = month
            new_data.loc[i, 'day'] = day
            new_data.loc[i, 'time'] = time
            new_data.loc[i, 'GBCC'] = GBCC.item()
            new_data.loc[i, 'snow1'] = cp.sum(res1)
            new_data.loc[i, 'snow_ratio1'] = new_data.loc[i, 'snow1'] / len(res1)
            new_data.loc[i, 'snow2'] = cp.sum(res2)
            new_data.loc[i, 'snow_ratio2'] = new_data.loc[i, 'snow2'] / len(res2)
            new_data.loc[i, 'snow3'] = cp.sum(res3)
            new_data.loc[i, 'snow_ratio3'] = new_data.loc[i, 'snow3'] / len(res3)
            for i in range(len(new_data)):

                if self.veg_type == 'SH' or self.veg_type == 'DB':

                    if new_data.loc[i, 'snow_ratio1'] <= 0.01:
                        new_data.loc[i, 'snow_ratio'] = new_data.loc[i, 'snow_ratio1'] * 100
                    else:
                        new_data.loc[i, 'snow_ratio'] = new_data.loc[i, 'snow_ratio2'] * 100
                else:
                    if new_data.loc[i, 'GBCC'] < self.GBCC_snow90:
                        new_data.loc[i, 'snow_ratio'] = new_data.loc[i, 'snow_ratio3'] * 100
                    else:
                        if new_data.loc[i, 'snow_ratio1'] <= 0.01:
                            new_data.loc[i, 'snow_ratio'] = new_data.loc[i, 'snow_ratio1'] * 100
                        elif new_data.loc[i, 'snow_ratio2'] <= 0.01:
                            new_data.loc[i, 'snow_ratio'] = new_data.loc[i, 'snow_ratio2'] * 100
                        else:
                            new_data.loc[i, 'snow_ratio'] = new_data.loc[i, 'snow_ratio3'] * 100

        

                

        new_data.to_csv(self.out_path+self.site+'_Snow_Ratio_exmodel.csv',index=False)
        
        return new_data[['site','year','month','day','time','snow_ratio']]
        
        
