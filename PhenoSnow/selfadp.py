import os
import cupy as cp
import numpy as np
import pandas as pd
from rich.progress import track
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse_score
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
from .preproccessor import *
from PhenoSnow import kmeansdec


def fill_missing_with_threshold(data_series, max_consecutive=14):


    mask_na = data_series.isna()
    groups = mask_na.ne(mask_na.shift()).cumsum()
    block_sizes = mask_na.groupby(groups).transform('sum')
    
    exclude_mask_na = (block_sizes > max_consecutive) & mask_na
    temp_data = data_series.where(~exclude_mask_na, np.inf)
    
    filled_data = temp_data.interpolate(method='linear')
    filled_data = filled_data.replace(np.inf, np.nan)
    
    return filled_data


class ParaAdp():
    
    def __init__(self,site,img_path,out_path,mask_inf,site_GBCC,training_data,add_path='None',snow_baseline=0,random_state=11,img_enhance=True):
        
        super(ParaAdp, self).__init__()
        self.site = site
        self.img_path = img_path
        self.out_path = out_path
        self.mask_inf = mask_inf
        self.site_GBCC = site_GBCC[site_GBCC['fog']==0]
        self.site_GBCC.reset_index(drop=True,inplace=True)
        self.data = training_data
        self.add_path = add_path
        self.snow_baseline = snow_baseline
        self.random_state = random_state
        self.vec1 = ['R0', 'G0', 'B0']
        self.vec2 = ['B1', 'B0']
        self.site_snow = site_GBCC[(site_GBCC['fog']==0)&(site_GBCC['GBCC']<self.snow_baseline)]
        self.site_snow.reset_index(drop=True,inplace=True)
        self.img_enhance=img_enhance
       
    def single_svm(self,svm_file_list,model,vec):
        new_data = pd.DataFrame()
        for i in track(range(len(svm_file_list)),description='Calculating Single_SVM',transient=True):

            if svm_file_list.loc[i,'img_type']=='or_img':
                BGR_data = load(self.img_path,svm_file_list.loc[i,'file'],self.mask_inf).img2BGRarray()
            else:
                BGR_data = load(self.out_path+'enhance/', svm_file_list.loc[i, 'file'], self.mask_inf).img2BGRarray()
            data_m = (BGR_data[:, 0] ** 2 + BGR_data[:, 1] ** 2 + BGR_data[:, 2] ** 2) ** (1 / 2)
            BGR_data0 = BGR_data / cp.tile(data_m.reshape(len(data_m), 1), 3)
            BGR_data0 = cp.nan_to_num(BGR_data0)
            #L_data = load(self.img_path, svm_file_list.loc[i,'file'], self.mask_inf).img2Larray()
            BGR_data1 = BGR_data / cp.max(BGR_data).item()
            data = cp.column_stack((BGR_data0, BGR_data1))
            data_frame = pd.DataFrame(columns=['B0', 'G0', 'R0', 'B1', 'G1', 'R1'], data=data.get())


            res = model.predict(data_frame[vec])

            res[res < 1] = 0

            file_inf = svm_file_list.loc[i,'file'].split('_')
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
            new_data.loc[i, 'snow'] = cp.sum(res)
            new_data.loc[i, 'snow_ratio'] = new_data.loc[i, 'snow'] / len(res)
            new_data.loc[i, 'snow_ratio_singlesvm'] = new_data.loc[i, 'snow_ratio']*100
            new_data.loc[i, 'img_type'] = svm_file_list.loc[i,'img_type']
        return new_data[['site','year','month','day','time','snow_ratio_singlesvm','img_type']]

    def Validation_nu1(self):
        
        print('Prepare Validation for Searching nu1')

        nu_snow_data = self.site_snow.copy()
        nu_snow_data['nu1_label'] = 1
        
        nu_snow0_data = self.site_GBCC.copy()
        nu_snow0_data['date'] = nu_snow0_data['year'].astype(str)+'-'+nu_snow0_data['month'].astype(str)+'-'+nu_snow0_data['day'].astype(str)
        nu_snow0_data['date'] = pd.to_datetime(nu_snow0_data['date'])
        
        time_start = min(nu_snow0_data['date'])
        time_end = max(nu_snow0_data['date'])
        timet = pd.date_range(time_start,time_end)
        timetdt = pd.DataFrame(columns=['date'],data=timet)
        

        
        nu_snow0_data = timetdt.merge(nu_snow0_data,how='left',on=['date'])
        nu_snow0_data['GCC_SG'] = savgol_filter(fill_missing_with_threshold(nu_snow0_data['GCC']), window_length=15, polyorder=3,mode='nearest')
        
        GCC_thr = 0.9*(nu_snow0_data['GCC_SG'].max()-nu_snow0_data['GCC_SG'].min())+nu_snow0_data['GCC_SG'].min()
        
        nu_snow0_data = nu_snow0_data[nu_snow0_data['GBCC']>0]

        nu_snow0_data = nu_snow0_data[nu_snow0_data['GCC']>GCC_thr]
        
        nu_snow0_data = nu_snow0_data[nu_snow0_data['month'].astype(int)>5]
        nu_snow0_data = nu_snow0_data[nu_snow0_data['month'].astype(int)<9]
        nu_snow0_data.reset_index(drop=True,inplace=True)
        nu_snow0_data['nu1_label'] = 0
        if len(nu_snow0_data)>2*len(nu_snow_data):

            nu_snow0_data1 = pd.DataFrame()
            for m in range(6,9):
                if len(nu_snow0_data[nu_snow0_data['month']==m])==1:
                    nu_snow0_data1 = pd.concat([nu_snow0_data1,nu_snow0_data[nu_snow0_data['month']==m]],ignore_index=True)
                    nu_snow0_data.drop(index=nu_snow0_data[nu_snow0_data['month'] == m].index,inplace=True)

            nu_snow0_data.reset_index(drop=True, inplace=True)
            nu_snow0_data = train_test_split(nu_snow0_data, test_size=2*len(nu_snow_data)-len(nu_snow0_data1), stratify=nu_snow0_data['month'], random_state=self.random_state)[1]
            nu_snow0_data = pd.concat([nu_snow0_data,nu_snow0_data1],ignore_index=True)

        nu1_test_data = pd.concat([nu_snow_data[['file','site','year','month','day','time','nu1_label']],nu_snow0_data[['file','site','year','month','day','time','nu1_label']]],ignore_index=True)
        nu1_test_data['img_type'] = 'or_img'

        if self.img_enhance==True:

            nu_snow0_data_eh = nu_snow0_data.copy()
            nu_snow0_data_eh['img_type'] = 'eh_img'

            nu1_test_data = pd.concat(
                [nu1_test_data[['file', 'site', 'year', 'month', 'day', 'time', 'nu1_label', 'img_type']],
                 nu_snow0_data_eh[['file', 'site', 'year', 'month', 'day', 'time', 'nu1_label', 'img_type']]], ignore_index=True)

            for n in range(len(nu_snow0_data_eh)):

                if os.path.exists(self.out_path+'enhance/'+nu_snow0_data.loc[n,'file']):
                    pass
                else:
                    img_or = cv.imread(self.img_path+nu_snow0_data.loc[n,'file'])
                    img_eh = img_or[:, :, [0, 0, 2]]
                    os.makedirs(self.out_path+'enhance/', exist_ok=True)
                    cv.imwrite(self.out_path+'enhance/'+nu_snow0_data.loc[n,'file'], img_eh)

        nu1_test_data.to_csv(self.out_path + self.site + '_file1.csv', index=False)

        return nu1_test_data
    
    

    def Search_nu1(self,nu1_lower=0.05,nu1_upper=0.7,interval=0.05,mini=True):
        
        valdata = self.Validation_nu1()
        valdata[['year', 'month', 'day', 'time']] = valdata[['year', 'month', 'day', 'time']].astype(int)
        file_list = valdata[['file','img_type']]

        nu1_list = np.arange(nu1_lower,nu1_upper+interval,interval)
        if mini==True and nu1_list[0]>0.01:
            nu1_list = np.insert(nu1_list,0,0.01)
            
        nu1_error = 9999
        

        print('Searching nu1')
        for j in nu1_list:
          
            sgdocsvm_1 = linear_model.SGDOneClassSVM(nu=j,random_state=self.random_state)

            sgdocsvm_1.fit(self.data[self.vec1])

            
            al_data = self.single_svm(file_list,sgdocsvm_1,self.vec1)
            al_data[['year', 'month', 'day', 'time']] = al_data[['year', 'month', 'day', 'time']].astype(int)
            temp_data = valdata.merge(al_data,how='left',on=['site','year','month','day','time','img_type'])
            x = temp_data['nu1_label'].values
            y = temp_data['snow_ratio_singlesvm'].values
            y[y <= 1] = 0
            y[y > 1] = 9
            xy = x + y
            err_nu1 = np.count_nonzero(xy == 9) / np.count_nonzero(x == 0) - np.count_nonzero(
                xy == 10) / np.count_nonzero(x == 1)
            mis_err = np.count_nonzero(xy == 9) / np.count_nonzero(x == 0)

            if err_nu1 < nu1_error:
                nu1_error = err_nu1
                opt_nu1 = j
                lb=1
            elif mis_err>0.5:
                nu1_error = err_nu1
                opt_nu1 = j
                lb=2
            elif lb==2:
                nu1_error = err_nu1
                opt_nu1 = j
            else:
                break

            if err_nu1 == -1:
                break
        print('Done','nu1=',str(opt_nu1))
        temp_data.to_csv(self.out_path + self.site + '_file1_temp.csv', index=False)
        nu1_res = pd.DataFrame(columns=['opt_nu1','nu1_error'],data=[[opt_nu1,nu1_error]])
        nu1_res.to_csv(self.out_path + self.site + '_nu1.csv', index=False)
        return opt_nu1,nu1_error


    def Validation_nu2(self,model='None'):
        print('Prepare Validation for Searching nu2')


        file_list2data = self.site_GBCC.copy()
        file_list2data = file_list2data[(file_list2data['month'].astype(int)>8) | (file_list2data['month'].astype(int)<6)]
        file_list2data.reset_index(drop=True,inplace=True)
        file_list2data['img_type'] = 'or_img'
        file_list2 = file_list2data[['file','img_type']]

        if model=='None':
            sgdocsvm_1 = linear_model.SGDOneClassSVM(nu=self.Search_nu1()[0],random_state=self.random_state)
            sgdocsvm_1.fit(self.data[self.vec1])
        else:
            sgdocsvm_1 = model

        file_list2snow = self.single_svm(file_list2,sgdocsvm_1,self.vec1)
        file_list2snow[['year','month','day','time']] = file_list2snow[['year','month','day','time']].astype(int)
        file_list2data[['year', 'month', 'day', 'time']] = file_list2data[['year', 'month', 'day', 'time']].astype(int)

        file_list2snow = file_list2snow.merge(file_list2data,how='left',on=['site','year','month','day','time','img_type'])
        file_list2snow = file_list2snow[file_list2snow['snow_ratio_singlesvm']>30]
        file_list2snow.reset_index(drop=True,inplace=True)
        if os.path.exists(self.out_path+self.site+'_kmeans.csv'):
            file_kmeans = pd.read_csv(self.out_path+self.site+'_kmeans.csv')
            file_kmeans = file_kmeans[['file','kmeans']]
            file_list2snow = file_list2snow.merge(file_kmeans,how='left',on='file')
        else:
            for n in track(range(len(file_list2snow)),description='Calculating Kmeans',transient=True):
                file_list2snow.loc[n,'kmeans'] = kmeansdec.SnowDetection(file_list2snow.loc[n,'file'],self.img_path,self.mask_inf).kmeans_classify()

        file_list2snow = file_list2snow[file_list2snow['kmeans']>30]
        file_list2snow.loc[file_list2snow[file_list2snow['kmeans']>60].index,'snow_es'] = file_list2snow[['kmeans','snow_ratio_singlesvm']].max(axis=1)
        file_list2snow.loc[file_list2snow[file_list2snow['kmeans']<=60].index,'snow_es'] = file_list2snow[['kmeans','snow_ratio_singlesvm']].min(axis=1)
        file_list2snow.reset_index(drop=True, inplace=True)
        file_list2snow.to_csv(self.out_path+self.site+'_file2.csv',index=False)

        return file_list2snow
    
    
    
    def Search_nu2(self,nu2_lower=0.1,nu2_upper=0.5,interval=0.1,nu1='None'):

        if nu1=='None':
            nu1=self.Search_nu1()[0]
        else:
            nu1=nu1
        
        sgdocsvm_1 = linear_model.SGDOneClassSVM(nu=nu1,random_state=self.random_state)
        sgdocsvm_1.fit(self.data[self.vec1])    
        
        nu2_list = np.arange(nu2_lower,nu2_upper+interval,interval)
        nu2_error=9999
        if os.path.exists(self.out_path+self.site+'_file2.csv'):
            file_list2snow = pd.read_csv(self.out_path+self.site+'_file2.csv')
        else:
            file_list2snow = self.Validation_nu2(model=sgdocsvm_1)

        file_list2snow['img_type'] = 'or_img'
        file_list2snow.rename(columns={'snow_ratio_singlesvm':'snow_ratio_singlesvm1'},inplace=True)
        file_list2snow[['year', 'month', 'day', 'time']] = file_list2snow[['year', 'month', 'day', 'time']].astype(int)
        file_snowlist2 = file_list2snow[['file','img_type']]

        print('Searching nu2')
        for k in nu2_list:

            sgdocsvm_2 = linear_model.SGDOneClassSVM(nu=k, random_state=self.random_state)

            sgdocsvm_2.fit(self.data[self.vec2])

            al_data = self.single_svm(file_snowlist2, sgdocsvm_2, self.vec2)
            al_data[['year', 'month', 'day', 'time']] = al_data[['year', 'month', 'day', 'time']].astype(int)

            temp_data = file_list2snow.merge(al_data,how='left',on=['site','year','month','day','time','img_type'])
            temp_data.to_csv(self.out_path+self.site+'_file2_temp.csv',index=False)
            err_nu2 = mse_score(temp_data['snow_es'],temp_data['snow_ratio_singlesvm'],squared=False)
            if err_nu2 < nu2_error:
                nu2_error = err_nu2
                opt_nu2 = k
            else:
                break
        print('Done','nu2=',str(opt_nu2))
        nu2_res = pd.DataFrame(columns=['opt_nu2', 'nu2_error'], data=[[opt_nu2, nu2_error]])
        nu2_res.to_csv(self.out_path + self.site + '_nu2.csv', index=False)
        return opt_nu2,nu2_error


