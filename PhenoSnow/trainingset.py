import pandas as pd
from rich.progress import track
from sklearn.cluster import KMeans
from .preproccessor import *
import cupy as cp

class CreateTrainingSet():

    def __init__(self, site, GBCC_data, img_path, out_path, mask_inf, snow_baseline=0, img_baseline=10, random_state=11):

        super(CreateTrainingSet, self).__init__()
        self.site = site
        self.GBCC_data = GBCC_data
        self.img_path = img_path
        self.out_path = out_path
        self.mask_inf = mask_inf
        self.snow_baseline = snow_baseline
        self.img_baseline = img_baseline
        self.random_state = random_state
        self.snow_baseline = snow_baseline

    def create_training(self):

        self.site_snow = self.GBCC_data[(self.GBCC_data['fog'] == 0) & (
            self.GBCC_data['GBCC'] < self.snow_baseline)]


        self.site_snow.reset_index(drop=True, inplace=True)

        file = self.site_snow.loc[:, 'file'].values

        training = pd.DataFrame()

        for i in track(range(len(file)), description='Creating The Training Set, Site:'+self.site):
            BGR_data = load(self.img_path, file[i], self.mask_inf).img2BGRarray()
            data_m = (BGR_data[:, 0] ** 2 + BGR_data[:, 1] ** 2 + BGR_data[:, 2] ** 2) ** (1 / 2)
            BGR_data0 = BGR_data / cp.tile(data_m.reshape(len(data_m), 1), 3)
            BGR_data0 = cp.nan_to_num(BGR_data0)
            L_data = load(self.img_path, file[i], self.mask_inf).img2Larray()
            BGR_data1 = BGR_data / cp.max(BGR_data).item()
            all_inf =  cp.column_stack((BGR_data0, L_data,BGR_data1,BGR_data))


            k_means = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)

            k_means.fit(BGR_data.get())

            if cp.nanmean(BGR_data[k_means.labels_==1]).item()>cp.nanmean(BGR_data).item():
                tra_data = all_inf[k_means.labels_==1]
            else:
                tra_data = all_inf[k_means.labels_==0]

            tra_frame = pd.DataFrame(columns=['B0','G0','R0','L','B1','G1','R1','B','G','R'],data=tra_data.get())




            if i != 0:
                training = pd.concat([training, tra_frame],
                                     axis=0, ignore_index=True)
                training.drop_duplicates(keep='first', inplace=True)
                training.reset_index(drop=True, inplace=True)
            else:
                training = tra_frame
                pass
            pass

        training.dropna(axis=0, inplace=True)
        training.to_csv(self.out_path+self.site+'_training.csv',index=False)
        return training
