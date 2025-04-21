from sklearn.cluster import KMeans
from .preproccessor import *
import cupy as cp


class SnowDetection():
    
    def __init__(self,filename,img_path,mask_inf,snow_baseline=0,random_state=11):
        
        super(SnowDetection, self).__init__()
        
        self.filename = filename
        self.img_path = img_path
        self.mask_inf = mask_inf
        self.snow_baseline = snow_baseline
        self.random_state = random_state
        

        
    def kmeans_classify(self):
              
        BGR_data = load(self.img_path,self.filename ,self.mask_inf,blur=True).img2BGRarray()
        k_means = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
        k_means.fit(BGR_data.get())
        if cp.nanmean(BGR_data[k_means.labels_ == 1]).item() > cp.nanmean(BGR_data).item():
            res = cp.sum(k_means.labels_)/len(k_means.labels_)
        else:
            res = 1-cp.sum(k_means.labels_)/len(k_means.labels_)

        res = res*100

                
        return res
 
    

