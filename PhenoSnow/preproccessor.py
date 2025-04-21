import cv2 as cv
import cupy as cp

class load():
    def __init__(self,img_path,file,mask,blur=True):
        
        super(load, self).__init__()
        self.img_cv = cv.imread(img_path+file)
        self.mask_cv = cv.imread(mask,0)
        self.img_shape = self.img_cv.shape
        self.blur = blur

    def create_mask(self):

        mask = self.mask_cv + 1

        return mask

    def mask_resize(self):

        mask = self.mask_cv + 1
        mask2 = cv.resize(mask, dsize=(self.img_shape[1], self.img_shape[0]), interpolation=cv.INTER_NEAREST)

        return mask2

    def img2BGRarray(self):

        if self.img_cv.shape[:2] == self.mask_cv.shape:
            mask = self.create_mask()
        else:
            mask = self.mask_resize()
        if self.blur==True:
            img = cv.GaussianBlur(self.img_cv, ksize=(5, 5), sigmaX=0, sigmaY=0)
        else:
            img = self.img_cv
        BGR_data = cp.asarray(img[mask == 1], dtype=cp.int64)

        return BGR_data

    def img2Larray(self):

        if self.img_cv.shape[:2] == self.mask_cv.shape:
            mask = self.create_mask()
        else:
            mask = self.mask_resize()
        if self.blur == True:
            img = cv.GaussianBlur(self.img_cv, ksize=(5, 5), sigmaX=0, sigmaY=0)

        lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)

        L_data = cp.asarray(lab[mask == 1], dtype=cp.int64)[:, 0]/255

        return L_data

class fog():
    
    def __init__(self,img_path,file,mask,thr_gray=60,thr_lp=600):
        
        super(fog, self).__init__()
        self.img_cv = cv.imread(img_path+file)
        self.img_gray = cv.cvtColor(self.img_cv, cv.COLOR_BGR2GRAY)
        self.mask_cv = cv.imread(mask,0)
        self.thr_gray = thr_gray
        self.thr_lp = thr_lp
        if self.img_cv.shape[:2] == self.mask_cv.shape:
            self.mask_inf = load(img_path,file,mask).create_mask()
        else:
            self.mask_inf = load(img_path,file,mask).mask_resize()
            
    def gray(self):

        gray_roi = cp.asarray(self.img_gray[self.mask_inf == 1], dtype=cp.int64)
        return cp.nanmean(gray_roi)

    def fuzzy(self):
        
        lp=cv.Laplacian(self.img_gray, cv.CV_64F)
        lp_roi = cp.asarray(lp[self.mask_inf == 1], dtype=cp.float64)

        return cp.nanvar(lp_roi)
    
    def label(self):
        
        if self.gray()<self.thr_gray and self.fuzzy()<self.thr_lp:
            return 1
        else:
            return 0
        
