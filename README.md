# ASDF
Codebase supporting "An automatic snow detection framework from repeat digital photography".
The automatic snow detection framework (ASDF) aims to estimate the fractional snow cover (FSC) from repeat digital photography for plant functional types (PFTs). ASDF provides a fully automatic processing flow without manually labeled samples, which makes ASDF much more convenient for numerous repeat digital images. The framework uses machine learning to obtain adjustable classification criteria automatically rather than using constant thresholds, resulting in a pixel-level classification for images. The ASDF was designed based on PhenoCam Network images and has been modeled and tested in 6 PFTs, including agriculture, grassland, tundra, shrub, deciduous broadleaf, and evergreen needleleaf. It has higher accuracy than previous unsupervised methods, especially for vegetation with complex surfaces. 

Requirements
Python 3.10
opencv-python==4.7.0.72
numpy==1.26.0
cupy-cuda11x==13.4.0
pandas==2.2.3
scikit-learn==1.3.0
scipy==1.11.3
joblib==1.2.0
rich==13.3.5



Imagery Dataset
The images used in this study are available from PhenoCam Dataset V2.0 (Seyednasrollah et al., 2019) and the PhenoCam Network (https://phenocam.nau.edu/webcam/network/download/). The detailed information on images used in this study can be found in image_list.csv.

├── image_list/     # Images used in this study
│   ├── site_name /   # Images of the site “site_name”
├── PhenoSnow/                # source code of ASDF
├── test_run.py            # Test script
│   ├── mask /   # masks used in this study
│   │   ├── site_name_mask.tif # Mask corresponding to the site “site_name”
├── output/             # The intermediate output file of the algorithm
│   ├── construction /   # Intermediate and output files in model construction
│   │   ├── site_name/# Files corresponding to the site “site_name”
│   │   │   ├── site_name/ #Examples of the original image with output of SVM1 and SVM2
│   │   │   ├── enhance/ Examples of enhanced image
│   │   │   ├── site_name_GBCC.csv # GBCC results 
│   │   │   ├── site_name_training.csv #Automatically generated training set (Formal example)
│   │   │   ├── site_name_file1.csv #file list of automatically generated validation dataset for searching nu1
│   │   │   ├── site_name_file2.csv # file list of automatically generated validation dataset for searching nu2
│   │   │   ├── site_name_nu1.csv # nu1 and error
│   │   │   ├── site_name_nu2.csv # nu2 and error
│   │   │   ├── site_name_sgdvm1.pkl # Trained model (SVM1)
│   │   │   ├── site_name_sgdvm2.pkl # Trained model (SVM2)
│   │   │   ├── site_name_Snow_Ratio.csv # FSC results of construction dataset
│   ├── test /   # output files for model test
│   │   ├── site_name_Snow_Ratio_exmodel.csv # FSC results of test dataset
├── demo/  # detect FSC using trained model
│   ├── input /
│   ├── output /
│   ├── mask.tif
│   ├── model1.pkl
│   ├── model2.pkl
│   ├── demo.py  #Run this script
└── requirements.txt     # development environment


