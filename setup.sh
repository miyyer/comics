#!/bin/bash

mkdir ./data
mkdir ./logs

# download page images
wget https://obj.umiacs.umd.edu/comics/raw_page_images.tar.gz

# download ocr 
wget https://obj.umiacs.umd.edu/comics/COMICS_ocr_file.csv -P ./data/

# download ad pages to filter out
wget https://obj.umiacs.umd.edu/comics/predadpages.txt -P ./data/

# download panel images
wget https://obj.umiacs.umd.edu/comics/raw_panel_images.tar.gz -P ./data/

# download vgg features
wget https://obj.umiacs.umd.edu/comics/vgg_features.h5 -P ./data/

# untar 
tar -xvzf ./data/raw_panel_images.tar.gz -C ./data/

# export PYTHONPATH
PWD=$(pwd)
export PYTHONPATH=$PYTHONPATH:$PWD

# preprocess the raw data to create hdf5
python create_hdf5.py
