
import logging
from typing import Dict, List, Union, Any
from pathlib import Path
from collections import defaultdict
import numpy as np
import h5py
import os
import torch
def load_local_hdf5(name, path: Path) -> Dict[str, Any]:
    data = {}
    with h5py.File(path, 'r') as hfile:
        grp = hfile[name]
        for k, v in grp.items():
            data[k] = hfile[name][k][:]
            
    return data
def read_instrincs(intrinsc_path):
    all_query_name =[]
    with open(intrinsc_path,'r') as file:
        for line in file:
            data_line=line.strip("\n").split(' ')
            
            img_name = os.path.basename(data_line[0])
            w,h,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]   #! :8
            all_query_name.append(img_name)
    cfg = {
        'model': 'PINHOLE',
        'width': int(w),
        'height': int(h),
        'params': [fx, fy, cx, cy]
            }    
    return cfg, all_query_name

input = "/home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization_github_complete/yanhao.h5"
intrinsics_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization_github_complete/AirLoc/queries_intrinsics/que/phone_day1_intrinsics.txt"
cfg, img_name = read_instrincs(intrinsics_path)
for query in img_name:
    corres = load_local_hdf5(query, input)  
    print(len(corres['points_2D']), len(corres['points_3D']))
    import ipdb; ipdb.set_trace();