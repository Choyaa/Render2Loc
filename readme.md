# Render & Compare

This project is for cross-view 6-DoF visual localization. 

## Installation

The code is written by PYTHON, and you will need the fowllowing package or more to support the project:

```
numpy cv2 pyrender trimesh matplotlib h5py os pycolmap scipy xmltodict json
```
## Run
### AirLoc

To use this project, just run:

 ```
 Python run.py --config_file ./config/config.json
 ```
 
 The `.json` file has five part, the detail will mentioned in the blow.

The `confige.json` hase five part, including `read_phoneRTK`, `read_queryGT`, `GPS_extraction`, `read_queryPrior` and `read_uavPrior`.
### Make your own dataset

###


