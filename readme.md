# Geo Relocalization & Render (GRR)

This project is for xxx...

## 0. Requirment

The code is written by PYTHON, and you will need the fowllowing package or more to support the project:

```
numpy cv2 pyrender trimesh matplotlib h5py os pycolmap scipy xmltodict json
```
 ## 1. Run

To use this project, just run:

 ```
 Python run.py --config_file ./config/config.json
 ```
 
 The `.json` file has five part, the detail will mentioned in the blow.

## 2. Confige files
The `confige.json` hase five part, including `read_phoneRTK`, `read_queryGT`, `GPS_extraction`, `read_queryPrior` and `read_uavPrior`.


### 2.4 render_from_obj

该步骤根据无人机的先验，从gps中获取位置，通过云台的角度获取姿态。然后，根据先验位姿的视角在三维模型中渲染出一张图像，为后面的query图像的匹配和重定位提供参考帧（render_reference）

```json
"render_from_obj":{
    "input_recons" : "direct to obj",
    "origin_coord" : [445613.9, 4611391.76, 0.0],
    "save_path" : save path for render images,
    "input_pose" : transdered posed from SRT files（query图的先验pose）
},
```

### 2.5 render2loc

该步骤将上一步所渲染的图像，与query图像进行匹配。然，根据渲染图对应的深度图和pose计算出匹配点所对应的三维点坐标，再通过pycolmap的PnP解算，将query图的pose计算出来。最后，根据计算的pose再渲染出该视角的图像，用于视频帧与三维场景的叠加。

```json
"render2loc":{
    "model_ckpt":"./weights/outdoor_ds.ckpt"(Loftr的与训练模型),
    "query_pth": "direct to query images"(query图像的路径),
    "results": "./results/loc_results/query_loc_pose.txt"(query图的精确pose),
    "input_recons" : "direct to obj",(obj模型的存储路径)
    "origin_coord" : [445613.9, 4611391.76, 0.0],
    "save_path" : "./results/loc2render/"（根据所计算的query图的精确pose最后输出的渲染图像）
    }
```

## 3. Output
+ 输出：均保存在results文件夹中，包括
    + query图像的pose
    + query图视角对应的深度图
    + query图视角对应的渲染图
