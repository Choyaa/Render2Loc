from venv import EnvBuilder
from cv2 import ROTATE_180
from scipy.spatial.transform import Rotation as R
import numpy as np
from pyproj import Proj
from pyproj import Transformer
from pyproj import CRS
import os
# data = []

def RtoQ(write_path, input_path):
    if os.path.isfile(write_path):
        os.remove(write_path)
    with open(write_path,'w') as file_w:
        with open(input_path,'r') as file:
            for line in file:
                data_line=line.strip("\n").split(' ')
                # print(data_line)

                xyz = list(map(float, list(data_line[1:4]))) # W2C                
                crs_CGCS2000=CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')    #degree
                # crs_CGCS2000=CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101,AUTORITY["EPSG", "1024"]],AUTHORITY["EPSG", "1043"]],PRIMEM["Greenwich",0.0,AUTHORITY["EPSG", "8901"]],UNIT["Degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4490"]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0,AUTHORITY["EPSG",“9001”]],AUTHORITY["EPSG","4547"]]')    #degree
                # crs_utm45N = CRS.from_epsg(32645)
                crs_WGS84=CRS.from_epsg(4326)
               
               # to_utm45N = crs_utm45N 
                # from_crs = crs_CGCS2000
                to_crs = crs_WGS84
                
                transformer = Transformer.from_crs(to_crs, crs_CGCS2000)
                new_x,new_y = transformer.transform(xyz[0],xyz[1])
                euler = data_line[4:]
                ret = R.from_euler('zxy',[float(euler[0]),90-float(euler[1]),float(euler[2])],degrees=True)
                Q= ret.as_quat()

                out_line = [data_line[0]]+list(Q)+[new_x]+[new_y]+[xyz[2]]
                out_line_str  = str(data_line[0])+' '+str(out_line[4])+' '+str(out_line[1])+' '+str(out_line[2])+' '+str(out_line[3])+' '+str(out_line[5])+' '+str(out_line[6])+' '+str(out_line[7])+' \n'

                file_w.write(out_line_str)