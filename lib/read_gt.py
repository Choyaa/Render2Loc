from unicodedata import name
from scipy.spatial.transform import Rotation as R
import numpy as np
from pyproj import Proj
from pyproj import Transformer
from pyproj import CRS
import os
import pandas as pd
import xmltodict


# data = []

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def GPStoProj(lat,lon):
    crs_CGCS2000 = CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
    crs_WGS84 = CRS.from_epsg(4326)
    # crs_utm45N = CRS.from_epsg(32645)
    from_crs = crs_CGCS2000
    to_crs = crs_WGS84

    transformer = Transformer.from_crs(to_crs, from_crs)
    new_x, new_y = transformer.transform(lat, lon)

    return new_x, new_y

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
                
                from_crs = crs_CGCS2000
                to_crs = crs_WGS84
                # to_utm45N = crs_utm45N
                transformer = Transformer.from_crs(to_crs, from_crs)
                new_x,new_y = transformer.transform(xyz[0],xyz[1])
                euler = data_line[4:]
                ret = R.from_euler('zxy',[float(euler[0]),90-float(euler[1]), 90-float(euler[2])],degrees=True)
                Q= ret.as_quat()
                # print(data_line[0])
                # out_line = [data_line[0]]+list(Q)+list(xyz_w2c[0])+list(xyz_w2c[1])+list(xyz_w2c[2])
                out_line = [data_line[0]]+list(Q)+[new_x]+[new_y]+[xyz[2]]
                # print(out_line)
                out_line_str  = str(data_line[0])+' '+str(out_line[4])+' '+str(out_line[1])+' '+str(out_line[2])+' '+str(out_line[3])+' '+str(out_line[5])+' '+str(out_line[6])+' '+str(out_line[7])+' \n'
                # print(out_line_str)
                file_w.write(out_line_str)

def RtoQ_csv(write_path, input_path):

    pd_data = pd.read_csv(input_path)

    with open(write_path, 'w') as file_w:
        for i in range(len(pd_data['img_name'])):
            gps_lon = pd_data['gps_lon'][i]
            gps_lat = pd_data['gps_lat'][i]
            gps_alt = pd_data['new_gps_alt'][i]

            xyz = list(map(float, list([gps_lat, gps_lon, gps_alt])))

            crs_CGCS2000 = CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')
            crs_WGS84 = CRS.from_epsg(4326)

            from_crs = crs_CGCS2000
            to_crs = crs_WGS84

            transformer = Transformer.from_crs(to_crs, from_crs)
            new_x, new_y = transformer.transform(xyz[0], xyz[1])

            euler = [pd_data['yaw'][i], pd_data['pitch'][i], pd_data['roll'][i]]

            ret = R.from_euler('zxy', [90-float(euler[0]), 90-float(euler[1]), 90-float(euler[2])], degrees=True)
            Q = ret.as_quat()

            out_line = [pd_data['img_name'][i]] + list(Q) + [new_x] + [new_y] + [xyz[2]]
            out_line_str = str(pd_data['img_name'][i]) + ' ' + str(out_line[4]) + ' ' + str(out_line[1]) + ' ' + str(
                out_line[2]) + ' ' + str(out_line[3]) + ' ' + str(out_line[5]) + ' ' + str(out_line[6]) + ' ' + str(
                out_line[7]) + ' \n'
            file_w.write(out_line_str)

def write_in(write_path, input_path):

    pd_data = pd.read_csv(input_path)
    with open(write_path, 'w') as file_w:
        for i in range(len(pd_data['img_name'])):
            out_line_str = str(pd_data['img_name'][i]) + ' ' + 'PINHOLE' + ' ' + str(4096) + ' ' + str(3072) + ' ' + str(3195.77388912733) + ' ' + str(3195.77388912733) \
            + ' '+ str(2039.25221718194) + ' ' + str(1554.460779394) + '\n'

            file_w.write(out_line_str)

def write_ground_truth(root_path,pose_path,intrinsics_path,meta_path,Photogroup_id, input_path):

    file_object = open(input_path, encoding='utf-8')
    try:
        all_the_xmlStr = file_object.read()
    finally:
        file_object.close()
        # transfer to dict
    dictdata = dict(xmltodict.parse(all_the_xmlStr))
    with open(pose_path, 'w') as file_p:
        with open(intrinsics_path, 'w') as file_i:
            with open(meta_path, 'w') as file_m:
                for idx in Photogroup_id: 
                    tilponts = dictdata['BlocksExchange']['Block']['Photogroups']['Photogroup'][idx]
                    # focal_fixed = 63.6952
                    if idx ==0:
                        # # focal = 63.5837
                        # focal = focal_fixed
                        img_type = 'cloud'
                    elif idx ==6:
                        # # focal = 64.699
                        # focal = focal_fixed
                        img_type = 'sun'
                    elif idx ==7:
                        # # focal = 64.3744
                        # focal = focal_fixed
                        img_type = 'rain'
                    # else:
                        # focal = focal_fixed
                        # img_type = 'cloud'
                    w = int(tilponts['ImageDimensions']['Width'])
                    h = int(tilponts['ImageDimensions']['Height'])
                    # focalPixel= focal* (np.sqrt(h**2+ w**2)) /35.0
                    focalPixel= 1108.0
                    fx= fy = str(focalPixel)+' '
                    img_width = str(w)+' '
                    img_Height = str(h)+' '
                    CameraModelType = 'PINHOLE'
                    cx = str(tilponts['PrincipalPoint']['x'])+' '
                    cy = str(tilponts['PrincipalPoint']['y'])

                    img_list = tilponts['Photo']
                    with open(root_path+img_type+'_meta_pose.txt', 'w') as file_img:
                        for i in range(len(img_list)):
                            img_name = img_list[i]['ImagePath'].split('/')[-1]
                            print(img_name)
                            pose_M00 = img_list[i]['Pose']['Rotation']['M_00']
                            pose_M01 = img_list[i]['Pose']['Rotation']['M_01']
                            pose_M02 = img_list[i]['Pose']['Rotation']['M_02']
                            pose_M10 = img_list[i]['Pose']['Rotation']['M_10']
                            pose_M11 = img_list[i]['Pose']['Rotation']['M_11']
                            pose_M12 = img_list[i]['Pose']['Rotation']['M_12']
                            pose_M20 = img_list[i]['Pose']['Rotation']['M_20']
                            pose_M21 = img_list[i]['Pose']['Rotation']['M_21']
                            pose_M22 = img_list[i]['Pose']['Rotation']['M_22']

                            pose_x = img_list[i]['Pose']['Center']['x']
                            pose_y = img_list[i]['Pose']['Center']['y']
                            pose_z = img_list[i]['Pose']['Center']['z']

                            R_gt =  np.array([pose_M00,pose_M01,pose_M02,
                                                                pose_M10,pose_M11,pose_M12,
                                                                pose_M20,pose_M21,pose_M22],dtype=float)
                            T_gt = np.array([pose_x,pose_y,pose_z],dtype=float).squeeze()
                            
                            gt_qvec = rotmat2qvec(R_gt).squeeze()
                            gt_qvec = ' '.join(map(str, gt_qvec))
                            gt_tvec = ' '.join(map(str, T_gt))

                            metadata_M00 = img_list[i]['Pose']['Metadata']['Rotation']['M_00']
                            metadata_M01 = img_list[i]['Pose']['Metadata']['Rotation']['M_01']
                            metadata_M02 = img_list[i]['Pose']['Metadata']['Rotation']['M_02']
                            metadata_M10 = img_list[i]['Pose']['Metadata']['Rotation']['M_10']
                            metadata_M11 = img_list[i]['Pose']['Metadata']['Rotation']['M_11']
                            metadata_M12 = img_list[i]['Pose']['Metadata']['Rotation']['M_12']
                            metadata_M20 = img_list[i]['Pose']['Metadata']['Rotation']['M_20']
                            metadata_M21 = img_list[i]['Pose']['Metadata']['Rotation']['M_21']
                            metadata_M22 = img_list[i]['Pose']['Metadata']['Rotation']['M_22']

                            metadata_x = img_list[i]['Pose']['Metadata']['Center']['x']
                            metadata_y = img_list[i]['Pose']['Metadata']['Center']['y']
                            metadata_z = img_list[i]['Pose']['Metadata']['Center']['z']

                            lon = float(metadata_x)
                            lat = float(metadata_y)
                            
                            #WGS 84经纬度坐标转换为CGCS_2000_3_Degree_GK_CM_114E坐标
                            metadata_x,metadata_y = GPStoProj(lat,lon)
                            
                            R_meta =  np.array([metadata_M00,metadata_M01,metadata_M02,
                                                                metadata_M10,metadata_M11,metadata_M12,
                                                                metadata_M20,metadata_M21,metadata_M22],dtype=float)
                            T_meta = np.array([metadata_x,metadata_y,metadata_z],dtype=float)

                            meta_qvec = rotmat2qvec(R_meta).squeeze()
                            meta_qvec = ' '.join(map(str, meta_qvec))
                            meta_tvec = ' '.join(map(str, T_meta))

                            gt_line_str = f'{img_name} {gt_qvec} {gt_tvec}\n'
                            intrinsic_line_str = f'{img_name} {CameraModelType} {img_width}{img_Height}{fx}{fy}{cx}{cy}\n'
                            meta_line_str = f'{img_name} {meta_qvec} {meta_tvec}\n'
                        
                            file_p.write(gt_line_str)
                            file_i.write(intrinsic_line_str)
                            file_m.write(meta_line_str)
                            file_img.write(meta_line_str)

    file_p.close()
    file_i.close()
    file_m.close()
                    
                




if __name__ == '__main__':
    # RtoQ_csv('../txt/tyg_gps_result.txt', '../txt/test_construct11_8.csv')
    # write_in('../txt/tyg_gps_intrisics.txt', '../txt/test_construct11_8.csv')
    # 直接从XML文件中读取先验pose 和 GTpose以及内参
    xml_path = 'GT/Thermal_final/红外包含QUERY GT模型.xml'
    Photogroup_id = [0,6,7]
    root_path = 'GT/Thermal_final/'
    pose_path = 'GT/Thermal_final/gt_pose.txt'
    intrinsics_path = 'GT/Thermal_final/intrinsic.txt'
    meta_path = 'GT/Thermal_final/meta_pose.txt'
    write_ground_truth(root_path,pose_path,intrinsics_path,meta_path,Photogroup_id,xml_path)