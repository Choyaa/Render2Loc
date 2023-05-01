from tkinter import image_types
from unicodedata import name
import exifread
import os
import glob
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyproj import CRS
from scipy.spatial.transform import Rotation as R
import xmltodict

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

def rotation_to_quat (rotation_metrix):
    a = []
    # for _,v1 in rotation_metrix.items():
    for v1 in rotation_metrix:
        a.append(float(v1))
    a_np = np.array(a)
    a_np = a_np.reshape(3, 3)
    a_qvec = rotmat2qvec(a_np)# w,x,y,z
    return a_qvec, a_np

def read_file(img_file):
    """
    :param img_file:
    :return:
    """

    img_list = []
    img_list += glob.glob(os.path.join(img_file, "*.{}").format('jpg'))

    if len(img_list) != 0:
        return img_list
    else:
        print("img_file is wrong, there is no image")

def trans(qz, qy ,qx ,qw):
    qvec = np.array([float(qw), float(qx), float(qy), float(qz)])
    qmat = qvec2rotmat(qvec)
    qmat = qmat.T
    new_qvec =  rotmat2qvec(qmat)
    w, x, y, z = new_qvec
    return str(w), str(x), str(y), str(z)

def Write_intrinscs(write_path, input_path, intrincs_):
    if os.path.isfile(write_path):
            os.remove(write_path)
    with open(write_path,'w') as file_w:
        with open(input_path,'r') as file:
            for line in file:
                data_line=line.strip("\n").split(' ')
                name = data_line[0]
                model_name = 'OPENCV_FISHEYE'
                intrincs_ = intrincs_
                data_line_str = str(name) + ' ' + model_name +' '+str(intrincs_[0])+' '+str(intrincs_[1])+' '+str(intrincs_[2])+' '+str(intrincs_[3])+' '+str(intrincs_[4])+' '+str(intrincs_[5])+' '+str(intrincs_[6])+' '+str(intrincs_[7])+' '+str(intrincs_[8])+' '+str(intrincs_[9])+' \n'
                file_w.write(data_line_str)
            print("Done with writting intrinscs.txt")
                
def EtoQ(write_path, input_path):  # Euler to Qvec
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
                crs_utm45N = CRS.from_epsg(32645)
                crs_WGS84=CRS.from_epsg(4326)
                
                # from_crs = crs_CGCS2000
                from_crs = crs_WGS84
                to_cgcs = crs_CGCS2000
                transformer = Transformer.from_crs(from_crs, to_cgcs)
                new_x,new_y = transformer.transform(xyz[1],xyz[0])
                euler = data_line[4:]
                ret = R.from_euler('zxy',[90-float(euler[0]),90-float(euler[1]),90-float(euler[2])],degrees=True)# -90
                Q= ret.as_quat()
                # print(data_line[0])
                # out_line = [data_line[0]]+list(Q)+list(xyz_w2c[0])+list(xyz_w2c[1])+list(xyz_w2c[2])
                # out_line = [data_line[0]]+list(Q)+[new_x]+[new_y]+[xyz[2]]
                out_line = [data_line[0]]+list(Q)+[new_x]+[new_y]+[str(xyz[2])]
                # print(out_line)
                out_line_str  = str(data_line[0])+' '+str(out_line[4])+' '+str(out_line[1])+' '+str(out_line[2])+' '+str(out_line[3])+' '+str(out_line[5])+' '+str(out_line[6])+' '+str(out_line[7])+' \n'
                # print(out_line_str)
                file_w.write(out_line_str)
            print("Done with writting pose.txt")

def QtoE(write_path, input_path): # Qvec to Euler
    if os.path.isfile(write_path):
        os.remove(write_path)
    with open(write_path,'w') as file_w:
        with open(input_path,'r') as file:
            for line in file:
                data_line=line.strip("\n").split(' ')
                # print(data_line)
                qvec = list(map(float, list(data_line[1:5])))
                xyz = list(map(float, list(data_line[5:8]))) # W2C                
                crs_CGCS2000=CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')    #degree
                # crs_CGCS2000=CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101,AUTORITY["EPSG", "1024"]],AUTHORITY["EPSG", "1043"]],PRIMEM["Greenwich",0.0,AUTHORITY["EPSG", "8901"]],UNIT["Degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4490"]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0,AUTHORITY["EPSG",“9001”]],AUTHORITY["EPSG","4547"]]')    #degree
                crs_utm45N = CRS.from_epsg(32645)
                crs_WGS84=CRS.from_epsg(4326)
                crs_114E = CRS.from_epsg(4547)
                
                # from_crs = crs_CGCS2000
                from_crs = crs_114E
                to_cgcs = crs_WGS84
                transformer = Transformer.from_crs(from_crs, to_cgcs)
                new_x,new_y = transformer.transform(xyz[1],xyz[0])
                # euler = data_line[4:]
                if qvec == [0, 0, 0, 0]:
                    out_line_str = str(data_line[0]) + ' 0 0 0 0 0 0' + '\n'
                    file_w.write(out_line_str)
                    continue
                ret = R.from_quat(qvec)# -90
                euler = ret.as_euler('xyz', degrees=True)
                # print(data_line[0])
                # out_line = [data_line[0]]+list(Q)+list(xyz_w2c[0])+list(xyz_w2c[1])+list(xyz_w2c[2])
                # out_line = [data_line[0]]+list(Q)+[new_x]+[new_y]+[xyz[2]]
                out_line = [data_line[0]]+list(euler)+[new_x]+[new_y]+[str(xyz[2])]
                # print(out_line)
                out_line_str  = str(data_line[0])+' '+str(out_line[1])+' '+str(out_line[2])+' '+str(out_line[3])+' '+str(out_line[4])+' '+str(out_line[5])+' '+str(out_line[6])+'\n'
                # print(out_line_str)
                file_w.write(out_line_str)
            print("Done with writting pose.txt")

def RtoQ(write_path, input_path): # Rotation to Qvec
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
                crs_utm45N = CRS.from_epsg(32645)
                crs_WGS84=CRS.from_epsg(4326)
                
                # from_crs = crs_CGCS2000
                from_crs = crs_WGS84
                to_cgcs = crs_CGCS2000
                transformer = Transformer.from_crs(from_crs, to_cgcs)
                new_x,new_y = transformer.transform(xyz[1],xyz[0])
                euler = data_line[4:]
                ret = R.from_euler('zxy',[90-float(euler[0]),90-float(euler[1]),90-float(euler[2])],degrees=True)# -90
                Q= ret.as_quat()
                # print(data_line[0])
                # out_line = [data_line[0]]+list(Q)+list(xyz_w2c[0])+list(xyz_w2c[1])+list(xyz_w2c[2])
                # out_line = [data_line[0]]+list(Q)+[new_x]+[new_y]+[xyz[2]]
                out_line = [data_line[0]]+list(Q)+[new_x]+[new_y]+[str(xyz[2])]
                # print(out_line)
                out_line_str  = str(data_line[0])+' '+str(out_line[4])+' '+str(out_line[1])+' '+str(out_line[2])+' '+str(out_line[3])+' '+str(out_line[5])+' '+str(out_line[6])+' '+str(out_line[7])+' \n'
                # print(out_line_str)
                file_w.write(out_line_str)

def dealwith_csv_qvec(csv_information, output_path):

    count = 1
    info_name = csv_information["img_name"]
    to_write = []

    fd_out = open(output_path, 'w')
    for j in range(len(info_name)):
        if eval(csv_information['gps_alt'][j]) == 0:
            continue
        else:
            fd_out.write((str(count)+'.jpg') + ' ')
            # fd_out.write(info_name[j] + ' ')
            qz, qy ,qx ,qw = csv_information['qz'][j], csv_information['qy'][j], csv_information['qx'][j], csv_information['qw'][j]
            # qw, qx, qy, qz =  trans(qz, qy ,qx ,qw) 

            to_write.extend([qw, qx ,qy ,qz])
            gps_lon = csv_information['gps_lon'][j]
            gps_lat = csv_information['gps_lat'][j]
            gps_alt = csv_information['gps_alt'][j]
            crs_CGCS2000=CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')    #degree
            crs_WGS84=CRS.from_epsg(4326)
            from_crs = crs_WGS84
            to_CGCS2000 = crs_CGCS2000
            transformer = Transformer.from_crs(from_crs, to_CGCS2000)
            new_x,new_y = transformer.transform(float(gps_lat), float(gps_lon))
            new_z = eval(gps_alt)
            to_write.extend([str(new_x), str(new_y) ,str(new_z)])
            line_ = " ".join([str(elem) for elem in to_write])
            fd_out.write(line_ + "\n")
            count += 1
        to_write=[]

    fd_out.close()


def dealwith_csv_euler(csv_information, output_path):

    count = 1
    info_name = csv_information["img_name"]
    to_write = []

    fd_out = open(output_path, 'w')
    for j in range(len(info_name)):
        if float(csv_information['gps_alt'][j]) == 0:
            continue
        else:
            fd_out.write((str(count)+'.jpg') + ' ')
            # fd_out.write(info_name[j] + ' ')

            gps_lon = csv_information['gps_lon'][j]
            gps_lat = csv_information['gps_lat'][j]
            gps_alt = float(csv_information['gps_alt'][j])
            to_write.extend([str(gps_lon), str(gps_lat) ,str(gps_alt)])
            yaw, pitch, roll = csv_information['yaw'][j], csv_information['pitch'][j], csv_information['roll'][j]

            to_write.extend([roll, pitch ,yaw])
            
            line_ = " ".join([str(elem) for elem in to_write])
            fd_out.write(line_ + "\n")
            count += 1
        to_write=[]
    print("Done with writting euler.txt")
    fd_out.close()


def write_ground_truth(write_path, input_path, sequence):
    
    file_object = open(input_path, encoding='utf-8')
    try:
        all_the_xmlStr = file_object.read()
    finally:
        file_object.close()
        # transfer to dict
    dictdata = dict(xmltodict.parse(all_the_xmlStr))
    # import ipdb;ipdb.set_trace();

    tilponts = dictdata['BlocksExchange']['Block']['Photogroups']['Photogroup'][sequence]  #!

    img_list = tilponts['Photo']
    
    constructed_image = []

    with open(write_path, 'w') as file_w:
        for i in range(len(img_list)): #len(img_list)
            img_path = img_list[i]['ImagePath'].split('/')[-1]  
            pose_M00 = img_list[i]['Pose']['Rotation']['M_00']
            pose_M01 = img_list[i]['Pose']['Rotation']['M_01']
            pose_M02 = img_list[i]['Pose']['Rotation']['M_02']
            pose_M10 = img_list[i]['Pose']['Rotation']['M_10']
            pose_M11 = img_list[i]['Pose']['Rotation']['M_11']
            pose_M12 = img_list[i]['Pose']['Rotation']['M_12']
            pose_M20 = img_list[i]['Pose']['Rotation']['M_20']
            pose_M21 = img_list[i]['Pose']['Rotation']['M_21']
            pose_M22 = img_list[i]['Pose']['Rotation']['M_22']
            a_qvec, a_np = rotation_to_quat([pose_M00, pose_M01, pose_M02,
                                            pose_M10, pose_M11, pose_M12,
                                            pose_M20, pose_M21, pose_M22])
            qw, qx, qy, qz = a_qvec[0], a_qvec[1], a_qvec[2], a_qvec[3]

            # degug
            # R = qvec2rotmat([qw, qx, qy, qz])

            x = img_list[i]['Pose']['Center']['x']
            y = img_list[i]['Pose']['Center']['y']
            z = str(float(img_list[i]['Pose']['Center']['z']  ))  

            
            

            out_line = img_path + ' ' + str(qw) + ' ' + str(qx) + ' ' + str(qy) + ' ' + str(qz) + ' ' + str(x) + ' ' + str(y) + ' ' + str(z) + '\n'
            constructed_image.append(img_path)
            # print(image_path)
            file_w.write(out_line)
    return constructed_image
def delete_image(path_list, constructed_image):
    image_path = os.listdir(path_list)
    image_path.sort(key=lambda x:int(x[-8:-4]))  #!compared name number
    raw_image = []
    for img in image_path:
        raw_image.append(img.split('/')[-1])
    unconstructed_images = list(set(raw_image).difference(set(constructed_image)))
    print("unconstructed images: ", len(unconstructed_images))
    print("constructed images: ", len(raw_image) - len(unconstructed_images) )
    for i in range(len(unconstructed_images)):
        delete_path = os.path.join(path_list, unconstructed_images[i])
        if os.path.exists(delete_path):
            os.remove(delete_path)
        else:
            print(delete_path)
    
def change_name(inputfile):
    path_list = os.listdir(inputfile)
    path_list.sort(key=lambda x:int(x[-8:-4]))  #!
    cnt = 1
    for i in path_list:
        name = str(cnt)+ '.jpg'
        old_name = os.path.join(inputfile, i)
        new_name = os.path.join(inputfile, name)
        os.rename(old_name, new_name)
        cnt+=1
                   
        
        
def main(config):
    # ==== Extracting Ground Truth from .xml ====
    sequence_name = config["sequence_name"]
    save_path = config["save_path"]
    input_path_xml = config["input_path_xml"]
    image_path = config["image_path"]  

    for i in range(len(sequence_name)):
        gt_save_path = save_path + sequence_name[i] + '.txt'
        constructed_image = write_ground_truth(gt_save_path, input_path_xml, i)
        sequence_path = os.path.join(image_path, sequence_name[i])
        print(sequence_path)
        # delete_image(sequence_path, constructed_image)
        # change_name(sequence_path)
        
    # ====generate reference gt from .xml
    # sequence_name = ['h', 'q', 'x','y', 'z']
    # input_path_xml = "/home/ubuntu/Documents/1-pixel/1-jinxia/拍摄1126/query_pic/2-gt/Block_1126_REGERENCE_2.xml"
    # save_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/拍摄1126/query_pic/2-reference_gt/"
    # for i in range(len(sequence_name)):
    #     gt_save_path = save_path + sequence_name[i] + '.txt'
    #     constructed_image = write_ground_truth(gt_save_path, input_path_xml, i)
        
    
        
        

