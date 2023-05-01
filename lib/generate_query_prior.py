import os
import sys
import numpy as np
import pandas as pd
from pyproj import Transformer
from pyproj import CRS
from scipy.spatial.transform import Rotation as R
import math   
import exifread
import torch
from lib.rotation_transformation import matrix_to_euler_angles, qvec2rotmat, quaternion_to_matrix

def qvec2rotmat(qvec):  #!wxyz
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



def EtoQ(write_path, input_path, GT_path, a):  # Euler to Qvec
    if os.path.isfile(write_path):
        os.remove(write_path)
    fd_out = open(write_path, 'w')
    with open(GT_path,'r') as file_w:
        with open(input_path,'r') as file:
            for line,line_2 in zip(file, file_w):
                to_w = []
                data_line=line.strip("\n").split(' ')
                data_line_2=line_2.strip("\n").split(' ')

                print(data_line[7:])
                qw, qx, qy, qz = data_line[7:]
                qvec = [float(qx), float(qy), float(qz), float(qw)]
                ret = R.from_quat(qvec)
                euler_zxy = ret.as_euler(a, degrees=True)
                euler_zxy = [round(aa,4) for aa in euler_zxy]


                euler_GT = list(map(float, list(data_line_2[1:4])))
                euler_GT = [round(aa,4) for aa in euler_GT]
                # print("G T: ", euler_GT)
                to_w.extend([a])
                to_w.extend(euler_zxy)
                to_w.extend(["G T"])
                to_w.extend(euler_GT)
                to_w = " ".join([str(elem) for elem in to_w])
                fd_out.write(to_w + "\n")
                print(a, ": ", euler_zxy, " ", "G T: ", euler_GT)
            print("Done with writting pose.txt")
    fd_out.close()
choice = ['xyz', 'xzy', 'yxz', 'yzx', 'zyx', 'zxy']
def parse_pose_list(path):
    poses = {}
    # write_path
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
        # with open(output,'w') as file_w:
        #     for line in file_r:
        #     data_line=line.strip("\n").split(' ')
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])
            quat = torch.from_numpy(q)
            matrix = quaternion_to_matrix(quat)
            euler = matrix_to_euler_angles(matrix.T, "XYZ")
            euler = euler.numpy().tolist()
            euler_prior_angle = [euler[0] *180 / math.pi, euler[1] *180 / math.pi, euler[2] *180 / math.pi]
            
            
            # R = np.asmatrix(qvec2rotmat(q)).transpose()  #c2w
            
            # T = np.identity(4)
            # T[0:3,0:3] = R

            # if origin_coord is not None:
            #     origin_coord = np.array(origin_coord)
            #     T[0:3,3] -= origin_coord
            # transf = np.array([
            #     [1,0,0,0],
            #     [0,-1,0,0],
            #     [0,0,-1,0],
            #     [0,0,0,1.],
            # ])
            # T = T @ transf

def trans_euler(input, output):
    output = "/home/ubuntu/Documents/1-pixel/1-jinxia/1126render/prior/GT&PRIOR/day2/prior_day1_t.txt"
    with open(input,'r') as file_r:
        with open(output,'w') as file_w:
            for line in file_r:
                data_line=line.strip("\n").split(' ')
                qw, qx, qy, qz = data_line[7:]
                qvec = [ float(qw),float(qx), float(qy), float(qz)]

                # qmat = qvec2rotmat(qvec)
                # qmat = qmat.T
                # qvec = rotmat2qvec(qmat)

                # from prior qvec to euler
                qv = [ float(qvec[1]),float(qvec[2]), float(qvec[3]), float(qvec[0])]
                ret = R.from_quat(qv)
                euler_xyz = ret.as_euler('xyz', degrees=True)
                print(euler_xyz)
                print("=====")
                euler_xyz[0] =  euler_xyz[0] - 180

                euler_xyz[2] =  euler_xyz[2] + 180

                # # euler to matrix
                ret_2 = R.from_euler('xyz', euler_xyz, degrees=True)
                new_maticx = ret_2.as_matrix()
                
                new_vec = rotmat2qvec(new_maticx.T)
                print(euler_xyz)
                           
                xyz = list(map(float, list(data_line[1:4]))) # W2C                
                crs_CGCS2000=CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')    #degree
                crs_utm45N = CRS.from_epsg(32645)
                crs_WGS84=CRS.from_epsg(4326)
                
                # from_crs = crs_CGCS2000
                from_crs = crs_WGS84
                to_cgcs = crs_CGCS2000
                transformer = Transformer.from_crs(from_crs, to_cgcs)
                new_x,new_y = transformer.transform(xyz[1],xyz[0])

                out_line = [data_line[0]]+list(new_vec)+[new_x]+[new_y]+[str(xyz[2])]
                out_line_str  = str(data_line[0])+' '+str(out_line[1])+' '+str(out_line[2])+' '+str(out_line[3])+' '+str(out_line[4])+' '+str(out_line[5])+' '+str(out_line[6])+' '+str(out_line[7])+' \n'
                
                file_w.write(out_line_str)

    print("Done with writting pose.txt")

def trans_euler_gt(input, output):
    output = "/home/ubuntu/Documents/1-pixel/1-jinxia/1126render/prior/GT&PRIOR/day2/gt_euler.txt"
    with open(input, 'r') as f:
        with open(output,'w') as file_w:  
            for data in f.read().rstrip().split('\n'):
                data = data.split()
                name = data[0]
                q, t = np.split(np.array(data[1:], float), [4])
                # qvec = [float(q[1]), float(q[2]), float(q[3]), float(q[0])]
                qmat = qvec2rotmat(q)
                qmat = qmat.T
                qvec = rotmat2qvec(qmat)
                
                qv = [ float(qvec[1]),float(qvec[2]), float(qvec[3]), float(qvec[0])]
                ret = R.from_quat(qv)
                euler_xyz = ret.as_euler('xyz', degrees=True)
                
                out_line = [name]+list(euler_xyz)
                out_line_str  = name+' '+str(out_line[1])+' '+str(out_line[2])+' '+str(out_line[3])+' \n'
                
                file_w.write(out_line_str)

    print("Done with writting pose.txt")
    
def trans_euler_prior(input, output):
    output = "/home/ubuntu/Documents/1-pixel/1-jinxia/1126render/prior/GT&PRIOR/day2/prior_euler.txt"
    with open(input,'r') as file_r:
        with open(output,'w') as file_w:
            for line in file_r:
                data_line=line.strip("\n").split(' ')
                qw, qx, qy, qz = data_line[7:]
                qvec = [ float(qw),float(qx), float(qy), float(qz)]
                qv = [ float(qvec[1]),float(qvec[2]), float(qvec[3]), float(qvec[0])]
                ret = R.from_quat(qv)
                euler_xyz = ret.as_euler('xyz', degrees=True)
                print(euler_xyz)

                out_line = [data_line[0]]+list(euler_xyz)
                out_line_str  = str(data_line[0])+' '+str(out_line[1])+' '+str(out_line[2])+' '+str(out_line[3])+' \n'
                
                file_w.write(out_line_str)

    print("Done with writting pose.txt")

      
def generate_prior_perImage(sequence_txt_path, sequence_image_path, sequence_csv_path):
    '''
    extract raw prior data(quen & gps) per image from complete csv document
    '''
    count = 1
    csv_information = pd.read_csv(sequence_csv_path, encoding = 'gb2312')
    info_name = csv_information["loggingSample(N)"]
    to_write = []
    path_list = os.listdir(sequence_image_path)
    path_list.sort(key=lambda x:int(x[4:-4]))
    fd_out = open(sequence_txt_path, 'w')
    for img in path_list:
        img_information = exifread.process_file(open(os.path.join(sequence_image_path, img), 'rb'))
        # import ipdb;ipdb.set_trace();
        time_img = img_information['Image DateTime'].values
        new_time = time_img[11:17] # 手机时间 #time_img[:4] + '-' + time_img[5:7] + '-' + 
        second = time_img[17:]
        new_time = new_time + str(int(second))
        
        flag = count
        for j in range(len(info_name)):
            if float(csv_information['locationAltitude(m)'][j]) == 0:
                print("locationAltitude = 0000")
                continue
            else:
                prior_time = csv_information['loggingTime(txt)'][j][11:13]+':'+csv_information['loggingTime(txt)'][j][14:16]+':'+ str(int(csv_information['loggingTime(txt)'][j][17:19])+2)
                if new_time == prior_time:
                    print(new_time)
                    fd_out.write(img + ' ')
                    count += 1
                    gps_lon = csv_information['locationLongitude(WGS84)'][j]
                    gps_lat = csv_information['locationLatitude(WGS84)'][j]
                    gps_alt = float(csv_information['locationAltitude(m)'][j]) 
                    to_write.extend([str(gps_lon), str(gps_lat) ,str(gps_alt)])
                    yaw, pitch, roll = csv_information['motionYaw(rad)'][j]*180/math.pi, csv_information['motionPitch(rad)'][j]*180/math.pi, csv_information['motionRoll(rad)'][j]*180/math.pi

                    qx, qy, qz, qw = csv_information['motionQuaternionX(R)'][j], csv_information['motionQuaternionY(R)'][j], csv_information['motionQuaternionZ(R)'][j],csv_information['motionQuaternionW(R)'][j]

                    to_write.extend([yaw, pitch ,roll])
                    to_write.extend([qw, qx, qy ,qz])

                    line_ = " ".join([str(elem) for elem in to_write])
                    fd_out.write(line_ + "\n")
                    break
            to_write=[]
        if flag == count:  #if csv did not record prior for current image, giving the last data
            print("dddd")
            fd_out.write(img + ' ')
            to_write.extend([str(gps_lon), str(gps_lat) ,str(gps_alt)])
            to_write.extend([yaw, pitch ,roll])
            to_write.extend([qw, qx, qy ,qz] )
            print(img)
            line_ = " ".join([str(elem) for elem in to_write])
            fd_out.write(line_ + "\n")
            count += 1
            to_write=[]
    print("Done with writting iphone.txt")
    fd_out.close()
def generate_prior_colmap(input, output):
    '''
    transform raw prior data into colmap format
    '''
    with open(input,'r') as file_r:
        with open(output,'w') as file_w:
            for line in file_r:
                data_line=line.strip("\n").split(' ')
                qw, qx, qy, qz = data_line[7:]
                qvec = [ float(qw),float(qx), float(qy), float(qz)]

                # from prior qvec to euler
                qv = [ float(qvec[1]),float(qvec[2]), float(qvec[3]), float(qvec[0])] #!xyzw
                ret = R.from_quat(qv)
                euler_xyz = ret.as_euler('xyz', degrees=True)

                euler_xyz[0] =  euler_xyz[0] - 180
                euler_xyz[2] =  euler_xyz[2] - 180   #- 50, -180(for day1)
                print(euler_xyz)

                # # euler to matrix
                ret_2 = R.from_euler('xyz', euler_xyz, degrees=True)
                new_matrix = ret_2.as_matrix()
                
                new_vec = rotmat2qvec(new_matrix.T)  #!wxyz
              
                xyz = list(map(float, list(data_line[1:4]))) # W2C                
                crs_CGCS2000=CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')    #degree
                crs_WGS84=CRS.from_epsg(4326)
                
                # from_crs = crs_CGCS2000
                from_crs = crs_WGS84
                to_cgcs = crs_CGCS2000
                transformer = Transformer.from_crs(from_crs, to_cgcs)
                new_x,new_y = transformer.transform(xyz[1],xyz[0])
                
                # xyz w2c
                # t_c2w = [new_x, new_y, xyz[2]]
                # t_w2c = -(new_matrix.T).dot(t_c2w)

                out_line = [data_line[0]]+list(new_vec)+[new_x]+[new_y]+[str(xyz[2])]
                out_line_str  = str(data_line[0])+' '+str(out_line[1])+' '+str(out_line[2])+' '+str(out_line[3])+' '+str(out_line[4])+' '+str(out_line[5])+' '+str(out_line[6])+' '+str(out_line[7])+' \n'
                
                file_w.write(out_line_str)
        file_w.close()
    print("Done with writting pose.txt")    
    
# def generate_prior_colmap(input, output):  
    '''
    update z based on dsm, z = dsm(x, y) + 1.5
    add seed for prior
    x [+- 5]
    y [+- 5]
    yaw [+- 30]
    '''  
    
      
def main(config):
    # ==== extract prior from Iphone =====

    sequence_name = config["sequence_name"]
    img_list = config["img_list"]
    save_txt_path = config["save_txt_path"]
    raw_prior_path = config["raw_prior_path"] 
    save_colmap_prior_path = config["save_colmap_prior_path"]
    
    
    # =====generate iphone prior output txt(per image)
    for i in range(len(sequence_name)):
        sequence_image_path = img_list+ sequence_name[i]
        sequence_txt_path = save_txt_path  + sequence_name[i] +'_prior'+ '.txt'
        if not os.path.exists(sequence_txt_path):
            print("generate prior txt")
            sequence_csv_path = raw_prior_path  + sequence_name[i] +'_prior' + '.csv'
            print(sequence_csv_path)
            print(sequence_image_path)
            print(sequence_txt_path)
            print("==========")
            generate_prior_perImage(sequence_txt_path, sequence_image_path, sequence_csv_path)  #!wxyz
    
    # ====txt to colmap
    for i in range(len(sequence_name)):
        sequence_txt_path = save_txt_path + sequence_name[i] +'_prior' + '.txt'
        sequence_colmap_path = save_colmap_prior_path + sequence_name[i] +'_prior' +'.txt'
        if not os.path.exists(sequence_colmap_path):
            print("generate prior colmap")
            print(sequence_colmap_path)
            print("==========")
            generate_prior_colmap(sequence_txt_path, sequence_colmap_path)   
        
        
        
        
