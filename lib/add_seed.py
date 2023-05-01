# from osgeo import gdal
import numpy as np
from lib.rotation_transformation import matrix_to_euler_angles, qvec2rotmat,matrix_to_quaternion,euler_angles_to_matrix,quaternion_to_matrix,rotmat2qvec
from scipy.spatial.transform import Rotation as R
import os


# ====txt to colmap- seed             
def query_add_seed_txt(dsm_filepath, save_filepath, query_sequence, dev = 'phone'):
    '''
    transform raw prior data into colmap format
    '''
    dataset = gdal.Open(dsm_filepath)  # dsm
    
    geotrans = dataset.GetGeoTransform()
    originX = geotrans[0]
    originY = geotrans[3]
    pixelWidth = geotrans[1]
    pixelHeight = geotrans[5]
    band = dataset.GetRasterBand(1)

    delta_x = [0, 5, -5]
    delta_y = [0, 5, -5]
    euler_seed = []
    with open(query_sequence,'r') as file_r:
        with open(save_filepath,'w') as file_w:
            for line in file_r:
                data_line=line.strip("\n").split(' ')
                name_raw = data_line[0]
                qw, qx, qy, qz = data_line[7:]
                qvec = [ float(qw),float(qx), float(qy), float(qz)]

                # from prior qvec to euler
                qv = [ float(qvec[1]),float(qvec[2]), float(qvec[3]), float(qvec[0])] #!xyzw
                ret = R.from_quat(qv)
                euler_xyz0 = ret.as_euler('xyz', degrees=True)
                euler_xyz1 = ret.as_euler('xyz', degrees=True)
                euler_xyz2 = ret.as_euler('xyz', degrees=True)
                euler_xyz0[0] = euler_xyz0[0] - 180
                
                euler_xyz1[0] = euler_xyz1[0] - 180
                euler_xyz1[2] = euler_xyz1[2] - 60
                
                euler_xyz2[0] = euler_xyz2[0] - 180
                euler_xyz2[2] = euler_xyz2[2] + 60
                
                #==== add seed
                euler_seed.append(euler_xyz0)
                euler_seed.append(euler_xyz1)
                euler_seed.append(euler_xyz2)
                # # euler to matrix
                new_qvec = []
                for i in range(len(euler_seed)):     
                    print(euler_seed[i]) 
                    ret_0 = R.from_euler('xyz', euler_seed[i], degrees=True)
                    new_matrix0 = ret_0.as_matrix()
                    new_vec0 = rotmat2qvec(new_matrix0.T)  #!wxyz
                    new_qvec.append(new_vec0)
                

              
                xyz = list(map(float, list(data_line[1:4]))) # W2C                
                crs_CGCS2000=CRS.from_wkt('PROJCS["CGCS_2000_3_Degree_GK_CM_114E",GEOGCS["GCS_China_Geodetic_Coordinate_System_2000",DATUM["D_China_2000",SPHEROID["CGCS2000",6378137.0,298.257222101]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Gauss_Kruger"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",114.0],PARAMETER["Scale_Factor",1.0],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]')    #degree
                crs_WGS84=CRS.from_epsg(4326)
                
                # from_crs = crs_CGCS2000
                from_crs = crs_WGS84
                to_cgcs = crs_CGCS2000
                transformer = Transformer.from_crs(from_crs, to_cgcs)
                x, y = transformer.transform(xyz[1],xyz[0])
                
                #==========add seed

                ### add a line
                index = 0
                for i in range(len(delta_x)):
                    x = x + delta_x[i]
                    for j in range(len(delta_y)):
                        y = y + delta_y[j]
                        if dev == 'phone':
                            xOffset = int((x - originX) / pixelWidth)
                            yOffset = int((y - originY) / pixelHeight)
                            z = band.ReadAsArray(xOffset, yOffset, 1, 1) + 1.5  #z = 1.5
                        else:
                            z = xyz[2]
                        for k in range(len(new_qvec)): 
                            name = name_raw[:-4] + '_' +str(index) +'.jpg'
                            index += 1
                            q = new_qvec[k]
                            out_line_str  = name+' '+ str(q[0])+' '+str(q[1])+' '+str(q[2])+' '+str(q[3])+' '+str(x)+' '+str(y)+' '+str(z[0][0])+' \n'
                            file_w.write(out_line_str)
                

    print("Done with writting pose.txt")    
                                  

def query_add_seed(dsm_filepath, save_filepath, query_sequence,name_list, dev = 'phone'):
    dataset = gdal.Open(dsm_filepath)  # dsm
    
    geotrans = dataset.GetGeoTransform()
    originX = geotrans[0]
    originY = geotrans[3]
    pixelWidth = geotrans[1]
    pixelHeight = geotrans[5]
    band = dataset.GetRasterBand(1)
    delta = [[0, 0], [0, 5], [0, -5], [5, 0], [-5, 0]]
    
    with open(query_sequence, 'r') as f:
        with open(save_filepath,'w') as file_w:
            for data in f.read().rstrip().split('\n'):
                index = 0
                data = data.split()
                name_raw = data[0].split('/')[-1]
                q, t = np.split(np.array(data[1:], float), [4])       
                ### d = 1.5m
                x = t[0]
                y = t[1]
                ### add seed
                qmat = qvec2rotmat(q)
                qmat = qmat.T
                qvec = rotmat2qvec(qmat)  #!w2c

                for i in range(len(delta)):
                    x = t[0] + delta[i][0]
                    y = t[1] + delta[i][1]
                    if dev == 'phone':
                        xOffset = int((x - originX) / pixelWidth)
                        yOffset = int((y - originY) / pixelHeight)
                        z = band.ReadAsArray(xOffset, yOffset, 1, 1) + 1.5  #z = 1.5
                        z = z[0][0]
                        # ==== q, q+60, q -60
                        qvec1, qvec2 = yaw_seed(qvec)  #! output wxyz
                        name = name_raw[:-4] + '_' +str(index) +'.png'
                        out_line_str  = name_list+name_raw[:-4]+'/' + name+' '+str(q[0])+' '+str(q[1])+' '+str(q[2])+' '+str(q[3])+' '+str(x)+' '+str(y)+' '+str(z)+' \n'
                        file_w.write(out_line_str)
                        index += 1
                        name = name_raw[:-4] + '_' +str(index) +'.png'
                        out_line_str  = name_list +name_raw[:-4]+'/'+name + ' '+str(qvec1[0])+' '+str(qvec1[1])+' '+str(qvec1[2])+' '+str(qvec1[3])+' '+str(x)+' '+str(y)+' '+str(z)+' \n'
                        file_w.write(out_line_str)
                        index += 1
                        name = name_raw[:-4] + '_' +str(index) +'.png'
                        out_line_str  = name_list +name_raw[:-4]+'/'+name + ' '+str(qvec2[0])+' '+str(qvec2[1])+' '+str(qvec2[2])+' '+str(qvec2[3])+' '+str(x)+' '+str(y)+' '+str(z)+' \n'
                        file_w.write(out_line_str)
                        index += 1
                    else:
                        z = t[2]
                        name = name_raw[:-4] +  +'.png'
                        out_line_str  = name_list+name_raw[:-4]+'/' + name+' '+str(q[0])+' '+str(q[1])+' '+str(q[2])+' '+str(q[3])+' '+str(x)+' '+str(y)+' '+str(z)+' \n'
                        file_w.write(out_line_str)

                    
    print("Done with writting pose.txt")           
def yaw_seed(qvec):
    
    qv = [ float(qvec[1]),float(qvec[2]), float(qvec[3]), float(qvec[0])]
    
    ret = R.from_quat(qv)
    euler_xyz = ret.as_euler('xyz', degrees=True)
    
    euler_xyz_2 = ret.as_euler('xyz', degrees=True)

    euler_xyz[2] = euler_xyz[2] + 30
    euler_xyz_2[2] = euler_xyz_2[2] - 30
    
    
    # # euler to matrix
    ret_1 = R.from_euler('xyz', euler_xyz, degrees=True)
    ret_2 = R.from_euler('xyz', euler_xyz_2, degrees=True)
    new_matrix1 = ret_1.as_matrix()
    new_matrix2 = ret_2.as_matrix()
    
    new_qvec1 = rotmat2qvec(new_matrix1.T)
    new_qvec2 = rotmat2qvec(new_matrix2.T)  
    return new_qvec1, new_qvec2


def main(config):
    # ==== extract prior from Iphone =====

    dsm_file= config["dsm_file"]
    save_seed_filepath = config["save_seed_filepath"]
    sequence_colmap_path = config["sequence_colmap_path"]
    name_list = ['render/phone/night/sequence4/RGB/','render/phone/night/sequence3/RGB/','render/phone/night/sequence2/RGB/','render/phone/night/sequence1/RGB/',
                 'render/phone/day/sequence3/RGB/','render/phone/day/sequence2/RGB/','render/phone/day/sequence1/RGB/']


    sequence_name = ['phone_night4', 'phone_night3', 'phone_night2','phone_night1', 'phone_day3', 'phone_day2', 'phone_day1']
    for i in range(len(sequence_name)):
        query_sequence = sequence_colmap_path  +  sequence_name[i] +'_prior'+'.txt'
        save_seed_path = save_seed_filepath  +  sequence_name[i] +'_seed'+'.txt'
        if not os.path.exists(save_seed_path):
            query_add_seed(dsm_file, save_seed_path, query_sequence, name_list[i], 'phone')
    name_list = [  'render/UAV/day/sequence2/RGB/','render/UAV/day/sequence1/RGB/',
                 'render/UAV/night/sequence1/RGB/']
    sequence_name = ['uav_day2','uav_day1', 'uav_night1']
    for i in range(len(sequence_name)):
        query_sequence = sequence_colmap_path +  sequence_name[i] +'_prior'+'.txt'
        save_seed_path = save_seed_filepath + sequence_name[i] +'_seed'+'.txt'
        if not os.path.exists(save_seed_path):
            query_add_seed(dsm_file, save_seed_path, query_sequence,  name_list[i],'dji')
