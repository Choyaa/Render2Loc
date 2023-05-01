import exifread
import os
import glob
import pandas as pd


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
        
def Read_RTK_of_query( img_list, rtk_path, csv_save_path):
    ground_truth_information = pd.read_csv(rtk_path, encoding = 'gb2312')  
    
    img_name_list = []
    point_list = []
    
    gps_lon_list = []
    gps_lat_list = []
    gps_alt_list = []

    for img in img_list:
        img_information = exifread.process_file(open(img, 'rb'))
        # import ipdb;ipdb.set_trace();
        time_img = img_information['Image DateTime'].values
        new_time = time_img[:4] + '-' + time_img[5:7] + '-' + time_img[8:] # 手机时间
        flag = 0
        for k in range(len(ground_truth_information["time"])):
            ground_truth_time = ground_truth_information["time"][k]
            if new_time == ground_truth_time:
                flag = 1
                img_name_list.append(img.split('/')[-1])
                gps_lon_list.append(ground_truth_information['lon'][k])
                gps_lat_list.append(ground_truth_information['lat'][k])
                gps_alt_list.append(ground_truth_information['H'][k])
                point_list.append(ground_truth_information['name'][k])
        #### delete unconstrcted photos in CC
        if flag == 0:
            print("unconstrcted photos:", img)
            if os.path.exists(img):
                os.remove(img)  #!if you need change name, go in src
                
                
    dataframe = pd.DataFrame({'img_name':img_name_list,  'gps_lon':gps_lon_list, 'gps_lat':gps_lat_list, 'gps_alt':gps_alt_list, 'point': point_list})
    dataframe.to_csv(csv_save_path,index=False,sep=',')
    
def main(config):
    '''sssss
    only for phone, uav will automatically read by CC
    RTK file can not have CH
    '''
    sequence_name = config["sequence_name"]
    rtk_path =  config["rtk_path"]
    csv_save_path = config["csv_save_path"]
    file_path = config["file_path"]
    for i in range(len(sequence_name)):
        img_list = read_file(file_path + sequence_name[i])   #!image
        sequence_save_path = csv_save_path + sequence_name[i] + '.csv'
        Read_RTK_of_query(img_list, rtk_path, sequence_save_path)  #!
