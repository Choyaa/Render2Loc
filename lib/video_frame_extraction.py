import enum
from fileinput import filename
# from importlib.resources import path
from typing import ValuesView
import cv2
import numpy as np
import os
import math
import re
import numpy as np
def VideoFrameExtraction(intputfile, outputfile):
    """_summary_
    视频抽帧
    Args:
        intputfile (_type_): 视频路径（根目录）
        outputfile (_type_): 保存图片路径
    """
    
    video_root_file = []
    for i in os.listdir(intputfile):
        full_path = os.path.join(intputfile, i)
        if full_path.endswith('.MP4'):  #可修改为其他后缀
            video_root_file.append(full_path)
    
    if os.path.isdir(outputfile):
        pass
    else:
        os.makedirs(outputfile) 
    
    video_num = len(video_root_file)
    
    if video_num != 0:
        State = True
    else:
        print('视频路径错误，或者视频个数为0')
    
    
    for i in range(video_num):
        # 遍历每一个视频进行抽帧和保存
        cap = cv2.VideoCapture(video_root_file[i])
        fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        print("fps:",fps)
        print("h,w:",height, width)
        cap.set(cv2.CAP_PROP_FPS, int(fps//2))  #从fps/2开始读取
        
        video_name = video_root_file[i].split('/')[2][:-4]
        video_name_file = os.path.join(outputfile, video_name)
        if os.path.isdir(video_name_file):
            pass
        else:
            os.makedirs(video_name_file)
        
        imageNum = 0
        
        while State:
            capState, frame = cap.read()
            imageNum += 1
            if capState == True and (imageNum % fps) and (imageNum >= 900) and (imageNum <=2100) == 0:
                cv2.imwrite(video_name_file + "/" + str(int(imageNum / fps)) + '.jpg', frame)
            if capState == False:
                cap.release()
                break
    
    print("读取结束")

def SingleVideoFrameExtraction(video_root_file, outputfile, a, b, fre):
    """_summary_
    视频抽帧
    Args:
        intputfile (_type_): 视频路径（根目录）
        outputfile (_type_): 保存图片路径
    """
    if os.path.isdir(outputfile):
        pass
    else:
        os.makedirs(outputfile) 
    if os.path.exists(video_root_file):
        State = 1
        cap = cv2.VideoCapture(video_root_file)
        fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # float
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        print("fps:",fps)
        print("h,w:",height, width)
        cap.set(cv2.CAP_PROP_FPS, int(fps//2))  #从fps/2开始读取
        
        video_name = video_root_file.split('/')[2][:-4]
        if os.path.isdir(outputfile):
            pass
        else:
            os.makedirs(outputfile)
        
        imageNum = 0

        while State:
            capState, frame = cap.read()
            
            imageNum += 1
            if capState == True and (imageNum > a) and (imageNum <=b) and imageNum % fre == 0:
                cv2.imwrite(outputfile + "/" + str(int(imageNum / fre-a / fre)) + '.jpg', frame)   #??
            if (imageNum >b) and capState == False:
                cap.release()
                break
    else:
        print("video does not exist!")

    
    print("读取结束")
    return width, height

def Camera_intrincs(path, output):
    files = os.listdir(path)
    for file in files:
        f = open(file, 'r')
        AllData = f.read()
        f.close()
        ResumRe = re.compile('focal_len: (.+?)\]')   #.+ focal_len到] 中间所有字符； \]：提取[]加转义字符\ ； ？：非贪婪提取； (?= 末尾字符): 不包含]符号； 
        result = ResumRe.findall(AllData)
        for i, v in enumerate(result) : result[i] = float(v)
        
def save_txt(camera_id, model, focal, w,h, cx,cy, save_path):
    if os.path.isfile(save_path):
        os.remove(save_path)
    with open(save_path, "a") as f:
        for i in range(len(camera_id)):
            # f.write('{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n'.format(camera_id[i], model[i], w[i], h[i],focal[i], cx[i],cy[i]))
            f.write('{:} {:} {:} {:} {:} {:} {:} {:}\n'.format(camera_id[i], model[i], w[i], h[i],focal[i],focal[i], cx[i],cy[i]))
           
def generate_intrinsics(h, w, start, end, fre, SRT_file, save_path):
    """
    视频抽帧
    Args:
        H, W:  视频分辨率
        start, end :视频抽帧起止秒
        fre: 抽帧间隔,  fps = 30/ fre
        name: query文件名字  #fps = 30/ fre
        SRT_file: 无人机SRT文件路径
        save_path: intrinsics文件保存路径
    """
    # Focal
    # read SRT file
    f = open(SRT_file, 'r')
    AllData = f.read()
    f.close()
    AllResume = {}
    ResumRe = re.compile('focal_len: (.+?)\]')   #.+ focal_len到] 中间所有字符； \]：提取[]加转义字符\ ； ？：非贪婪提取； (?= 末尾字符): 不包含]符号； 
    result = ResumRe.findall(AllData)
    focal = result[start:end: fre]
    for i, v in enumerate(focal) : focal[i] = float(v)
    focal_np = np.array(focal).astype(float)
    focal_np = focal_np * (np.sqrt(h**2+ w**2)) /35.0  # transform focal 
    focal = focal_np.tolist()
    
    # HW
    H = [int(h)] * len(focal) 
    W = [int(w)] * len(focal) 
    # cxcy
    cx = [int(h/2)] * len(focal) 
    cy = [int(w/2)] * len(focal) 
    # MODEL
    model = ["PINHOLE"] * len(focal)  
    #Camera ID
    Camera_id = [ str(i) +'.jpg' for i in range(1, len(focal)+1)]
    
    print(len(focal), len(Camera_id))
    
    # save intrinsics.txt
    save_txt(Camera_id, model, focal, W,H, cy, cx, save_path)  
         
def main(config):
    #video 
    input = config['input']
    print(input )
    output = config['output']
    #SRT file 
    Filename = config['Filename']
    print(Filename)
    #intrinsics file from SRT
    save_path = config['intrinsics_save_path']
    #extract the video from start to end, 帧数 = b -a 
    start = config['start']   
    end = config['end'] 
    #fps = 30/ fre
    fre = config['fre'] 

    w, h = SingleVideoFrameExtraction(input, output, start, end, fre) 
    print("W, H:", w, h)
    print("start, end:", start, end)
    print("fps:", 30/ fre)
    
    # w = 1920
    # h = 1080
    generate_intrinsics(h, w, start, end, fre, Filename, save_path)     
      
    print("Video extraction successed!")


if __name__ == "__main__":
    '''
    w, h
    a, b
    fre
    '''
    #video 
    input = "/home/ubuntu/Documents/1-pixel/视频抽帧/video/142.MP4"
    output = "/home/ubuntu/Documents/1-pixel/视频抽帧/image/142"
    
    #SRT file 
    Filename = "/home/ubuntu/Documents/1-pixel/视频抽帧/txt/142.txt"
    #intrinsics file from SRT
    intrinsics_save_path = "/home/ubuntu/Documents/1-pixel/视频抽帧/txt/142_30_intrinsics.txt"
    #extract the video from start to end, 帧数 = b -a 
    start = 3450   
    end = 4350 
    #fps = 30/ fre
    fre = 30  
    
    config_manual = {
        "input" : input,
        "output" : output,
        "Filename" : Filename,
        "intrinsics_save_path" : intrinsics_save_path,
        "start" : start,   
        "end" : end, 
        "fre" : fre
    }

    main(config_manual)
    