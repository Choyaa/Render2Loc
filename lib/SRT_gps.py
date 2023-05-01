
from distutils.command.config import config
import re
import os
# from tracemalloc import start

#8700,9150
#0.990
def save_txt(camera_id, latitude, longitude, abs_alt, yaw, pitch, roll, save_path):
    if os.path.isfile(save_path):
        os.remove(save_path)
    with open(save_path, "a") as f:
        for i in range(0, len(camera_id)):
            # f.write('{:}\t{:}\t{:}\t{:}\t{:}\t{:}\t{:}\n'.format(camera_id[i], model[i], w[i], h[i],focal[i], cx[i],cy[i]))
            # f.write('{:} {:} {:} {:} {:} {:} {:}\n'.format(i, latitude[i], longitude[i], abs_alt[i], yaw[i], pitch[i], roll[i]))
            f.write('{:} {:} {:} {:} {:} {:} {:}\n'.format(str(i+1)+'.jpg', latitude[i], longitude[i], abs_alt[i], yaw[i], pitch[i], roll[i]))

def main(config):
    #extract gps  from query SRT.txt
    #gps: [latitude, longitude, abs_alt, yaw, pitch, roll]
    Filename = config['input']
    fre = config['fre']
    #results save path
    save_path = config['save_path']
    f = open(Filename, 'r')
    AllData = f.read()
    f.close()
    a = config['start']
    b = config['end']
    AllResume = {}

    ResumRe = re.compile('latitude: (.+?)\]')   #.+ focal_len到] 中间所有字符； \]：提取[]加转义字符\ ； ？：非贪婪提取； (?= 末尾字符): 不包含]符号； 
    result1 = ResumRe.findall(AllData)
    lenn = len(result1)
    result1 = result1[a:b:fre]
    for i, v in enumerate(result1) : result1[i] = float(v)

    ResumRe = re.compile('longitude: (.+?)\]')   #.+ focal_len到] 中间所有字符； \]：提取[]加转义字符\ ； ？：非贪婪提取； (?= 末尾字符): 不包含]符号； 
    result2 = ResumRe.findall(AllData)
    result2 = result2[a:b:fre]
    for i, v in enumerate(result2) : result2[i] = float(v)

    ResumRe = re.compile('abs_alt: (.+?)\]')   #.+ focal_len到] 中间所有字符； \]：提取[]加转义字符\ ； ？：非贪婪提取； (?= 末尾字符): 不包含]符号； 
    result3 = ResumRe.findall(AllData)
    result3 = result3[a:b:fre]
    for i, v in enumerate(result3) : result3[i] = float(v)

    ResumRe = re.compile('gb_yaw: (.+?)g')   #.+ focal_len到] 中间所有字符； \]：提取[]加转义字符\ ； ？：非贪婪提取； (?= 末尾字符): 不包含]符号； 
    result4 = ResumRe.findall(AllData)
    result4 = result4[a:b:fre]
    for i, v in enumerate(result4) : result4[i] = float(v)

    ResumRe = re.compile('gb_pitch: (.+?)g')   #.+ focal_len到] 中间所有字符； \]：提取[]加转义字符\ ； ？：非贪婪提取； (?= 末尾字符): 不包含]符号； 
    result5 = ResumRe.findall(AllData)
    result5 = result5[a:b:fre]
    for i, v in enumerate(result5) : result5[i] = float(v)

    ResumRe = re.compile('gb_roll: (.+?)\]')   #.+ focal_len到] 中间所有字符； \]：提取[]加转义字符\ ； ？：非贪婪提取； (?= 末尾字符): 不包含]符号； 
    result6 = ResumRe.findall(AllData)
    result6 = result6[a:b:fre]
    for i, v in enumerate(result6) : result6[i] = float(v)
    # result3 = [1617.0] * len(result1) 
    # id = [str(i)+'.jpg' for i in range(1,b-a+1)]
    # id = [str(int(i/30)+1)+'.jpg' for i in range(1,b-a+1,30)]

    id = [str(i)+'.jpg' for i in range(a,b,fre)]
    print(len(id), len(result1))
    save_txt(id,result1, result2, result3, result4, result5, result6,save_path)

if __name__ == "__main__":          
    #extract gps  from query SRT.txt
    #gps: [latitude, longitude, abs_alt, yaw, pitch, roll]
    Filename = "/home/ubuntu/Documents/1-pixel/视频抽帧/txt1/multi_people.txt"
    fre = 1
    #results save path
    save_path = "/home/ubuntu/Documents/1-pixel/视频抽帧/txt1/multi_people"+str(fre)+"100_gps_result.txt"
    start = 8700
    end = 9150

    config_manual = {
        "input": Filename,
        "fre" : fre,
        "save_path" : save_path,
        "start":start,
        "end":end
    }

    main(config_manual)
