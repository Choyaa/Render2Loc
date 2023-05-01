from sys import exec_prefix
import numpy as np
import logging
from matplotlib import image, pyplot as plt 
import shutil
from operator import itemgetter, attrgetter
# import os
# import sys
# # print(sys.path)
# os.chdir("./geo_relocalization")
from sklearn import datasets
logger = logging.getLogger(__name__)
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


def eval_absolute_poes(gt_list,pred_list,match_list,weather,result):
    errors_t = []
    errors_t_X = []
    errors_t_Y = []
    errors_t_Z = []
    errors_R = []

    matched_errors_t = []
    matched_errors_t_X = []
    matched_errors_t_Y = []
    matched_errors_t_Z = []
    matched_errors_R = []
    matched_errors_R_list = []
    matched_errors_t_list = []
    match_count= []
    # img_list = pred_list[]
    image_num = len(pred_list)
    for name in pred_list:
    # for name in test_list:
        if name not in gt_list:
            e_t = np.inf
            e_R = 180.
            continue
        else:
            
            gt = gt_list[name]
            R_gt, t_gt = qvec2rotmat(gt[:4]), np.array(gt[4:])
            pred = pred_list[name]
            qvec, t = pred[0][:4],pred[0][4:]
            R = qvec2rotmat(qvec)
            # e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
            e_t = np.linalg.norm(t_gt - t, axis=0)
            e_X =abs(t_gt[0] - t[0])
            e_Y = abs(t_gt[1] - t[1])
            e_Z = abs(t_gt[2] - t[2])
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            match_point = match_list[name][0]
        if match_point>8:
            matched_errors_t.append(e_t)
            matched_errors_R.append(e_R)
            matched_errors_t_list.append([name,e_t])
            matched_errors_R_list.append([name,e_R])
            matched_errors_t_X.append(e_X)
            matched_errors_t_Y.append(e_Y)
            matched_errors_t_Z.append(e_Z)
        match_count.append(match_point)
        errors_t.append(e_t)
        errors_R.append(e_R)
        
        errors_t_X.append(e_X)
        errors_t_Y.append(e_Y)
        errors_t_Z.append(e_Z)

    matched_errors_t_list = sorted(matched_errors_t_list, key=itemgetter(1))
    matched_errors_R_list = sorted(matched_errors_R_list, key=itemgetter(1))

    for item in matched_errors_t_list[:5]:
        image_name = item[0]
        shutil.copyfile('results/loc_results/match_vis/'+image_name,'results/eval_result/'+weather+'/best_t_5/'+str(item[1])+'_'+image_name)
    for item in matched_errors_t_list[-10:]:
        image_name = item[0]
        shutil.copyfile('results/loc_results/match_vis/'+image_name,'results/eval_result/'+weather+'/worst_t_10/'+str(item[1])+'_'+image_name)
    for item in matched_errors_R_list[:5]:
        image_name = item[0]
        shutil.copyfile('results/loc_results/match_vis/'+image_name,'results/eval_result/'+weather+'/best_R_5/'+str(item[1])+'_'+image_name)
    for item in matched_errors_R_list[-10:]:
        image_name = item[0]
        shutil.copyfile('results/loc_results/match_vis/'+image_name,'results/eval_result/'+weather+'/worst_R_10/'+str(item[1])+'_'+image_name)
    match_count = np.array(match_count)
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    # t_his = np.histogram(errors_t,bins =  [0,1,2,3,4,5,6,7,8,9,10,11])
    plt.figure()
    plt.subplot(221)
    plt.title(weather+"_errors_t") 
    plt.hist(errors_t,bins =  [0,1,2,3,4,5,6,7,8,9,10,11] ) #bins =  [0,1,2,3,4,5,6,7,8,9,10,11]
    # plt.savefig('/home/yan/code/geo_relocalization/results/eval_result/errors_t_'+weather+'_his.jpg')
    # plt.show()

    # plt.figure()
    plt.subplot(222)
    plt.title(weather+"_errors_R") 
    plt.hist(errors_R,bins =  [0,1,2,3,4,5,6,7,8,9,10,11])#bins =  [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15] 
    # plt.show()
    # plt.savefig('/home/yan/code/geo_relocalization/results/eval_result/errors_R_'+weather+'_his.jpg')

    mean_match = np.mean(match_count)
    failed_match = len(match_count[match_count<8])

    matched_errors_t = np.array(matched_errors_t)
    matched_errors_R = np.array(matched_errors_R)

    matched_med_R = np.median(matched_errors_R)
    matched_med_t = np.median(matched_errors_t)
    matched_med_t_X = np.median(matched_errors_t_X)
    matched_med_t_Y = np.median(matched_errors_t_Y)
    matched_med_t_Z = np.median(matched_errors_t_Z)

    matched_mean_R = np.mean(matched_errors_R)
    matched_mean_t = np.mean(matched_errors_t)
    matched_mean_t_X = np.mean(matched_errors_t_X)
    matched_mean_t_Y = np.mean(matched_errors_t_Y)
    matched_mean_t_Z = np.mean(matched_errors_t_Z)
    
    matched_max_R = np.max(matched_errors_R)
    matched_max_t = np.max(matched_errors_t)
    matched_max_t_X = np.max(matched_errors_t_X)
    matched_max_t_Y = np.max(matched_errors_t_Y)
    matched_max_t_Z = np.max(matched_errors_t_Z)

    matched_min_R = np.min(matched_errors_R)
    matched_min_t = np.min(matched_errors_t)
    matched_min_t_X = np.min(matched_errors_t_X)
    matched_min_t_Y = np.min(matched_errors_t_Y)
    matched_min_t_Z = np.min(matched_errors_t_Z)

    # plt.figure()
    plt.subplot(223)
    plt.title(weather+"_matched_errors_t")
    plt.hist(matched_errors_t,bins =  [0,1,2,3,4,5,6,7,8,9,10,11] )
    # plt.savefig('/home/yan/code/geo_relocalization/results/eval_result/matched_errors_t_'+weather+'_his.jpg')
    # plt.show()
    # plt.figure()
    plt.subplot(224)
    plt.title(weather+"_matched_errors_R") 
    plt.hist(matched_errors_R,bins =  [0,1,2,3,4,5,6,7,8,9,10,11])#bins =  [0,1,2,3,4,5,6,7,8,9,10,12,13,14,15] 

    plt.tight_layout()
    # plt.show()
    # plt.savefig('/home/yan/code/geo_relocalization/results/eval_result/matched_errors_R_'+weather+'_his.jpg')
    plt.savefig('/home/yan/code/geo_relocalization/results/eval_result/'+weather+'_his.jpg')


    med_R = np.median(errors_R)
    med_t = np.median(errors_t)
    med_t_X = np.median(errors_t_X)
    med_t_Y = np.median(errors_t_Y)
    med_t_Z = np.median(errors_t_Z)

    mean_R = np.mean(errors_R)
    mean_t = np.mean(errors_t)
    mean_t_X = np.mean(errors_t_X)
    mean_t_Y = np.mean(errors_t_Y)
    mean_t_Z = np.mean(errors_t_Z)
    
    max_R = np.max(errors_R)
    max_t = np.max(errors_t)
    max_t_X = np.max(errors_t_X)
    max_t_Y = np.max(errors_t_Y)
    max_t_Z = np.max(errors_t_Z)

    min_R = np.min(errors_R)
    min_t = np.min(errors_t)
    min_t_X = np.min(errors_t_X)
    min_t_Y = np.min(errors_t_Y)
    min_t_Z = np.min(errors_t_Z)
    

    out = f'Weather: {weather} Test image nums: {image_num}'
    out += f'\nMean match: {mean_match}, failed match : {failed_match}'
    out += f'\n==================================================='
    out += f'\nALL statics'
    out += f'\nMedian X: {med_t_X*100:.3f}cm, Median Y: {med_t_Y*100:.3f}cm,Median Z: {med_t_Z*100:.3f}cm'
    out += f'\nMean X: {mean_t_X*100:.3f}cm, Mean Y: {mean_t_Y*100:.3f}cm,Mean Z: {mean_t_Z*100:.3f}cm'
    out += f'\nMax X: {max_t_X*100:.3f}cm, Max Y: {max_t_Y*100:.3f}cm,Max Z: {max_t_Z*100:.3f}cm'
    out += f'\nMin X: {min_t_X*100:.3f}cm, Min Y: {min_t_Y*100:.3f}cm,Min Z: {min_t_Z*100:.3f}cm'

    out += f'\nMedian errors: {med_t*100:.3f}cm, {med_R:.3f}deg'
    out += f'\nMean errors: {mean_t*100:.3f}cm, {mean_R:.3f}deg'
    out += f'\nMax errors: {max_t*100:.3f}cm, {max_R:.3f}deg'
    out += f'\nMin errors: {min_t*100:.3f}cm, {min_R:.3f}deg'
    # print(out)
    out += '\nPercentage of test images localized within:'
    # threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    # threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    threshs_t = [ 0.5, 1.5, 2.0, 5.0]
    threshs_R = [1, 1.5 , 2.0, 5.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f'\n\t{th_t*100:.1f}cm, {th_R:.1f}deg : {ratio*100:.2f}%'

    out += f'\n==================================================='
    out += f'\nMATCHED statics'
    # out += f'\nMean match: {mean_match}, failed match : {failed_match}'
    out += f'\nMedian X: {matched_med_t_X*100:.3f}cm, Median Y: {matched_med_t_Y*100:.3f}cm,Median Z: {matched_med_t_Z*100:.3f}cm'
    out += f'\nMean X: {matched_mean_t_X*100:.3f}cm, Mean Y: {matched_mean_t_Y*100:.3f}cm,Mean Z: {matched_mean_t_Z*100:.3f}cm'
    out += f'\nMax X: {matched_max_t_X*100:.3f}cm, Max Y: {matched_max_t_Y*100:.3f}cm,Max Z: {matched_max_t_Z*100:.3f}cm'
    out += f'\nMin X: {matched_min_t_X*100:.3f}cm, Min Y: {matched_min_t_Y*100:.3f}cm,Min Z: {matched_min_t_Z*100:.3f}cm'

    out += f'\nMedian errors: {matched_med_t*100:.3f}cm, {matched_med_R:.3f}deg'
    out += f'\nMean errors: {matched_mean_t*100:.3f}cm, {matched_mean_R:.3f}deg'
    out += f'\nMax errors: {matched_max_t*100:.3f}cm, {matched_max_R:.3f}deg'
    out += f'\nMin errors: {matched_min_t*100:.3f}cm, {matched_min_R:.3f}deg'
    # print(out)
    out += '\nPercentage of test images localized within:'
    # threshs_t = [0.01, 0.02, 0.03, 0.05, 0.25, 0.5, 5.0]
    # threshs_R = [1.0, 2.0, 3.0, 5.0, 2.0, 5.0, 10.0]
    threshs_t = [ 0.5, 1.5, 2.0, 5.0]
    threshs_R = [1, 1.5 , 2.0, 5.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((matched_errors_t < th_t) & (matched_errors_R < th_R))
        out += f'\n\t{th_t*100:.1f}cm, {th_R:.1f}deg : {ratio*100:.2f}%'
    print(out)
    with open(result,'w') as f:
            f.writelines(out)
    f.close()
    # logger.info(out)

def load_file(path,file_type=None,weather=None):
    if file_type=='pred':
        file = open(path,'r')
        line_list=[]
        for l in file:
            line_list.append(l)
        d = {}
        if weather=='cloud':
            sub_list = line_list[:338]
        elif weather=='sun':
            sub_list = line_list[338:356]
        elif weather=='rain':
            sub_list = line_list[356:]
        for line in sub_list:
            key = line.split(' ')[0]
            value = [list(map(float,line.split(' ')[1:])),weather]
            d[key] = value
    else:
            file = open(path,'r')
            d = {}
            for line in file:
                key = line.split(' ')[0]
                value = list(map(float,line.split(' ')[1:]))
                d[key] = value
    return d

if __name__=='__main__':
    scene = ['jinxia']
    weather_list=['sun','cloud','rain']
    method = 'Loftr_render2match_thermal_ref_0.8pred-gt'
    # dataset = '/media/yan/data/CCTV7/Loftr/'

    # dataset = '/home/yan/code/image-processing-from-scratch/d2net/'
    # dataset = '/home/yan/code/patch2pix/abs_pose/'
    for scene_id in scene:
        
        gt_path = '/home/yan/code/geo_relocalization/results/gt_pose.txt'
        # pred_path = '/home/yan/code/pixloc/outputs/results/cctv/pixloc_'+scene_id+'.txt'
        pred_path ='/home/yan/code/geo_relocalization/results/loc_results/query_loc_pose.txt'
        match_path ='/home/yan/code/geo_relocalization/results/loc_results/match_count.txt'
        # test_path = '/media/yan/data/CCTV7/Loftr/'+scene_id+'/data_split/test_list.txt'
        
        gt_list = load_file(gt_path)
        # pred_list = load_file(pred_path)
        match_list = load_file(match_path)
        # test_list = []
        # with open(test_path,'r') as f:
        #     for line in f :
        #         line = line.strip()
        #         test_list.append(line)
        
        # weather_list =[sun,cloud,rain]
        for weather in weather_list:
            pred_list = load_file(pred_path,'pred',weather)
            result = '/home/yan/code/geo_relocalization/results/eval_result/'+method+'-'+scene_id+'-'+weather+'.txt'
            eval_absolute_poes(gt_list,pred_list,match_list ,weather,result)

        # eval_absolute_poes(gt_list,pred_list,match_list,weather ,result)

        # with open(T_pose,"r") as t:
        # for line in t.readlines():
        #     line = line.strip('\n')  #去掉列表中每一个元素的换行符
        #     line = line.strip()
        #     if line!= '':
        #         line = line.split(' ')
        #         name = line[0]
        #         qvec = np.array(tuple(map(float, line[1:5])))
        #         tvec = np.array(tuple(map(float, line[5:])))
        #         T_name_list.append(name)
        #         T_abs_pose.append([qvec,tvec])
        #     else:
        #         continue