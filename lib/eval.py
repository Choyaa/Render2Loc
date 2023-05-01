import cv2
import logging
import numpy as np
import os

from scipy.spatial.transform import Rotation
from collections import defaultdict


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



def evaluate(results, gt, only_localized=False):
    predictions = {}
    with open(results, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            q, t = np.split(np.array(data[1:], float), [4])

            predictions[os.path.splitext(os.path.basename(name))[0]] = (qvec2rotmat(q), t)

    test_names = []
    gts = {}
    with open(gt, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0].split('/')[-1]
            q, t = np.split(np.array(data[1:], float), [4])

            gts[os.path.splitext(os.path.basename(name))[0]] = (qvec2rotmat(q), t)
            test_names.append(os.path.splitext(os.path.basename(name))[0])
    
    errors_t = []
    errors_R = []
    index = 0
    for name in test_names:
        if name not in predictions:
            print("nm",name)
            if only_localized:
                continue
            e_t = np.inf
            e_R = 180.
        else:
            index+=1
            R_gt, t_gt = gts[name]
            R, t = predictions[name]
            
            e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)

                
            
            # if e_t > 0.5:
            #     print (name, e_t)
            cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
            e_R = np.rad2deg(np.abs(np.arccos(cos)))
            if '155147' in name :
                # print("xyz_gt: ", t_gt)
                # print("xyz_estiamted: ", t)
                print(name)
                print("er: ", e_R)
                print("et", e_t)

        errors_t.append(e_t)
        errors_R.append(e_R)
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)
    med_t = np.median(errors_t)
    std_t = np.std(errors_t)
    med_R = np.median(errors_R)
    std_R = np.std(errors_R)
    # out = f'Results for file {results.name}:'
    out = f'\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg'
    out += f'\nStd errors: {std_t:.3f}m, {std_R:.3f}deg'

    out += '\nPercentage of test images localized within:'
    threshs_t = [0.25, 0.5, 1.0]
    threshs_R = [2.0, 5.0, 10.0]
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f'\n\t{th_t*100:.0f}cm, {th_R:.0f}deg : {ratio*100:.2f}%'
    #logger.info(out)
    print (out)
    
def main(config):
    # gt path
  
    # dji_day_gt.txt
    dji_day_gt = config["dji_day_gt"]
    # dji_night_gt.txt
    dji_night_gt = config["dji_night_gt"]
    # phone_day_gt.txt
    phone_day_gt = config["phone_day_gt"]
    # phone_night_gt.txt
    phone_night_gt = config["phone_night_gt"]


    # render1 pose.txt path
    render1_pose = config["render1_pose"]
    # render2 pose.txt path
    render2_pose = config["render2_pose"]
    # render3 pose.txt path
    render3_pose = config["render3_pose"]

    
    # # # ---------------------------------
    print("=====superpoint superglue")
    print ('Evaluation DJI Day render_1 localization')
    evaluate(render1_pose, dji_day_gt)
    
    print ('Evaluation DJI Day render_2 localization')
    evaluate(render2_pose, dji_day_gt)
    
    print ('Evaluation DJI Day render_3 localization')
    evaluate(render3_pose, dji_day_gt)

    # # ---------------------------------
    # print("=====superpoint superglue")
    # print ('Evaluation DJI Night render_1 localization')
    # evaluate(render1_pose, dji_night_gt)
    
    # print ('Evaluation DJI Night render_2 localization')
    # evaluate(render2_pose, dji_night_gt)
    
    # print ('Evaluation DJI Nihgt render_3 localization')
    # evaluate(render3_pose, dji_night_gt)   
    # ---------------------------------
    # print("=====fixed xy")
    # print ('Evaluation Phone Day render_1 localization')
    # evaluate(render1_pose, phone_day_gt)
    
    # print ('Evaluation Phone Day render_2 localization')
    # evaluate(render2_pose, phone_day_gt)
    
    # print ('Evaluation Phone Day render_3 localization')
    # evaluate(render3_pose, phone_day_gt)
 
 
    # # ---------------------------------
    # print("=====superpoint superglue")
    # print ('Evaluation Phone Night render_1 localization')
    # evaluate(render1_pose, phone_night_gt)
    
    # print ('Evaluation Phone Night render_2 localization')
    # evaluate(render2_pose, phone_night_gt)
    
    # print ('Evaluation Phone Nihgt render_3 localization')
    # evaluate(render3_pose, phone_night_gt)     
    
    
    
    


