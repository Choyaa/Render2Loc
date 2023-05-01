import os
import numpy as np 
import torch
import cv2
from src.utils.plotting import make_matching_figure
import matplotlib.cm as cm

def show_match(img0_raw, img1_raw, mkpts0, mkpts1,mconf, save_path):
    mkpts0 = mkpts0.cpu().numpy()
    mkpts1 = mkpts1.cpu().numpy()
    mconf = mconf.cpu().numpy()
    color = cm.jet(mconf)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]
    make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, path = save_path)

def main(config, data, matcher):
    pairs = data["pairs"]
    
    matches_data = []
    save_loc_path =  config["results"]
    for imgq_pth, render_list in pairs.items():
        max_correct = 0
        match_res = {}
        imgq_name = os.path.basename(imgq_pth)
        for render_pth in render_list:
            imgr_pth = render_pth[0]
            exrr_pth = render_pth[1]
            imgr_name = os.path.basename(imgr_pth)
            matches, _, _, mconf = matcher(imgq_pth, imgr_pth)   
            F, mask = cv2.findFundamentalMat(matches[:,:2], matches[:,2:], cv2.FM_RANSAC,12, 0.99)
            index = np.where(mask == 1)
            new_matches = matches[index[0]]
            mconf = torch.tensor(mconf[index[0]]).float()
            mkpts0q = torch.tensor(new_matches[:,:2]).float()
            mkpts1r = torch.tensor(new_matches[:,2:]).float()
            correct = torch.tensor(mkpts0q.shape[0])
#============visual
            if correct > max_correct:
                max_correct = correct
                imgr_name_final = imgr_name
                imgr_pth_final = imgr_pth
                exrr_pth_final = exrr_pth
                mkpts1r_final = mkpts1r
                mkpts0q_final = mkpts0q
                if not os.path.exists(save_loc_path+'pic/'):
                    os.makedirs(save_loc_path+'pic/')
                match_vis_path = save_loc_path+'pic/'+ imgq_name.split('.')[0] + '.png'
                imgq_raw = cv2.imread(imgq_pth, cv2.IMREAD_GRAYSCALE)
                imgr_raw = cv2.imread(imgr_pth_final, cv2.IMREAD_GRAYSCALE)
                show_match(imgq_raw,imgr_raw,mkpts0q_final,mkpts1r_final,mconf,match_vis_path) 
                
        match_res["imgq_name"] = imgq_name
        match_res["imgr_name"] = imgr_name_final
        match_res["exrr_pth"] = exrr_pth_final 
        match_res["mkpts_r"] = mkpts1r_final
        match_res["mkpts_q"] = mkpts0q_final
        match_res["correct"] = max_correct
        matches_data.append(match_res)
    return matches_data
        
        
        