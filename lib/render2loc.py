import os
os.chdir("..")
from copy import deepcopy
import h5py
import logging
import pickle
import cv2
import pycolmap
from collections import defaultdict
import os
import torch
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from src.utils.plotting import make_matching_figure
from scipy.spatial.transform import Rotation as R
from src.loftr import LoFTR, default_cfg
from tqdm import tqdm
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

def show_match(img0_raw, img1_raw, mkpts0, mkpts1,mconf, path):
    mkpts0 = mkpts0.cpu().numpy()
    mkpts1 = mkpts1.cpu().numpy()
    mconf = mconf.cpu().numpy()
    color = cm.jet(mconf)
    text = [
        'LoFTR',
        'Matches: {}'.format(len(mkpts0)),
    ]
    fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text=text, path = path)
    return fig

def filter_mkp(mkpq,mkpr,mconf,conf_thres=0.7):
    correct = [idx for idx , value in enumerate(mconf) if value>conf_thres]
    filter_mkpr = mkpr[correct] 
    filter_mkpq = mkpq[correct]
    filter_mconf = mconf[correct]
    return filter_mkpq,filter_mkpr,filter_mconf ,len(correct)
def match_pair(imgq_pth,imgr_pth,matcher):
    # imgr_pth = '/home/yan/code/geo_relocalization/results/1-11.png'
    scale = 8
    scale1 = 2
    
    imgq_raw = cv2.imread(imgq_pth, cv2.IMREAD_GRAYSCALE)
    imgr_raw = cv2.imread(imgr_pth, cv2.IMREAD_GRAYSCALE)
    imgq_raw = cv2.resize(imgq_raw, (int(imgq_raw.shape[1]/scale1), int(imgq_raw.shape[0]/scale1)))
    imgq_resize = cv2.resize(imgq_raw, (int(imgq_raw.shape[1]/scale), int(imgq_raw.shape[0]/scale)))
    imgr_resize = cv2.resize(imgr_raw, (int(imgr_raw.shape[1]/scale), int(imgr_raw.shape[0]/scale)))
    imgq_8 = cv2.resize(imgq_resize, (imgq_resize.shape[1]//8*8, imgq_resize.shape[0]//8*8))  # input size shuold be divisible by 8
    imgr_8 = cv2.resize(imgr_resize, (imgr_resize.shape[1]//8*8, imgr_resize.shape[0]//8*8))
    print("size:======================", imgq_8.shape, imgr_8.shape)
    # imgq_raw = cv2.resize(imgq_raw, (640, 640))  # input size shuold be divisible by 8
    # imgr_raw = cv2.resize(imgr_raw, (640, 640))

    imgq = torch.from_numpy(imgq_8)[None][None].cuda() / 255.
    imgr = torch.from_numpy(imgr_8)[None][None].cuda() / 255.
    batch = {'image0': imgq, 'image1': imgr} 
    # random_index1 = np.random.randint(0, mkpts0.shape[0], size= 20)
    # print('batch',batch)
    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0q = batch['mkpts0_f'] * scale
        mkpts1r = batch['mkpts1_f'] * scale
        mconf = batch['mconf'] 

    return mkpts0q,mkpts1r,mconf, imgq_raw,imgr_raw

def read_renderimg_pose(r_pose_path):
    
    all_pose_r = []
    with open(r_pose_path,'r') as file:
        for line in file:
                data_line=line.strip("\n").split(' ')
            #     print(data_line)
                idx = data_line[0]
                w,x,y,z= data_line[1],data_line[2],data_line[3],data_line[4]
                r_matrix = R.from_quat([x,y,z,w]).as_matrix()
                t = np.array([ float(data_line[5]),float(data_line[6]), float(data_line[7]),1.0]).reshape(4,)
                pose_r = np.zeros((4,4))
                pose_r[:3, :3]=r_matrix
                pose_r[:4, 3] = t
                pose_r= np.linalg.inv(pose_r)
                all_pose_r.append([idx,pose_r])
        return all_pose_r


def interpolate_depth(pos, depth):

    ids = torch.arange(0, pos.shape[0])
    depth = depth[:,:,0]
    h, w = depth.size()
    
    
    
    i = pos[:, 0]
    j = pos[:, 1]

    # Valid corners 验证坐标是否越界
    i_top_left = torch.floor(i).long()
    j_top_left = torch.floor(j).long()
    valid_top_left = torch.min(i_top_left >= 0, j_top_left >= 0)

    i_top_right = torch.floor(i).long()
    j_top_right = torch.ceil(j).long()
    valid_top_right = torch.min(i_top_right >= 0, j_top_right < w)

    i_bottom_left = torch.ceil(i).long()
    j_bottom_left = torch.floor(j).long()
    valid_bottom_left = torch.min(i_bottom_left < h, j_bottom_left >= 0)

    i_bottom_right = torch.ceil(i).long()
    j_bottom_right = torch.ceil(j).long()
    valid_bottom_right = torch.min(i_bottom_right < h, j_bottom_right < w)

    valid_corners = torch.min(
        torch.min(valid_top_left, valid_top_right),
        torch.min(valid_bottom_left, valid_bottom_right)
    )

    i_top_left = i_top_left[valid_corners]
    j_top_left = j_top_left[valid_corners]

    i_top_right = i_top_right[valid_corners]
    j_top_right = j_top_right[valid_corners]

    i_bottom_left = i_bottom_left[valid_corners]
    j_bottom_left = j_bottom_left[valid_corners]

    i_bottom_right = i_bottom_right[valid_corners]
    j_bottom_right = j_bottom_right[valid_corners]

    # ids = ids[valid_corners]

    # Valid depth验证深度
    valid_depth = torch.min(
        torch.min(
            depth[i_top_left, j_top_left] > 0,
            depth[i_top_right, j_top_right] > 0
        ),
        torch.min(
            depth[i_bottom_left, j_bottom_left] > 0,
            depth[i_bottom_right, j_bottom_right] > 0
        )
    )

    i_top_left = i_top_left[valid_depth]
    j_top_left = j_top_left[valid_depth]

    i_top_right = i_top_right[valid_depth]
    j_top_right = j_top_right[valid_depth]

    i_bottom_left = i_bottom_left[valid_depth]
    j_bottom_left = j_bottom_left[valid_depth]

    i_bottom_right = i_bottom_right[valid_depth]
    j_bottom_right = j_bottom_right[valid_depth]

    # 深度有效的点的index
    ids = ids[valid_depth]
    
    # Interpolation 插值深度
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    #插值出来的深度
    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]

# 读取深度
def read_valid_depth(depth_exr,mkpts1r):
    depth = cv2.imread(depth_exr, cv2.IMREAD_UNCHANGED)
    depth = torch.tensor(depth)  #?
    mkpts1r_a = torch.unsqueeze(mkpts1r.cpu()[:,0],0)
    mkpts1r_b =  torch.unsqueeze(mkpts1r.cpu()[:,1],0)
    mkpts1r_inter = torch.cat((mkpts1r_b ,mkpts1r_a),0).transpose(1,0)

    depth, _, valid = interpolate_depth(mkpts1r_inter , depth)

    return depth,valid

def loc_query(imgq_pth,mkpq,mkp3d,K_w2c, w, h, max_error=5):
    height, width = int(h), int(w)
    cx = K_w2c[0][2]
    cy = K_w2c[1][2]
    focal_length = K_w2c[0][0]
    #TODO:这里的相机模型是否要改变
    cfg = {
            'model': 'SIMPLE_PINHOLE',
            'width': width,
            'height': height,
            'params': [focal_length, cx, cy]
            
        }
    ret = pycolmap.absolute_pose_estimation(
            mkpq, mkp3d, cfg, max_error)

    return ret


def read_img_list(query_path,render_path):
    query_list = os.listdir(query_path)
    render_list = os.listdir(render_path)
    # query_list.sort(key=lambda x:int(x.split('.')[0]))
    # render_list.sort(key=lambda x:int(x.split('.')[0]))
    return query_list,render_list

def Get_Points3D(depth, R, t, K, points):   #c2w  points[n,2]
    if points.shape[-1] != 3:
        points_2D = torch.cat([points, torch.ones_like(points[ :, [0]])], dim=-1)
        points_2D = points_2D.T  #[3, n]
    t = torch.unsqueeze(t,-1).repeat(1, points_2D.shape[-1])
    Points_3D = R @ K @ (depth * points_2D) + t   
    return Points_3D    #[3,n]

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

def parse_pose_list(path):
    all_pose_c2w = []
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = data[0]
            if len(data) > 1:
                w2c_q, w2c_t = np.split(np.array(data[1:], float), [4])

                R = np.asmatrix(qvec2rotmat(w2c_q))   #!c2w
                t = -R.T @ w2c_t
                R = R.T
                Pose_c2w = np.identity(4)
                Pose_c2w[0:3,0:3] = R
                Pose_c2w[0:3, 3] = t
                all_pose_c2w.append([name, Pose_c2w, w2c_q, w2c_t])
        
    return all_pose_c2w

def read_instrincs(intrinsc_path):
    all_K = []
    # all_query_name =[]
    with open(intrinsc_path,'r') as file:
        for line in file:
            data_line=line.strip("\n").split(' ')

            img_name = data_line[0]
            w,h,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]   #! :8
            # focal_length = fx
            K_w2c = np.array([
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 
            # all_query_name.append(img_name)
            all_K.append([img_name,K_w2c,w, h])
    
    return all_K
def get_render_candidate(all_render_name, query_name):
    render_candidate = []
    query = (query_name.split('/')[-1]).split('.')[0]
    for render_name in all_render_name:
        if query in render_name:
            render_candidate.append(render_name)
    return render_candidate
def get_have_done(output_file):
    have_done_list = []
    with open(output_file, 'r') as f:
        for line in f:
            data_line=line.strip("\n").split(' ')
            img_name = data_line[0] 
            have_done_list.append(img_name.split('.')[0])  #!short name .png
    return have_done_list
            
        
                  
    
def main(config):
    num_iter = config["iteration_nums"]
    model_ckpt = config["model_ckpt"]
    images_path = config["images_path"]
    render_path = config["render_path"]
    intrinscs_path = config["query_camera"]
    r_pose_path = config["render_pose"] 
    result_path = config["results"]
    python_rgb_path = config["python_rgb_path"]
    python_depth_path = config["python_depth_path"]
    rgb_path = config["rgb_path"]
    depth_path = config["depth_path"]
    blender_path = config["blender_path"]
    sequence =config["sequence"]
    render_camera = config["render_camera"]
    
    #1.定义loftr
    matcher = LoFTR(config=default_cfg)
    matcher.load_state_dict(torch.load(model_ckpt)['state_dict'])
    matcher = matcher.eval().cuda()
    

    # 2.读取图像列表
    # 3.读取render图像深度文件
    #4.读取内参文件
    all_K_query= read_instrincs(intrinscs_path)
    all_K_render = read_instrincs(render_camera)
    all_query_name = list(np.array(all_K_query)[:,0])
    #5.读取render图像的pose文件
    t1 = time.time()
    loftr_thres = [0.2, 0.2, 0.5, 0.75]
    for iterative_num in range(2, num_iter+1):
        r_pose_path = result_path +  str(iterative_num-1) + '_query_estimated_pose.txt' #!##output_file  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
        print("==============", r_pose_path)
        all_pose_c2w = parse_pose_list(r_pose_path)
        save_loc_path = result_path #+ 'iterative' + str(iterative_num) +'/'
        if not os.path.exists(save_loc_path):
            os.makedirs(save_loc_path)
        output_file = save_loc_path + str(iterative_num) +'_query_estimated_pose.txt' #!

        all_render_name = list(np.array(all_pose_c2w)[:, 0])
        # 两幅图像匹配
        poses={}
        no_match_count = 0
        query_havedone_list = []
        #! imgr_name , imgq_name  全称
        pbar = tqdm(total=len(all_query_name), unit='pts')
        with open(output_file, 'w') as f:
            for imgq_name in all_query_name:
                imgq_pth = images_path + imgq_name
                imgq = imgq_name.split('/')[-1]  #! get short name
                render_candidate = get_render_candidate(all_render_name, imgq)
                max_correct = 0 

                #load and initialize render candidate
                if len(render_candidate) == 0:
                    name = 'render'+str(iterative_num)+'/'+sequence+imgq.split('.')[0] + '.png'
                    f.write(f'{name}\n') #!w2c
                else:
                    imgr_name_final = render_candidate[0]
                    for imgr_name in render_candidate: 
                        imgr_pth = render_path + imgr_name
                        imgr = imgr_name.split('/')  #!list
                        if iterative_num == 1: 
                            depth_exr = render_path + imgr[0] +'/' + imgr[1]+'/' + imgr[2] +'/'+ imgr[3] +'/' +'depth/'+imgq.split('.')[0]+'/'+ imgr[-1].split('.')[0]+'0001.exr'  #!
                        else:
                            depth_exr = render_path + imgr_name.split('.')[0]+'0001.exr' 
                        mkpts0q,mkpts1r,mconf, imgq_raw,imgr_raw = match_pair(imgq_pth,imgr_pth,matcher)
                        # print("mconf:",mconf)
                        mkpts0q,mkpts1r,mconf,correct = filter_mkp(mkpts0q,mkpts1r,mconf,loftr_thres[iterative_num-1])  #!
                        #=====
                        F, mask = cv2.findFundamentalMat(mkpts0q.cpu().numpy(), mkpts1r.cpu().numpy(), cv2.FM_RANSAC,12, 0.99)
                        index = np.where(mask == 1)
                        # print(index)
                        mkpts0q = mkpts0q[index[0]]
                        mkpts1r = mkpts1r[index[0]]
                        mconf = mconf[index[0]]
                        #======
                        # 可视化匹配结果
                        if not os.path.exists(save_loc_path+'render'+str(iterative_num)+'/'):
                            os.makedirs(save_loc_path+'render'+str(iterative_num)+'/')
                        match_vis_path = save_loc_path+'render'+str(iterative_num)+'/'+imgr[-1].split('.')[0] + '.png'
                        # imgq_raw = cv2.imread(imgq_pth, cv2.IMREAD_GRAYSCALE)
                        # imgr_raw = cv2.imread(imgr_pth, cv2.IMREAD_GRAYSCALE)
                        show_match(imgq_raw,imgr_raw,mkpts0q,mkpts1r,mconf,match_vis_path)
                        # fig = show_match(imgq_raw,imgr_raw,mkpts0q,mkpts1r,mconf)
                        # match_vis_path = save_loc_path+ imgq.split('.')[0]+'/'+'match_vis/'
                        # if not os.path.exists(match_vis_path):
                        #     os.makedirs(match_vis_path)
                        # plt.savefig(match_vis_path+ imgr[-1])
                    # 读取render图像深度 
                        if correct > max_correct:
                            max_correct = correct
                            imgr_name_final = imgr_name
                            depth_exr_final = depth_exr
                            mkpts1r_final = mkpts1r
                            mkpts0q_final = mkpts0q
                    if iterative_num == 1 or max_correct == 0:
                            render_prior_idx = all_render_name.index(imgr_name_final)
                            qvec, tvec = all_pose_c2w[render_prior_idx][2], all_pose_c2w[render_prior_idx][3]
                            qvec = ' '.join(map(str, qvec))
                            tvec = ' '.join(map(str, tvec))
                            name = 'render'+str(iterative_num)+'/'+sequence+imgq.split('.')[0] + '.png'
                            f.write(f'{name} {qvec} {tvec}\n') #!w2c 
                    else:    
                        depth,valid=read_valid_depth(depth_exr_final,mkpts1r_final)  #!depth render name bingo

                        #设置内参
                        render_idx = all_query_name.index(imgq_name)   #!query 内参缩小一倍
                        K_w2c = all_K_render[render_idx][1]   
                        K_w2c = torch.tensor(K_w2c).cpu().float()
                        K_c2w = K_w2c.inverse()

                        #读取render图的pose
                        render_prior_idx = all_render_name.index(imgr_name_final)
                        pose_c2w = torch.tensor(all_pose_c2w[render_prior_idx][1]).float()

                        # 根据render图像的深度图计算3D点
                        mkpts1r_final = mkpts1r_final.cpu()
                        mkpts0q_final = mkpts0q_final.cpu()
                        
                        mkpr_final = mkpts1r_final[valid]
                        mkpq_final = mkpts0q_final[valid].numpy()
                    
                        Points_3D = Get_Points3D(depth, pose_c2w[:3, :3], pose_c2w[:3, 3], K_c2w, mkpr_final)  #!k_c2w
                        Points_3D = Points_3D.T
                        Points_3D = Points_3D.numpy()
                        
                        # 重定位计算pose
                        query_idx = all_query_name.index(imgq_name)  
                        # focal_length = all_K_query[query_idx][2]
                        K_w2c_query = all_K_query[query_idx][1]
                        w, h = all_K_query[query_idx][2], all_K_query[query_idx][3]

                        # print('query_focal',focal_length)
                        ret = loc_query(imgq_pth,mkpq_final,Points_3D,K_w2c_query, w, h, max_error=12)
                        if ret['success'] ==True:
                            poses[imgq_name] = (ret['qvec'], ret['tvec'])
                            qvec, tvec = poses[imgq_name]
                            qvec = ' '.join(map(str, qvec))
                            tvec = ' '.join(map(str, tvec))
                            name = 'render'+str(iterative_num)+'/'+sequence+imgq.split('.')[0] + '.png'
                            f.write(f'{name} {qvec} {tvec}\n') #!w2c  
                            # print(imgq_name ,poses[imgq_name])
                        else:
                            no_match_count += 1
                            render_prior_idx = all_render_name.index(imgr_name_final)
                            qvec, tvec = all_pose_c2w[render_prior_idx][2], all_pose_c2w[render_prior_idx][3]
                            qvec = ' '.join(map(str, qvec))
                            tvec = ' '.join(map(str, tvec))
                            print("22222222222no matches ><")
                            name = 'render'+str(iterative_num)+'/'+sequence+imgq.split('.')[0] + '.png'
                            f.write(f'{name} {qvec} {tvec}\n') #!w2c
                pbar.update(1)
            pbar.close()  
        print("render....")
        image_save_path = render_path +'render'+str(iterative_num)+'/'+sequence
        #======================render===============
        cmd = '{} -b {} -P {} -- {} {} {}'.format(blender_path, 
                                                    rgb_path, 
                                                    python_rgb_path, 
                                                    render_camera,
                                                    output_file, 
                                                    image_save_path,
                                        )
        print(cmd)
        os.system(cmd)
        cmd = '{} -b {} -P {} -- {} {} {}'.format(blender_path, 
                                                    depth_path, 
                                                    python_depth_path, 
                                                    render_camera,
                                                    output_file, 
                                                    image_save_path,
                                        )
        os.system(cmd)
        r_pose_path = output_file
        #=====================render=============
        # 输出
        print('failed match:',no_match_count, 'percentage',(no_match_count/len(all_query_name))*100,'%' )
    t2 = time.time()
    print("time:", t2 - t1)
if __name__=="__main__":
    main()


