import numpy as np
import os
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

def read_instrincs(intrinsc_path):
    all_K = []
    # all_query_name =[]
    with open(intrinsc_path,'r') as file:
        for line in file:
            data_line=line.strip("\n").split(' ')
            
            img_name = os.path.basename(data_line[0])
            w,h,fx,fy,cx,cy = list(map(float,data_line[2:8]))[:]   #! :8
            focal_length = fx
            K_w2c = np.array([
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 
            # all_query_name.append(img_name)
            all_K.append([data_line[0], img_name,K_w2c,focal_length, w, h])
    
    return all_K
def parse_pose_list(path):
    all_pose_c2w = []
    with open(path, 'r') as f:
        for data in f.read().rstrip().split('\n'):
            data = data.split()
            name = os.path.basename(data[0])
            if len(data) > 1:
                w2c_q, w2c_t = np.split(np.array(data[1:], float), [4])

                R = np.asmatrix(qvec2rotmat(w2c_q))   #!c2w
                t = -R.T @ w2c_t
                R = R.T
                Pose_c2w = np.identity(4)
                Pose_c2w[0:3,0:3] = R
                Pose_c2w[0:3, 3] = t
                all_pose_c2w.append([data[0], name, Pose_c2w, w2c_q, w2c_t])
        
    return all_pose_c2w

def get_pairs(all_render_name, all_query_name, render_path, query_path, iterative_num):   
    render_dir = {}
    for query_name in all_query_name:
        renders = []
        render_candidate = []
        render = []
        query = (query_name.split('/')[-1]).split('.')[0]
        imgq_pth = query_path + query_name
        for render_name in all_render_name:
            if query in render_name:
                render_candidate.append(render_name)
        for imgr_name in render_candidate: 
            imgr_pth = render_path + imgr_name
            imgr = imgr_name.split('/')
            if iterative_num == 0: 
                print(imgr_pth)
                exrr_pth = render_path + imgr[0] +'/' + imgr[1]+'/' + imgr[2] +'/'+ imgr[3] +'/' +'depth/'+query+'/'+ imgr[-1].split('.')[0]+'0001.exr'  #!
                
            else:
                exrr_pth = render_path + imgr_name.split('.')[0]+'0001.exr' 
            # exrr_pth = render_path + imgr.split('.')[0]+'0001.exr' #!
            render = [imgr_pth, exrr_pth]
            renders.append(render)
        render_dir[imgq_pth] = renders
           
    return render_dir  #return pairs dict
def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            if len(p) == 0:
                continue
            q, r = p.split()
            retrieval[q].append(r)
    return dict(retrieval)
def get_pairs_imagepath(pairs, depth_path, image_path):
    
    render_dir = {}
    for query_name, imgr_name_list in pairs.items():
        renders = []
        imgq_pth = image_path + query_name
        for imgr_name in imgr_name_list:
            imgr_pth = image_path + imgr_name
            imgr = imgr_name.split('/')[-1] 
            exrr_pth = depth_path + imgr.split('.')[0]+'0005.exr' #!  
            render = [imgr_pth, exrr_pth] 
            renders.append(render)
        render_dir[imgq_pth] = renders
    return render_dir
def main(config, data, iter):

    image_path = config["images_path"]
    render_path = config["render_path"]
    intrinscs_path = config["query_camera"]
    reference_camera = config["reference_camera"]
    r_pose_path = config["render_pose"] 
    retrieval_path = config["retrieval_path"]
    depth_path = config["depth_render_path"]
    # r_pose_path = result_path + str(iter-1) +'_query_estimated_pose.txt' #! #!##output_file  #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
       
    
    if "query name" not in data.keys():    
        all_K = read_instrincs(intrinscs_path)
        all_db_K = read_instrincs(reference_camera)
        all_query_name = list(np.array(all_K)[:, 1])
        all_query_dir = list(np.array(all_K)[:, 0])
        data["query_name"] = all_query_name
        data["intrinsics"] = all_K
        data["db_intrinsics"] = all_db_K
    else:
        all_query_name = data["query_name"]
    
    
    
  
    all_pose_c2w = parse_pose_list(r_pose_path)
    all_render_name = list(np.array(all_pose_c2w)[:, 1])
    
    # all_render_dir = list(np.array(all_pose_c2w)[:, 0])
    data["render_name"] = all_render_name
    data["pose"] = all_pose_c2w
    # all_pairs = get_pairs(all_render_dir, all_query_dir, render_path, image_path, iter)

    
    
    # assert retrieval_path.exists(), retrieval_path
    pairs = parse_retrieval(retrieval_path)
    all_pairs_path = get_pairs_imagepath(pairs, depth_path, image_path)
    data["pairs"] = all_pairs_path
    data["iter"] = iter   
    

    return data
