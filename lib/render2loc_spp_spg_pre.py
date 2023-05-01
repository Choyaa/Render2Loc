import os
os.chdir("..")
import cv2
import pycolmap
import os
import torch
import numpy as np
import matplotlib.cm as cm
from src.utils.plotting import make_matching_figure
from tqdm import tqdm


os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

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

def filter_mkp(mkpq,mkpr,mconf,conf_thres=0.7):
    correct = [idx for idx , value in enumerate(mconf) if value>conf_thres]
    filter_mkpr = mkpr[correct] 
    filter_mkpq = mkpq[correct]
    filter_mconf = mconf[correct]
    return filter_mkpq,filter_mkpr,filter_mconf ,len(correct)

def match_pair(imgq_pth,imgr_pth,matcher):

    scale = 8
    imgq_raw = cv2.imread(imgq_pth, cv2.IMREAD_GRAYSCALE)
    imgr_raw = cv2.imread(imgr_pth, cv2.IMREAD_GRAYSCALE)
    imgq_resize = cv2.resize(imgq_raw, (int(imgq_raw.shape[1]/scale), int(imgq_raw.shape[0]/scale)))
    imgr_resize = cv2.resize(imgr_raw, (int(imgr_raw.shape[1]/scale), int(imgr_raw.shape[0]/scale)))
    imgq_8 = cv2.resize(imgq_resize, (imgq_resize.shape[1]//8*8, imgq_resize.shape[0]//8*8))  # input size shuold be divisible by 8
    imgr_8 = cv2.resize(imgr_resize, (imgr_resize.shape[1]//8*8, imgr_resize.shape[0]//8*8))

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

def loc_query(mkpq, mkp3d, K_w2c, w, h, max_error=5):
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
            mkpq.numpy(), mkp3d.numpy(), cfg, max_error)

    return ret

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
            focal_length = fx
            K_w2c = np.array([
            [fx,0.0,cx],
            [0.0,fy,cy],
            [0.0,0.0,1.0],
            ]) 
            # all_query_name.append(img_name)
            all_K.append([img_name,K_w2c,focal_length])
    
    return all_K
def blender_load(blender_path, project_path, script_path, intrinscs_path, extrinsics_path, image_save_path):
    '''
    blender_path: .exe path, start up blender
    project_path: .blend path, 
    script_path: .py path, batch rendering script
    intrinscs_path: colmap format
    extrinsics_path: colmap format
    image_save_path: rendering image save path
    '''
    cmd = '{} -b {} -P {} -- {} {} {}'.format(blender_path, 
                                            project_path, 
                                            script_path, 
                                            intrinscs_path,
                                            extrinsics_path, 
                                            image_save_path,
                                )
    os.system(cmd)           
    
def main(config, data, matches):
    render_path = config["render_path"]
    intrinscs_path = config["query_camera"]
    save_loc_path = config["results"]
    python_rgb_path = config["python_rgb_path"]
    python_depth_path = config["python_depth_path"]
    rgb_path = config["rgb_path"]
    depth_path = config["depth_path"]
    blender_path = config["blender_path"]
    sequence =config["sequence"]
    
    iterative_num = data["iter"]
    all_K= data["intrinsics"]
    all_query_name = data["query_name"]
    all_pose_c2w = data["pose"]
    all_render_name = data["render_name"]
    
    if not os.path.exists(save_loc_path):
        os.makedirs(save_loc_path)
    output_file = save_loc_path + str(iterative_num) +'_query_estimated_pose.txt' #!
    
    poses={}
    no_match_count = 0

    pbar = tqdm(total=len(all_query_name), unit='pts')
    with open(output_file, 'w') as f:
        for match in matches:  
            imgr_name_final = match["imgr_name"]
            imgq_name = match["imgq_name"]
            depth_exr_final = match["exrr_pth"]
            max_correct = match["correct"]
            mkpts1r_final = match["mkpts_r"]
            mkpts0q_final = match["mkpts_q"]
            if iterative_num == 1 or max_correct == 0:
                print("no match")
                render_prior_idx = all_render_name.index(imgr_name_final)
                qvec, tvec = all_pose_c2w[render_prior_idx][3], all_pose_c2w[render_prior_idx][4]
                qvec = ' '.join(map(str, qvec))
                tvec = ' '.join(map(str, tvec))
                name = 'render'+str(iterative_num)+'/'+sequence+imgq_name.split('.')[0] + '.png'
                f.write(f'{name} {qvec} {tvec}\n') 
            else:    
                depth, valid=read_valid_depth(depth_exr_final,mkpts1r_final)  

                # read instinsics for query and render
                idx = all_query_name.index(imgq_name)   
                K_w2c = all_K[idx][2]   
                weight = all_K[idx][4]
                height = all_K[idx][5]
                K_w2c = torch.tensor(K_w2c).cpu().float()
                K_c2w = K_w2c.inverse()

                # read extinsics
                render_prior_idx = all_render_name.index(imgr_name_final)
                pose_c2w = torch.tensor(all_pose_c2w[render_prior_idx][2]).float()

                # pnp
                mkpr_final = mkpts1r_final[valid].cpu()
                mkpq_final = mkpts0q_final[valid].cpu()
            
                Points_3D = Get_Points3D(depth, pose_c2w[:3, :3], pose_c2w[:3, 3], K_c2w, mkpr_final)  
                Points_3D = Points_3D.T
                
                result = loc_query(mkpq_final, Points_3D, K_w2c, weight, height, max_error=12)
                if result['success'] ==True:
                    poses[imgq_name] = (result['qvec'], result['tvec'])
                    qvec, tvec = poses[imgq_name]
                    qvec = ' '.join(map(str, qvec))
                    tvec = ' '.join(map(str, tvec))
                    name = 'render'+str(iterative_num)+'/'+sequence+imgq_name.split('.')[0] + '.png'
                    f.write(f'{name} {qvec} {tvec}\n')  

                else:
                    render_prior_idx = all_render_name.index(imgr_name_final)
                    qvec, tvec = all_pose_c2w[render_prior_idx][3], all_pose_c2w[render_prior_idx][4]
                    qvec = ' '.join(map(str, qvec))
                    tvec = ' '.join(map(str, tvec))
                    print("reloc unsuccess")
                    name = 'render'+str(iterative_num)+'/'+sequence+imgq_name.split('.')[0] + '.png'
                    f.write(f'{name} {qvec} {tvec}\n') 
            pbar.update(1)
        pbar.close()  
    print("render....")
    image_save_path = render_path +'render'+str(iterative_num)+'/'+sequence
    # render rgb and depth images.
    blender_load(blender_path, rgb_path, python_rgb_path, intrinscs_path, output_file, image_save_path)
    blender_load(blender_path, depth_path, python_depth_path, intrinscs_path, output_file, image_save_path)

    # update pose 
    config["render_pose"]  =  output_file


if __name__=="__main__":
    main()


