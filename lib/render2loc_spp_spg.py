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
import h5py
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


def interpolate_depth(pos, depth):
    ids = torch.arange(0, pos.shape[0])
    depth = depth[:,:,0]
    h, w = depth.size()
    
    
    
    i = pos[:, 0]
    j = pos[:, 1]

    # Valid corners, check whether it is out of range
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


    # Valid depth
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

    # vaild index
    ids = ids[valid_depth]
    
    i = i[ids]
    j = j[ids]
    dist_i_top_left = i - i_top_left.float()
    dist_j_top_left = j - j_top_left.float()
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    #depth is got from interpolation
    interpolated_depth = (
        w_top_left * depth[i_top_left, j_top_left] +
        w_top_right * depth[i_top_right, j_top_right] +
        w_bottom_left * depth[i_bottom_left, j_bottom_left] +
        w_bottom_right * depth[i_bottom_right, j_bottom_right]
    )

    pos = torch.cat([i.view(1, -1), j.view(1, -1)], dim=0)

    return [interpolated_depth, pos, ids]

# read depth
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
    

    cfg = {
            'model': 'SIMPLE_PINHOLE',
            'width': width,
            'height': height,
            'params': [focal_length, cx, cy]
            
        }
    ret = pycolmap.absolute_pose_estimation(
            mkpq.numpy(), mkp3d.numpy(), cfg, max_error)

    return ret

def loc_query_wa(mkpq, mkp3d, K_w2c, w, h, max_error=5):
    height, width = int(h), int(w)
    cx = K_w2c[0][2]
    cy = K_w2c[1][2]
    fx = K_w2c[0][0]
    fy = K_w2c[1][1]
    for i in range(0, mkpq.shape[0]):
        keypoint = mkpq
    
    
    
    cfg = {
            'model': 'PINHOLE',
            'width': width,
            'height': height,
            'params': [fx, fy, cx, cy]
            
        }
    
    ret = pycolmap.absolute_pose_estimation(
            mkpq.numpy(), mkp3d.numpy(), cfg, max_error)

    return ret

def Get_Points3D(depth, R, t, K, points):   # points[n,2]
    if points.shape[-1] != 3:
        points_2D = torch.cat([points, torch.ones_like(points[ :, [0]])], dim=-1)
        points_2D = points_2D.T  
    t = torch.unsqueeze(t,-1).repeat(1, points_2D.shape[-1])
    Points_3D = R @ K @ (depth * points_2D) + t   
    return Points_3D    #[3,n]

def blender_engine(blender_path, project_path, script_path, intrinscs_path, extrinsics_path, image_save_path):
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
    save_loc_path = config["results"]
    sequence =config["sequence"]
    
    iterative_num = data["iter"]
    all_K= data["intrinsics"]
    all_db_K= data["db_intrinsics"]
    all_query_name = data["query_name"]
    all_pose_c2w = data["pose"]
    all_render_name = data["render_name"]
    
    if not os.path.exists(save_loc_path):
        os.makedirs(save_loc_path)
    output_file = save_loc_path + str(iterative_num) +'_query_estimated_pose.txt' #!
    
    pbar = tqdm(total=len(all_query_name), unit='pts')
    
    with open(output_file, 'w') as f:
        for match in matches:  
            imgr_name_final = match["imgr_name"]
            imgq_name = match["imgq_name"]
            depth_exr_final = match["exrr_pth"]
            max_correct = match["correct"]
            mkpts1r_final = match["mkpts_r"]
            mkpts0q_final = match["mkpts_q"]
            if max_correct == 0:
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
                
                K_db_w2c = all_K[0][2]   
                K_db_w2c = torch.tensor(K_db_w2c).cpu().float()
                K_db_c2w = K_db_w2c.inverse()                
            
                Points_3D = Get_Points3D(depth, pose_c2w[:3, :3], pose_c2w[:3, 3], K_db_c2w, mkpr_final)  
                Points_3D = Points_3D.T

                result = loc_query_wa(mkpq_final, Points_3D, K_w2c, weight, height, max_error=12)
                if result['success'] ==True:
                    qvec, tvec = result['qvec'], result['tvec']
                    qvec = ' '.join(map(str, qvec))
                    tvec = ' '.join(map(str, tvec))
                    name = 'render'+str(iterative_num+1)+'/'+sequence+imgq_name.split('.')[0] + '.png'
                    f.write(f'{name} {qvec} {tvec}\n')  
                else:
                    render_prior_idx = all_render_name.index(imgr_name_final)
                    qvec, tvec = all_pose_c2w[render_prior_idx][3], all_pose_c2w[render_prior_idx][4]
                    qvec = ' '.join(map(str, qvec))
                    tvec = ' '.join(map(str, tvec))
                    print("reloc unsuccess")
                    name = 'render'+str(iterative_num+1)+'/'+sequence+imgq_name.split('.')[0] + '.png'
                    f.write(f'{name} {qvec} {tvec}\n') 
                if iterative_num == 3:
                    
                    d23 = dict()
                    feature_path = "/home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization_github_complete/yanhao.h5"
                    # file = h5py.File(feature_path1, 'w')
                    with h5py.File(str(feature_path), 'a') as fd:
                        
                        grp = fd.create_group(imgq_name)
                    #     cx = K_w2c[0][2]
                    #     cy = K_w2c[1][2]
                    #     fx = K_w2c[0][0]
                    #     fy = K_w2c[1][1]
                    
                    #     cfg = {
                    #         'model': 'PINHOLE',
                    #         'width': int(weight),
                    #         'height': int(height),
                    #         'params': [fx, fy, cx, cy]
                    #             }
                    #     d23["cfg"] = cfg
                        d23["points_2D"] = mkpq_final
                        d23["points_3D"] = Points_3D
                        for k, v in d23.items():
                            grp.create_dataset(k, data=v)                       
            pbar.update(1)
        pbar.close()  

    # update pose 
    config["render_pose"]  =  output_file


if __name__=="__main__":
    main()


