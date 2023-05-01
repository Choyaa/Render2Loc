import argparse
from lib import render2loc_meshloc, render2loc_spp_spg_meshloc
from lib.RtoQ import RtoQ
import json
import os
import sys


os.chdir("/home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',default='./config/config_evaluate.json', type=str, help='configuration file')
    args = parser.parse_args()
     
    with open(args.config_file) as fp:
        config = json.load(fp)
        
    method = 'superglue'
    print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Visualize matches of {method}')
    config_file = f'./config/{method}.yml'
    
    #========= Read .xml file =========== 
    # Extract 'camera.txt'+'image.txt'+'points.txt'
    # xml_read_scale.main(config["read_xml"])

    #========= Video extraction ==========
    # Video extraction from query
    # Intrincs extraction from query
    # video_frame_extraction.main(config["video_extraction"])

    # # # # #GPS extraction from query
    # SRT_gps.main(config["GPS_extraction"])

    # # # # #======= Render from .obj =============
    # # # # Input: .obj and video prior pose
    # # # # Output: rendered image

    # # # # First transfer the GPS to 45N
    # RtoQ(config["render_from_obj"]["input_pose"], config["GPS_extraction"]["save_path"])
    # # Update --input_camera
    # config["render_from_obj"].update({"input_camera" : config["video_extraction"]["intrinsics_save_path"]})

    # render_pyrender.main(config["render_from_obj"])

    #======= Render2loc ============
    # config["render2loc"].update({
    #     "render_path" : config["render_from_obj"]["save_path"],
    #     "input_camera" : config["video_extraction"]["intrinsics_save_path"],
    #     "r_pose_path" : config["render_from_obj"]["input_pose"],
    #     "depth_path"  : config["render_from_obj"]["depth_path"]
    # })
    
    
    # render2loc_spp_spg_meshloc.main(config["render2loc"], config_file)
    
    # eval.main(config["evaluate"])
    
    one_iterative_render2loc.main(config["render2loc"])
    
    # render2loc_spp_spg.main(config["render2loc"], config_file)

    # # render by Render2loc's results
    # config["render2loc"].update({
    #     "input_pose": config["render2loc"]["results"],
    #     "depth_path"  : config["render2loc"]["final_depth_path"]
    #     })
    # render_pyrender.main(config["render2loc"])
