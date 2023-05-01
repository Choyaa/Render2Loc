import argparse
from lib import iterative_relocalization, video_frame_extraction , render_pyrender ,SRT_gps, render2loc, eval
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
    
    
    # render2loc.main(config["render2loc"])
    eval.main(config["evaluate"])
    
    # iterative_relocalization.main(config["render2loc"])

    # # render by Render2loc's results
    # config["render2loc"].update({
    #     "input_pose": config["render2loc"]["results"],
    #     "depth_path"  : config["render2loc"]["final_depth_path"]
    #     })
    # render_pyrender.main(config["render2loc"])
