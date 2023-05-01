import os
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
    
def main(config, data):
    render_path = config["render_path"]
    intrinscs_path = config["query_camera"]
    extinsics_path = config["render_pose"]
    python_rgb_path = config["python_rgb_path"]
    python_depth_path = config["python_depth_path"]
    rgb_path = config["rgb_path"]
    depth_path = config["depth_path"]
    blender_path = config["blender_path"]
    sequence =config["sequence"]
    render_path = config["render_path"]
    
    iterative_num = data["iter"]
    
    print("render....")
    image_save_path = render_path +'render'+str(iterative_num)+'/'+sequence
    
    # render rgb and depth images.
    blender_engine(blender_path, rgb_path, python_rgb_path, intrinscs_path, extinsics_path, image_save_path)
    blender_engine(blender_path, depth_path, python_depth_path, intrinscs_path, extinsics_path, image_save_path)
    