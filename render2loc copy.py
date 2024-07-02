from pathlib import Path
import json
import time
from lib import (
    localize_render2loc,
    match_feature,
    pair_from_seed,
    eval,
    generate_seed
)
from utils.blender import (
    blender_engine, 
    blender_obj_loader
)


class Render2Loc:
    def __init__(self, config_file='configs/config_demo.json'):
        with open(config_file) as fp:
            self.config = json.load(fp)
        
        self.dataset = Path(self.config['render2loc']['datasets'])
        self.images = self.dataset / 'images/images_upright'
        self.outputs = self.dataset / self.config['render2loc']['results']
        
        self.render_camera = self.dataset / self.config['render2loc']['query_camera']
        self.query_camera = self.dataset / self.config['render2loc']['query_camera']
        self.gt_pose = self.dataset / self.config['evaluate']['gt_pose']
        self.dsm_filepath = self.dataset / self.config['render2loc']['dsm_file']
        self.prior_poses = self.dataset / self.config['render2loc']['prior_pose']
        
        self.engine = self.config['render2loc']['engine']
        self.dev = self.config['render2loc']['dev']
        self.distortion = self.config['render2loc']['distortion']
        self.data = dict()
        
        # init texture model
        blender_obj_loader.main(self.config)
        
        # init matcher
        method = 'superglue'
        print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Visualize matches of {method}')
        config_file = f'configs/{method}.yml'
        self.matcher = match_feature.ImageMatcher(config_file = config_file)
        
        # init localizer
        self.localizer = localize_render2loc.QueryLocalizer(config=self.config)
        
    def UAV_localization_with_prior(self):
        render_poses = self.dataset / self.config['render2loc']['render_poses']
        render_images = self.dataset / self.config['render2loc']['render_images']
        start_time = time.time()
        data = dict()
        # TODO: uav or phone
        # distortation
        
        # exploration 
        generate_seed.main(str(self.dsm_filepath), render_poses, self.prior_poses, dsm=True)

        # exploitation
        for iter in range(1):
            data = pair_from_seed.main(
                           self.images, 
                           render_images, 
                           self.query_camera,
                           self.render_camera,
                           render_poses,
                           data)
            blender_engine.main(self.config, 
                        self.render_camera, 
                        render_poses, 
                        render_images)
            self.matcher.main(data, self.outputs)
           
            render_poses = self.localizer.main(data, iter)  # update pose file
            
        blender_engine.main(self.config, 
                        self.render_camera, 
                        render_poses, 
                        render_images)
        eval.pose(self.gt_pose, render_poses)
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 50
        print(f"执行时间：{elapsed_time}秒")   



if __name__ == "__main__":
    # 初始化
    render2loc = Render2Loc('configs/config_demo.json')

    # 定位
    ret = render2loc.UAV_localization_with_prior()
    
    

    
    
    