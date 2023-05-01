

day1
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/queriess.blend \
-P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_rgb.py \
--  2736 3648 27   /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/phone_day1_gt.txt \
/home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/   


/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/depth_render.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_depth.py --  2736 3648 27 /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/phone_day1_gt.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/   


day2-night4
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/night1/reference_rgb.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_rgb.py --  2736 3648 27   /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/phone_day2_gt.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/   
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/night1/reference.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_depth.py --  2736 3648 27 /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/results_gt/phone_day_sequence1/iterative3/query_estimated_pose.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/images/render_upright/render3/phone_day_sequence1/

dji day1 
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/queriess.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/dji_rgb.py --  /home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/queries_gt/test_focal/dj_day1_intrinsics.txt  /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/dj_day1_gt.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/   
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/depth_render.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/dji_depth.py --  /home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/queries_gt/test_focal/dj_day1_intrinsics.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/dj_day1_gt.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/   

night1

/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/night1/reference_rgb.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_rgb.py --  2736 3648 27   /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/phone_day2_gt.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_focal/   
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/night1/reference.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_depth.py --  2736 3648 27 /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/results_gt/phone_day_sequence1/iterative3/query_estimated_pose.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/images/render_upright/render3/phone_day_sequence1/


dji day2
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/djiday2/reference.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/r_blender_depth.py  --  5472 3648 24 /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/results/UAV_day_sequence2/iterative1/query_estimated_pose.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/images/images_upright/render2/UAV_day_sequence2/
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/djiday2/reference_rgb.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/r_blender_depth.py  --  5472 3648 24 /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/results/UAV_day_sequence2/iterative1/query_estimated_pose.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/images/images_upright/render2/UAV_day_sequence2/


gt/estimated
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/queriess.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_rgb.py --  /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/phone_day1_intrinsics.txt   /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/4_query_estimated_pose.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/   
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/queriess.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_rgb.py --  /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/phone_day1_intrinsics.txt   /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/phone_day1_gt.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/   



/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/night1/reference_rgb.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_rgb.py -- /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_intrinsics/que/phone_day1_intrinsics.txt   /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/results_gt/phone_night_sequence1/3_query_estimated_pose.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/images/render_upright/render3/phone_night_sequence1/   
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/night1/reference.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_depth.py --  /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_intrinsics/que/phone_day1_intrinsics.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/results_gt/phone_day_sequence1/iterative3/query_estimated_pose.txt  /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/images/render_upright/render3/phone_day_sequence1/


/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/reference_rgb.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_rgb.py  --  /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/dj_day_intrinsics.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/4_query_estimated_pose.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/


/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/reference_rgb.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/refer_rgb.py --  /home/ubuntu/Documents/blender_render/referencee/refer_camera.txt   /home/ubuntu/Documents/blender_render/referencee/images.txt /home/ubuntu/Documents/blender_render/referencee/   
/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/reference.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/refer_depth.py --  /home/ubuntu/Documents/blender_render/referencee/refer_camera.txt   /home/ubuntu/Documents/blender_render/referencee/render2loc_images.txt /home/ubuntu/Documents/blender_render/referencee/   



/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender \
-b /home/ubuntu/Documents/1-pixel/blender_demo/djiday2/reference_rgb.blend \
-P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/refer_rgb.py \
-- /home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/queries_intrinsics/que/dj_day1_intrinsics.txt  \
/home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/AirLoc/queries_gt/single/dj_day1_gt.txt \
/home/ubuntu/Documents/blender_render/nerf/   

/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender \
-b /home/ubuntu/Documents/1-pixel/render/blender_demo/djiday2/reference.blend \
-P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/r_blender_depth.py  \
-- /home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/queries_intrinsics/que/dj_day1_intrinsics.txt  \
/home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/AirLoc/queries_gt/single/dj_day1_gt.txt \
/home/ubuntu/Documents/blender_render/nerf/ 



 /home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender \
 -b /home/ubuntu/Documents/1-pixel/blender_demo/reference_rgb.blend \
 -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_rgb.py \
 --  /home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/queries_intrinsics/que/phone_day2_intrinsics.txt \
/home/ubuntu/Documents/1-pixel/1-jinxia/jinxia_datasets/queries_gt/single/phone_day2_gt.txt   \
/home/ubuntu/Documents/blender_render/nerf/   


/home/ubuntu/Downloads/blender-3.3.1-linux-x64/blender -b /home/ubuntu/Documents/1-pixel/render/blender_demo/queriess.blend -P /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/lib/phone_rgb.py --  /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/phone_day1_intrinsics.txt   /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/phone_day1_gt.txt /home/ubuntu/Documents/1-pixel/1-jinxia/icme_relocalization/jinxia/queries_gt/test_gt/   
 
