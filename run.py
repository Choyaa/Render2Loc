import os
import sys
import argparse
from lib import (
    match_feature,
    pair_from_pose,
    render2loc,
    eval,
    render2loc_spp_spg,
    blender_engine,
)
import json
import yaml

os.chdir(sys.path[0])
import immatch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_file',
        default='./config/config_evaluate.json',
        type=str,
        help='configuration file',
    )
    args = parser.parse_args()

    with open(args.config_file) as fp:
        config = json.load(fp)

    method = 'superglue'
    print(f'\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Visualize matches of {method}')
    config_file = f'./config/{method}.yml'

    with open(config_file, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)['airloc']
        if 'ckpt' in args:
            args['ckpt'] = os.path.join('..', args['ckpt'])
        class_name = args['class']

    # Init model
    model = immatch.__dict__[class_name](args)
    matcher = lambda im1, im2: model.match_pairs(im1, im2)

    # ======= Render2loc ============
    # ========== Read query RTK(CC aquirement) from RTK equipment
    # read_queryRTK_cc.main(config["read_phoneRTK"])

    # ========== generate query GT
    # generate_query_GT.main(config["read_queryGT"])

    # ========== generate query prior
    # generate_query_prior.main(config["read_queryPrior"])
    # generate_DJI_query_prior.main(config["read_uavPrior"])

    # ========== generate seed
    # add_seed.main(config["add_seed"])
    # ==========
    data = dict()

    data = pair_from_pose.main(config["render2loc"], data, 0)

    matches = match_feature.main(config["render2loc"], data, matcher)

    render2loc_spp_spg.main(config["render2loc"], data, matches)  #!update pose file
    for iter in range(1, 4):

        data = pair_from_pose.main(config["render2loc"], data, iter)

        blender_engine.main(config["render2loc"], data)

        matches = match_feature.main(config["render2loc"], data, matcher)

        render2loc_spp_spg.main(config["render2loc"], data, matches)  #!update pose file

    # ============= eval
    # eval.main(config["evaluate"])
