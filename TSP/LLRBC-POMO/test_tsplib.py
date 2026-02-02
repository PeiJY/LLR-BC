##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
# USE_CUDA = not DEBUG_MODE
import torch
USE_CUDA = not DEBUG_MODE and torch.cuda.is_available()
CUDA_DEVICE_NUM = 0
import json

##########################################################################################
# Path Config
import time
import os
import sys
import argparse
import numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPlibTester import TSPTester as Tester


##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    'distribution': 'U',
    'distribution_params':[0,0]
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_tsp100_model2_longTrain',  # directory path of pre-trained model and log files saved.
        'epoch': 1200,  # epoch version of pre-trained model to laod.
    },
    # 'test_episodes': 100*1000,
    # 'test_batch_size': 10000,
    'test_episodes': 10000,
    'test_batch_size': 1000,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 1000,
    # ----- added ----
    'test_data_load': {
        'enable': True,
        'filename': '../TSP100_test_seed1234.pt'
    },
    # ----------------
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test',
        'filename': 'test_log.txt'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()
        
    
    
    create_logger(**logger_params)
    _print_config()

    for distribution in ['U','GM','R','E']:
        global env_params
        env_params['distribution'] = distribution
        global tester_params
        tester_params['test_data_load']['filename'] = '../generated_test_instances/TSPinstances_10000_{}_{}_0'.format(env_params['problem_size'],distribution)
        
        tester = Tester(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)

        copy_all_src(tester.result_folder)

        tester.run()


def test_tsplib(model_path,folder=None):

    device = torch.device('cuda', CUDA_DEVICE_NUM)
    t_test = time.time()
    if DEBUG_MODE:
        _set_debug_mode()
        
    create_logger(**logger_params)
    _print_config()
    results = {}

    json_file_path = folder+'/quick_test_tsplib_performance.json'
    if os.path.exists(json_file_path):
        print('load previous test result: ',json_file_path)
        with open(json_file_path, 'r', encoding='utf-8') as file:
            tresults = json.load(file)  
        if len(tresults.keys()) > len(results.keys()):
            results = tresults

    # all tsplib instance
    instance_folder = '../tsplib/'
    instance_names = sorted(os.listdir(instance_folder)) 
    instance_names = [name.split('.')[0] for name in instance_names]
    instance_path_list = [os.path.join(instance_folder, f+'.tsp') for f in instance_names]
    assert instance_path_list[-1].endswith(".tsp")
    
    # test
    t0 = time.time()
    for instance_index in range(len(instance_names)):
        path = instance_path_list[instance_index]
        name = instance_names[instance_index]
        if name in results.keys():
            print(f' -- {name} tested')
            continue
        print(f'-- begin test {name}')
        
        t1 = time.time()
        file = open(path, "r")
        lines = [ll.strip() for ll in file]
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.startswith("DIMENSION"):
                dimension = int(line.split(':')[1])
            elif line.startswith('NODE_COORD_SECTION'):
                locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=float)
                i = i + dimension
            i += 1
        original_locations = locations[:, 1:]
        original_locations = np.expand_dims(original_locations, axis=0)  # [1, n, 2]
        locations = torch.Tensor(original_locations / original_locations.max()).to(device)  # Scale location coordinates to [0, 1]
        loc_scaler = original_locations.max()
        env_params['problem_size'] = locations.size(1)
        env_params['pomo_size'] = locations.size(1)
        env_params['loc_scaler'] = loc_scaler
        # env = Env(**env_params)
        tester = Tester(env_params=env_params,
                        model_params=model_params,
                        tester_params=tester_params)
        score, aug_score = tester._test_one_batch(batch_size = locations.size(0),test_data=locations)
        # score = torch.round(score * loc_scaler).long()
        # aug_score = torch.round(aug_score * loc_scaler).long()
        score = score * loc_scaler
        aug_score = aug_score * loc_scaler

        results[name] = {'score':score,'aug_score':aug_score}
        sorted_result = {k: results[k] for k in sorted(results)}
        with open(json_file_path,'w') as file:
            print('         test finish, seond used: '+str(time.time()-t1) +', save test result as ',json_file_path)
            json.dump(sorted_result,file,indent=4)
    print('== total time used for test: ',time.time()-t0)


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    # --------- for lifelong --------
    parser = argparse.ArgumentParser(description="training parameters")
    parser.add_argument('--runname',type=str,help="run name")
    parser.add_argument('--size',type=int,help='instance size')
    parser.add_argument('--model',type=str,help='the path of model file to test')
    args = parser.parse_args()
    logger_params['log_file']['desc'] = args.runname
    size = args.size
    env_params['problem_size'] = size
    env_params['pomo_size'] = size
    tester_params['model_load']['path'] = args.model
    # ---------------- end ----------
    tester_params['model_load']['path'] = args.model.split('checkpoint-')[0]
    tester_params['model_load']['epoch'] = int(args.model.split('checkpoint-')[1].split('.')[0])
    folder = './tsplib_test/'
    if not os.path.exists(folder):
        os.mkdir(folder)
    
    folder += args.model.split('/')[-2] 
    if not os.path.exists(folder):
        os.mkdir(folder)
    test_tsplib(args.model,folder)
