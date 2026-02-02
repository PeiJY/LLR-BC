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

import os
import sys
import argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src
import time
import random
import numpy as np
##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    'distribution': 'U',
    "sizes" : {'U':20,'Ring':20,'E':50,'Grid':50,'GM':100,'C':100},
    "batch_sizes" : {'U':32,'Ring':32,'E':32,'Grid':32,'GM':32,'C':32},
    "task_order": ['U','E','Ring','Grid','C','GM'],
    "task_interval":200,
    "batch_per_epoch":128,
    'problem_sample_log':False,
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

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [191,],
        'gamma': 0.1
    }
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_tsp100_model2_longTrain',  # directory path of pre-trained model and log files saved.
        'epoch': 3100,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 1*1000,
    'test_batch_size': 64,
    "episodes" : {'U':10 * 1000,'Ring':10 * 1000,'E':4 * 1000,'Grid':4 * 1000,'GM':2 * 1000,'C':2 * 1000},
    # "batch_sizes" : {'U':64,'Ring':64,'E':32,'Grid':32,'GM':16,'C':16},
    "batch_sizes" : {'U':32,'Ring':32,'E':32,'Grid':32,'GM':32,'C':32},
    'augmentation_enable': False,
    'aug_factor': 1,
    'aug_batch_size': 1000,
    # ----- added ----
    'test_data_load': {
        'enable': True,
        'filename': '../TSP100_test_seed1234.pt'
    },
    'baseline_solution_load':
    {
        'enable': True,
        'filename': '../vrp100_test_seed1234_HGS_solution'
    }
    # ----------------
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'train',
        'filename': 'train_log.txt'
    }
}



##########################################################################################
from TSPTester import TSPTester as Tester
import re

def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################

def test_main(se,ee,it=None,folder=None):
    t_test = time.time()
    if DEBUG_MODE:
        _set_debug_mode()
        
    create_logger(**logger_params)
    _print_config()
    results = {}
    all_distributions = ['U','GM','E','C','Ring','Grid']
    
    
    
    json_file_path = folder+'quick_test_performance.json'
    if os.path.exists(json_file_path):
        print('load previous test result: ',json_file_path)
        with open(json_file_path, 'r', encoding='utf-8') as file:
            tresults = json.load(file) 
        if len(tresults.keys()) > len(results.keys()):
            results = tresults

    # Regular expression pattern to match 'checkpoint-<number>'
    pattern = re.compile(r'checkpoint-(\d+)')

    # List to store the extracted numbers
    epochs = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder):
        # Match the filename against the pattern
        match = pattern.match(filename)
        if match:
            # Extract the number and convert it to an integer
            number = int(match.group(1))
            if number >= se and number <= ee:
                if it is not None and not number % it ==0:
                    continue
                epochs.append(number)
    epochs=sorted(epochs)    
    
    for epoch in epochs:
        print('test checkpoint at epoch: ',epoch)
        if str(epoch) in results.keys():
            print('epoch {} already tested'.format(epoch))
            to_test_distribution = [d for d in all_distributions if d not in results[str(epoch)].keys()]
            print('---------- distribution already test: ',results[str(epoch)].keys(),', to test: ',to_test_distribution)
        else:
            results[str(epoch)] = {}
            to_test_distribution = all_distributions
        t0 = time.time()
        for distribution in to_test_distribution:
            t1 = time.time()
            print(' -- test distribution: ',distribution)
            global env_params
            env_params['distribution'] = distribution
            size = env_params["sizes"][distribution]
            env_params['problem_size'] = size
            env_params['pomo_size'] = size
            global tester_params
            # tester_params['test_data_load']['filename'] = '../generated_test_instances/CVRPinstances_{}_{}_100000_seed0'.format(distribution,env_params['problem_size'])
            tester_params['test_data_load']['filename'] = '../generated_test_instances/TSPinstances_{}_{}_10000_seed0'.format(distribution,env_params['problem_size'])
            if not os.path.exists(tester_params['model_load']['path']+'checkpoint-{}.pt'.format(epoch)):
                print('model not exist: ',tester_params['model_load']['path']+'checkpoint-{}.pt'.format(epoch))
                continue

            tester_params['baseline_solution_load']['filename'] = tester_params['test_data_load']['filename'] + '_Gurobi_solution'
            if not os.path.exists(tester_params['baseline_solution_load']['filename']):
                print('baseline solution not exist: ',tester_params['baseline_solution_load']['filename'])
                tester_params['baseline_solution_load']['enable'] = False
                tester_params['baseline_solution_load']['filename'] = None
            
            tester_params['model_load']['epoch'] = epoch
            tester_params['test_batch_size']=tester_params['batch_sizes'][distribution]
            tester_params['aug_batch_size'] = tester_params['batch_sizes'][distribution]/8
            print('test with batch size and aug batch size as ',tester_params['test_batch_size'],tester_params['aug_batch_size'])
            tester = Tester(env_params=env_params,
                            model_params=model_params,
                            tester_params=tester_params)

            # copy_all_src(tester.result_folder)
            if tester_params['baseline_solution_load']['enable']:
                score,aug_score,gap = tester.run()
                results[str(epoch)][distribution] = {'score':score,'aug_score':aug_score,'gap':gap}
            else:
                score,aug_score = tester.run()
                results[str(epoch)][distribution] = {'score':score,'aug_score':aug_score}
            sorted_result = {k: results[k] for k in sorted(results, key=int)}
            with open(json_file_path,'w') as file:
                print('test finish, seond used: '+str(time.time()-t1) +', save test result as ',json_file_path)
                json.dump(sorted_result,file,indent=4)
        print(f'== second used for epoch {epoch}: ',time.time()-t0)
    print('== total time used for test: ',time.time()-t_test)

#########################################################################################

orders = [
    ['E', 'C', 'Grid', 'U', 'Ring', 'GM'],
    ['U', 'GM', 'E', 'Ring', 'Grid', 'C'],
    ['E', 'Grid', 'Ring', 'C', 'U', 'GM'],
    ['Grid', 'GM', 'E', 'U', 'Ring', 'C'],
    ['Grid', 'C', 'Ring', 'U', 'GM', 'E']
    ]
import time
if __name__ == "__main__":
    t0 = time.time()
    # --------- for lifelong --------
    parser = argparse.ArgumentParser(description="training parameters")
    parser.add_argument('--runname',type=str,default='debug',help="run name")
    parser.add_argument('--folder',type=str,help="folder of model checkpoints to test")
    args = parser.parse_args()
    logger_params['log_file']['desc'] = args.runname

    check_point_folder = args.folder

    # test
    tester_params['model_load']['path'] = check_point_folder

    
    test_main(0,1201,1,check_point_folder)
    
    print(' total hours used: ',(time.time()-t0)/3600)