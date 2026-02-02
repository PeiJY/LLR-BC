##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
# USE_CUDA = not DEBUG_MODE
import torch
USE_CUDA = not DEBUG_MODE and torch.cuda.is_available()
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys
import argparse
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils
import random
import numpy as np

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from TSPTrainer import TSPTrainer as Trainer
from TSPTrainer import TSPTrainer_EWC as Trainer_EWC
from TSPTrainer import TSPTrainer_LiBOG as Trainer_LiBOG
from TSPTrainer import TSPTrainer_LLRBC as Trainer_LLRBC
import time
from collections import Counter
##########################################################################################
# parameters
env_params = {
    'problem_size': 100,
    'pomo_size': 100,
    'distribution': 'U',
    "sizes" : {'U':20,'Ring':20,'E':50,'Grid':50,'GM':100,'C':100},
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

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 200,
    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
    'aug_train':False,
    'prev_model_path': None,
    'fixed_init_model_path':None,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': 'toset',  # directory path of pre-trained model and log files saved.
        'epoch': 0,  # epoch version of pre-trained model to laod.

    },
    # ------ for switch task ----
    "episodes" : {'U':10 * 1000,'Ring':10 * 1000,'E':4 * 1000,'Grid':4 * 1000,'GM':2 * 1000,'C':2 * 1000},
    "batch_sizes" : {'U':64,'Ring':64,'E':32,'Grid':32,'GM':16,'C':16},
    # ------ for EWC -----
    'EWC_fisher_batch_number':64,
    'EWC_lambda':1,
    'fisher_normalise':False,
    # ------ for ITC ------
    'ITC_weight' : 1e-3,
    # ----- for LLRBC -----
    'consolidation_loss_weight':100,
    'reverse_KLD_clone':False,
    'replay_buffer_size': 1000,
    'clone_sample_size': 16,
    'temperature_clone':1.0,
    'confidence_weight_KLD':False,
    'confidence_weight_add':False,
    'confidence_method':'var',
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
    "batch_sizes" : {'U':64,'Ring':64,'E':32,'Grid':32,'GM':16,'C':16},
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 1000,
    'test_data_load': {
        'enable': True,
        'filename': '../TSP100_test_seed1234.pt'
    },
    'baseline_solution_load': {
        'enable': True,
        'filename': '../vrp100_test_seed1234_HGS_solution'
    },
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
            if not os.path.exists(tester_params['test_data_load']['filename']+'_Gurobi_solution'):
                print('baseline solution not exist: ',tester_params['baseline_solution_load']['filename'])
                tester_params['baseline_solution_load']['filename'] = None
                tester_params['baseline_solution_load']['enable'] = False
            tester_params['model_load']['epoch'] = epoch
            tester_params['test_batch_size']=tester_params['batch_sizes'][distribution]
            tester_params['aug_batch_size'] = tester_params['batch_sizes'][distribution]/8
            print('test with batch size and aug batch size as ',tester_params['test_batch_size'],tester_params['aug_batch_size'])
            tester = Tester(env_params=env_params,
                            model_params=model_params,
                            tester_params=tester_params)

            # copy_all_src(tester.result_folder)

            if not tester_params['baseline_solution_load']['enable']:
                score,aug_score = tester.run()
                results[str(epoch)][distribution] = {'score':score,'aug_score':aug_score}
            else:
                score, aug_score, gap = tester.run()
                results[str(epoch)][distribution] = {'score':score,'aug_score':aug_score,'gap':gap}
            sorted_result = {k: results[k] for k in sorted(results, key=int)}
            with open(json_file_path,'w') as file:
                print('test finish, seond used: '+str(time.time()-t1) +', save test result as ',json_file_path)
                json.dump(sorted_result,file,indent=4)
        print(f'== second used for epoch {epoch}: ',time.time()-t0)
    print('== total time used for test: ',time.time()-t_test)


##########################################################################################
# main
import json
import math
def main(task_order=['U','E','GM','C','Ring','Grid'],llmethod = 'FT',from_st=False):
    global trainer_params
    t0 = time.time()
    if DEBUG_MODE:
        _set_debug_mode()
        
    create_logger(**logger_params)
    _print_config()

    epochs_each_task = trainer_params['epochs']
    current_task_index = 0
    if trainer_params['model_load']['enable']:
        epoch = trainer_params['model_load']['epoch'] + 1
        current_task_index = math.floor(epoch / epochs_each_task)
        first_epochs_each_task= epochs_each_task-(epoch % epochs_each_task)
    else:
        epoch = 0
        first_epochs_each_task = epochs_each_task

    if llmethod == 'LLRBC':
        trainer = Trainer_LLRBC(env_params=env_params,
                    model_params=model_params,
                    optimizer_params=optimizer_params,
                    trainer_params=trainer_params)
    elif llmethod == 'FT':
        print('creat trainer: Trainer')
        trainer = Trainer(env_params=env_params,
                        model_params=model_params,
                        optimizer_params=optimizer_params,
                        trainer_params=trainer_params)
    elif llmethod == 'EWC':
        print('creat trainer: Trainer_EWC')
        trainer = Trainer_EWC(env_params=env_params,
                        model_params=model_params,
                        optimizer_params=optimizer_params,
                        trainer_params=trainer_params)
    elif llmethod == 'LiBOG':
        print('creat trainer: Trainer_LiBOG')
        trainer = Trainer_LiBOG(env_params=env_params,
                        model_params=model_params,
                        optimizer_params=optimizer_params,
                        trainer_params=trainer_params)
    else:
        raise ValueError('lifelong learning method {} not exist'.format(llmethod))

    # copy_all_src(trainer.result_folder)

    check_point_folder = trainer.result_folder +'/'

    # save training hyper_parameters
    hyper_parameters = {}
    hyper_parameters['env_params'] = env_params
    hyper_parameters['model_params'] = model_params
    hyper_parameters['optimizer_params'] = optimizer_params
    hyper_parameters['trainer_params'] = trainer_params
    with open(check_point_folder+'hyper_parameters.json', 'w') as f:
        json.dump(hyper_parameters, f)

    first_task = True  # for recoving training
    for dis_index in range(len(task_order)):
    # for distribution in task_order:
        if dis_index < current_task_index: # recover to the task corresponding to the load epoch
            continue
        distribution = task_order[dis_index]
        print('== train on distribution: ',distribution)
        size = env_params['sizes'][distribution]
        env_params['distribution'] = distribution
        env_params['pomo_size'] = size
        env_params['problem_size'] = size
        # --- switch task --- 
        if llmethod in ['EWC','LiBOG']:
            if first_task:
                trainer.change_env(env_params,calculate_save_fisher=False)
            else:
                trainer.change_env(env_params,calculate_save_fisher=True)
        elif llmethod == 'LLRBC':
            if first_task:
                trainer.change_env(env_params,re_init_optimizer=False,do_buffer_copy=False)
            else:
                trainer.change_env(env_params)
        else: # FT
            if first_task:
                trainer.change_env(env_params,re_init_optimizer=False)
            else:
                trainer.change_env(env_params)
        # --------------------
        trainer.start_epoch = epoch + 1
        if first_task:  
            trainer.trainer_params['epochs'] = epoch + first_epochs_each_task
        else:   
            trainer.trainer_params['epochs'] = epoch + epochs_each_task
        trainer.run()
        if first_task:  
            epoch = epoch + first_epochs_each_task
        else:   
            epoch = epoch + epochs_each_task
        first_task=False


        if len(trainer.old_experience_buffer.buffer)>0 and 'task' in trainer.old_experience_buffer.buffer[0].keys():
            lst = [exp['task'] for exp in trainer.old_experience_buffer.buffer]
            counts = Counter(lst)
            print('task belonging count in old_experience_buffer:', counts) 
        if len(trainer.updating_experience_buffer.buffer)>0 and 'task' in trainer.updating_experience_buffer.buffer[0].keys():
            lst = [exp['task'] for exp in trainer.updating_experience_buffer.buffer]
            counts = Counter(lst)
            print('task belonging count in updating_experience_buffer:', counts) 
    print('== time used for train: ',time.time()-t0)
    return check_point_folder
    





def _set_debug_mode():
    print('in debug mode')
    global trainer_params
    trainer_params['epochs'] = 3
    trainer_params['train_episodes'] = 20
    trainer_params['aug_batch_size']= 20
    trainer_params["episodes"]= {'U':20,'E':20,'GM':20,'C':20,'Grid':20,'Ring':20}
    trainer_params['validate_episodes']= 3
    

def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################
orders = [
    ['E', 'C', 'Grid', 'U', 'Ring', 'GM'],
    ['U', 'GM', 'E', 'Ring', 'Grid', 'C'],
    ['E', 'Grid', 'Ring', 'C', 'U', 'GM'],
    ['Grid', 'GM', 'E', 'U', 'Ring', 'C'],
    ['Grid', 'C', 'Ring', 'U', 'GM', 'E']
    ]
if __name__ == "__main__":
    t0 = time.time()
    # --------- for lifelong --------
    parser = argparse.ArgumentParser(description="training parameters")
    parser.add_argument('--runname',type=str,default='debug',help="run name")
    parser.add_argument('--llmethod',type=str,default='FT',help='lifelong learning method, in [FT, EWC, LiBOG, LLRBC]')
    parser.add_argument('--debug',action='store_true',help='set into debug model with very small epoch and batch size')
    parser.add_argument('--aug_train',action='store_true',help='use 8 augment in training')
    parser.add_argument('--task_order',type=int,default='0',help='order of tasks')
    parser.add_argument('--recover_path',type=str,help="the path of model saved that is going to be loaded for recovering trianing")
    parser.add_argument('--recover_epoch',type=int,help="the epoch number of model saved that is going to be loaded for recovering trianing")
    parser.add_argument('--fixed_init_model',action='store_true',help='use a pregenerated fixed initial model for lifelong learning')
    parser.add_argument('--keep_optimizer',action='store_true',help='do not re initial optimizer after each task interval')
    parser.add_argument('--problem_sample_log',action='store_true',help='print detailed log for traiing problem instance generation')
    # --- for EWC ---
    parser.add_argument('--EWC_lambda',type=float,default=1.0,help='EWC lambda')
    parser.add_argument('--EWC_fisher_batch_number',type=int,default=16,help='number of batch for calcualte fisher information matrix')
    parser.add_argument('--fisher_norm',action='store_true',help='normalise the fisher matrix of each task')
    # --- for LiBOG ---
    parser.add_argument('--ITC_weight',type=float,default=1e-3,help='ITC weight in loss')
    # --- for LLR-BC --- 
    parser.add_argument('--consolidation_loss_weight',type=float,default=100.0,help='weight of clone loss')
    parser.add_argument('--buffer_size',type=int,default=1000,help='replay buffer size')
    parser.add_argument('--clone_sample_size',type=int,default=16,help='sample size from the buffer')
    parser.add_argument('--temperature_clone',type=float,default=1.0,help='temperature of clone loss (KLD) calculation')
    parser.add_argument('--confidence_weight_KLD',action='store_true',help='use confidence as weight in KLD calculation in batch experience setting')
    parser.add_argument('--confidence_method',type=str,default='var',help='method for calculate confidence, var, t2m or entropy')
    parser.add_argument('--confidence_scale',type=float,default=1.0,help='the exponent scale of confidence weight')
    parser.add_argument('--RKLDw',type=float,default=1.0,help='the weight of RKLD in clone loss, 0.0 for use KLD only, 1.0 for use RKLD only')
    parser.add_argument('--buffer_in_each_epoch',action='store_true',help='bufferring experience per epoch, default only on the last epoch of each task')
    # ---------------- end ----------

    args = parser.parse_args()
    logger_params['log_file']['desc'] = args.runname
    trainer_params['EWC_lambda'] = args.EWC_lambda
    trainer_params['EWC_fisher_batch_number'] = args.EWC_fisher_batch_number
    trainer_params['fisher_normalise'] = args.fisher_norm
    trainer_params['ITC_weight'] = args.ITC_weight
    trainer_params['confidence_weight_KLD']=args.confidence_weight_KLD
    trainer_params['confidence_method']=args.confidence_method
    trainer_params['confidence_scale']=args.confidence_scale
    trainer_params['RKLD_weight']=args.RKLDw
    trainer_params['aug_train']=args.aug_train

    assert args.confidence_method in ['var','t2m','entropy']
    
    if args.fixed_init_model:
        trainer_params['fixed_init_model_path'] = './init_model.pt'
    else:
        trainer_params['fixed_init_model_path'] = None
    if args.debug:
        _set_debug_mode()
    task_order = orders[args.task_order]

    env_params['task_order'] = task_order

    trainer_params['aug_train']=args.aug_train
    print('args.aug_train', args.aug_train)
    
    if args.buffer_in_each_epoch:
        trainer_params['experience_add_per_epochs'] = 1
    else:
        trainer_params['experience_add_per_epochs'] = trainer_params['epochs']
  
    

    if (not args.recover_epoch is None) and (not args.recover_path is None):
        trainer_params['model_load']['enable'] = True
        trainer_params['model_load']['path'] = args.recover_path
        trainer_params['model_load']['epoch'] = args.recover_epoch
    else:
        trainer_params['model_load']['enable'] = False


    trainer_params['consolidation_loss_weight']= args.consolidation_loss_weight
    trainer_params['replay_buffer_size']= args.buffer_size
    trainer_params['clone_sample_size']= args.clone_sample_size
    trainer_params['temperature_clone']= args.temperature_clone
    env_params['problem_sample_log'] = args.problem_sample_log

    trainer_params['re_init_optimizer'] = True

    # ----------------- train ----------

    check_point_folder = main(task_order,args.llmethod)

     

    
    # ---------------- test ----------
    tester_params['model_load']['path'] = check_point_folder
    test_main(0,1201,1,check_point_folder)
    
    print(' total hours used: ',(time.time()-t0)/3600)