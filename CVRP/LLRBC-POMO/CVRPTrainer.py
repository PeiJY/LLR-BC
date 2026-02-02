
import torch
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import torch.nn as nn
from utils.utils import *
import random
import copy
import torch.nn.functional as F

def check_tensor(name, t):
    if t is None:
        print(f"{name} is None")
        return
    if not torch.is_tensor(t):
        print(f"{name} is not a tensor, type = {type(t)}")
        return
    
    if torch.isnan(t).any():
        print(f"[NaN] detected in {name}")
    if torch.isinf(t).any():
        print(f"[Inf] detected in {name}")
    if not torch.isfinite(t).all():
        print(f"[Non-finite] detected in {name}")
    with torch.no_grad():
        print(
            f"{name}: min={t.min().item():.5e}, "
            f"max={t.max().item():.5e}, "
            f"mean={t.mean().item():.5e}"
        )

class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        if not trainer_params['fixed_init_model_path'] is None:
            checkpoint = torch.load(trainer_params['fixed_init_model_path'], map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.device_count() > 1:
            print("use ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        if not trainer_params['fixed_init_model_path'] is None:
            checkpoint = torch.load(trainer_params['fixed_init_model_path'], map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!: '+'{path}/checkpoint-{epoch}.pt'.format(**model_load))

        # utility
        self.time_estimator = TimeEstimator()


    def change_env(self,env_params,overwrite_self=True,change_trianer_params=True,print_log = True,re_init_optimizer=True):
        if overwrite_self:
            self.env_params = env_params
            self.env = Env(**self.env_params)
        else:
            self.env = Env(**env_params)
        if change_trianer_params:
            d = env_params['distribution']
            self.trainer_params['train_batch_size'] = self.trainer_params['batch_sizes'][d]
            self.trainer_params['train_episodes'] = self.trainer_params['episodes'][d]
        if print_log:
            self.logger.info(' -- switch env to {}-{}, with train batch size as {},  and episodes per epoch as {}'.format(
            env_params['distribution'],env_params['problem_size'],self.trainer_params['train_batch_size'],self.trainer_params['train_episodes']))
        if re_init_optimizer:
            self.logger.info(' -- re init optimizer for learning model')
            self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        
    def run(self):
        self.time_estimator.reset(self.start_epoch)
        first_epoch = True
        
        
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            
            # ---------------- save the init model -------------
            if epoch  == 1:
                self.logger.info("Saving initial_model")
                checkpoint_dict = {
                    'epoch': 0,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, 0))
            # ----------------------------------------------------
                

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0 or first_epoch or (epoch == self.trainer_params['epochs']-1)  or (epoch == self.trainer_params['epochs']-2):
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
                first_epoch = False
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.pre_forward(reset_state)
        else:
            self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob, probs  = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

class CVRPTrainer_multitask:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        
        self.batch_level = trainer_params['batch_level']

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        if not trainer_params['fixed_init_model_path'] is None:
            checkpoint = torch.load(trainer_params['fixed_init_model_path'], map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.device_count() > 1:
            print("use ", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        if not trainer_params['fixed_init_model_path'] is None:
            checkpoint = torch.load(trainer_params['fixed_init_model_path'], map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!: '+'{path}/checkpoint-{epoch}.pt'.format(**model_load))

        # utility
        self.time_estimator = TimeEstimator()


    def change_env(self,env_params,overwrite_self=True,change_trianer_params=True,print_log = True,re_init_optimizer=True):
        if overwrite_self:
            self.env_params = env_params
            self.env = Env(**self.env_params)
        else:
            self.env = Env(**env_params)
        if change_trianer_params:
            d = env_params['distribution']
            self.trainer_params['train_batch_size'] = self.trainer_params['batch_sizes'][d]
            self.trainer_params['train_episodes'] = self.trainer_params['episodes'][d]
        if print_log:
            self.logger.info(' -- switch env to {}-{}, with train batch size as {},  and episodes per epoch as {}'.format(
            env_params['distribution'],env_params['problem_size'],self.trainer_params['train_batch_size'],self.trainer_params['train_episodes']))
        if re_init_optimizer:
            self.logger.info(' -- re init optimizer for learning model')
            self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        
    def run(self,tasks):
        self.tasks = tasks
        self.time_estimator.reset(self.start_epoch)
        first_epoch = True
        
        
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            
            # ---------------- save the init model -------------
            if epoch  == 1:
                self.logger.info("Saving initial_model")
                checkpoint_dict = {
                    'epoch': 0,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, 0))
            # ----------------------------------------------------
                

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0 or first_epoch or (epoch == self.trainer_params['epochs']-1)  or (epoch == self.trainer_params['epochs']-2):
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
                first_epoch = False
            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        if not self.batch_level:
            distribution = random.choice(self.tasks)
            env_params = copy.deepcopy(self.env_params)
            size = env_params['sizes'][distribution]
            env_params['distribution'] = distribution
            env_params['pomo_size'] = size
            env_params['problem_size'] = size
            self.change_env(env_params,re_init_optimizer=False)

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        if self.batch_level:
            temp = [self.trainer_params['episodes'][d] for d in self.tasks]
            train_num_episode = int(sum(temp)/len(temp))
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:


            if self.batch_level:
                distribution = random.choice(self.tasks)
                # print('== train on distribution: ',distribution)
                env_params = copy.deepcopy(self.env_params)
                size = env_params['sizes'][distribution]
                env_params['distribution'] = distribution
                env_params['pomo_size'] = size
                env_params['problem_size'] = size
                self.change_env(env_params,re_init_optimizer=False)

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            avg_score, avg_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.pre_forward(reset_state)
        else:
            self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob, probs  = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()

class CVRPTrainer_EWC:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        if not trainer_params['fixed_init_model_path'] is None:
            checkpoint = torch.load(trainer_params['fixed_init_model_path'], map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.device_count() > 1:
            print("use ", torch.cuda.device_count(), " GPUs")
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        # ------------------ EWC components ----------------------
        self.ewc_lambda = trainer_params['EWC_lambda']  # Regularization strength
        self.fisher_matrix_list = []  # To store Fisher information
        self.old_params_list = []  # To store old task parameters
        # --------------------------------------------------------

        if not trainer_params['fixed_init_model_path'] is None:
            checkpoint = torch.load(trainer_params['fixed_init_model_path'], map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.old_params_list=checkpoint['old_params_list'] if 'old_params_list' in checkpoint.keys() else []
            self.fisher_matrix_list=checkpoint['fisher_matrix_list'] if 'fisher_matrix_list' in checkpoint.keys() else []
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        
        

    def change_env(self,env_params,overwrite_self=True,calculate_save_fisher=True,change_trianer_params=True,print_log = True):
        if calculate_save_fisher:
            self.old_params_list.append([p.clone().detach() for p in self.model.parameters()])
            self.logger.info(' -- model parameters saved')
            self.fisher_matrix_list.append(self.compute_fisher_matrix())
            self.logger.info(' -- fisher matrix calculated')
        if overwrite_self:
            self.env_params = env_params
            self.env = Env(**self.env_params)
        else:
            self.env = Env(**env_params)
        if change_trianer_params:
            d = env_params['distribution']
            self.trainer_params['train_batch_size'] = self.trainer_params['batch_sizes'][d]
            self.trainer_params['train_episodes'] = self.trainer_params['episodes'][d]
        if print_log:
            self.logger.info(' -- switch env to {}-{}, with train batch size as {},  and episodes per epoch as {}'.format(
            env_params['distribution'],env_params['problem_size'],self.trainer_params['train_batch_size'],self.trainer_params['train_episodes']))
        self.logger.info(' -- re init optimizer for learning model')
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        
    def compute_fisher_matrix(self):
        """
        Compute Fisher information matrix for EWC
        """
        fisher_matrix = [torch.zeros_like(p) for p in self.model.parameters()]
        
        # Sample some episodes to estimate Fisher information
        for i in range(self.trainer_params['EWC_fisher_batch_number']):
            EWC_batch_size = self.trainer_params['train_batch_size']
            self.env.load_problems(EWC_batch_size)
            reset_state, _, _ = self.env.reset()
            if isinstance(self.model, nn.DataParallel):
                self.model.module.pre_forward(reset_state)
            else:
                self.model.pre_forward(reset_state)
            
            # Run episode and collect gradients
            state, reward, done = self.env.pre_step()
            # prob_list = torch.zeros(size=(1, self.env.pomo_size, 0))
            prob_list = torch.zeros(size=(EWC_batch_size, self.env.pomo_size, 0))
            
            while not done:
                selected, prob, probs  = self.model(state)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            
            # Compute gradients for Fisher information
            self.model.zero_grad()
            log_prob = prob_list.log().sum(dim=2)
            log_prob.mean().backward()
            
            # Update Fisher matrix
            for j, param in enumerate(self.model.parameters()):
                if param.grad is not None:
                    fisher_matrix[j] += param.grad.data.pow(2) / self.trainer_params['train_batch_size']
        
        # normalise
        if self.trainer_params['fisher_normalise']:
            all_values = torch.cat([p.view(-1) for p in fisher_matrix])  
            min_val, max_val = all_values.min(), all_values.max()

            fisher_matrix = [(p - min_val) / (max_val - min_val + 1e-8) for p in fisher_matrix]
        return fisher_matrix
    

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        first_epoch = True
        
        
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            
            # ---------------- save the init model -------------
            if epoch  == 1:
                self.logger.info("Saving initial_model")
                checkpoint_dict = {
                    'epoch': 0,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'old_params_list':self.old_params_list,
                    'fisher_matrix_list':self.fisher_matrix_list,
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, 0))
            # ----------------------------------------------------
            
            

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss, EWC_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0 or first_epoch or (epoch == self.trainer_params['epochs']-1):
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'old_params_list':self.old_params_list,
                    'fisher_matrix_list':self.fisher_matrix_list,
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
                first_epoch = False

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        EWC_loss_AM = AverageMeter()
        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            avg_score, avg_loss, EWC_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            EWC_loss_AM.update(EWC_loss, batch_size)
            episode += batch_size

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}, EWC loss in Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg, EWC_loss_AM.avg))

        return score_AM.avg, loss_AM.avg, EWC_loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.pre_forward(reset_state)
        else:
            self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        while not done:
            selected, prob, probs  = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # --------------- EWC in loss ------------------
        # Add EWC regularization if old task exists
        ewc_loss = 0.0
        for old_task_index in range(len(self.old_params_list)):
            for (param, old_param, fisher) in zip(
                self.model.parameters(), 
                self.old_params_list[old_task_index], 
                self.fisher_matrix_list[old_task_index]
            ):
                ewc_loss += (fisher * (param - old_param).pow(2)).sum() 
        loss_mean += self.ewc_lambda * ewc_loss 
        # ------------------------------------------------
        
        
        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item(), ewc_loss

class CVRPTrainer_LiBOG:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        if not trainer_params['fixed_init_model_path'] is None:
            checkpoint = torch.load(trainer_params['fixed_init_model_path'], map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.device_count() > 1:
            print("use ", torch.cuda.device_count(), " GPUs!")
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        self.fisher_matrix_list = []  # To store Fisher information
        self.old_params_list = []  # To store old task parameters
        self.bsf_model = None


        if not trainer_params['fixed_init_model_path'] is None:
            checkpoint = torch.load(trainer_params['fixed_init_model_path'], map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.old_params_list=checkpoint['old_params_list'] if 'old_params_list' in checkpoint.keys() else []
            self.fisher_matrix_list=checkpoint['fisher_matrix_list'] if 'fisher_matrix_list' in checkpoint.keys() else []
            self.bsf_model = Model(**self.model_params)
            if 'bsf_model' in checkpoint.keys():
                self.bsf_model.load_state_dict(checkpoint['bsf_model'])
            else:
                self.bsf_model.load_state_dict(self.model.state_dict())
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()
        
        # ------------------ EWC and ITC components ----------------------
        self.ewc_lambda = trainer_params['EWC_lambda']  # Regularization strength
        self.ITC_weight = trainer_params['ITC_weight']
        # --------------------------------------------------------

    def change_env(self,env_params,overwrite_self=True,calculate_save_fisher=True,change_trianer_params=True,print_log = True):
        if calculate_save_fisher:
            self.old_params_list.append([p.clone().detach() for p in self.model.parameters()])
            self.logger.info(' -- model parameters saved')
            self.fisher_matrix_list.append(self.compute_fisher_matrix())
            self.logger.info(' -- fisher matrix calculated')
        if overwrite_self:
            self.env_params = env_params
            self.env = Env(**self.env_params)
        else:
            self.env = Env(**env_params)
        if change_trianer_params:
            d = env_params['distribution']
            self.trainer_params['train_batch_size'] = self.trainer_params['batch_sizes'][d]
            self.trainer_params['train_episodes'] = self.trainer_params['episodes'][d]
        if print_log:
            self.logger.info(' -- switch env to {}-{}, with train batch size as {},  and episodes per epoch as {}'.format(
            env_params['distribution'],env_params['problem_size'],self.trainer_params['train_batch_size'],self.trainer_params['train_episodes']))
        self.bsf_model = None
        self.logger.info(' -- re init optimizer for learning model')
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        
    def compute_fisher_matrix(self):
        """
        Compute Fisher information matrix for EWC
        """
        fisher_matrix = [torch.zeros_like(p) for p in self.model.parameters()]
        
        # Sample some episodes to estimate Fisher information
        for i in range(self.trainer_params['EWC_fisher_batch_number']):
            EWC_batch_size = self.trainer_params['train_batch_size']
            self.env.load_problems(EWC_batch_size)
            reset_state, _, _ = self.env.reset()
            if isinstance(self.model, nn.DataParallel):
                self.model.module.pre_forward(reset_state)
            else:
                self.model.pre_forward(reset_state)
            
            # Run episode and collect gradients
            state, reward, done = self.env.pre_step()
            # prob_list = torch.zeros(size=(1, self.env.pomo_size, 0))
            prob_list = torch.zeros(size=(EWC_batch_size, self.env.pomo_size, 0))
            
            while not done:
                selected, prob, probs = self.model(state)
                state, reward, done = self.env.step(selected)
                prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
                
            
            # Compute gradients for Fisher information
            self.model.zero_grad()
            log_prob = prob_list.log().sum(dim=2)
            log_prob.mean().backward()
            
            # Update Fisher matrix
            for j, param in enumerate(self.model.parameters()):
                if param.grad is not None:
                    fisher_matrix[j] += param.grad.data.pow(2) / self.trainer_params['train_batch_size']
        
        # normalise
        if self.trainer_params['fisher_normalise']:
            all_values = torch.cat([p.view(-1) for p in fisher_matrix])  
            min_val, max_val = all_values.min(), all_values.max()

            fisher_matrix = [(p - min_val) / (max_val - min_val + 1e-8) for p in fisher_matrix]
        return fisher_matrix
    

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        first_epoch = True
        
        bsf_loss = -float('inf')
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('================= epoch {} ============================================='.format(epoch))
            print('number of old parameters saved: ',len(self.old_params_list))
            # ---------------- save the init model -------------
            if epoch  == 1:
                self.logger.info("Saving initial_model")
                checkpoint_dict = {
                    'epoch': 0,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'old_params_list':self.old_params_list,
                    'fisher_matrix_list':self.fisher_matrix_list,
                    'bsf_model':None if self.bsf_model is None else self.bsf_model.state_dict(),
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, 0))
            # ----------------------------------------------------
            
            

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss, _, _ = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)
            
            # for ITC
            if train_loss > bsf_loss:
                bsf_loss = train_loss 
                self.bsf_model = Model(**self.model_params)
                self.logger.info('copy model (loss {}) to bsf model (loss {})'.format(train_loss,bsf_loss))
                self.bsf_model.load_state_dict(self.model.state_dict())
                self.bsf_model.encoder.load_state_dict(self.model.encoder.state_dict())
                self.bsf_model.decoder.load_state_dict(self.model.decoder.state_dict())
                self.bsf_model.eval()
            # else:
            #     print('not copy model (loss {}) to bsf model (loss {})'.format(train_loss,bsf_loss))
            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0 or first_epoch or (epoch == self.trainer_params['epochs']-1):
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'old_params_list':self.old_params_list,
                    'fisher_matrix_list':self.fisher_matrix_list,
                    'bsf_model':None if self.bsf_model is None else self.bsf_model.state_dict()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
                first_epoch = False

            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        EWC_loss_AM = AverageMeter()
        ITC_loss_AM = AverageMeter()
        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            avg_score, avg_loss, EWC_loss, ITC_loss = self._train_one_batch(batch_size)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            EWC_loss_AM.update(EWC_loss, batch_size)
            ITC_loss_AM.update(ITC_loss,batch_size)
            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}, EWC loss in Loss: {:.4f} , ITC loss in Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg, EWC_loss_AM.avg, ITC_loss_AM.avg))

        return score_AM.avg, loss_AM.avg, EWC_loss_AM.avg, ITC_loss_AM.avg

    def _train_one_batch(self, batch_size):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.pre_forward(reset_state)
            if not self.bsf_model is None:
                self.bsf_model.module.pre_forward(reset_state)
        else:
            self.model.pre_forward(reset_state)
            if not self.bsf_model is None:
                self.bsf_model.pre_forward(copy.deepcopy(reset_state))

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # ---- for ITC -----
        probs_list = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.problem_size+1, 0))
        bsf_probs_list = torch.zeros(size=(batch_size, self.env.pomo_size, self.env.problem_size+1, 0))
        # ------------------
        
        
        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob, probs  = self.model(state)
            # print('shape of prob is {}, of probs is {}'.format(prob.shape, probs.shape))
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            # ---- for ITC -----
            if not self.bsf_model is None:
                _,_,bsf_probs = self.bsf_model(copy.deepcopy(state))
                bsf_probs_list = torch.cat((bsf_probs_list, bsf_probs.unsqueeze(-1)), dim=3)
                probs_list = torch.cat((probs_list, probs.unsqueeze(-1)), dim=3)
            # ------------------
            

        # Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()
        
        # --------------- ITC  in loss --------------
        ITC_loss = 0.0
        if not self.bsf_model is None:
            kld = self.compute_kl_divergence(probs_list.clone().to('cpu'),bsf_probs_list.clone().detach().to('cpu'))
            # print('KLD',kld)
            ITC_loss = self.ITC_weight * kld 
        # -------------------------------------------
        # --------------- EWC in loss ------------------
        # Add EWC regularization if old task exists
        ewc_loss = 0.0
        for old_task_index in range(len(self.old_params_list)):
            for (param, old_param, fisher) in zip(
                self.model.parameters(), 
                self.old_params_list[old_task_index], 
                self.fisher_matrix_list[old_task_index]
            ):
                ewc_loss += (fisher * (param - old_param).pow(2)).sum() 
        ewc_loss += self.ewc_lambda * ewc_loss 
        # ------------------------------------------------
        loss_mean += ewc_loss + ITC_loss
        
        
        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), loss_mean.item(), ewc_loss, ITC_loss

    def compute_kl_divergence(self,P_probs,Q_probs):
        # reshape to (T * B * pomo, num_nodes)
        T, B, P, N = P_probs.shape
        eps = 1e-8
        P_probs = torch.clamp(P_probs, min=eps, max=1.0)
        Q_probs = torch.clamp(Q_probs, min=eps, max=1.0)
        P_probs = P_probs.reshape(-1, N)
        Q_probs = Q_probs.reshape(-1, N)
        
        kl_loss = F.kl_div(
            input=(Q_probs+1e-8).log(),     # logQ (student)
            target=P_probs,          # P (teacher)
            reduction='batchmean'
        )
        return kl_loss

def calculate_confidence(probs_tensor,method='var'):
    """
    compute confidence
    - var_confidence: variance / max_variance
    - top2_margin
    - entropy_confidence: entropy / max_entropy
    - method: in var, t2m, entropy
    """
    # probs shape [..., n_actions]
    n_actions = probs_tensor.shape[-1]
    
    if method == 'var':
        # ============ Variance Confidence ===============
        mean_probs = probs_tensor.mean(dim=-1, keepdim=True)            # shape [..., 1]
        squared_diff = (probs_tensor - mean_probs) ** 2                 # shape [..., n_actions]
        variance = squared_diff.mean(dim=-1)                            # shape [...]

        # Compute max_variance
        mean_value = 1.0 / n_actions
        diff_first = (1.0 - mean_value) ** 2
        diff_others = (0.0 - mean_value) ** 2 * (n_actions - 1)
        total_squared_diff = diff_first + diff_others
        max_variance = total_squared_diff / n_actions

        var_confidence = variance / (max_variance + 1e-8)               # shape [...]
        return var_confidence

    if method == 't2m':
        # ============ Top-2 Margin ===============
        top2_probs, _ = torch.topk(probs_tensor, k=2, dim=-1)           # shape [..., 2]
        top2_margin = top2_probs[..., 0] - top2_probs[..., 1]           # shape [...]
        return top2_margin

    if method == 'entropy':
        # ============ Entropy Confidence ===============

        log_probs = torch.log(probs_tensor + 1e-8)                      
        entropy = -torch.sum(probs_tensor * log_probs, dim=-1)          # shape [...]

        if  torch.isnan(log_probs).any():
            print("NaN in log_probs:")
        if torch.isnan(entropy).any():
            print("NaN in entropy:")
        max_entropy = torch.log(torch.tensor(float(n_actions)))         # scalar
        entropy_confidence = 1.0 - (entropy / (max_entropy + 1e-8))     
        if torch.isnan(entropy_confidence).any():
            print("NaN in entropy_confidence:")
        return entropy_confidence


class ExperienceBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.n_seen = 0
        self.add_num = 0

    def add(self, reset_state, state, actions, probs, device='cpu', task=None):
        """
        Store the entire state objects and tensors without slicing.
        """
        exp = {
            'reset_state': copy.deepcopy(reset_state),
            'state': copy.deepcopy(state),
            'probs': probs.detach().to(device),
            'task':task,
        }

        
        self.n_seen += 1
        reservoir_prob = self.capacity / self.n_seen
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            self.add_num+=1
        else:
            if random.random() < reservoir_prob:
                idx = random.randint(0, self.capacity - 1)
                self.buffer[idx] = exp
                self.add_num+=1

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            return []
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def get_state(self):
        return {
            'capacity': self.capacity,
            'n_seen': self.n_seen,
            'buffer': self.buffer
        }


    def load_state(self, state_dict):
        self.capacity = state_dict['capacity']
        self.n_seen = state_dict['n_seen']
        self.buffer = copy.deepcopy(state_dict['buffer'])

import time
def behaviour_clone_loss(model, sampled_experiences, temperature=1.0, device='cuda',confidence_weight_KLD=False,confidence_method='var',confidence_scale=1.0,RKLD_weight=0.0):
    # t0 = time.time()
    kl_list = []
    # confidence_list = []
    for exp in sampled_experiences:
        # t1 = time.time()
        reset_state = exp['reset_state']
        state = exp['state']
        old_probs = exp['probs'].to(device)
        # Move all tensors in reset_state to device
        for attr in vars(reset_state).keys():
            v = getattr(reset_state, attr)
            if v is not None and torch.is_tensor(v):
                setattr(reset_state, attr, v.to(device).float())

        # Also move tensors in Step_State
        for attr in vars(state).keys():
            v = getattr(state, attr)
            if v is not None and torch.is_tensor(v):
                if attr in ['BATCH_IDX', 'POMO_IDX', 'current_node', 'finished', 'selected_count']:
                    v = v.to(device).long()
                else:
                    v = v.to(device).float()
                setattr(state, attr, v)
        # Pre-forward
        model.pre_forward(reset_state)
        # Forward
        outputs = model(state)
        if isinstance(outputs, tuple):
            current_logits = outputs[2]  # [batch, pomo, num_nodes+1]
        else:
            current_logits = outputs

    
        current_logits_scaled = F.softmax(current_logits / temperature,dim=-1)
        old_logits_scaled = F.softmax(old_probs / temperature,dim=-1)

        current_log_prob = torch.log(current_logits_scaled + 1e-9)
        old_log_prob = torch.log(old_logits_scaled + 1e-9)


        if RKLD_weight==1.0:
            kl = F.kl_div(
                old_log_prob,
                current_logits_scaled,
                reduction='none'
            )
        elif RKLD_weight==0.0:
            kl = F.kl_div(
                current_log_prob,
                old_logits_scaled,
                reduction='none'
            )
        else:
            rkl = F.kl_div(
                old_log_prob,
                current_logits_scaled,
                reduction='none'
            )
            kl = F.kl_div(
                current_log_prob,
                old_logits_scaled,
                reduction='none'
            )
            kl = RKLD_weight*rkl + (1-RKLD_weight)*kl
        kl = kl.sum(dim=-1)
        kl = kl * (temperature ** 2)

        if confidence_weight_KLD:
            confidence = calculate_confidence(old_probs,method=confidence_method)

            weights = (1- confidence + 1e-9)**confidence_scale

            kl_flat = kl.reshape(-1)
            weightsflat = weights.reshape(-1)

            mean_weights = weightsflat.mean()       
            

            weights = weightsflat / (mean_weights + 1e-9)

            wsum = weights.sum()

            weighted_mean = (weights * kl_flat).sum() / wsum

            kl_list.append(weighted_mean)
        else:
            kl_mean = kl.mean()
            kl_list.append(kl_mean)

    if len(kl_list) == 0:
        return torch.tensor(0.0, device=device)
    
    return torch.stack(kl_list).mean()

class CVRPTrainer_LLRBC:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        if trainer_params['aug_train']:
            self.aug_factor = 8
        else:
            self.aug_factor = 1

        print('env aug_factor', self.aug_factor)

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # -------- for replay ----------
        self.buffer_size = self.trainer_params['replay_buffer_size']
        self.old_experience_buffer = ExperienceBuffer(capacity=self.buffer_size)
        self.updating_experience_buffer = ExperienceBuffer(capacity=self.buffer_size)
        self.consolidation_loss_weight = self.trainer_params['consolidation_loss_weight']
        self.clone_sample_size = self.trainer_params['clone_sample_size']
        self.temperature = self.trainer_params['temperature_clone']
        self.confidence_weight_KLD=self.trainer_params['confidence_weight_KLD']
        self.confidence_method=self.trainer_params['confidence_method']
        self.confidence_scale=self.trainer_params['confidence_scale']
        self.RKLD_weight=self.trainer_params['RKLD_weight']
        # -----------------------------

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            self.device = device
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            self.device = device
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        if not trainer_params['fixed_init_model_path'] is None:
            checkpoint = torch.load(trainer_params['fixed_init_model_path'], map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        if torch.cuda.device_count() > 1:
            print("use", torch.cuda.device_count(), "GPU!")
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])
        if not trainer_params['fixed_init_model_path'] is None:
            checkpoint = torch.load(trainer_params['fixed_init_model_path'], map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            if 'old_experience_buffer' in checkpoint.keys():
                self.old_experience_buffer.load_state(checkpoint['old_experience_buffer'])
            if 'updating_experience_buffer' in checkpoint.keys():
                self.updating_experience_buffer.load_state(checkpoint['updating_experience_buffer'])
            self.logger.info('Saved Model Loaded !!: '+'{path}/checkpoint-{epoch}.pt'.format(**model_load))


        # utility
        self.time_estimator = TimeEstimator()


    def change_env(self,env_params,overwrite_self=True,change_trianer_params=True,print_log = True,re_init_optimizer=True, do_buffer_copy = True):
        if overwrite_self:
            self.env_params = env_params
            self.env = Env(**self.env_params)
        else:
            self.env = Env(**env_params)
        if change_trianer_params:
            d = env_params['distribution']
            self.trainer_params['train_batch_size'] = self.trainer_params['batch_sizes'][d]
            self.trainer_params['train_episodes'] = self.trainer_params['episodes'][d]
        if print_log:
            self.logger.info(' -- switch env to {}-{}, with train batch size as {},  and episodes per epoch as {}'.format(
            env_params['distribution'],env_params['problem_size'],self.trainer_params['train_batch_size'],self.trainer_params['train_episodes']))
        if re_init_optimizer:
            self.logger.info(' -- re init optimizer for learning model')
            self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        if do_buffer_copy:
            self.logger.info(' -- replace experience buffer')
            self.old_experience_buffer = copy.deepcopy(self.updating_experience_buffer)
        
            

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        first_epoch = True
        
        
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')
            
            # ---------------- save the init model -------------
            if epoch  == 1:
                self.logger.info("Saving initial_model")
                checkpoint_dict = {
                    'epoch': 0,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'old_experience_buffer':self.old_experience_buffer.get_state(),
                    'updating_experience_buffer':self.updating_experience_buffer.get_state()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, 0))
            # ----------------------------------------------------
                

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            ############################
            # Logs & Checkpoint
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                    self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                    self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0 or first_epoch or (epoch == self.trainer_params['epochs']-1)  or (epoch == self.trainer_params['epochs']-2):
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'old_experience_buffer':self.old_experience_buffer.get_state(),
                    'updating_experience_buffer':self.updating_experience_buffer.get_state()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))
                first_epoch = False

   
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()
        if self.consolidation_loss_weight > 0:
            RL_loss_AM = AverageMeter()
            clone_loss_AM = AverageMeter()


        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            if epoch % self.trainer_params['experience_add_per_epochs'] == 0:
                do_experience_collect = True 
            else:
                do_experience_collect = False
            avg_score, avg_loss, loss_terms = self._train_one_batch(batch_size,do_experience_collect)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)
            if self.consolidation_loss_weight > 0:
                RL_loss_AM.update(loss_terms[0])
                clone_loss_AM.update(loss_terms[1])

            episode += batch_size


        # Log Once, for each epoch
        loss_terms_info = ''
        if self.consolidation_loss_weight > 0:
            loss_terms_info = ', RL loss: {:.4f}, Clone loss: {:.4f} (weight {}), buffer added {}'.format(RL_loss_AM.avg,clone_loss_AM.avg,self.consolidation_loss_weight,self.updating_experience_buffer.add_num)
                    
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg)+loss_terms_info)

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size, do_experience_collect):

        # Prep
        ###############################################
        self.model.train()
        self.env.load_problems(batch_size,self.aug_factor)
        reset_state, _, _ = self.env.reset()
        if isinstance(self.model, nn.DataParallel):
            self.model.module.pre_forward(reset_state)
        else:
            self.model.pre_forward(reset_state)

        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        ###############################################
        state, reward, done = self.env.pre_step()

        step_couter = 0
        while not done:
            selected, prob, probs  = self.model(state)
            # ---- add to buffer ----
            if do_experience_collect and step_couter >= 2: # the first 2 steps are not included, step 1 depot, step 2 uniformly a new route starting node
                self.updating_experience_buffer.add(state=state,reset_state=reset_state,actions=selected,probs=probs,task=self.env.distribution)
                
            # -----------------------
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            step_couter += 1

        # RL Loss
        ###############################################
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()
        RL_loss_value = loss_mean.item()
        self.model.zero_grad()
        if self.consolidation_loss_weight > 0 and len(self.old_experience_buffer.buffer) > 0:
            # ---- Behaviour clone loss ----
            if torch.isnan(loss_mean).any():
                print("before consolidation loss_mean has NaN:", loss)
            if torch.isinf(loss_mean).any():
                print("before consolidation loss_mean has Inf:", loss)
            loss_mean.backward(retain_graph=True)
            del loss_mean
            sampled_experiences = self.old_experience_buffer.sample(self.clone_sample_size)
            clone_loss = behaviour_clone_loss(self.model,sampled_experiences,temperature=self.temperature,
                                              device=self.device,confidence_weight_KLD=self.confidence_weight_KLD,confidence_method=self.confidence_method,
                                              confidence_scale=self.confidence_scale,RKLD_weight=self.RKLD_weight)
            clone_loss = clone_loss * self.consolidation_loss_weight
            if torch.isnan(clone_loss).any():
                print("clone_loss has NaN:", torch.isnan(clone_loss).any().item())
            if torch.isinf(clone_loss).any():
                print("clone_loss has Inf:", torch.isinf(clone_loss).any().item())
            clone_loss.backward()
            clone_loss_value = clone_loss.item()
        else:
            if torch.isnan(loss_mean).any():
                print("loss_mean has NaN:", loss)
            if torch.isinf(loss_mean).any():
                print("loss_mean has Inf:", loss)
            loss_mean.backward()
            clone_loss_value = 0.0


        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        
        # loss_mean.backward()
        self.optimizer.step()
        return score_mean.item(), clone_loss_value+RL_loss_value, (RL_loss_value,clone_loss_value)
