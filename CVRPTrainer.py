# CVRPTrainer.py

import torch
from logging import getLogger
import numpy as np
from CVRPEnv import CVRPEnv as Env
from CVRPModel_training import CVRPModel as Model
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from utils import *

class CVRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):

        self.seed = env_params.get('seed', 1234)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

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
        else:
            device = torch.device('cpu')

        self.device = device
        self.env_params['device'] = device

        # Main Components
        self.model = Model(**self.model_params).to(device)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self._ensure_checkpoint_compatible(checkpoint)
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as exc:
                raise RuntimeError(
                    "Checkpoint load failed. Ensure ablation/gate settings match the training run."
                ) from exc
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            epoch_seed = self.seed + epoch
            torch.manual_seed(epoch_seed)
            np.random.seed(epoch_seed)
            self.logger.info(f"Epoch {epoch}: Reset random seed to {epoch_seed}")

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data(),
                    'model_params': self.model_params,
                    'env_params': self.env_params,
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

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

            if loop_cnt < 5:
                self.logger.info('Epoch {:3d}: Batch {:3d}/{:3d}  Score: {:.4f},  Loss: {:.4f}'
                                 .format(epoch, (episode//batch_size), (train_num_episode//batch_size),
                                         score_AM.avg, loss_AM.avg))
            loop_cnt += 1

        self.logger.info('Epoch {:3d}: Train Avg Score: {:.4f},  Avg Loss: {:.4f}'
                         .format(epoch, score_AM.avg, loss_AM.avg))
        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size):
        self.model.train()
        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0), device=self.device)
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        advantage = reward - reward.float().mean(dim=1, keepdim=True)
        log_prob = prob_list.log().sum(dim=2)
        loss = -advantage * log_prob
        loss_mean = loss.mean()

        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = -max_pomo_reward.float().mean()

        self.model.zero_grad()
        loss_mean.backward()
        self.optimizer.step()
        
        return score_mean.item(), loss_mean.item()

    def _ensure_checkpoint_compatible(self, checkpoint):
        ckpt_params = checkpoint.get('model_params')
        if not ckpt_params:
            return
        ckpt_flags = {
            'enable_encoder_global': ckpt_params.get('enable_encoder_global', True),
            'enable_encoder_cluster': ckpt_params.get('enable_encoder_cluster', True),
            'enable_decoder_cluster': ckpt_params.get('enable_decoder_cluster', True),
            'decoder_fusion': ckpt_params.get(
                'decoder_fusion',
                'learned_gate' if ckpt_params.get('use_learned_gate', False) else 'rule'
            ),
        }
        current_flags = {
            'enable_encoder_global': self.model_params.get('enable_encoder_global', True),
            'enable_encoder_cluster': self.model_params.get('enable_encoder_cluster', True),
            'enable_decoder_cluster': self.model_params.get('enable_decoder_cluster', True),
            'decoder_fusion': self.model_params.get('decoder_fusion', 'rule'),
        }
        mismatches = []
        for key, ckpt_value in ckpt_flags.items():
            if current_flags.get(key) != ckpt_value:
                mismatches.append(f"{key}: ckpt={ckpt_value} current={current_flags.get(key)}")
        if mismatches:
            detail = ", ".join(mismatches)
            raise ValueError(
                "Checkpoint config mismatch for ablation/gate settings: " + detail
            )
