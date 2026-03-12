import torch
import os
from logging import getLogger

from CVRPEnv import CVRPEnv as Env
from CVRPModel_training import CVRPModel as Model
from CVRProblemDef import load_pdtsp_instance
from utils import *

class CVRPTester:
    def __init__(self, env_params, model_params, tester_params):
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.logger = getLogger(name='tester')
        
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            device = torch.device("cuda", cuda_device_num)
            torch.cuda.set_device(cuda_device_num)
        else:
            device = torch.device('cpu')
        
        self.device = device
        self.env_params['device'] = device # Pass device to env params

        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params).to(self.device)

        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        self.logger.info(f"Loading model checkpoint: {checkpoint_fullname}")
        checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
        self._ensure_checkpoint_compatible(checkpoint)
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint load failed. Ensure ablation/gate settings match the training run."
            ) from exc
        self.model.eval()

    def _ensure_checkpoint_compatible(self, checkpoint):
        ckpt_params = checkpoint.get('model_params')
        if not ckpt_params:
            return
        ckpt_flags = {
            'enable_encoder_cluster': ckpt_params.get('enable_encoder_cluster', True),
            'enable_decoder_cluster': ckpt_params.get('enable_decoder_cluster', True),
            'use_learned_gate': ckpt_params.get('use_learned_gate', False),
        }
        current_flags = {
            'enable_encoder_cluster': self.model_params.get('enable_encoder_cluster', True),
            'enable_decoder_cluster': self.model_params.get('enable_decoder_cluster', True),
            'use_learned_gate': self.model_params.get('use_learned_gate', False),
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

    def solve_instance(self, filepath):
        self.logger.info(f"Solving instance: {filepath}")

        depot_xy, node_xy, node_demand, cluster_list, pairing, scale, id_to_idx, depot_id = \
            load_pdtsp_instance(filepath, device=self.device)
        
        self.env.use_saved_problems(depot_xy, node_xy, node_demand, cluster_list, pairing)

        aug_factor = self.tester_params.get('aug_factor', 1) if self.tester_params.get('augmentation_enable', False) else 1
        
        with torch.no_grad():
            # The batch_size for a single instance is always 1
            self.env.load_problems(1, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            state, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                state, reward, done = self.env.step(selected)

        # Reshape reward based on aug_factor and original pomo_size
        pomo_size = self.env.pomo_size
        reward = reward.reshape(aug_factor, pomo_size)
        
        best_pomo_reward, best_pomo_idx = reward.max(dim=1)
        best_aug_reward, best_aug_idx = best_pomo_reward.max(dim=0)
        
        best_solution_pomo_idx = best_pomo_idx[best_aug_idx]
        
        # Adjust batch index for augmentation
        # The selected_node_list has a batch dimension equal to aug_factor
        best_solution_batch_idx = best_aug_idx.item()

        best_tour_indices = self.env.selected_node_list[best_solution_batch_idx, best_solution_pomo_idx, :]
        
        idx_to_id = {v: k for k, v in id_to_idx.items()}
        
        best_tour_original_ids = []
        for env_idx in best_tour_indices.tolist():
            if env_idx == 0:
                best_tour_original_ids.append(depot_id)
            else:
                tensor_idx = env_idx - 1
                best_tour_original_ids.append(idx_to_id[tensor_idx])
        
        final_score = -best_aug_reward.item() * scale

        self.logger.info(f"Best solution tour length: {final_score:.4f}")
        self.logger.info(f"Best solution tour (original IDs): {best_tour_original_ids}")
        
        return final_score, best_tour_original_ids
