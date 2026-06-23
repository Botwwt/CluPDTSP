# experiment.py

import argparse
import glob
import logging
import os
import time

import numpy as np
import torch

from utils import create_logger, copy_all_src
from CVRPTrainer import CVRPTrainer as Trainer
from CVRPTester import CVRPTester as Tester
from CVRProblemDef import load_pdp_pkl_instance


ABLATION_CHOICES = [
    'full',
    'no_enc_cluster',
    'cluster_only',
    'no_dec_cluster',
    'avg_fusion',
    'no_cluster',
    'pomo',
    'no_pomo',
]


def resolve_ablation(ablation):
    mapping = {
        'full': {
            'enable_encoder_global': True,
            'enable_encoder_cluster': True,
            'enable_decoder_cluster': True,
            'decoder_fusion': 'learned_gate',
        },
        'no_enc_cluster': {
            'enable_encoder_global': True,
            'enable_encoder_cluster': False,
            'enable_decoder_cluster': True,
            'decoder_fusion': 'learned_gate',
        },
        'cluster_only': {
            'enable_encoder_global': False,
            'enable_encoder_cluster': True,
            'enable_decoder_cluster': True,
            'decoder_fusion': 'learned_gate',
        },
        'no_dec_cluster': {
            'enable_encoder_global': True,
            'enable_encoder_cluster': True,
            'enable_decoder_cluster': False,
            'decoder_fusion': 'single',
        },
        'avg_fusion': {
            'enable_encoder_global': True,
            'enable_encoder_cluster': True,
            'enable_decoder_cluster': True,
            'decoder_fusion': 'average',
        },
        'no_cluster': {
            'enable_encoder_global': True,
            'enable_encoder_cluster': False,
            'enable_decoder_cluster': False,
            'decoder_fusion': 'single',
        },
        'pomo': {
            'enable_encoder_global': True,
            'enable_encoder_cluster': False,
            'enable_decoder_cluster': False,
            'decoder_fusion': 'single',
        },
        'no_pomo': {
            'enable_encoder_global': True,
            'enable_encoder_cluster': True,
            'enable_decoder_cluster': True,
            'decoder_fusion': 'learned_gate',
        },
    }
    if ablation not in mapping:
        raise ValueError(f"Unknown ablation: {ablation}")
    return mapping[ablation]


def apply_ablation(model_params, ablation, disable_gate):
    ablation_params = resolve_ablation(ablation)
    model_params.update(ablation_params)
    if disable_gate and model_params['decoder_fusion'] == 'learned_gate':
        model_params['decoder_fusion'] = 'rule'
    if not model_params['enable_decoder_cluster']:
        model_params['decoder_fusion'] = 'single'
    model_params['use_learned_gate'] = model_params['decoder_fusion'] == 'learned_gate'
    return model_params


def build_train_configs(opts):
    env_params = {
        'problem_size': opts.problem_size,
        'pomo_size': 1 if (opts.single_rollout or opts.ablation == 'no_pomo') else opts.problem_size,
        'seed': opts.seed,
        'distribution': opts.distribution,
    }

    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'eval_type': 'greedy',
        'use_learned_gate': not opts.disable_gate,
    }
    model_params = apply_ablation(model_params, opts.ablation, opts.disable_gate)

    optimizer_params = {
        'optimizer': {
            'lr': 1e-4,
            'weight_decay': 1e-6,
        },
        'scheduler': {
            'milestones': [8001],
            'gamma': 0.1,
        },
    }

    trainer_params = {
        'use_cuda': torch.cuda.is_available(),
        'cuda_device_num': opts.cuda_device_num,
        'epochs': opts.epochs,
        'train_episodes': opts.train_episodes,
        'train_batch_size': opts.train_batch_size,
        'logging': {
            'model_save_interval': opts.checkpoint_interval,
            'img_save_interval': 100,
            'log_image_params_1': {
                'json_foldername': 'log_image_style',
                'filename': 'style_cvrp_100.json',
            },
            'log_image_params_2': {
                'json_foldername': 'log_image_style',
                'filename': 'style_loss_1.json',
            },
        },
        'model_load': {
            'enable': False,
        },
    }

    if opts.resume_path and opts.resume_epoch is not None:
        trainer_params['model_load'] = {
            'enable': True,
            'path': opts.resume_path,
            'epoch': opts.resume_epoch,
        }

    log_desc = f"train_pdtsp_n{opts.problem_size}_{opts.distribution}_{opts.ablation}"
    if model_params.get('use_learned_gate'):
        log_desc += '_gate'
    if env_params['pomo_size'] == 1:
        log_desc += '_single_rollout'

    logger_params = {
        'log_file': {
            'desc': log_desc,
            'filename': 'run_log',
        }
    }
    if opts.result_dir:
        logger_params['log_file']['filepath'] = os.path.join(opts.result_dir, '{desc}')

    return env_params, model_params, optimizer_params, trainer_params, logger_params


def run_train(opts):
    env_params, model_params, optimizer_params, trainer_params, logger_params = build_train_configs(opts)

    create_logger(**logger_params)
    logger = logging.getLogger('root')
    logger.info(f"env_params: {env_params}")
    logger.info(f"model_params: {model_params}")
    logger.info(f"optimizer_params: {optimizer_params}")
    logger.info(f"trainer_params: {trainer_params}")

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    if not opts.resume_path:
        copy_all_src(trainer.result_folder)
    trainer.run()


def run_test(opts):
    if opts.decode_strategy == 'greedy':
        pomo_size = 1
        eval_type = 'greedy'
    elif opts.decode_strategy in ['sampling', 'pomo']:
        pomo_size = opts.width
        eval_type = 'sampling'
    else:
        raise ValueError(f"Unknown decode strategy: {opts.decode_strategy}")

    env_params = {
        'problem_size': 0,
        'pomo_size': pomo_size,
    }

    model_params = {
        'embedding_dim': 128,
        'sqrt_embedding_dim': 128**(1/2),
        'encoder_layer_num': 6,
        'qkv_dim': 16,
        'head_num': 8,
        'logit_clipping': 10,
        'ff_hidden_dim': 512,
        'use_learned_gate': not opts.disable_gate,
        'eval_type': eval_type,
    }
    model_params = apply_ablation(model_params, opts.ablation, opts.disable_gate)

    if not opts.model_path or opts.epoch is None:
        raise ValueError("Test requires --model_path and --epoch")

    tester_params = {
        'use_cuda': torch.cuda.is_available(),
        'cuda_device_num': opts.cuda_device_num,
        'model_load': {
            'path': opts.model_path,
            'epoch': opts.epoch,
        },
        'augmentation_enable': False,
    }

    log_desc = f"eval_{os.path.basename(opts.model_path)}_e{opts.epoch}_{opts.decode_strategy}_w{pomo_size}_{opts.ablation}"
    if model_params.get('use_learned_gate'):
        log_desc += '_gate'
    logger_params = {
        'log_file': {
            'desc': log_desc,
            'filename': 'log.txt',
        }
    }

    create_logger(**logger_params)
    logger = logging.getLogger('root')

    instance_paths = []
    for pattern in opts.instances:
        instance_paths.extend(glob.glob(pattern))

    instance_paths = sorted(set(instance_paths))
    if not instance_paths:
        raise ValueError(f"No instance files found matching: {opts.instances}")

    logger.info(f"Tester params: {tester_params}")
    logger.info(f"Testing {len(instance_paths)} instances.")
    logger.info(f"Model: {opts.model_path}, epoch: {opts.epoch}")
    logger.info(f"Decode strategy: {opts.decode_strategy}, width/samples: {pomo_size}")
    logger.info(f"Ablation: {opts.ablation}, use_learned_gate: {model_params.get('use_learned_gate')}")

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    results_summary = {}
    all_scores = []
    all_times = []

    for path in instance_paths:
        if not os.path.exists(path):
            logger.error(f"Instance file not found, skipping: {path}")
            continue

        logger.info(f"Solving instance: {os.path.basename(path)}")

        depot_xy, node_xy, node_demand, cluster_list, pairing, scale =             load_pdp_pkl_instance(path, device=tester.device)

        tester.env.use_saved_problems(depot_xy, node_xy, node_demand, cluster_list, pairing)

        start_time = time.time()

        with torch.no_grad():
            tester.env.load_problems(1)
            reset_state, _, _ = tester.env.reset()
            tester.model.pre_forward(reset_state)

            state, reward, done = tester.env.pre_step()
            while not done:
                selected, _ = tester.model(state)
                state, reward, done = tester.env.step(selected)

        solve_time = time.time() - start_time
        all_times.append(solve_time)

        best_reward, _ = reward.max(dim=1)
        final_score = -best_reward.item() * scale
        all_scores.append(final_score)

        filename = os.path.basename(path)
        results_summary[filename] = {'score': final_score, 'time': solve_time}

        logger.info(f"  >> Solved in {solve_time:.2f}s. Best Tour Length: {final_score:.4f}")

    avg_score = np.mean(all_scores)
    std_score = np.std(all_scores)
    avg_time = np.mean(all_times)
    std_time = np.std(all_times)

    print("\n" + "=" * 70)
    print("                          COMPLETE TEST SUMMARY")
    print("=" * 70)
    for filename, result in sorted(results_summary.items()):
        print(f"Instance: {filename:<40s} Score: {result['score']:<12.4f} Time: {result['time']:.2f}s")
    print("-" * 70)
    print(f"Decode Strategy: {opts.decode_strategy}")
    if opts.decode_strategy in ['sampling', 'pomo']:
        print(f"Width/Samples:   {pomo_size}")
    print("-" * 70)
    print(f"Average Score:   {avg_score:.5f}  (Std Dev: {std_score:.5f})")
    print(f"Average Time:    {avg_time:.3f}s   (Std Dev: {std_time:.3f}s)")
    print("=" * 70)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', required=True, choices=['train', 'test'])
    parser.add_argument('--ablation', type=str, default='full', choices=ABLATION_CHOICES)
    parser.add_argument('--disable_gate', action='store_true')
    parser.add_argument('--single_rollout', action='store_true')

    parser.add_argument('--problem_size', type=int, default=60)
    parser.add_argument('--distribution', type=str, default='clustered', choices=['uniform', 'clustered'])
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--checkpoint-interval', type=int, default=100)
    parser.add_argument('--train_episodes', type=int, default=2816)
    parser.add_argument('--train_batch_size', type=int, default=256)
    parser.add_argument('--cuda_device_num', type=int, default=0)
    parser.add_argument('--resume_path', type=str, default=None)
    parser.add_argument('--resume_epoch', type=int, default=None)
    parser.add_argument('--result_dir', type=str, default=None)

    parser.add_argument('instances', nargs='*')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--decode_strategy', type=str, default='greedy', choices=['greedy', 'sampling', 'pomo'])
    parser.add_argument('--width', type=int, default=10)

    return parser.parse_args(argv)


def main(argv=None):
    opts = parse_args(argv)

    if opts.task == 'train':
        run_train(opts)
    elif opts.task == 'test':
        if not opts.instances:
            raise ValueError("Test requires instance paths or patterns as positional args")
        run_test(opts)
    else:
        raise ValueError(f"Unknown task: {opts.task}")


if __name__ == '__main__':
    main()
