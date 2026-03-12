# CluAttention (PDTSP)

`experiment.py` is the central entry point. The four wrapper scripts fix the ablation setting:

- `run_full.py`: full model
- `run_no_enc_cluster.py`: encoder cluster attention disabled
- `run_no_dec_cluster.py`: decoder cluster attention disabled
- `run_pomo.py`: POMO baseline (cluster attention disabled in both encoder and decoder)

These wrappers only toggle the cluster-attention modules. The rest of the model structure, training setup, and evaluation options stay the same.

## Requirements

- Python 3.9+
- PyTorch 1.12+ (install the CUDA build that matches your driver if needed)

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Example PyTorch installation for CUDA 11.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Data Format

Evaluation data is stored as `.pkl` files. Two formats are supported:

- List format: `[(depot_xy, node_xy), ...]`
- Dict format: `{'data': [(depot_xy, node_xy), ...], 'scale_factor': float}`

Where:

- `depot_xy` is the depot coordinate `(x, y)`
- `node_xy` is the list of customer node coordinates

For `.pkl` instances, the code assumes the first half of `node_xy` are pickup nodes and the second half are their paired delivery nodes.

Evaluation expects each `.pkl` file to contain a single instance by default.

### Generate a Test Set

Example: 100-node instances, 100 files, clustered distribution.

```bash
python generate_pdp_dataset.py --name test --problem pdp --data_distribution clustered --cluster_std 0.1 --graph_sizes 100 --dataset_size 1 --num_files 100 --seed 10000 --data_dir data
```

Example output path:

`data/pdp/pdp100_test_clustered_std0.1_seed10000.pkl`

## Training

Full model:

```bash
python run_full.py --task train --problem_size 100 --distribution clustered --train_batch_size 128
```

Encoder cluster attention disabled:

```bash
python run_no_enc_cluster.py --task train --problem_size 100 --distribution clustered --train_batch_size 128
```

Decoder cluster attention disabled:

```bash
python run_no_dec_cluster.py --task train --problem_size 100 --distribution clustered --train_batch_size 128
```

POMO baseline:

```bash
python run_pomo.py --task train --problem_size 100 --distribution clustered --train_batch_size 128
```

Disable the learned gate:

```bash
python run_full.py --task train --problem_size 100 --distribution clustered --disable_gate
```

## Evaluation

Greedy decoding:

```bash
python run_full.py --task test "data/pdp/pdp100_test_clustered_std0.1_seed*.pkl" --model_path /path/to/result_dir --epoch 800 --decode_strategy greedy
```

Sampling with width 1280:

```bash
python run_full.py --task test "data/pdp/pdp100_test_clustered_std0.1_seed*.pkl" --model_path /path/to/result_dir --epoch 800 --decode_strategy sampling --width 1280
```

Sampling with width 12800:

```bash
python run_full.py --task test "data/pdp/pdp100_test_clustered_std0.1_seed*.pkl" --model_path /path/to/result_dir --epoch 800 --decode_strategy sampling --width 12800
```

To evaluate another ablation, replace `run_full.py` with one of:

- `run_no_enc_cluster.py`
- `run_no_dec_cluster.py`
- `run_pomo.py`

## Project Structure

- `experiment.py`: shared training and evaluation entry point
- `run_full.py`: full-model wrapper
- `run_no_enc_cluster.py`: wrapper without encoder cluster attention
- `run_no_dec_cluster.py`: wrapper without decoder cluster attention
- `run_pomo.py`: POMO baseline wrapper
- `CVRPModel_training.py`: model definition
- `CVRPEnv.py`: environment and state transitions
- `CVRPTrainer.py`: training loop
- `CVRPTester.py`: checkpoint loading and evaluation
- `generate_pdp_dataset.py`: test-set generator
- `utils.py`: logging and utility helpers
- `data/`: dataset directory (not version-controlled)

## References

**Cluster-Aware Attention-Based Deep Reinforcement Learning for Pickup and Delivery Problems** Wentao Wang, Lifeng Han, and Guangyu Zou. *arXiv preprint arXiv:2603.10053*, 2026.  
[[Paper]](https://arxiv.org/abs/2603.10053)

<details>
<summary>BibTeX (Click to expand)</summary>

```bibtex
@misc{wang2026clusteraware,
      title={Cluster-Aware Attention-Based Deep Reinforcement Learning for Pickup and Delivery Problems}, 
      author={Wentao Wang and Lifeng Han and Guangyu Zou},
      year={2026},
      eprint={2603.10053},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={[https://arxiv.org/abs/2603.10053](https://arxiv.org/abs/2603.10053)}, 
}
<details>
