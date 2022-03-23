This is the official codes for [EXACT: Scalable Graph Neural Networks Training via Extreme Activation Compression](https://openreview.net/forum?id=vkaMaq95_rX).

## Install
This code is tested with Python 3.8 and CUDA 11.0. We note that during our experiments, we found that the version of Pytorch, Pytorch Sparse, and Pytorch Scatter can significantly impact the running speed of the baseline. To reproduce the results in our paper, please follow the below configuration.

- Requirements
```
torch == 1.9.0
torch_geometric == 1.7.2
torch_scatter == 2.0.8
torch_sparse == 0.6.12
```

- Build
```bash
cd exact
pip install -v -e .
```

## Reproduce results
### Important note

The default setting is applying INT8 quantization to activations. If you want to get the results without quantization. Please add ```--act_fp``` kwargs

### Reproduce ogbn-arxiv results.
```bash
cd mem_speed_bench
python ./arxiv/train_full_batch.py --conf ./arxiv/conf/$MODEL.yaml --n_bits $BIT_WIDTH --kept_frac $FRAC
```
MODEL must be chosen from {gcn, sage, gcn2, gat}, BIT_WIDTH must be chosen from {1,2,4,8}, FRAC is pretty flexible. it can be any float-point number <= 1.0. If FRAC == 1.0, then the random projection will not be applied.

If you do not want to apply any quantization, you can change the commend to 
```
python ./arxiv/train_full_batch.py --conf ./arxiv/conf/$MODEL.yaml --act_fp --kept_frac $FRAC
```

### Reproduce Flickr, Yelp and Reddit results.
For full-batch training, 
```bash
cd mem_speed_bench
python ./non_ogbn_datasets/train_full_batch.py --conf ./non_ogbn_datasets/conf/$MODEL.yaml --n_bits $BIT_WIDTH --kept_frac $FRAC --dataset $DATASET --grad_norm $GRAD_NORM
```
MODEL must be chosen from {gcn, sage, gcn2}. BIT_WIDTH must be chosen from {1,2,4,8}, FRAC can be any float-point number <= 1.0. 
DATASET must be chosen from {flickr, reddit2, yelp}.
For GRAD_NORM, it can found in Table 11-15.

For mini-batch training, 
```bash
cd mem_speed_bench
python ./non_ogbn_datasets/train_mini_batch.py --conf ./non_ogbn_datasets/conf/$MODEL.yaml --n_bits $BIT_WIDTH --kept_frac $FRAC --dataset $DATASET --grad_norm $GRAD_NORM
```
MODEL must be chosen from {saint_sage, cluster_gcn}. BIT_WIDTH must be chosen from {1,2,4,8}, FRAC can be any float-point number <= 1.0. 
DATASET must be chosen from {flickr, reddit2, yelp}
For GRAD_NORM, it can found in Table 11-15.


### Reproduce ogbn-products results.
For full-batch training, 
```bash
cd mem_speed_bench
python ./products/train_full_batch.py --conf ./products/conf/sage.yaml --n_bits $BIT_WIDTH --kept_frac $FRAC
```
BIT_WIDTH must be chosen from {1,2,4,8}, FRAC is pretty flexible. it can be any float-point number <= 1.0. If FACT == 1.0, then the random projection will not be applied.

For mini-batch training, 
```bash
cd mem_speed_bench
python ./yaml/train_mini_batch.py --conf ./yaml/conf/$MODEL.yaml --n_bits $BIT_WIDTH --kept_frac $FRAC
```
MODEL must be chosen from {cluster_sage, saint_sage}. BIT_WIDTH must be chosen from {1,2,4,8}, FRAC can be any float-point number <= 1.0.

### Get the occupied memory and training throughout.
Add the flag **--deug_mem** and **--test_speed** to the above commends. For example,
```
python ./arxiv/train_full_batch.py --conf ./arxiv/conf/$MODEL.yaml --n_bits $BIT_WIDTH --kept_frac $FRAC --debug_mem --test_speed
```

### Combining EXACT and AMP
Add the flag **--amp** to the above commends.

## Acknowledgment about our implementation
For quantization, our code is based on the official code of [ActNN](https://arxiv.org/abs/2104.14129) and [BLPA](https://github.com/ayanc/blpa).

For SpMM, our code is based on the official code of [torch sparse](https://github.com/rusty1s/pytorch_sparse)

For the overall code structure, our code is based on the official code of [GNNAutoScale](https://github.com/rusty1s/pyg_autoscale)