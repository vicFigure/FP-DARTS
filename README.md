# FP-DARTS: Search Loss Gradient-aware Pruning for one-shot Neural Architecture Search

This repository is the official implementation of [FP-DARTS: Search Loss Gradient-aware Pruning for one-shot Neural Architecture Search]. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Searching
We utilize first-order optimization of DARTS to alternately update operation parameters and architecture weights.

Refer to ```scripts/run_search.sh``` for further introduction.

### DARTS's search space
To search the model architectures by FP-DARTS-s1, run this command:
```search by s1
python train_search.py --sal_type task
```

To search the model architectures by FP-DARTS-s2, run this command:
```search by s2
python train_search.py --sal_type task --sal_second
```

To search the model architectures by FP-DARTS-s1r, run this command:
```search by s1r
python train_search.py --sal_type task --num_compare 5
```

### Larger search space
To search the model architectures by FP-DARTS-s1r on larger search space, run this command:
```search by s1r w/o constraints
python train_search.py --sal_type task --num_compare 5 --no_restrict
```



## Evaluation
To evaluate the searched model(s) in the paper, we need to train the model from scratch and then evaluate the ultimate performance. 

We follow the configuration of DARTS: use cutout and auxiliary training strategy for CIFAR10 dataset, and use auxiliary training strategy for ImageNet dataset. 

Refer to ```scripts/run_fulltrain.sh``` for further introduction.

### DARTS's search space
To evaluate the searched model on CIFAR10 under DARTS's search space, run this command:
```eval_cifar
python train.py --mode train --cutout --auxiliary --base_dir <path_to_the_dir_of_model>
```

To evaluate the searched model on ImageNet, run:
```eval_imagenet
python train_imagenet.py --mode train --auxiliary --base_path <path_to_the_dir_of_model> --genotype_name <the_name_of_genotype_by_model_selection>
```

### Larger search space
To evaluate the searched model on CIFAR10 under larger search space, run this command:
```eval_cifar
python train.py --mode train --no_restrict --cutout --auxiliary --base_dir <path_to_the_dir_of_model>
```

To evaluate the searched model on ImageNet, run:
```eval_imagenet
python train_imagenet.py --mode train --no_restrict --auxiliary --base_path <path_to_the_dir_of_model> --genotype_name <the_name_of_genotype_by_model_selection>
```



## Results
### DARTS's search space
Our model achieves the following performance on CIFAR10 under DARTS's search space. All results are averaged among three searches with different random seeds.

| Model name         | Top 1 Acc | Params (M) |
| ------------------ |---------- |----------- |
| FP-DARTS-s1   |   97.41%  |    3.3     |
| FP-DARTS-s2   |   97.45%  |    3.3     |
| FP-DARTS-s1r   |   97.55%  |    3.5     |

Our model achieves the following performance on ImageNet:

| Model name         | Top 1 Acc | Params (M) |
| ------------------ |---------- |----------- |
| FP-DARTS-s2        |   75.0%   |     4.9    |
| FP-DARTS-s1r       |   75.4%   |     5.1    |

### Larger search space
Our model achieves the following performance on CIFAR10 under DARTS's search space:

| Model name         | Top 1 Acc | Params (M) |
| ------------------ |---------- |----------- |
| FP-DARTS-s1r   |   97.52%  |    3.4     |

Our model achieves the following performance on ImageNet:

| Model name         | Top 1 Acc | Params (M) |
| ------------------ |---------- |----------- |
| FP-DARTS-s1r      |   75.3%   |     5.1    |


