
## Preparing datasets

### Cartoon
- [CartoonSet](https://google.github.io/cartoonset/)
    - download the dataset
        - cartoonset100k.tar (move to **./data/cartoonset100k**)
    - unzip data
```bash
python scripts/preprocess_cartoonset.py
```

### CelebAMask-HQ
  - [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ)
    - download the dataset
      - CelebA-HQ-img (move to **./data**)
      - CelebAMask-HQ-mask-anno (move to **./data**)

```bash
python scripts/preprocess_celebahq.py
```

## Training
```bash
python train.py --exp_name name
```
## Evaluation

```bash
python generate.py --exp_name name
python pytorch_fid/fid_score.py experiments/name/results/gt experiments/name/results/pred
```