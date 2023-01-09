# Forest-Point-MAE for ScanObjectNN

## Self-supervised Learning on 3D Data with Masked Autoencoders


## 0. Description 


This approach to point cloud classification is mainly based upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE) (Pang et al., 2022). In a first step, the point cloud is partitioned into patches via the farthest point sampling algorithm. Those patches are then encoded with a Mini-PointNet (Qi et al., 2017). and processed with as tokens with a standard transformer in an encoder/decoder design. The encoder/decoder design is a result of the self-supervised pretraining setup, that is identical to Point-MAE and employs the ShapeNet dataset (Chang et al., 2015). The new part in this approach lies in a novel data augmentation, that yields accuracy improvements over all ScanObjectNN benchmarks of more than 1 %. For the data augmentation, patch dropout, after partitioning the point cloud into patches a random ratio of those patches is set to the first patch. With this, those patches become effectively invisible to the network. And the network is presented with a point cloud with “holes” in it. This makes the classification task harder and similar holes might also occur in other scanned point clouds for example because of occlusion. For image vision transformers, it was shown that a similar augmentation is especially useful in making a model more robust to missing data (Liu et al., 2022). I find that a maximum dropout ratio of 90 % of the patches leads to the highest increase in accuracy of more than 1 % in comparison to no Patch Dropout over all variation of ScanObjectNN.


*Liu, Y., Matsoukas, C., Strand, F., Azizpour, H., and Smith, K. Patch Dropout: Economizing Vision Transformers Using Patch Dropout. 2022. 10.48550/arxiv.2208.07220.
*Qi, C. R., Su, H., Mo, K., and Guibas, L. J. Pointnet: Deep learning on point sets for 3d classification and segmentation. In: Proceedings of the IEEE conference on computer vision and pattern recognition. 2017, pp. 652-660.
*Pang, Y., Wang, W., Tay, F. E. H., Liu, W., Tian, Y., and Yuan, L. Masked Autoencoders for Point Cloud Self-supervised Learning. 2022. 10.48550/arxiv.2203.06604.


## 1. Setup

```
bash scripts/setup.sh
```

The data must be stored in data/scanobjectnn in the main project folder.

This codebase uses Weights and Biases for logging. Logging without an account is enabled via anonymous logging in line 68 of main.py.


## 2. Pretraining

The pretrained model from Point-MAE is used and can be downloaded from the [Point-MAE-Repo](https://github.com/Pang-Yatian/Point-MAE/releases/download/main/pretrain.pth). It has to be saved in experiments/pretraining/pretrain_official/pretrain.pth.


## 3. Finetuning

To run the training on the different variants of the ScanObjectNN dataset with different patch dropout parameters run: 

```
bash scripts/scanobject_drop0[x].sh
```

The used GPU can be chosen by setting CUDA_VISIBLE_DEVICES in the scripts. If only one gpu is available, set to 0. To adjust training parameters like the batch size or learning rate change the values in the .yaml files in cfgs/classification.

## 4. Results

The table gives the overall accuracy on the test set.

Dropout rate | T25 | T25R | T50R | T50RS
--- | --- | --- | --- | --- 
0.0 | 90.4 | 89.5 | 88.3 | 88.1
0.3 | 89.7 | 90.0 | 88.5 | 89.0
0.7 | 91.4 | 89.9 | 88.8 | 88.7
0.9 | 91.7 | 90.6 | 89.8 | 89.56



## Acknowledgements

The code is mainly based upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch).
