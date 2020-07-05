# MetaPruning

This is the pytorch implementation of our paper "MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning", https://arxiv.org/abs/1903.10258, published in ICCV 2019. 

<img width=60% src="https://github.com/liuzechun0216/images/blob/master/figure1.jpg"/>

Traditional pruning decides pruning which channel in each layer and pays human effort in setting the pruning ratio of each layer. MetaPruning can automatically search for the best pruning ratio of each layer (i.e., number of channels in each layer). 

MetaPruning contains two steps: 
1. train a meta-net (PruningNet), to provide reliable weights for all the possible combinations of channel numbers in each layer (Pruned Net structures).
2. search for the best Pruned Net by evolutional algorithm and evaluate one best Pruned Net via training it from scratch.

# Citation

If you use the code in your research, please cite:

	@inproceedings{liu2019metapruning,
	  title={Metapruning: Meta learning for automatic neural network channel pruning},
	  author={Liu, Zechun and Mu, Haoyuan and Zhang, Xiangyu and Guo, Zichao and Yang, Xin and Cheng, Kwang-Ting and Sun, Jian},
	  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
	  pages={3296--3305},
	  year={2019}
	}

# Run

1. Requirements:
    * python3, pytorch 1.1.0, torchvision 0.3.0

2. ImageNet data:
    * You need to split the original training images into sub-validation dataset,  which contains 50000 images randomly selected from the training images with 50 images in each 1000-class, and sub-training dataset with the rest of images. Training the PruningNet with the sub-training dataset and searching the pruned network with the sub-validation dataset for inferring model accuracy. 

3. Steps to run:
    * Step1:  training
    * Step2:  searching 
    * Step3:  evaluating
    
    * After training the Pruning Net, checkpioint.pth.tar will be generated in the training folder, which will be loaded by the searching algorithm. After searching is done, the top1 encoding vector will be shown in the log. By simply copying the encoding vector to the rngs = \[ \] in evaluate.py, you can evaluate the Pruned Network corresponding to this encoding vector. 

# Models

MobileNet v1

| | Uniform Baselines | | Meta Pruning| | | 
| --- | --- | --- | --- | --- | --- | 
| Ratio | Top1-Acc | FLOPs | Top1-Acc | FLOPs | Model |
| 1x | 70.6% | 569M | - | - | - |
| 0.75x | 68.4% | 325M | 70.9% | 316M | [Model-MetaP-Mbv1-0.75](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EXuAHjKVTa9Gkj4ZHET0s58BU9QGI9O88iEVLopWu-usdw?e=b0VcpJ) |
| 0.5x  | 63.7% | 149M | 66.1% | 142M | [Model-MetaP-Mbv1-0.5 ](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/ERb0bJ7ggL5Du8v4mrLeVlkBEontkyhTWdDKIoMZQwHC2w?e=5pXdDh) |
| 0.25x | 50.6% | 41M  | 57.2% | 41M  | [Model-MetaP-Mbv1-0.25](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EQpBwbDTmCxLmpG8BCzG3xMBJsRoYOURAwG53HzkIqOzKQ?e=UrABgZ) |


MobileNet v2

| Uniform Baselines | | Meta Pruning| | | 
| --- | --- | --- | --- | --- | 
| Top1-Acc | FLOPs | Top1-Acc | FLOPs | Model |
| 74.7% | 585M | - | - | - |
| 72.0% | 313M | 72.7% | 303M | [Model-MetaP-Mbv2-300M](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EWtiXOwKblRNnQs_xtzpM8oBuQ7wlAXGzrlJgEPZ7aXc7Q?e=h1vn4s) |
| 67.2% | 140M | 68.2% | 140M | [Model-MetaP-Mbv2-140M](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EcCbtJSanrdJs9xANHx6qSgBF3FzCN00uNTlDv2vJlZlNw?e=HoQmtY) |
| 54.6% | 43M  | 58.3% | 43M  | [Model-MetaP-Mbv2-40M](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EX_Lna862JpHhLz7eWUBRqABdCh1_7wyxtaW4bE7PC3wuw?e=dEkWyv)  |


ResNet

| | Uniform Baselines | | Meta Pruning| | | 
| --- | --- | --- | --- | --- | --- | 
| Ratio | Top1-Acc | FLOPs | Top1-Acc | FLOPs | Model |
| 1x | 76.6% | 4.1G | - | - | - |
| 0.75x | 74.8% | 2.3G | 75.4% | 2.0G | [Model-MetaP-ResN-0.75](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EUpHJYfrtaFMn46Af94vqn4BgNr9AAZ6hskoWahtA8r5Tg?e=8ovp6p) |
| 0.5x  | 72.0% | 1.1G | 73.4% | 1.0G | [Model-MetaP-ResN-0.5 ](https://hkustconnect-my.sharepoint.com/:u:/g/personal/zliubq_connect_ust_hk/EX8NhySdkw1NrUx9EYVCH0sBEJzgwM4ZS0Opv6WG0intJA?e=xMLY07) |
