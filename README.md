# MDL-Net
This repository is the official implementation of **3D Multimodal Fusion Network with Disease-induced Joint Learning for Early Alzheimer’s Disease Diagnosis.** [MDL-Net](https://ieeexplore.ieee.org/document/10498133)   
If you use the codes and models from this repo, please cite our work. Thanks!  
``` 
@ARTICLE{10498133,
  author={Qiu, Zifeng and Yang, Peng and Xiao, Chunlun and Wang, Shuqiang and Xiao, Xiaohua and Qin, Jing and Liu, Chuan-Ming and Wang, Tianfu and Lei, Baiying},
  journal={IEEE Transactions on Medical Imaging}, 
  title={3D Multimodal Fusion Network with Disease-induced Joint Learning for Early Alzheimer’s Disease Diagnosis}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Three-dimensional displays;Feature extraction;Brain modeling;Neuroimaging;Imaging;Deep learning;Fuses;Alzheimer’s disease diagnosis;3D Multimodal fusion network;Interpretability;Disease-induced joint learning},
  doi={10.1109/TMI.2024.3386937}}
    
```   
## Data Preparation
**Batch process nii data to save as pth file**.  
If you want to use the 'train.py' file for training, you can first try the 'nii2pth.py' file for data preprocessing, the purpose of 'nii2pth.py' is to save all the data of a category as a pth file for subsequent use.
