# [AAAI 2024] Unveiling Details in the Dark: Simultaneous Brightening and Zooming for Low-Light Image Enhancement [[Paper]](https://ojs.aaai.org/index.php/AAAI/article/view/28515)
By Ziyu Yue, Jiaxin Gao,  Zhixun Su

## Pipeline
<p align="center">
    <img src="./Figures/pipeline.png" alt="pipeline" width="80%">
</p>


## Dependencies
```
pip install -r requirements.txt
````

## Download the raw training and evaluation datasets
### Paired dataset 
RELLISUR dataset: Andreas Aakerberg, Kamal Nasrollahi, Thomas Moeslund. "RELLISUR: A Real Low-Light Image Super-Resolution Dataset". NeurIPS Datasets and Benchmarks 2021. [RELLISUR](https://vap.aau.dk/rellisur/)

### Unpaired dataset 
Please refer to DARK FACE dataset: Yang, Wenhan and Yuan, Ye and Ren, Wenqi and Liu, Jiaying and Scheirer, Walter J. and Wang, Zhangyang and Zhang, and et al. "DARK FACE: Face Detection in Low Light Condition". IEEE Transactions on Image Processing, 2020. [DARK FACE](https://flyywh.github.io/CVPRW2019LowLight/)

Please refer to Dark Zurich dataset: Christos Sakaridis, Dengxin Dai, Luc van Gool. "Guided Curriculum Model Adaptation and Uncertainty-Aware Evaluation for Semantic Nighttime Image Segmentation". ICCV, 2019. [Dark Zurich](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/)


## Pre-trained Models 
You can download our pre-trained model from [[Google Drive]](https://drive.google.com/drive/folders/1m3t15rWw76IDDWJ0exLOe5P0uEnjk3zl?usp=drive_link) and [[Baidu Yun (extracted code:cjzk)]](https://pan.baidu.com/s/1fPLVgnZbdY1n75Flq54bMQ)

## How to train?
You need to modify ```datasets/dataset.py``` slightly for your environment
Then train MIRNet_v2 with default settings, run
```
cd BrZoNet
sh train.sh
```

## How to test?
```
python basicsr/test.py -opt /Super_Resolution/Options/msc_retinex_srnet_v12_scale2_test.yml  # For x2 task
python basicsr/test.py -opt /Super_Resolution/Options/msc_retinex_srnet_v12_scale4_test.yml  # For x4 task
```


## Results
- Visual comparison
<p align="center">
    <img src="./Figures/result1.png" alt="result1" width="80%">
</p>

- Benchmark Evaluation
<p align="center">
    <img src="./Figures/result2.png" alt="result2" width="80%">
</p>

## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@inproceedings{yue2024unveiling,
  title={Unveiling Details in the Dark: Simultaneous Brightening and Zooming for Low-Light Image Enhancement},
  author={Yue, Ziyu and Gao, Jiaxin and Su, Zhixun},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={7},
  pages={6899--6907},
  year={2024}
}
```

## Acknowledgement
Part of the code is adapted from previous works: [BasicSR](https://github.com/XPixelGroup/BasicSR) and [MIRNet](https://github.com/swz30/MIRNet) (code structure). We thank all the authors for their contributions.

Please contact me if you have any questions at: 11901015@mail.dlut.edu.cn

