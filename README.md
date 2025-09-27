# WDMamba: When Wavelet Degradation Prior Meets Vision Mamba for Image Dehazing (TCSVT2025)
This is the office implementation of ***WDMamba: When Wavelet Degradation Prior Meets Vision Mamba for Image Dehazing,TCSVT2025***

Jie Sun, Heng Liu, Yongzhen Wang, Xiao-Ping Zhang and Mingqiang Wei
<br>

<hr />

> **Abstract:** *In this paper, we reveal a novel haze-specific wavelet degradation prior observed through wavelet transform analysis, which shows that haze-related information predominantly resides in low-frequency components. Exploiting this insight,  we propose a novel dehazing framework, WDMamba, which decomposes the image dehazing task into two sequential stages: low-frequency restoration followed by detail enhancement. This coarse-to-fine strategy enables WDMamba to effectively capture features specific to each stage of the dehazing process, resulting in high-quality restored images. Specifically, in the low‐frequency restoration stage, we integrate Mamba blocks to reconstruct global structures with linear complexity,  efficiently removing overall haze and producing a coarse restored image. Thereafter, the detail enhancement stage reinstates fine‐grained information that may have been overlooked during the previous phase, culminating in the final dehazed output. Furthermore, to enhance detail retention and achieve more natural dehazing, we introduce a self-guided contrastive regularization during network training. By utilizing the coarse restored output as a hard negative example, our model learns more discriminative representations, substantially boosting the overall dehazing performance. Extensive evaluations on public dehazing benchmarks demonstrate that our method surpasses state-of-the-art approaches both qualitatively and quantitatively.* 
<hr />

## Dependencies and Installation

- Ubuntu >= 22.04
- CUDA >= 11.8
- Pytorch>=2.0.1
- Other required packages in `requirements.txt`
```
cd WDMamba

# create new anaconda env
conda create -n WDMamba python=3.9
conda activate WDMamba

# install python dependencies
pip3 install -r requirements.txt
python setup.py develop
```

## Train the model

```
bash train_haze4k.sh
```

## Inference

```
bash test.sh
```
or
```
python inference.py
```

## Dehazing Results

[Haze4K, RESIDE-6K, NH-HAZE, Dense-HAZE, O-HAZE(1200×1600)] (https://pan.baidu.com/s/1VdqpPY-Y1gMmpK4ej37wmg?pwd=y9e6)

## Pre-trained Models

[Haze4K, RESIDE-6K, NH-HAZE, Dense-HAZE, O-HAZE(1200×1600)] (https://pan.baidu.com/s/1HIs-nHXEaLxwBb1279PVbw?pwd=98j9)

## Citation
```
@ARTICLE{11180084,
  author={Sun, Jie and Liu, Heng and Wang, Yongzhen and Zhang, Xiao-Ping and Wei, Mingqiang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={WDMamba: When Wavelet Degradation Prior Meets Vision Mamba for Image Dehazing}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={WDMamba;Wavelet degradation prior;Vision Mamba;Image dehazing;Contrastive regularization},
  doi={10.1109/TCSVT.2025.3614173}}
```

## Acknowledgement

This project is based on [BasicSR](https://github.com/xinntao/BasicSR) and [Wave-Mamba](https://github.com/AlexZou14/Wave-Mamba). 
