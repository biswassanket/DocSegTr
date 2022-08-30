# DocSegTr
 
## Description
Official Pytorch implementation of the paper [DocSegTr: An Instance-Level End-to-End Document Image Segmentation Transformer](https://arxiv.org/abs/2201.11438). This model is implemented on top of the [adelaidet](https://github.com/aim-uofa/AdelaiDet) and [detectron2](https://github.com/facebookresearch/detectron2) frameworks. The paper proposes a novel bottom-up instance segmentation strategy using Transformers to segment instances(document layouts) in scientific document images from the [PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) benchmark.

<p align="center">
  <img src="https://github.com/biswassanket/DocSegTr/blob/master/images/architecture.png">
  <br>
  <br>
  <b><i>DocSegtr builds on a simple CNN feature extractor with FPN on the input document image. The multi-scaled feature maps(P2-P6) from FPN are combined with positional embedding information to feed into transformer layers, to predict document instances and generate corresponding kernel dynamically. The layerwise feature aggregation module combines the local FPN features and global transformer feature from P5 to segment the instances on the document image</i></b>
</p>

## Getting Started 

### Step 1: Clone this repository and change directory to repository root
```bash
git clone https://github.com/biswassanket/DocSegTr.git 
cd DocSegTr
```

### Step 2: Setup and activate the conda environment with required dependencies:

```bash
conda env create -f environment.yml
conda activate instaseg
```
### Step 3: Build detectron 2 and adelaidet from source:

Building detectron2 v0.2.1 from source using the following [link](https://github.com/facebookresearch/detectron2/archive/refs/tags/v0.2.1.zip) to download.    

```bash
cd detectron2-0.2.1
python setup.py build develop
```
To build [adelaidet](https://github.com/aim-uofa/AdelaiDet) from source you need this simple command:

```bash
cd .. //going back to original work dir
python setup.py build develop
```

### Step 4: Downloading dataset 

* To download **PubLayNet** dataset: `curl -o <YOUR_TARGET_DIR>/publaynet.tar.gz https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz`

### Step 5: For testing our model, download the best pretrained model weights from this link [best model](https://drive.google.com/file/d/1N5FLCbnJIq_1cvrN4D8OmXqiaxcd5ql4/view?usp=sharing)

```bash
python tools/train_net_custom.py \
    --config-file configs/SOTR/R_101_DCN_doc.yaml \
    --eval-only \
    --num-gpus 1 \
    MODEL.WEIGHTS work_dir/.../model_final.pth
```

### Step 6: For training the model from scratch, use this magic command for training on 'n' GPUs:


```bash
python tools/train_net_custom.py \
    --config-file configs/SOTR/R_101_DCN_doc.yaml \
    --num-gpus 2
```
### Step 7: For visualizing qualitative result predictions on PubLayNet, use the following:

```bash
python tools/visualize_publaynet.py \
    --input /path to JSON created by trained model/ \
    --output /path to output_dir \
    --dataset publaynet_minival \
    --conf-threshold 0.6
```
## Results

<p align="center">
  <img src="https://github.com/ayanban011/DocSegTr/blob/master/images/Qualitative_analysis.png">
  <br>
  <br>
  <b><i>Qualitative analysis on the PubLayNet dataset by DocSegTr . Here first, second and third columns represent original image, ground truth and our proposed DocSegTr results, respectively.</i></b>
</p>

### Citation

If you find this code useful in your research then please cite

```
@article{biswas2022docsegtr,
  title={DocSegTr: An Instance-Level End-to-End Document Image Segmentation Transformer},
  author={Biswas, Sanket and Banerjee, Ayan and Llad{\'o}s, Josep and Pal, Umapada},
  journal={arXiv preprint arXiv:2201.11438},
  year={2022}
}
```

## Acknowledgement 
Our project has adapted and borrowed the code structure from [SOTR](https://github.com/easton-cau/SOTR). 
We thank the authors. This research has been partially supported by the Spanish projects RTI2018-095645-B-C21, and FCT-19-15244, and the Catalan projects 2017-SGR-1783, the CERCA Program / Generalitat de Catalunya and PhD Scholarship from AGAUR (2021FIB-10010).
  
## Author
* [Sanket Biswas](https://github.com/biswassanket)
* [Ayan Banerjee](https://github.com/ayanban011)

  
### Conclusion
Thank you and sorry for the bugs!









