![Generic badge](https://img.shields.io/badge/python-3.8-blue.svg)
![Generic badge](https://img.shields.io/badge/pytorch-1.7.1-blue.svg)
![Generic badge](https://img.shields.io/badge/ubuntu-16.04-orange.svg)
![Generic badge](https://img.shields.io/badge/gpu-RTX_3090-yellowgreen.svg)

# ArcFace_pytorch
- original paper : https://arxiv.org/abs/1801.07698
- ArcFace in Pedestrian Re-Identification

## Index
  - [Overview](#overview) 
  - [Getting Started](#getting-started)
  - [Authors](#authors)
  - [License](#license)
<!--  Other options to write Readme
  - [Deployment](#deployment)
  - [Used or Referenced Projects](Used-or-Referenced-Projects)
-->

## Overview
<!-- Write Overview about this project -->
**Additive Angular Margin Loss for Pedestrian Re-Identification**
- InputPipeline
- Backbone network : ResNet50
- Loss function
- Learning parameters
- Inference
- Evaluation result

**part of object tracking in multi camera** 

<img width="70%" src="https://user-images.githubusercontent.com/22339360/147850656-c4737588-8ed1-479d-b2b7-ced47fd6ed40.png"/>


## Getting Started
**Image Recognition in opened dataset**
<!--
### Depencies
 Write about need to install the software and how to install them 
-->
<!-- A step by step series of examples that tell you how to get a development 
env running

Say what the step will be

    Give the example

And repeat

    until finished
-->
**If you want to learn a model again**
1. Click [AI Hub Korean Re-ID](https://aihub.or.kr/aidata/7977) | [PRID dataset](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/prid11/) button to download dataset (link to download page)
2. Set data (categorize images each ID)
- train data structure example
<img width="70%" src="https://user-images.githubusercontent.com/22339360/147875647-ac5f9bc8-2325-40e6-9376-f4e6b797b395.png"/>
3. `clone`  this repository

### **You can Download model**

[GoogleDrive](https://drive.google.com/file/d/1ycxiUJ--SYfgZ0F2OZ0TBhKuhMVQL_Y5/view?usp=sharing)
<!--
## Deployment
 Add additional notes about how to deploy this on a live system
 -->

## Performance

## Authors
  - [wonjeong lee](https://github.com/one-jeong) - **WonJeong Lee** - <itzmewj97@gmail.com>


## Used or Referenced Projects
- [ronghuaiyang/arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch)

## License

```
MIT License

Copyright (c) 2021 one-jeong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
