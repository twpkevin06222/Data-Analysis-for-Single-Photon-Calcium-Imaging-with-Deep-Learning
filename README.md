# Data-Analysis-for-Single-Photon-Calcium-Imaging-with-Deep-Learning
Affliated with Leibniz Institute for Neurobiology(LIN) Magdeburg and OVGU Magdeburg, we implemented data analysis for single photon calcium imaging with deep learning. This repo consisted of the work for deep learning team project under the guidance of Prof. Sebastian Stober from [AI Lab](https://ai.ovgu.de/), Faculty of Computer Science and Dr. Michael Lippert from [LIN](https://www.neuroscience-magdeburg.de/research/professor-jazz/michael-lippert/). 
```
Associated team members: Wai Po Kevin Teng, Praveen Putti and Lisa Schneider. 
```
Final report of this project can be viewed [[here](DeepCalciumImagingAnalysis_report.pdf)]

## Motivation
Calcium Imaging is a powerful method to simultaneously capture the activity of cells in the brain of rodents. Especially one-photon calcium imaging with head-mounted miniature
microscopes became increasingly popular since it allows the observation of freely behaving rodents. However, single-photon imaging data processing remains a challenging task due to missing depth information, background fluctuation, low image contrast, and movement artifacts. Distinct cells are not clearly delineated and might overlap with others. A novel deep learning architecture is proposed to detect cells within one-photon calcium imaging sessions. Our approach introduces learnable coordinates, that learn to locate themselves in the center of cell activations. Our deep learning framework is presented in an end-to-end unsupervised learning manner, given the lack of annotated data for each frame. The performance of the proposed method is benchmarked against manually annotated data, generated by two neuroscientists, that aggregates the occurrence of neurons over the whole session.

## Baseline 
Calcium images are captured via head-mounted miniature microscopes on rodents. These calcium images are preprocessed follow by annotation enhancement by domain experts using the CAVE tools, a GUI implementation in MatLab by [Tegtmeier et al](https://doi.org/10.3389/fnins.2018.00958). from LIN in order to retrieve region of interest(ROI). 
<p align="center">
<img src="Fig/data_pipeline.png">
</p>

## Methods 
In our proposed model pipeline, we extensively implemented CoordConv by [Liu et al.](https://arxiv.org/abs/1807.03247) to capture the essence of positional features in the pixel space, hypothesising that this method would aid the allocation of neuron activation in the calcium images along the time frame. 
<p align="center">
<img src="Fig/CoordConvLayer.png">
</p>

## Model 
Since this is an unsupervised learning task, our team adopted an autoencoder inspired model architecture in the hope to learn the neuron activations from calcium imaging via reconstruction. The output of various pipeline in the model are as shown below. 
<p align="center">
<img src="Fig/row1_title.png">
</p>
- Complete model pipeline: 
<p align="center">
  <img src="Fig/Full_Model_Image.jpg", width=600, height=500>
</p>
- Encoder network: 
<p align="center">
<img src="Fig/final_enc_network.jpeg", width=800, height=500>
</p>
