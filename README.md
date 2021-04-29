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
Calcium images are captured via head-mounted miniature microscopes on rodents. These calcium images are preprocessed follow by annotation enhancement by domain experts using the [CAVE](https://doi.org/10.3389/fnins.2018.00958) tools, a GUI implementation in MatLab designed by the researchers from LIN. 
<p align="center">
<img src="Fig/data_pipeline">
</p>
