# Bridging the Gap between Model Explanations in Partially Annotated Multi-label Classification (CVPR 2023) | [Paper](https://arxiv.org/abs/2304.01804)

Youngwook Kim<sup>1</sup>, Jae Myung Kim<sup>2</sup>, Jieun Jeong<sup>1,3</sup>, Cordelia Schmid<sup>4</sup>, Zeynep Akata<sup>2,5</sup>, and Jungwoo Lee<sup>1,3</sup>

<sup>1</sup> <sub>Seoul National Univeristy</sub>  <sup>2</sup> <sub>University of T&uuml;bingen</sub> <sup>3</sup> <sub>HodooAI Lab</sub> <sup>4</sup> <sub>Inria, Ecole normale sup\'erieure, CNRS, PSL Research University</sub> <sup>5</sup> <sub>MPI for Intelligent Systems</sub>  

Primary contact : [ywkim@cml.snu.ac.kr](ywkim@cml.snu.ac.kr)

## Abstract
Due to the expensive costs of collecting labels in multi-label classification datasets, partially annotated multi-label classification has become an emerging field in computer vision. One baseline approach to this task is to assume unobserved labels as negative labels, but this assumption induces label noise as a form of false negative. To understand the negative impact caused by false negative labels, we study how the model's explanation is affected by these labels. We observe that the explanation of the model trained with full labels and partial labels highlights similar regions but with different scaling where the latter tends to have lower attribution scores. Based on these findings, we propose to boost the attribution scores of the model trained with partial labels to make its explanation resemble that of the model trained with full labels. Even with the conceptually simple approach, the multi-label classification performance improves by a large margin in three different datasets on single positive label setting and one dataset on large-scale partial label setting.

## Coming Soon!

## Acknowledgements
Our code is heavily built upon [Multi-Label Learning from Single Positive Labels](https://github.com/elijahcole/single-positive-multi-label) and [Large Loss Matters in Weakly Supervised Multi-Label Classification](https://github.com/snucml/LargeLossMatters).
