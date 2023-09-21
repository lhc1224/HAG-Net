# HAG-Net
1. [📎 Paper Link](#1)
6. [✉️ Statement](#6)
3. [🔍 Citation](#8)

## 📎 Paper Link <a name="1"></a> 
* Learning Visual Affordance Grounding from Demonstration Videos[[pdf](https://arxiv.org/pdf/2108.05675.pdf)] 

## 💡 Abstract <a name="2"></a> 
Visual affordance grounding aims to segment all possible interaction regions between people and objects from an image/video, which benefits many applications, such as robot grasping and action recognition. Prevailing methods predominantly depend on the appearance feature of the objects to segment each region of the image, which encounters the following two problems: (i) there are multiple possible regions in an object that people interact with; (ii) there are multiple possible human interactions in the same object region. To address these problems, we propose a Hand-aided Affordance Grounding Network (HAG-Net) that leverages the aided clues provided by the position and action of the hand in demonstration videos to eliminate the multiple possibilities and better locate the interaction regions in the object. Specifically, HAG-Net adopts a dual-branch structure to process the demonstration video and object image data. For the video branch, we introduce hand-aided attention to enhance the region around the hand in each video frame and then use the LSTM network to aggregate the action features. For the object branch, we introduce a semantic enhancement module (SEM) to make the network focus on different parts of the object according to the action classes and utilize a distillation loss to align the output features of the object branch with that of the video branch and transfer the knowledge in the video branch to the object branch. Quantitative and qualitative evaluations on two challenging datasets show that our method has achieved state-of-the-art results for affordance grounding. 
<p align="center">
    <img src="./img/fig1.png" width="600"/> <br />
    <em> 
    </em>
</p>



Pre-processed selection frames can be downloaded OPRA from [Baidu Pan](https://pan.baidu.com/s/1scLEtDTd59aQgnTdL6OIbQ) (ktpt) ].
and the EPIC dataset can be downloaded from [[Baidu Pan(https://pan.baidu.com/s/1PA0e6CIzIfcZRrPqZYiCSw)(myit)]]


## 🔍 Citation <a name="8"></a> 

```
@article{luo2021learning,
  title={Learning Visual Affordance Grounding from Demonstration Videos},
  author={Luo, Hongchen and Zhai, Wei and Zhang, Jing and Cao, Yang and Tao, Dacheng},
  journal={arXiv preprint arXiv:2108.05675},
  year={2021}
}
```
