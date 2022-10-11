<h1 align='center'> Gradient Gating for Deep Multi-Rate Learning on Graphs </h1>

This repository contains the implementation of **Gradient Gating (G^2)** 
from the preprint: [Gradient Gating for Deep Multi-Rate Learning on Graphs](https://arxiv.org/abs/2210.00513)

<p align="center">
<img align="middle" src="./imgs/gradient_gating_scheme2.png" width="400" />
</p>

### Requirements
Main dependencies (with python >= 3.7):<br />
torch==1.9.0<br />
torch-cluster==1.5.9<br />
torch-geometric==2.0.3<br />
torch-scatter==2.0.9<br />
torch-sparse==0.6.12<br />
torch-spline-conv==1.2.1<br />

Commands to install all the dependencies in a new conda environment <br />
*(python 3.7 and cuda 10.2 -- for other cuda versions change accordingly)*
```
conda create --name gradientgating python=3.7
conda activate gradientgating

pip install torch==1.9.0

pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
pip install torch-geometric
pip install scipy
pip install numpy
```

# Citation
If you found our work useful in your research, please cite our paper at:
```bibtex
@article{rusch2022gradient,
  title={Gradient Gating for Deep Multi-Rate Learning on Graphs},
  author={Rusch, T Konstantin and Chamberlain, Benjamin P and Mahoney, Michael W and Bronstein, Michael M and Mishra, Siddhartha},
  journal={arXiv preprint arXiv:2210.00513},
  year={2022}
}
```
(Also consider starring the project on GitHub.)
