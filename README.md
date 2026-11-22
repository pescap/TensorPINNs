# TensorPINNs


## About



Experiments in:

[PINNs for operator equations with stochastic data](https://arxiv.org/abs/2211.10344).


This plug-in allows computing statistical moments for operator equations with stochastic data of the form:

$$
A u (\omega) = f (\omega) \text{ for } \omega \in \Omega.
$$

The revised version of the paper (separable low-rank TPINNs, classical baselines, and the scripts that regenerate
every table and figure) is implemented in JAX in [`separable/`](separable/README.md).

The original scripts can be run with DeepXDE 1.7.1. A Docker image for DeepXDE can be launched with Docker:

```
 nvidia-docker run -p 8888:8888 pescapil/deepxde:latest
```


## Cite
```
@article{escapil2022physics,
  title={Physics-informed neural networks for operator equations with stochastic data},
  author={Escapil-Inchausp{\'e}, Paul and Ruz, Gonzalo A},
  journal={arXiv preprint arXiv:2211.10344},
  year={2022}
}
```
