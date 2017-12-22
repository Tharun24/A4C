# A4C

This repository contains the code for A4C introduced in the paper

[A4C:Anticipatory Asynchronous Actor Critic](https://openreview.net/pdf?id=rkKkSzb0b)

Tharun Medini, Xun Luan, [Anshumali Shrivastava](https://www.cs.rice.edu/~as143/)

### Citation

If you find the idea useful, please cite

@article{
  anonymous2018anticipatory,
  title={Anticipatory Asynchronous Advantage Actor-Critic (A4C): The power of Anticipation in Deep Reinforcement Learning},
  author={Anonymous},
  journal={International Conference on Learning Representations},
  year={2018},
  url={https://openreview.net/forum?id=rkKkSzb0b}
}

### Usage

The repository contains two folders, one for the  baseline GA3C and the other for Switching version of A4C. The other 2 variants **Dependent Updating(DU)** and **Indpenednt Updating(IU)** can be obtained by tweaking **Switching**. The code structure is essentially similar to [GA3C](https://github.com/NVlabs/GA3C) with critical differences in **Config.py, ProcessAgent.py, GameManager.py** and **Environment.py**. To run the A4C code, please change
your directory to either of these and run:

```
sh _clean.sh
sh _train.sh
```

### Configuration

The folder **Switching** has the file **Config.py** with the following chunk of code:

```
meta_size = 2 # aka step_size
begin_time = time.time()
switching_time = 9000 
decay_parameter = 1e-3
```

Here, **meta_size** is the max length of multi-step actions. By default, we use upto 2-step actions.

**switching_time** is the time after which we switch(in seconds). This should roughly be the time when starts to decay. 
