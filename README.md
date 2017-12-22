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

The repository contains two folders, one for the  baseline GA3C and the other for Switching version of A4C. To run the code, please change
your directory to either of these and run:

```
sh _clean.sh
sh _train.sh
```

### Configuration

Each of these folders has Config.py file where we can specify the game_name, step_size, switching_time can be configured 
