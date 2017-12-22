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

The repository contains two folders, one for the  baseline GA3C and the other for Switching version of A4C. The other 2 variants * *Dependent Updating(DU)* * and * *Indpenednt Updating(IU)* * can be obtained by tweaking * *Switching* * . The code structure is essentially similar to [GA3C](https://github.com/NVlabs/GA3C) with critical differences in * *Config.py, ProcessAgent.py, GameManager.py* * and * *Environment.py* * . To run the A4C code, please change
your directory to * *Switching* * and run:

```
sh _clean.sh
sh _train.sh
```

### Configuration

The folder * *Switching* * has the file * *Config.py* * with the following chunk of code:

```
meta_size = 2 # aka step_size
begin_time = time.time()
switching_time = 9000
```

Here, * *meta_size* * is the max length of multi-step actions. By default, we use upto 2-step actions.

* *switching_time* * is the time after which we switch(in seconds). This should roughly be the time when * *Dependent Updating* * starts to decay(9000 for Pong, Qbert and SpaceInvaders; 18000 for BeamRider). Automatig this is a little difficult because the rewards are extremely variant even when we perform a moving average over the last 1000 episodes. It's hard to judge whether the rewards have saturated or not. We are working on tracking the moving median of rewards of 1000 episodes and switching when it reduces. We'll update the code when we get robust results.

### Plotting Results

The codes in both GA3C and Switching folders give out text file each with total rewards for each episode. Running the script * *plot.py* * directly plots the comparison of both approaches. We can plot both wrt time and episodes by changing the variable * *plotby* * in the following chunk:

```
plotby = 'time'

if plotby=='episodes':
    plt1, = plt.plot(r_mean1,'r',label='GA3C')
    plt2, = plt.plot(r_mean2,'g',label='A4C')
    .....
```

