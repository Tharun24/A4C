# Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import gym
from Config import Config

class GameManager:
    def __init__(self, game_name, display):
        self.game_name = game_name
        self.display = display
        self.meta_size = Config.meta_size
        self.env = gym.make(game_name)
        self.reset()

    def reset(self):
        observation = self.env.reset()
        return observation

    def get_num_actions(self):
        return int(self.env.action_space.n*(self.env.action_space.n**self.meta_size - 1)/(self.env.action_space.n-1))

    def num_basic_actions(self):
        return self.env.action_space.n

    '''
    def action_sequence(self, x, b):
        #x -- action number
        #b -- number of basic actions
        assert(x >= 0)
        assert(1< b < 37)
        act_seq = []
        while x > 0:
            act_seq.append(x % b)
            x //= b
        if act_seq==[]:
            act_seq=[0]
        elif len(act_seq)>1:
            act_seq[-1] -= 1
            act_seq.reverse()
        return act_seq
	'''
	
    def step(self, actions):
        self._update_display()
        rewards = []
        observations = []
        #state_sequence = []
        #total_reward = 0        
        #actn_seq  = self.action_sequence(action,self.env.action_space.n)
        #for a in actn_seq:
        for a in actions:
            obs, rew, done, info = self.env.step(a)
            rewards.append(rew)
            observations.append(obs)
            #state_sequence.append(observation)
            #total_reward += r 
            if done:
                break
        return observations, rewards, done, info

    def _update_display(self):
        if self.display:
            self.env.render()
