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

from datetime import datetime
from multiprocessing import Process, Queue, Value

import numpy as np
import time

from Config import Config
from Environment import Environment
from Experience import Experience

def action_idx(actions, b):
    #x -- action number
    #b -- number of basic actions
    if len(actions)==1:
        assert(actions[0] < b)
        return actions[0]
    else:
        act_idx = 0
        count = 0
        for itm in reversed(actions):
            assert(itm < b)
            act_idx += b**count * itm
            count += 1
        count -= 1  
        act_idx += b**count
        return act_idx

class ProcessAgent(Process):
    def __init__(self, id, prediction_q, training_q, episode_log_q):
        super(ProcessAgent, self).__init__()

        self.id = id
        self.prediction_q = prediction_q
        self.training_q = training_q
        self.episode_log_q = episode_log_q

        self.env = Environment()
        self.num_actions = self.env.get_num_actions()
        self.num_basic_actions = self.env.num_basic_actions()
        self.actions = np.arange(self.num_actions)

        self.discount_factor = Config.DISCOUNT
        # one frame at a time
        self.wait_q = Queue(maxsize=1)
        self.exit_flag = Value('i', 0)

    @staticmethod
    def _accumulate_rewards_iu(experiences, discount_factor, terminal_reward, num_basic_actions):
        reward_sum = terminal_reward
        for t in reversed(range(0, len(experiences)-1)):
            r = np.clip(sum(experiences[t].rewards), Config.REWARD_MIN, Config.REWARD_MAX)
            reward_sum = discount_factor * reward_sum + r
            experiences[t].rewards = reward_sum
            experiences[t].states = experiences[t].states[0]
            experiences[t].actions = experiences[t].action
        return experiences[:-1]

    @staticmethod
    def _accumulate_rewards_du(experiences, discount_factor, terminal_reward, num_basic_actions):
        reward_sum = terminal_reward
        experiences_new = []
        for t in reversed(range(0, len(experiences)-1)):
            #if len(experiences[t].rewards)<len(experiences[t].actions):
            #    experiences[t].actions = experiences[t].actions[:len(experiences[t].rewards)] 
            for i in reversed(range(len(experiences[t].rewards))):
                r = np.clip(experiences[t].rewards[i], Config.REWARD_MIN, Config.REWARD_MAX)
                reward_sum = discount_factor * reward_sum + r
                exp = Experience(experiences[t].states[i],
                    action_idx(experiences[t].actions[:i+1],num_basic_actions), 
                    experiences[t].action, reward_sum, experiences[t].done)
                experiences_new.append(exp)
        return list(reversed(experiences_new))[:-1]

    def convert_data(self, experiences):
        x_ = np.array([exp.states for exp in experiences])
        a_ = np.eye(self.num_actions)[np.array([exp.actions for exp in experiences])].astype(np.float32)
        r_ = np.array([exp.rewards for exp in experiences])
        return x_, r_, a_

    def predict(self, state):
        # put the state in the prediction q
        self.prediction_q.put((self.id, state))
        # wait for the prediction to come back
        p, v = self.wait_q.get()
        return p, v

    def select_action(self, prediction):
        if Config.PLAY_MODE:
            action = np.argmax(prediction)
        else:
            action = np.random.choice(self.actions, p=prediction)
        return action

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

    def run_episode(self):
        self.env.reset()
        done = False
        experiences = []

        time_count = 0
        reward_sum = 0.0

        while not done:
            # very first few frames
            if self.env.current_state is None:
                self.env.step([0])  # 0 == NOOP
                continue

            prediction, value = self.predict(self.env.current_state)
            action = self.select_action(prediction)
            actions = self.action_sequence(action,self.num_basic_actions)
            rewards, done = self.env.step(actions)
            reward_sum += sum(rewards)
            exp = Experience(self.env.previous_states, actions, action, rewards, done)
            #for i in range(len(actions)):
            #   exp = Experience(self.env.previous_states[i], actions[i], prediction, rewards[i], done)
            #   experiences.append(exp)    
            experiences.append(exp)

            if done or time_count == Config.TIME_MAX:
                terminal_reward = 0 if done else value
                if int(time.time()-Config.begin_time)<72000:
                    prob = Config.prob_array[int(time.time()-Config.begin_time)]
                else:
                    prob = Config.prob_array[-1]
                
                if np.random.rand()<prob:
                    updated_exps = ProcessAgent._accumulate_rewards_du(experiences, self.discount_factor, terminal_reward, self.num_basic_actions)
                else:
                    updated_exps = ProcessAgent._accumulate_rewards_iu(experiences, self.discount_factor, terminal_reward, self.num_basic_actions)

                if len(updated_exps)!=0:
                    x_, r_, a_ = self.convert_data(updated_exps)
                    yield x_, r_, a_, reward_sum
                # reset the tmax count
                time_count = 0
                # keep the last experience for the next batch
                experiences = [experiences[-1]]
                reward_sum = 0.0

            time_count += 1

    def run(self):
        # randomly sleep up to 1 second. helps agents boot smoothly.
        time.sleep(np.random.rand())
        np.random.seed(np.int32(time.time() % 1 * 1000 + self.id * 10))

        while self.exit_flag.value == 0:
            total_reward = 0
            total_length = 0
            for x_, r_, a_, reward_sum in self.run_episode():
                total_reward += reward_sum
                total_length += len(r_) + 1  # +1 for last frame that we drop
                self.training_q.put((x_, r_, a_))
            self.episode_log_q.put((datetime.now(), total_reward, total_length))
