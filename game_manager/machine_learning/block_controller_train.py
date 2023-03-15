#!/usr/bin/python3
# -*- coding: utf-8 -*-

import glob
from random import random, sample, randint
import yaml
import numpy as np
import shutil
from collections import deque
from tensorboardX import SummaryWriter
import os
from datetime import datetime
import pprint
# import random
import copy
import torch
import torch.nn as nn
import sys
sys.path.append("game_manager/machine_learning/")
# import omegaconf
# from hydra import compose, initialize
# import subprocess

###################################################
###################################################
# ?????????
###################################################
###################################################


class Block_Controller(object):

    ####################################
    # ??????
    # init parameter
    board_backboard = 0
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0
    CurrentShape_class = 0
    NextShape_class = 0

    # ?2weight
    # ??????
    weight2_available = False
    # ?????????????
    weight2_enable = False
    predict_weight2_enable_index = 0
    predict_weight2_disable_index = 0

    # Debug ??
    debug_flag_shift_rotation = 0
    debug_flag_shift_rotation_success = 0
    debug_flag_try_move = 0
    debug_flag_drop_down = 0
    debug_flag_move_down = 0

    ####################################
    # ??????
    ####################################
    def __init__(self):
        # init parameter
        self.mode = None
        # train
        self.init_train_parameter_flag = False
        # predict
        self.init_predict_parameter_flag = False

    ####################################
    # Yaml ?????????
    ####################################
    def yaml_read(self, yaml_file):
        with open(yaml_file, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg

    ####################################
    # ?? parameter ???
    ####################################
    def set_parameter(self, yaml_file=None, predict_weight=None):
        self.result_warehouse = "outputs/"
        self.latest_dir = self.result_warehouse+"/latest"
        predict_weight2 = None

        ########
        # Config Yaml ????
        if yaml_file is None:
            raise Exception('Please input train_yaml file.')
        elif not os.path.exists(yaml_file):
            raise Exception(
                'The yaml file {} is not existed.'.format(yaml_file))
        cfg = self.yaml_read(yaml_file)

        ########
        # ?????
        if self.mode == "train" or self.mode == "train_sample" or self.mode == "train_sample2":
            # ouput dir ?????????????
            dt = datetime.now()
            self.output_dir = self.result_warehouse + \
                dt.strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(self.output_dir, exist_ok=True)

            # weight_dir ??? output_dir ?? trained model ????? output_dir ?????
            self.weight_dir = self.output_dir+"/trained_model/"
            self.best_weight = self.weight_dir + "best_weight.pt"
            os.makedirs(self.weight_dir, exist_ok=True)
        ########
        # ?????
        else:
            # Config Yaml ??????
            predict_weight_cfg = True
            if ('predict_weight' in cfg["common"]) \
                    and (predict_weight == "outputs/latest/best_weight.pt"):
                predict_weight = cfg["common"]["predict_weight"]
                predict_weight_cfg = True
            else:
                predict_weight_cfg = False

            dirname = os.path.dirname(predict_weight)
            self.output_dir = dirname + "/predict/"
            os.makedirs(self.output_dir, exist_ok=True)

            # ?2 model
            self.weight2_available = False
            self.weight2_enable = False
            # config yaml ? weight2_available ? True, ?? predict_weight2 ????? predict_weight ????????
            if ('weight2_available' in cfg["common"]) \
                    and cfg["common"]["weight2_available"] \
                    and cfg["common"]["predict_weight2"] != None \
                    and predict_weight_cfg:
                self.weight2_available = True
                predict_weight2 = cfg["common"]["predict_weight2"]
                self.predict_weight2_enable_index = cfg["common"]["predict_weight2_enable_index"]
                self.predict_weight2_disable_index = cfg["common"]["predict_weight2_disable_index"]

        ####################
        # default.yaml ? output_dir ????????
        # subprocess.run("cp config/default.yaml %s/"%(self.output_dir), shell=True)
        shutil.copy2(yaml_file, self.output_dir)

        # Tensorboard ????????
        self.writer = SummaryWriter(
            self.output_dir+"/"+cfg["common"]["log_path"])

        ####################
        # ????????
        ########
        # ?????
        if self.mode == "predict" or self.mode == "predict_sample":
            self.log = self.output_dir+"/log_predict.txt"
            self.log_score = self.output_dir+"/score_predict.txt"
            self.log_reward = self.output_dir+"/reward_predict.txt"
        ########
        # ?????
        else:
            self.log = self.output_dir+"/log_train.txt"
            self.log_score = self.output_dir+"/score_train.txt"
            self.log_reward = self.output_dir+"/reward_train.txt"

        # ??
        with open(self.log, "w") as f:
            print("start...", file=f)

        # ?????
        with open(self.log_score, "w") as f:
            print(0, file=f)

        # ????
        with open(self.log_reward, "w") as f:
            print(0, file=f)

        # Move Down ?????
        if 'move_down_flag' in cfg["train"]:
            self.move_down_flag = cfg["train"]["move_down_flag"]
        else:
            self.move_down_flag = 0

        # ??????????
        if cfg["model"]["name"] == "DQN" and ('predict_next_num' in cfg["train"]):
            self.predict_next_num = cfg["train"]["predict_next_num"]
        else:
            self.predict_next_num = 0

        # ??????????
        if cfg["model"]["name"] == "DQN" and ('predict_next_steps' in cfg["train"]):
            self.predict_next_steps = cfg["train"]["predict_next_steps"]
        else:
            self.predict_next_steps = 0

        # ?????????? (???)
        if cfg["model"]["name"] == "DQN" and ('predict_next_num_train' in cfg["train"]):
            self.predict_next_num_train = cfg["train"]["predict_next_num_train"]
        else:
            self.predict_next_num_train = 0

        # ?????????? (???)
        if cfg["model"]["name"] == "DQN" and ('predict_next_steps_train' in cfg["train"]):
            self.predict_next_steps_train = cfg["train"]["predict_next_steps_train"]
        else:
            self.predict_next_steps_train = 0

        # ??????
        if 'time_disp' in cfg["common"]:
            self.time_disp = cfg["common"]["time_disp"]
        else:
            self.time_disp = False

        ####################
        # =====Set tetris parameter=====
        # Tetris ?????
        # self.board_data_width , self.board_data_height ??????????
        self.height = cfg["tetris"]["board_height"]
        self.width = cfg["tetris"]["board_width"]

        # ???????
        self.max_tetrominoes = cfg["tetris"]["max_tetrominoes"]

        ####################
        # ???????????????
        self.state_dim = cfg["state"]["dim"]
        # ??+????
        print("model name: %s" % (cfg["model"]["name"]))

        # config/default.yaml ???
        # MLP ???
        if cfg["model"]["name"] == "MLP":
            # =====load MLP=====
            # model/deepnet.py ? MLP ????
            from machine_learning.model.deepqnet import MLP
            # ??????? MLP ???????????
            self.model = MLP(self.state_dim)
            # ??????
            self.initial_state = torch.FloatTensor(
                [0 for i in range(self.state_dim)])
            # ?????
            self.get_next_func = self.get_next_states
            self.reward_func = self.step
            # ??????
            self.reward_weight = cfg["train"]["reward_weight"]
            # ?????????????
            self.hole_top_limit = 1
            # ???????????????????
            self.hole_top_limit_height = -1

        # DQN ???
        elif cfg["model"]["name"] == "DQN":
            # =====load Deep Q Network=====
            from machine_learning.model.deepqnet import DeepQNetwork
            # DQN ???????????
            self.model = DeepQNetwork()
            if self.weight2_available:
                self.model2 = DeepQNetwork()

            # ??????
            self.initial_state = torch.FloatTensor(
                [[[0 for i in range(10)] for j in range(22)]])
            # ?????
            self.get_next_func = self.get_next_states_v2
            self.reward_func = self.step_v2
            # ??????
            self.reward_weight = cfg["train"]["reward_weight"]

            if 'tetris_fill_reward' in cfg["train"]:
                self.tetris_fill_reward = cfg["train"]["tetris_fill_reward"]
            else:
                self.tetris_fill_reward = 0
            print("tetris_fill_reward:", self.tetris_fill_reward)

            if 'tetris_fill_height' in cfg["train"]:
                self.tetris_fill_height = cfg["train"]["tetris_fill_height"]
            else:
                self.tetris_fill_height = 0
            print("tetris_fill_height:", self.tetris_fill_height)

            if 'height_line_reward' in cfg["train"]:
                self.height_line_reward = cfg["train"]["height_line_reward"]
            else:
                self.height_line_reward = 0
            print("height_line_reward:", self.height_line_reward)

            if 'hole_top_limit_reward' in cfg["train"]:
                self.hole_top_limit_reward = cfg["train"]["hole_top_limit_reward"]
            else:
                self.hole_top_limit_reward = 0
            print("hole_top_limit_reward:", self.hole_top_limit_reward)

            # ?????????????
            if 'hole_top_limit' in cfg["train"]:
                self.hole_top_limit = cfg["train"]["hole_top_limit"]
            else:
                self.hole_top_limit = 1
            print("hole_top_limit:", self.hole_top_limit)

            # ???????????????????
            if 'hole_top_limit_height' in cfg["train"]:
                self.hole_top_limit_height = cfg["train"]["hole_top_limit_height"]
            else:
                self.hole_top_limit_height = -1
            print("hole_top_limit_height:", self.hole_top_limit_height)

            if 'left_side_height_penalty' in cfg["train"]:
                self.left_side_height_penalty = cfg["train"]["left_side_height_penalty"]
            else:
                self.left_side_height_penalty = 0
            print("left_side_height_penalty:", self.left_side_height_penalty)

        # ????????
        if 'bumpiness_left_side_relax' in cfg["train"]:
            self.bumpiness_left_side_relax = cfg["train"]["bumpiness_left_side_relax"]
        else:
            self.bumpiness_left_side_relax = 0
        print("bumpiness_left_side_relax:", self.bumpiness_left_side_relax)

        if 'max_height_relax' in cfg["train"]:
            self.max_height_relax = cfg["train"]["max_height_relax"]
        else:
            self.max_height_relax = 0
        print("max_height_relax:", self.max_height_relax)

        ####################
        # ????? ??????? torch?????? model ?????
        if self.mode == "predict" or self.mode == "predict_sample":
            if not predict_weight == "None":
                if os.path.exists(predict_weight):
                    print("Load {}...".format(predict_weight))
                    # ??????????
                    self.model = torch.load(predict_weight)
                    # ?????????????????
                    self.model.eval()
                else:
                    print("{} is not existed!!".format(predict_weight))
                    exit()
            else:
                print("Please set predict_weight!!")
                exit()

            # ?2 model
            if self.weight2_available and (not predict_weight2 == "None"):
                if os.path.exists(predict_weight2):
                    print("Load2 {}...".format(predict_weight2))
                    # ??????????
                    self.model2 = torch.load(predict_weight2)
                    # ?????????????????
                    self.model2.eval()
                else:
                    print("{} is not existed!!(predict 2)".format(predict_weight))
                    exit()

        ####################
        # finetune ???
        # (?????????????
        elif cfg["model"]["finetune"]:
            # weight ????(?????????)???
            self.ft_weight = cfg["common"]["ft_weight"]
            if not self.ft_weight is None:
                # ?????????????
                self.model = torch.load(self.ft_weight)
                # ?????
                with open(self.log, "a") as f:
                    print("Finetuning mode\nLoad {}...".format(
                        self.ft_weight), file=f)

        # GPU ??????????
#        if torch.cuda.is_available():
#            self.model.cuda()

        # =====Set hyper parameter=====
        #  ????????(???????, ?????????????)
        self.batch_size = cfg["train"]["batch_size"]
        # lr = learning rate????
        self.lr = cfg["train"]["lr"]
        # pytorch ??????float ???
        if not isinstance(self.lr, float):
            self.lr = float(self.lr)
        # ??????????
        self.replay_memory_size = cfg["train"]["replay_memory_size"]
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        # ?? Episode ??? = ????????
        # 1 Episode = 1 ?????
        self.max_episode_size = self.max_tetrominoes
        self.episode_memory = deque(maxlen=self.max_episode_size)
        # ?????????? EPOCH ??(1 EPOCH = 1???)
        self.num_decay_epochs = cfg["train"]["num_decay_epochs"]
        # EPOCH ?
        self.num_epochs = cfg["train"]["num_epoch"]
        # epsilon: ??????????????? initial ?????final ????
        # Fine Tuning ?? initial ?????
        self.initial_epsilon = cfg["train"]["initial_epsilon"]
        self.final_epsilon = cfg["train"]["final_epsilon"]
        # pytorch ??????float ???
        if not isinstance(self.final_epsilon, float):
            self.final_epsilon = float(self.final_epsilon)

        # ????????????????????????(ADAM or SGD) ???
        # =====Set loss function and optimizer=====
        # ADAM ??? .... ?????????????????? ? ???????????????RMSProp ?????????
        if cfg["train"]["optimizer"] == "Adam" or cfg["train"]["optimizer"] == "ADAM":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr)
            self.scheduler = None
        # ADAM ?????SGD (???????????????? STEP SIZE ???????????????)
        else:
            # ??????????????????????????????????????????
            self.momentum = cfg["train"]["lr_momentum"]
            # SGD ???
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=self.momentum)
            # ??????????? EPOCH ?
            self.lr_step_size = cfg["train"]["lr_step_size"]
            # ???????...  Step Size ??? EPOCH ? gammma ??????????
            self.lr_gamma = cfg["train"]["lr_gamma"]
            # ?????????
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        # ???? - MSELoss ??????
        self.criterion = nn.MSELoss()

        # ?????????
        # =====Initialize parameter=====
        # 1EPOCH ... 1??
        self.epoch = 0
        self.score = 0
        self.max_score = -99999
        self.epoch_reward = 0
        self.cleared_lines = 0
        self.cleared_col = [0, 0, 0, 0, 0]
        self.iter = 0
        # ??????
        self.state = self.initial_state
        # ?????0
        self.tetrominoes = 0

        # ?????? Drop ??????????? (-1: ?????, ????: ????)
        # third_y, forth_direction, fifth_x
        self.skip_drop = [-1, -1, -1]

        # ? ??? = ??????????????
        self.gamma = cfg["train"]["gamma"]
        # ???1 ?????????????????????
        self.reward_clipping = cfg["train"]["reward_clipping"]

        self.score_list = cfg["tetris"]["score_list"]
        # ??????
        self.reward_list = cfg["train"]["reward_list"]  #???????????
        # Game Over ?? = Penalty
        self.penalty = self.reward_list[5]  # ??????????????

        ########
        # ??? 1 ??????????????...Q?????????
        # =====Reward clipping=====
        if self.reward_clipping:
            # ???????????(GAMEOVER ??)?????????????
            self.norm_num = max(max(self.reward_list), abs(self.penalty))
            # ????????????????????
            self.reward_list = [r/self.norm_num for r in self.reward_list]
            # ????????????????
            self.penalty /= self.norm_num
            # max_penalty ??? penalty ??????????? ?????????
            self.penalty = min(cfg["train"]["max_penalty"], self.penalty)

        #########
        # =====Double DQN=====
        self.double_dqn = cfg["train"]["double_dqn"]
        self.target_net = cfg["train"]["target_net"]
        if self.double_dqn:
            self.target_net = True

        # Target_net ON ???
        if self.target_net:
            print("set target network...")
            # ?????????
            self.target_model = copy.deepcopy(self.model)
            self.target_copy_intarval = cfg["train"]["target_copy_intarval"]

        ########
        # =====Prioritized Experience Replay=====
        # ???????????????
        self.prioritized_replay = cfg["train"]["prioritized_replay"]
        if self.prioritized_replay:
            from machine_learning.qlearning import PRIORITIZED_EXPERIENCE_REPLAY as PER
            # ????????????
            self.PER = PER(self.replay_memory_size,
                           gamma=self.gamma, alpha=0.7, beta=0.5)

        ########
        # =====Multi step learning=====
        self.multi_step_learning = cfg["train"]["multi_step_learning"]
        if self.multi_step_learning:
            from machine_learning.qlearning import Multi_Step_Learning as MSL
            self.multi_step_num = cfg["train"]["multi_step_num"]
            self.MSL = MSL(step_num=self.multi_step_num, gamma=self.gamma)

    ####################################
    # ???????????? episode memory ? penalty ??
    # ???????? episode_memory ? replay_memory ??
    ####################################
    def stack_replay_memory(self):
        if self.mode == "train" or self.mode == "train_sample" or self.mode == "train_sample2":
            self.score += self.score_list[5]

            # [next_state, reward, next2_state, done]
            self.episode_memory[-1][1] += self.penalty
            self.episode_memory[-1][3] = True  # store False to done lists.
            self.epoch_reward += self.penalty
            #
            if self.multi_step_learning:
                self.episode_memory = self.MSL.arrange(self.episode_memory)

            # ???????? episode_memory ? replay_memory ??
            self.replay_memory.extend(self.episode_memory)
            # ????????
            self.episode_memory = deque(maxlen=self.max_episode_size)
        else:
            pass

    ####################################
    # Game ? Reset ??? (Game Over?)
    # nextMove["option"]["reset_callback_function_addr"] ???
    ####################################
    def update(self):

        ##############################
        # ?????
        ##############################
        if self.mode == "train" or self.mode == "train_sample" or self.mode == "train_sample2":
            # ???????????? episode memory ? penalty ??
            # replay_memory ? episode memory ??
            self.stack_replay_memory()

            ##############################
            # ????
            ##############################
            # ????????1/10??????????
            if len(self.replay_memory) < self.replay_memory_size / 10:
                print("================pass================")
                print("iter: {} ,meory: {}/{} , score: {}, clear line: {}, block: {}, col1-4: {}/{}/{}/{} ".format(self.iter,
                                                                                                                   len(self.replay_memory), self.replay_memory_size / 10, self.score, self.cleared_lines, self.tetrominoes, self.cleared_col[1], self.cleared_col[2], self.cleared_col[3], self.cleared_col[4]))
            # ??????????????
            else:
                print("================update================")
                self.epoch += 1
                # ??????????????
                if self.prioritized_replay:
                    # replay batch index ??
                    batch, replay_batch_index = self.PER.sampling(
                        self.replay_memory, self.batch_size)
                # ???????
                else:
                    # batch ??????????????????????????????????????????? batch_size ??
                    batch = sample(self.replay_memory, min(
                        len(self.replay_memory), self.batch_size))

                # batch ??????????
                # (episode memory ???)
                state_batch, reward_batch, next_state_batch, done_batch = zip(
                    *batch)
                state_batch = torch.stack(
                    tuple(state for state in state_batch))
                reward_batch = torch.from_numpy(
                    np.array(reward_batch, dtype=np.float32)[:, None])
                next_state_batch = torch.stack(
                    tuple(state for state in next_state_batch))

                done_batch = torch.from_numpy(np.array(done_batch)[:, None])

                ###########################
                # ???? Q ???? (model ? __call__ ? forward)
                ###########################
                # max_next_state_batch = torch.stack(tuple(state for state in max_next_state_batch))
                q_values = self.model(state_batch)

                ###################
                # Traget net ????
                if self.target_net:
                    if self.epoch % self.target_copy_intarval == 0 and self.epoch > 0:
                        print("target_net update...")
                        # self.target_copy_intarval ??? best_weight ? target ?????
                        self.target_model = torch.load(self.best_weight)
                        # self.target_model = copy.copy(self.model)
                    # ?????????????????
                    self.target_model.eval()
                    # ======predict Q(S_t+1 max_a Q(s_(t+1),a))======
                    # ????????????????
                    with torch.no_grad():
                        # ?????? batch ??
                        # ???????????? batch ?? "?????" ????? q ?????
                        next_prediction_batch = self.target_model(
                            next_state_batch)
                else:
                    # ?????????????????
                    self.model.eval()
                    # ????????????????
                    with torch.no_grad():
                        # ???????????? batch ????? Q ???? (model ? __call__ ? forward)
                        next_prediction_batch = self.model(next_state_batch)

                ##########################
                # ????????
                ##########################
                self.model.train()

                ##########################
                # Multi Step lerning ???
                if self.multi_step_learning:
                    print("multi step learning update")
                    y_batch = self.MSL.get_y_batch(
                        done_batch, reward_batch, next_prediction_batch)

                # Multi Step lerning ?????
                else:
                    # done_batch, reward_bach, next_prediction_batch(Target net ?????? batch)
                    # ????????? done ? True ?? reward, False (Gameover ?? reward + gammma * prediction Q?)
                    # ? y_batch??? (gamma ????)
                    y_batch = torch.cat(
                        tuple(reward if done[0] else reward + self.gamma * prediction for done, reward, prediction in
                              zip(done_batch, reward_batch, next_prediction_batch)))[:, None]
                # ?????????????????? 0 ??? (???backward ????)
                self.optimizer.zero_grad()
                #########################
                # ???? - ???
                #########################
                # ?????????????
                if self.prioritized_replay:
                    # ?????????????
                    # ?????batch index
                    # ?????batch ??
                    # ?????batch ? Q ?
                    # ???????batch ? Q ? (Target model ????? Target model ??)
                    loss_weights = self.PER.update_priority(
                        replay_batch_index, reward_batch, q_values, next_prediction_batch)
                    # print(loss_weights *nn.functional.mse_loss(q_values, y_batch))
                    # ??????????? (q_values ??? ?????, y_batch ?????[Target net])
                    loss = (loss_weights *
                            self.criterion(q_values, y_batch)).mean()
                    # loss = self.criterion(q_values, y_batch)

                    # ???-????
                    loss.backward()
                else:
                    loss = self.criterion(q_values, y_batch)
                    # ???-????
                    loss.backward()
                # weight ??????????
                self.optimizer.step()
                # SGD ???
                if self.scheduler != None:
                    # ?????
                    self.scheduler.step()

                ###################################
                # ?????
                log = "Epoch: {} / {}, Score: {},  block: {},  Reward: {:.4f} Cleared lines: {}, col: {}/{}/{}/{} ".format(
                    self.epoch,
                    self.num_epochs,
                    self.score,
                    self.tetrominoes,
                    self.epoch_reward,
                    self.cleared_lines,
                    self.cleared_col[1],
                    self.cleared_col[2],
                    self.cleared_col[3],
                    self.cleared_col[4]
                )
                print(log)
                with open(self.log, "a") as f:
                    print(log, file=f)
                with open(self.log_score, "a") as f:
                    print(self.score, file=f)

                with open(self.log_reward, "a") as f:
                    print(self.epoch_reward, file=f)

                # TensorBoard ????
                self.writer.add_scalar(
                    'Train/Score', self.score, self.epoch - 1)
                self.writer.add_scalar(
                    'Train/Reward', self.epoch_reward, self.epoch - 1)
                self.writer.add_scalar(
                    'Train/block', self.tetrominoes, self.epoch - 1)
                self.writer.add_scalar(
                    'Train/clear lines', self.cleared_lines, self.epoch - 1)

                self.writer.add_scalar(
                    'Train/1 line', self.cleared_col[1], self.epoch - 1)
                self.writer.add_scalar(
                    'Train/2 line', self.cleared_col[2], self.epoch - 1)
                self.writer.add_scalar(
                    'Train/3 line', self.cleared_col[3], self.epoch - 1)
                self.writer.add_scalar(
                    'Train/4 line', self.cleared_col[4], self.epoch - 1)

            ###################################
            # EPOCH ??????????
            if self.epoch > self.num_epochs:
                # ????
                with open(self.log, "a") as f:
                    print("finish..", file=f)
                if os.path.exists(self.latest_dir):
                    shutil.rmtree(self.latest_dir)
                os.makedirs(self.latest_dir, exist_ok=True)
                shutil.copyfile(self.best_weight,
                                self.latest_dir+"/best_weight.pt")
                for file in glob.glob(self.output_dir+"/*.txt"):
                    shutil.copyfile(file, self.latest_dir +
                                    "/"+os.path.basename(file))
                for file in glob.glob(self.output_dir+"/*.yaml"):
                    shutil.copyfile(file, self.latest_dir +
                                    "/"+os.path.basename(file))
                with open(self.latest_dir+"/copy_base.txt", "w") as f:
                    print(self.best_weight, file=f)
                ####################
                # ??
                exit()

        ###################################
        # ?????
        else:
            self.epoch += 1
            log = "Epoch: {} / {}, Score: {},  block: {}, Reward: {:.4f} Cleared lines: {}- {}/ {}/ {}/ {}".format(
                self.epoch,
                self.num_epochs,
                self.score,
                self.tetrominoes,
                self.epoch_reward,
                self.cleared_lines,
                self.cleared_col[1],
                self.cleared_col[2],
                self.cleared_col[3],
                self.cleared_col[4]
            )

        ###################################
        # ???????????
        self.reset_state()

    ####################################
    # ??????? (Game Over ?)
    ####################################

    def reset_state(self):
        # ?????
        if self.mode == "train" or self.mode == "train_sample" or self.mode == "train_sample2":
            # ???,500 epoch ?????
            if self.score > self.max_score or self.epoch % 500 == 0:
                torch.save(
                    self.model, "{}/tetris_epoch{}_score{}.pt".format(self.weight_dir, self.epoch, self.score))
                self.max_score = self.score
                torch.save(self.model, self.best_weight)
        # ???????
        self.state = self.initial_state
        self.score = 0
        self.cleared_lines = 0
        self.cleared_col = [0, 0, 0, 0, 0]
        self.epoch_reward = 0
        # ????? 0 ?
        self.tetrominoes = 0
        # ?????? Drop ??????????? (-1: ?????, ????: ????)
        # third_y, forth_direction, fifth_x
        self.skip_drop = [-1, -1, -1]

    ####################################
    # ?????Line????
    ####################################
    def check_cleared_rows(self, reshape_board):
        board_new = np.copy(reshape_board)
        lines = 0
        empty_line = np.array([0 for i in range(self.width)])
        for y in range(self.height - 1, -1, -1):
            blockCount = np.sum(reshape_board[y])
            if blockCount == self.width:
                lines += 1
                board_new = np.delete(board_new, y, 0)
                board_new = np.vstack([empty_line, board_new])
        return lines, board_new

    ####################################
    # ?????, ????, ????, ????????
    ####################################
    def get_bumpiness_and_height(self, reshape_board):
        # ????? 0 ?????(???????????)???
        # (0,1,2,3,4,5,6,7) ? ?????? True, ?? False ???
        mask = reshape_board != 0
        # pprint.pprint(mask, width = 61, compact = True)

        # ??? ??????????????index???
        # ????????????????
        # ??? ??????????????????(?? width)???
        invert_heights = np.where(
            mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        # ??????????? (??)
        heights = self.height - invert_heights
        # ???????? (????)
        total_height = np.sum(heights)
        # ?????????? (????)
        max_height = np.max(heights)
        # ?????????? (????)
        min_height = np.min(heights)
        min_height_l = np.min(heights[1:])    #? ?????????

        # ??????? ????
        # currs = heights[:-1]
        currs = heights[1:-1]

        # ???2?????????
        # nexts = heights[1:]
        nexts = heights[2:]

        # ??????????????
        diffs = np.abs(currs - nexts)
        # ???? self.bumpiness_left_side_relax ??????
        if heights[1] - heights[0] > self.bumpiness_left_side_relax or heights[1] - heights[0] < 0:
            diffs = np.append(abs(heights[1] - heights[0]), diffs)

        # ???????????????????
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height, max_height, min_height, heights[0], min_height_l

    ####################################
    # ???, ??????? Penalty, ????????????
    # reshape_board: 2???????
    # min_height: ??????????1???????????? -1 ??? hole_top_penalty ??
    ####################################
    def get_holes(self, reshape_board, min_height):
        # ???
        num_holes = 0
        # ?????????????
        hole_top_penalty = 0
        # ????? list
        highest_grounds = [-1] * self.width
        # ?????? list
        highest_holes = [-1] * self.width
        # ????????
        for i in range(self.width):
            # ???
            col = reshape_board[:, i]
            # print(col)
            ground_level = 0
            # ????? 0(??????) ???????, ground_level ????????
            while ground_level < self.height and col[ground_level] == 0:
                ground_level += 1
            # ????????list ???
            cols_holes = []
            for y, state in enumerate(col[ground_level + 1:]):
                # ???????list???, list????????????
                if state == 0:
                    # num_holes += 1
                    cols_holes.append(self.height - (ground_level + 1 + y) - 1)
            # ? 1 liner ???????
            # cols_holes = [x for x in col[ground_level + 1:] if x == 0]
            # list ???????????????
            num_holes += len(cols_holes)

            # ???????
            highest_grounds[i] = self.height - ground_level - 1

            # ??????????
            if len(cols_holes) > 0:
                highest_holes[i] = cols_holes[0]
            else:
                highest_holes[i] = -1

        # ?????????
        max_highest_hole = max(highest_holes)

        # ??????????1????????????
        if min_height > 0:
            # ?????????????
            highest_hole_num = 0
            # ????????
            for i in range(self.width):
                # ?????????????
                if highest_holes[i] == max_highest_hole:
                    highest_hole_num += 1
                    # ???????hole_top_limit_height????
                    # ??????????? Penalty
                    if highest_holes[i] > self.hole_top_limit_height and \
                            highest_grounds[i] >= highest_holes[i] + self.hole_top_limit:
                        hole_top_penalty += highest_grounds[i] - \
                            (highest_holes[i])
            # ???????????????????????????????????????????
            hole_top_penalty /= highest_hole_num
            # debug
            # print(['{:02}'.format(n) for n in highest_grounds])
            # print(['{:02}'.format(n) for n in highest_holes])
            # print(hole_top_penalty, hole_top_penalty*max_highest_hole)
            # print("==")

        return num_holes, hole_top_penalty, max_highest_hole

    ####################################
    # ?????????????? (MLP
    ####################################
    def get_state_properties(self, reshape_board):
        # ?????????
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        # ???
        holes, _, _ = self.get_holes(reshape_board, -1)
        # ??????
        bumpiness, height, max_height, min_height, _, _ = self.get_bumpiness_and_height(reshape_board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    ####################################
    # ??????????????????? ????????
    ####################################
    def get_state_properties_v2(self, reshape_board):
        # ?????????
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        # ???
        holes, _, _ = self.get_holes(reshape_board, -1)
        # ??????
        bumpiness, height, max_row, min_height, _, _ = self.get_bumpiness_and_height(
            reshape_board)
        # ????
        # max_row = self.get_max_height(reshape_board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height, max_row])

    ####################################
    # ????????
    # get_bumpiness_and_height ???????????
    ####################################
    def get_max_height(self, reshape_board):
        # X ??????????
        sum_ = np.sum(reshape_board, axis=1)
        # print(sum_)
        row = 0
        # X ?????0??? Y ????
        while row < self.height and sum_[row] == 0:
            row += 1
        return self.height - row

    ####################################
    # ????????????
    ####################################
    def get_tetris_fill_reward(self, reshape_board, piece_id):
        # ?????
        if self.tetris_fill_height == 0:
            return 0

        # ??
        reward = 0
        lines = 0   #??????????????
        max_reward = self.tetris_fill_height
        # ????? 0 ?????(???????????)???
        # (0,1,2,3,4,5,6,7) ? ?????? True, ?? False ???
        mask = reshape_board != 0
        # X ??????????
        sum_ = np.sum(mask, axis=1)
        # ??mask???1????????????????????????
        # print(sum_)

        # line (1 - self.tetris_fill_height)??????????????
        for i in range(1, self.tetris_fill_height):
            # ??????????????????????????????????????
            if self.get_line_right_fill(reshape_board, sum_, i):
                reward += 1
#                # 1???2?    ??????????????reward +5??????
#                if i == 1:
#                    reward += 5

#        # ??????????????I????????1???????????
#        if piece_id == 1:  # I
#            reward += 5

        return reward

    ####################################
    # line ??????????????
    ####################################
    def get_line_right_fill(self, reshape_board, sum_, line):
        # 1????????????
        if sum_[self.height - line] == self.width - 1 \
                and reshape_board[self.height - line][0] == 0:
            # or reshape_board[self.height-1][self.width-1] == 0 ):
            # print("line:", line)
            return True
        else:
            return False

    ####################################
    # ??????????(2???) DQN .... ?????? ????????? ??????????????????
    #  get_next_func ???????
    # curr_backboard ???
    # piece_id ????? I L J T O S Z
    # currentshape_class = status["field_info"]["backboard"]
    ####################################
    def get_next_states_v2(self, curr_backboard, piece_id, CurrentShape_class):
        # ??????
        states = {}

        # ???????????????
        x_range_min = [0] * 4
        x_range_max = [self.width] * 4

        # ??????? drop_y_list[(direction,x)] = height
        drop_y_list = {}
        # ?????? checked_board[(direction0, x0, drop_y)] =True
        checked_board = {}

        # ????????????????
        if piece_id == 5:  # O piece => 1
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:  # I, S, Z piece => 2
            num_rotations = 2
        else:  # the others => 4
            num_rotations = 4

        ####################
        # Drop Down ?? ????????
        # ????????????????
        for direction0 in range(num_rotations):
            # ??????????????????????
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            (x_range_min[direction0], x_range_max[direction0]) = (x0Min, x0Max)

            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                # ???????????????????????????????????????y?????
                board, drop_y = self.getBoard(
                    curr_backboard, CurrentShape_class, direction0, x0, -1)
                # ??????
                drop_y_list[(direction0, x0)] = drop_y
                checked_board[(direction0, x0, drop_y)] = True

                # ????????
                reshape_backboard = self.get_reshape_backboard(board)
                # numpy to tensor (???1????)
                reshape_backboard = torch.from_numpy(
                    reshape_backboard[np.newaxis, :, :]).float()
                # ?????x0? ????????? direction0 ???????????????? ??
                #  states
                #    Key = Tuple (????? Drop Down ?? ??????X??, ?????????
                #                 ????? Move Down ?? ?, ?????????X??, ?????????)
                #                 ... -1 ??? ?????
                #    Value = ???????
                # (action ?)
                states[(x0, direction0, -1, -1, -1)] = reshape_backboard

        # print(len(states), end='=>')

        # Move Down ???????
        if self.move_down_flag == 0:
            return states

        ####################
        # Move Down ?? ????????
        # ??????
        third_y = -1
        forth_direction = -1
        fifth_x = -1
        sixth_y = -1

        # ????????
        reshape_curr_backboard = self.get_reshape_backboard(curr_backboard)

        # ????? 0 ?????(???????????)???
        # (0,1,2,3,4,5,6,7) ? ?????? True, ?? False ???
        mask_board = reshape_curr_backboard != 0
        # pprint.pprint(mask_board, width = 61, compact = True)

        # ??? ??????????????index???
        # ????????????????
        # ??? ??????????????????(?? width)???
        invert_heights = np.where(mask_board.any(
            axis=0), np.argmax(mask_board, axis=0), self.height)
        # ??????????? (??)
        heights = self.height - invert_heights
        # ????
        # max_height = heights[np.argmax(heights)]
        invert_max_height = invert_heights[np.argmin(invert_heights)]

        # Debug
        if self.debug_flag_shift_rotation_success == 1:
            print("")
        if self.debug_flag_shift_rotation == 1 or self.debug_flag_shift_rotation_success == 1:
            print("==================================================")
            print(heights)
            print(invert_heights)
            print("first_direction:", num_rotations,
                  " | ", CurrentShape_class.shape)

        # 1 ??? ??
        for first_direction in range(num_rotations):
            if self.debug_flag_shift_rotation == 1:
                print(" 1d", first_direction, "/ second_x:",
                      x_range_min[first_direction], " to ", x_range_max[first_direction])
            # 2 ??? x ???
            for second_x in range(x_range_min[first_direction], x_range_max[first_direction]):
                # ????????-1?????????????????
                if drop_y_list[(first_direction, second_x)] < invert_max_height + 1:
                    continue
                # ??? ????-2??????????????????
                if invert_heights[second_x] < 2:
                    continue
                # y ?????? ?????????-1 ???
                if self.debug_flag_shift_rotation == 1:
                    print("   2x", second_x, "/ third_y: ", invert_max_height,
                          " to ", drop_y_list[(first_direction, second_x)]+1)

                # 3 ??? y ???
                for third_y in range(invert_max_height, drop_y_list[(first_direction, second_x)]+1):
                    # y ?????? ?????????-1 ???
                    if self.debug_flag_shift_rotation == 1:
                        print("    3y", third_y, "/ forth_direction: ")

                    # ??????????????
                    direction_order = [0] * num_rotations
                    # ??? first_direction
                    new_direction_order = first_direction
                    #
                    for order_num in range(num_rotations):
                        direction_order[order_num] = new_direction_order
                        new_direction_order += 1
                        if not (new_direction_order < num_rotations):
                            new_direction_order = 0

                    # print(first_direction,"::", direction_order)

                    # 4 ??? ?? (Turn 2)
                    # first_direction ?????????
                    for forth_direction in direction_order:
                        # y ?????? ?????????-1 ???
                        if self.debug_flag_shift_rotation == 1:
                            print("     4d", forth_direction, "/ fifth_x: ", 0,
                                  " to ", x_range_max[forth_direction] - second_x, end='')
                            print("//")
                            print("       R:", end='')
                        # 0 ????
                        start_point_x = 0
                        # ??????????????????????
                        if first_direction == forth_direction:
                            start_point_x = 1

                        # ????????
                        right_rotate = True

                        # 5 ??? x ??? (Turn 2)
                        # shift_x ?????????
                        for shift_x in range(start_point_x, x_range_max[forth_direction] - second_x):
                            fifth_x = second_x + shift_x
                            # ???????????????????
                            if not ((forth_direction, fifth_x) in drop_y_list):
                                if self.debug_flag_shift_rotation == 1:
                                    print(shift_x, ": False(OutRange) ", end='/ ')
                                break
                            if third_y <= drop_y_list[(forth_direction, second_x + shift_x)]:
                                if self.debug_flag_shift_rotation == 1:
                                    print(shift_x, ": False(drop) ", end='/ ')
                                break
                            # direction (????)??????2??????????????x,y???????????????
                            coordArray = self.getShapeCoordArray(
                                CurrentShape_class, forth_direction, fifth_x, third_y)
                            # x????????????????????
                            judge = self.try_move_(curr_backboard, coordArray)
                            if self.debug_flag_shift_rotation == 1:
                                print(shift_x, ":", judge, end='/')
                            # ?????
                            if judge:
                                ####
                                # ????????STATES ????
                                states, checked_board = \
                                    self.second_drop_down(curr_backboard, CurrentShape_class,
                                                          first_direction, second_x, third_y, forth_direction, fifth_x,
                                                          states, checked_board)
                            # ?????????
                            else:
                                # ????????????????????????????
                                if shift_x == 0 and judge == False:
                                    right_rotate = False
                                break

                        # ???????????????????????
                        if right_rotate == False:
                            if self.debug_flag_shift_rotation_success == 1:
                                print(" |||", CurrentShape_class.shape, "-", forth_direction,
                                      "(", second_x, ",", third_y, ")|||<=RotateStop|||")
                            break

                        # ??????
                        if self.debug_flag_shift_rotation == 1:
                            print("//")
                            print("       L:", end='')

                        ########
                        # shift_x ?????????
                        for shift_x in range(-1, -second_x - 1, -1):
                            fifth_x = second_x + shift_x
                            # ???????????????????
                            if not ((forth_direction, fifth_x) in drop_y_list):
                                break
                            if third_y <= drop_y_list[(forth_direction, fifth_x)]:
                                break
                            # direction (????)??????2??????????????x,y???????????????
                            coordArray = self.getShapeCoordArray(
                                CurrentShape_class, forth_direction, fifth_x, third_y)
                            # x????????????????????
                            judge = self.try_move_(curr_backboard, coordArray)

                            if self.debug_flag_shift_rotation == 1:
                                print(shift_x, ":", judge, end='/ ')

                            # ?????
                            if judge:
                                ####
                                # ????????STATES ????
                                states, checked_board = \
                                    self.second_drop_down(curr_backboard, CurrentShape_class,
                                                          first_direction, second_x, third_y, forth_direction, fifth_x,
                                                          states, checked_board)

                            # ?????????
                            else:
                                break
                        # ??????
                        if self.debug_flag_shift_rotation == 1:
                            print("//")
                        # end shift_x
                    # end forth
                # end third
            # end second
        # end first

        # Debug
        if self.debug_flag_shift_rotation_success == 1:
            print("")
        # print (len(states))
        # states (action) ???
        return states

    ####################################
    # ???????(1???) MLP  .... ?????? ????????? ??????????????????
    #  get_next_func ???????
    ####################################

    def get_next_states(self, curr_backboard, piece_id, CurrentShape_class):
        # ??????
        states = {}

        if piece_id == 5:  # O piece
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:
            num_rotations = 2
        else:
            num_rotations = 4

        ####################
        # Drop Down ?? ????????
        # ????????????????
        for direction0 in range(num_rotations):
            # ??????????????????????
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            # ???????????????
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                # ???????????????????????????????????????y?????
                board, drop_y = self.getBoard(
                    curr_backboard, CurrentShape_class, direction0, x0, -1)
                # ????????
                reshape_board = self.get_reshape_backboard(board)
                # ?????x0? ????????? direction0 ????????????????
                #  states
                #    Key = Tuple (????? Drop Down ?? ??????X??, ?????????
                #                 ????? Move Down ?? ?, ?????????X??, ?????????)
                #    Value = ???????
                states[(x0, direction0, 0, 0, 0)
                       ] = self.get_state_properties(reshape_board)

        return states

    ####################################
    # ????????? states ????? (?????)
    ####################################
    def second_drop_down(self, curr_backboard, CurrentShape_class,
                         first_direction, second_x, third_y, forth_direction, fifth_x, states, checked_board):
        # debug
        # self.debug_flag_drop_down = 1

        # ???????????????????????????????????????y?????
        new_board, drop_y = self.getBoard(
            curr_backboard, CurrentShape_class, forth_direction, fifth_x, third_y)
        # ????
        sixth_y = third_y + drop_y

        # debug
        if self.debug_flag_shift_rotation_success == 1:
            print(" ***", CurrentShape_class.shape, "-", forth_direction,
                  "(", fifth_x, ",", third_y, "=>", sixth_y, ")***", end='')

        # ?????????
        if not ((forth_direction, fifth_x, sixth_y) in checked_board):
            # debug
            if self.debug_flag_shift_rotation_success == 1:
                print("<=NEW***", end='')
            # ?????????? (????)
            checked_board[(forth_direction, fifth_x, sixth_y)] = True
            # ????????
            reshape_backboard = self.get_reshape_backboard(new_board)
            # numpy to tensor (???1????)
            reshape_backboard = torch.from_numpy(
                reshape_backboard[np.newaxis, :, :]).float()
            ####################
            # ?????x0? ?????????????????? ??
            #  states
            #    Key = Tuple (????? Drop Down ?? ??????X??, ?????????
            #                 ????? Move Down ?? ?, ?????????X??, ?????????)
            #                 ... -1 ??? ?????
            #    Value = ???????
            # (action ?)
            states[(second_x, first_direction, third_y,
                    forth_direction, fifth_x)] = reshape_backboard

        # debug
        if self.debug_flag_shift_rotation_success == 1:
            print("")

        return states, checked_board

    ####################################
    # ??????????
    # board: 1????
    # coordArray: ?????2????
    ####################################
    def try_move_(self, board, coordArray):
        # ?????????(???)???
        judge = True

        debug_board = [0] * self.width * self.height
        debug_log = ""

        for coord_x, coord_y in coordArray:
            debug_log = debug_log + \
                "==(" + str(coord_x) + "," + str(coord_y) + ") "

            # ???????coord_y ? ???????????(???????coord_y????????
            # ???????coord_x, ???????coord_y????????)
            if 0 <= coord_x and \
                coord_x < self.width and \
                coord_y < self.height and \
                (coord_y * self.width + coord_x < 0 or
                    board[coord_y * self.width + coord_x] == 0):

                # ???
                debug_board[coord_y * self.width + coord_x] = 1

            # ??????? False
            else:
                judge = False
                # ?????
                # self.debug_flag_try_move = 1
                if 0 <= coord_x and coord_x < self.width \
                   and 0 <= coord_y and coord_y < self.height:
                    debug_board[coord_y * self.width + coord_x] = 8

        # Debug ?
        if self.debug_flag_try_move == 1:
            print(debug_log)
            pprint.pprint(board, width=31, compact=True)
            pprint.pprint(debug_board, width=31, compact=True)
            self.debug_flag_try_move = 0
        return judge

    ####################################
    # ????????
    ####################################
    def get_reshape_backboard(self, board):
        board = np.array(board)
        # ??, ?? reshape
        reshape_board = board.reshape(self.height, self.width)
        # 1, 0 ???
        reshape_board = np.where(reshape_board > 0, 1, 0)
        return reshape_board

    ####################################
    # ?????(2???)
    # reward_func ????????
    # ????????????????????????
    ####################################
    def step_v2(self, curr_backboard, action, curr_shape_class, hold_shape_id):
        # ?? action ? index ?????
        # 0: 2?? X???
        # 1: 1?? ???????
        # 2: 3?? Y??? (-1: ? Drop)
        # 3: 4?? ??????? (Next Turn)
        # 4: 5?? X??? (Next Turn)
        x0, direction0, third_y, forth_direction, fifth_x = action
        # ???????????????????????????????????????y?????
        board, drop_y = self.getBoard(curr_backboard, curr_shape_class, direction0, x0, -1)
        # ????????
        reshape_board = self.get_reshape_backboard(board)
        # ?????????
        # ?????, ????, ????, ????????
        bampiness, total_height, max_height, min_height, left_side_height, min_height_l = self.get_bumpiness_and_height(reshape_board)
        # max_height = self.get_max_height(reshape_board)
        # ???, ??????? Penalty, ????????????
        hole_num, hole_top_penalty, max_highest_hole = self.get_holes(reshape_board, min_height)
        # ????????????
        tetris_reward = self.get_tetris_fill_reward(reshape_board, hold_shape_id)
        # ????????
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        
        ## ????????????????????????????????????I????????????????????????
        
        # ?????
        reward = self.reward_list[lines_cleared] * (1 + (self.height - max(0, max_height))/self.height_line_reward)

        # I????????????3???????????????????
        if hold_shape_id == 1 & lines_cleared < 3:
            reward += 0.001

#       ???????????4????????????????
#        reward += min(0, (min_height_l - 4))/self.height_line_reward
#       ???????????4??????????????
#        if min_height_l < 4:
#            reward /= 2
        # ????
        # reward += 0.01
        # ??????
        # ??????
        reward -= self.reward_weight[0] * bampiness
        # ??????->?????????????????????
        if max_height > self.max_height_relax:
            reward -= self.reward_weight[1] * max(0, max_height-self.max_height_relax)
        # ????
        reward -= self.reward_weight[2] * hole_num
        # ??????????
        reward -= self.hole_top_limit_reward * hole_top_penalty * max_highest_hole
        # ?????????????
        reward += tetris_reward * self.tetris_fill_reward
        # ???????????
        if left_side_height > self.bumpiness_left_side_relax:
            reward -= (left_side_height - self.bumpiness_left_side_relax) * self.left_side_height_penalty

        self.epoch_reward += reward

        # ?????
        self.score += self.score_list[lines_cleared]
        # ?????????
        self.cleared_lines += lines_cleared
        self.cleared_col[lines_cleared] += 1
        # ?????????????
        self.tetrominoes += 1
        return reward

    ####################################
    # ?????(1???)
    # reward_func ????????
    ####################################
    def step(self, curr_backboard, action, curr_shape_class):
        x0, direction0, third_y, forth_direction, fifth_x = action
        # ???????????????????????????????????????y?????
        board, drop_y = self.getBoard(curr_backboard, curr_shape_class, direction0, x0, -1)
        # ????????
        reshape_board = self.get_reshape_backboard(board)
        # ?????????
        bampiness, height, max_height, min_height, _, _ = self.get_bumpiness_and_height(reshape_board)
        # max_height = self.get_max_height(reshape_board)
        hole_num, _, _ = self.get_holes(reshape_board, min_height)
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        # ?????
        reward = self.reward_list[lines_cleared]
        # ????
        # reward += 0.01
        # ?
        reward -= self.reward_weight[0] * bampiness
        if max_height > self.max_height_relax:
            reward -= self.reward_weight[1] * max(0, max_height)
        reward -= self.reward_weight[2] * hole_num
        self.epoch_reward += reward

        # ?????
        self.score += self.score_list[lines_cleared]

        # ??????
        self.cleared_lines += lines_cleared
        self.cleared_col[lines_cleared] += 1
        self.tetrominoes += 1
        return reward

    ####################################
    ####################################
    ####################################
    ####################################
    # ??????: ?????????????????
    ####################################
    ####################################
    ####################################
    ####################################
    def GetNextMove(self, nextMove, GameStatus, yaml_file=None, weight=None):

        t1 = datetime.now()
        # RESET ???? callback function ?? (Game Over ?)
        nextMove["option"]["reset_callback_function_addr"] = self.update
        # mode ??? (train ???)
        self.mode = GameStatus["judge_info"]["mode"]

        ################
        # ???????????????????????
        if self.init_train_parameter_flag == False:
            self.init_train_parameter_flag = True
            self.set_parameter(yaml_file=yaml_file, predict_weight=weight)

        self.ind = GameStatus["block_info"]["currentShape"]["index"]
        curr_backboard = GameStatus["field_info"]["backboard"]

        ##################
        # default board definition
        # self.width, self.height ???
        self.board_data_width = GameStatus["field_info"]["width"]
        self.board_data_height = GameStatus["field_info"]["height"]

        curr_shape_class = GameStatus["block_info"]["currentShape"]["class"]
        next_shape_class = GameStatus["block_info"]["nextShape"]["class"]
        next_next_shape_class = GameStatus["block_info"]["nextShapeList"]["element2"]["class"]
        hold_shape_class = GameStatus["block_info"]["holdShape"]["class"]

        ##################
        # next shape info
        self.ShapeNone_index = GameStatus["debug_info"]["shape_info"]["shapeNone"]["index"]
        curr_piece_id = GameStatus["block_info"]["currentShape"]["index"]
        next_piece_id = GameStatus["block_info"]["nextShape"]["index"]
        next_next_piece_id = GameStatus["block_info"]["nextShapeList"]["element2"]["index"]
        hold_piece_id = GameStatus["block_info"]["holdShape"]["index"]

        # reshape_backboard = self.get_reshape_backboard(curr_backboard)
        # print(reshape_backboard)
        # self.state = reshape_backboard

        ###############################################
        # Move Down ? ????????????????????????
        if self.skip_drop != [-1, -1, -1]:
            # third_y, forth_direction, fifth_x
            nextMove["strategy"]["direction"] = self.skip_drop[1]
            # ???
            nextMove["strategy"]["x"] = self.skip_drop[2]
            # Move Down ??
            nextMove["strategy"]["y_operation"] = 1
            # Move Down ?? ?
            nextMove["strategy"]["y_moveblocknum"] = 1
            # ?????? Drop ?????????????? (-1: ?????, ????: ????)
            self.skip_drop = [-1, -1, -1]
            # ????
            if self.time_disp:
                print(datetime.now()-t1)
            # ??
            return nextMove

        ###################
        # ?????? ????????? ??????????????????
        # next_steps
        #    Key = Tuple (??????????X??, ?????????)
        #                 ????? Move Down ?? ?, ?????????X??, ?????????)
        #    Value = ???????
        next_steps = self.get_next_func(curr_backboard, curr_piece_id, curr_shape_class)

        # hold???????????
        if hold_piece_id == None:
            # ????hold???
            hold_steps = self.get_next_func(curr_backboard, next_piece_id, next_shape_class)
        else:
            # 2???????
            # print(hold_piece_id)
            hold_steps = self.get_next_func(curr_backboard, hold_piece_id, hold_shape_class)
        # print (len(next_steps), end='=>')

        ###############################################
        ###############################################
        # ?????
        ###############################################
        ###############################################
        if self.mode == "train" or self.mode == "train_sample" or self.mode == "train_sample2":
            # init parameter
            # epsilon = ?????????????????
            # num_decay_epochs ??????????? epsilon ????????
            # num_decay_ecpchs ??? final_epsilon??
            epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
                self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
            u = random()
            # epsilon ???? u ?????????????
            random_action = u <= epsilon

            # ?????????
            if self.predict_next_num_train > 0:
                ##########################
                # ????????
                ##########################
                self.model.train()
                # index_list [1??index, 2??index, 3??index ...] => q
                index_list = []
                hold_index_list = []
                # index_list_to_q (1??index, 2??index, 3??index ...) => q
                index_list_to_q = {}
                hold_index_list_to_q = {}
                ######################
                # ???????predict_next_steps_train???, 1????predict_next_num_train??????
                index_list, index_list_to_q, next_actions, next_states \
                    = self.get_predictions(self.model, True, GameStatus, next_steps, self.predict_next_steps_train, 1, self.predict_next_num_train, index_list, index_list_to_q, -60000)
                # print(index_list_to_q)
                # print("max")
                hold_index_list, hold_index_list_to_q, hold_next_actions, hold_next_states \
                    = self.get_predictions(self.model, True, GameStatus, hold_steps, self.predict_next_steps_train, 1, self.predict_next_num_train, hold_index_list, hold_index_list_to_q, -60000)
#???hold?????????????????
                # ?????? q
                max_index_list = max(index_list_to_q, key=index_list_to_q.get)
                hold_max_index_list = max(hold_index_list_to_q, key=hold_index_list_to_q.get)

                # ?????????Q???????????????????????????
                # ????????????action???????????????????????????????????????????action?????
                if index_list_to_q[tuple(max_index_list)] < hold_index_list_to_q[tuple(hold_max_index_list)]:
                    nextMove["strategy"]["use_hold_function"] = "y"
                    next_steps = hold_steps
                    index_list = hold_index_list
                    index_list_to_q = hold_index_list_to_q
                    next_actions = hold_next_actions
                    next_states = hold_next_states
                    max_index_list = hold_max_index_list

                # print(max(index_list_to_q, key=index_list_to_q.get))
                # print(max_index_list[0].item())
                # print (len(next_steps))
                # print("============================")
                # ??? epsilon ????????
                if random_action:
                    # index ??????
                    index = randint(0, len(next_steps) - 1)
                else:
                    # 1??? index ??
                    index = max_index_list[0].item()
            else:
                # ??????? action ? states ????
                #    next_actions  = Tuple (??????????X??, ?????????)???
                #    next_states = ??????? ??
                next_actions, next_states = zip(*next_steps.items())
                hold_next_actions, hold_next_states = zip(*hold_steps.items())
                # next_states (??????? ??) ???????? (????????list ???????????????)
                next_states = torch.stack(next_states)
                hold_next_states = torch.stack(hold_next_states)

                # GPU ??????????
#                if torch.cuda.is_available():
#                    next_states = next_states.cuda()

                ##########################
                # ????????
                ##########################
                self.model.train()
                # ????????????????(Tensor.backward() ???????????????)
                with torch.no_grad():
                    # ???? Q ???? (model ? __call__ ? forward)
                    predictions = self.model(next_states)[:, 0]
                    hold_predictions = self.model(hold_next_states)[:, 0]
                    # predict = self.model(next_states)[:,:]
                    # predictions = predict[:,0]
                    # print("input: ", next_states)
                    # print("predict: ", predict[:,0])

                if max(predictions) < max(hold_predictions):
                    nextMove["strategy"]["use_hold_function"] = "y"
                    next_steps = hold_steps
                    next_actions = hold_next_actions
                    next_states = hold_next_states
                    predictions = hold_predictions

                # ??? epsilon ????????
                if random_action:
                    # index ??????
                    index = randint(0, len(next_steps) - 1)
                else:
                    # index ??????????
                    index = torch.argmax(predictions).item()

            # ?? action states ???? index ????
            next_state = next_states[index, :]

            # index ???? action ???
            # action ? list
            # 0: 2?? X???
            # 1: 1?? ???????
            # 2: 3?? Y??? (-1: ? Drop)
            # 3: 4?? ??????? (Next Turn)
            # 4: 5?? X??? (Next Turn)
            action = next_actions[index]
            # step, step_v2 ???????
            if nextMove["strategy"]["use_hold_function"] == "y":
              if hold_piece_id == None: # ??hold
                reward = 0  # hold???????????max????
              else:
                reward = self.step_v2(curr_backboard, action, hold_shape_class, curr_piece_id)
            else:
              reward = self.step_v2(curr_backboard, action, curr_shape_class, hold_piece_id)

            done = False  # game over flag

            #####################################
            # Double DQN ???
            # ======predict max_a Q(s_(t+1),a)======
            # if use double dqn, predicted by main model
            if self.double_dqn:
                # ?????????????? ?????????????????????????y?????
                if nextMove["strategy"]["use_hold_function"] == "y":
                    if hold_piece_id == None:
                        next_backboard = curr_backboard
                        drop_y = 0  # ???????????????????
                    else:
                        next_backboard, drop_y = self.getBoard(curr_backboard, hold_shape_class, action[1], action[0], action[2])
                else:
                    next_backboard, drop_y = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0], action[2])

                # ?????? ????????? ??????????????????
                next2_steps = self.get_next_func(next_backboard, next_piece_id, next_shape_class)
                # ?????????????????????
                # if nextMove["strategy"]["use_hold_function"] == "y" and hold_piece_id == None:
                #     next2_steps = self.get_next_func(
                #         next_backboard, next_piece_id, next_shape_class)
                # else:
                #     next2_steps = self.get_next_func(
                #         next_backboard, next_piece_id, next_shape_class)
                #     hold_next2_steps = self.get_next_func(
                #         next_backboard, next_next_piece_id, next_next_shape_class)

                # ??????? action ? states ????
                next2_actions, next2_states = zip(*next2_steps.items())
                # next_states ????????
                next2_states = torch.stack(next2_states)
                # GPU ??????????
#                if torch.cuda.is_available():
#                    next2_states = next2_states.cuda()
                ##########################
                # ????????
                ##########################
                self.model.train()
                # ????????????????
                with torch.no_grad():
                    # ???? Q ???? (model ? __call__ ? forward)
                    next_predictions = self.model(next2_states)[:, 0]
                # ?? index ??????????
                next_index = torch.argmax(next_predictions).item()
                # ????? index ??????
                next2_state = next2_states[next_index, :]

            ################################
            # Target Next ???
            # if use target net, predicted by target model
#             elif self.target_net:
#                 # ?????????????? ?????????????????????????y?????
#                 next_backboard, drop_y = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0], action[2])
#                 # ?????? ????????? ??????????????????
#                 next2_steps = self.get_next_func(next_backboard, next_piece_id, next_shape_class)
#                 # ??????? action ? states ????
#                 next2_actions, next2_states = zip(*next2_steps.items())
#                 # next_states ????????
#                 next2_states = torch.stack(next2_states)
#                 # GPU ??????????
# #                if torch.cuda.is_available():
# #                    next2_states = next2_states.cuda()
#                 ##########################
#                 # ????????
#                 ##########################
#                 self.target_model.train()
#                 # ????????????????
#                 with torch.no_grad():
#                     # ??????????? Q???
#                     next_predictions = self.target_model(next2_states)[:, 0]
#                 # ?? index ??????????
#                 next_index = torch.argmax(next_predictions).item()
#                 # ????? index ??????
#                 next2_state = next2_states[next_index, :]

#             # if not use target net,predicted by main model
#             else:
#                 # ?????????????? ?????????????????????????y?????
#                 next_backboard, drop_y = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0], action[2])
#                 # ?????? ????????? ??????????????????
#                 next2_steps = self.get_next_func(next_backboard, next_piece_id, next_shape_class)
#                 # ??????? action ? states ????
#                 next2_actions, next2_states = zip(*next2_steps.items())
#                 # ????? index ??????
#                 next2_states = torch.stack(next2_states)

#                 # GPU ??????????
# #                if torch.cuda.is_available():
# #                    next2_states = next2_states.cuda()
#                 ##########################
#                 # ????????
#                 ##########################
#                 self.model.train()
#                 # ????????????????
#                 with torch.no_grad():
#                     # ???? Q ???? (model ? __call__ ? forward)
#                     next_predictions = self.model(next2_states)[:, 0]

#                 # epsilon = ?????????????????
#                 # num_decay_epochs ??????????? epsilon ????????
#                 # num_decay_ecpchs ??? final_epsilon??
#                 epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
#                     self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
#                 u = random()
#                 # epsilon ???? u ?????????????
#                 random_action = u <= epsilon

#                 # ??? epsilon ????????
#                 if random_action:
#                     # index ?????
#                     next_index = randint(0, len(next2_steps) - 1)
#                 else:
#                    # ?? index ??????????
#                     next_index = torch.argmax(next_predictions).item()
#                 # ????? index ?????
#                 next2_state = next2_states[next_index, :]

            # =======================================
            # Episode Memory ?
            # next_state  ?????1??
            # reward ??
            # next2_state ?????????????? (Target net ??)
            # done Game Over flag
            # self.replay_memory.append([next_state, reward, next2_state,done])
            self.episode_memory.append([next_state, reward, next2_state, done])
            # ???????????????
            if self.prioritized_replay:
                # ???????????????????
                self.PER.store()

            # self.replay_memory.append([self.state, reward, next_state,done])

            ###############################################
            # ??? ??????
            ###############################################
            # ?????? Drop ??????????? (-1: ?????, ????: ????)
            # third_y, forth_direction, fifth_x
            # self.skip_drop = [-1, -1, -1]

            # ???????
            nextMove["strategy"]["direction"] = action[1]
            # ???
            nextMove["strategy"]["x"] = action[0]
            ###########
            # Drop Down ?????
            if action[2] == -1 and action[3] == -1 and action[4] == -1:
                # Drop Down ??
                nextMove["strategy"]["y_operation"] = 1
                # Move Down ???
                nextMove["strategy"]["y_moveblocknum"] = 1
                # ?????? Drop ??????????? (-1: ?????, ????: ????)
                self.skip_drop = [-1, -1, -1]
            ###########
            # Move Down ?????
            else:
                # Move Down ??
                nextMove["strategy"]["y_operation"] = 0
                # Move Down ?? ?
                nextMove["strategy"]["y_moveblocknum"] = action[2]
                # ?????? Drop ??????????? (-1: ?????, ????: ????)
                # third_y, forth_direction, fifth_x
                self.skip_drop = [action[2], action[3], action[4]]
                # debug
                if self.debug_flag_move_down == 1:
                    print("Move Down: ", "(", action[0], ",", action[2], ")")

            ##########
            # ??????
            ##########
            # 1???(EPOCH)?????????????????????????
            if self.tetrominoes > self.max_tetrominoes:
                nextMove["option"]["force_reset_field"] = True
            # STATE = next_state ??
            self.state = next_state

        ###############################################
        ###############################################
        # ?? ???
        ###############################################
        ###############################################
        elif self.mode == "predict" or self.mode == "predict_sample":
            ##############
            # model ????
            if self.weight2_available:
                # ????????
                reshape_board = self.get_reshape_backboard(curr_backboard)
                # ????????????
                _, _, max_highest_hole = self.get_holes(reshape_board, -1)
                # model2 ??????
                if max_highest_hole < self.predict_weight2_enable_index:
                    self.weight2_enable = True
                # model1 ??????
                if max_highest_hole > self.predict_weight2_disable_index:
                    self.weight2_enable = False

                # debug
                print(GameStatus["judge_info"]["block_index"], self.weight2_enable, max_highest_hole)

            ##############
            # model ??
            predict_model = self.model
            if self.weight2_enable:
                predict_model = self.model2

            # ??????????
            predict_model.eval()

            # ?????????
            if self.predict_next_num > 0:

                # index_list [1??index, 2??index, 3??index ...] => q
                index_list = []
                hold_index_list = []
                # index_list_to_q (1??index, 2??index, 3??index ...) => q
                index_list_to_q = {}
                hold_index_list_to_q = {}
                ######################
                # ???????predict_next_steps_train???, 1????predict_next_num_train??????
#??                                                   ?????original?false??true???????
                index_list, index_list_to_q, next_actions, next_states \
                    = self.get_predictions(self.model, False, GameStatus, next_steps, self.predict_next_steps_train,
                     1, self.predict_next_num_train, index_list, index_list_to_q, -60000)
                # print(index_list_to_q)
                # print("max")
                hold_index_list, hold_index_list_to_q, hold_next_actions, hold_next_states\
                    = self.get_predictions(self.model, False, GameStatus, hold_steps, self.predict_next_steps_train,
                     1, self.predict_next_num_train, hold_index_list, hold_index_list_to_q, -60000)

                # ?????? q
                max_index_list = max(index_list_to_q, key=index_list_to_q.get)
                hold_max_index_list = max(hold_index_list_to_q, key=hold_index_list_to_q.get)

                # ?????????Q???????????????????????????
                # ????????????action???????????????????????????????????????????action?????
                if index_list_to_q[tuple(max_index_list)] < hold_index_list_to_q[tuple(hold_max_index_list)]:
                    nextMove["strategy"]["use_hold_function"] = "y"
                    next_steps = hold_steps
                    index_list = hold_index_list
                    index_list_to_q = hold_index_list_to_q
                    next_actions = hold_next_actions
                    next_states = hold_next_states
                    max_index_list = hold_max_index_list

                # print(max(index_list_to_q, key=index_list_to_q.get))
                # print(max_index_list[0].item())
                # print("============================")
                # 1??? index ??
                index = max_index_list[0].item()

            else:
                # ????????????? action ? states ????states ???
                next_actions, next_states = zip(*next_steps.items())
                hold_next_actions, hold_next_states = zip(*hold_steps.items())
                next_states = torch.stack(next_states)
                hold_next_states = torch.stack(hold_next_states)

                # ???? Q ???? (model ? __call__ ? forward)
                predictions = predict_model(next_states)[:, 0]
                hold_predictions = predict_model(hold_next_states)[:, 0]
                ## ???? index ??
                index = torch.argmax(predictions).item()
                hold_index = torch.argmax(hold_predictions).item()

                if max(predictions) < max(hold_predictions):
                    nextMove["strategy"]["use_hold_function"] = "y"
                    next_steps = hold_steps
                    next_actions = hold_next_actions
                    next_states = hold_next_states
                    predictions = hold_predictions
                    index = hold_index

            # ?? action ? index ?????
            # 0: 2?? X???
            # 1: 1?? ???????
            # 2: 3?? Y??? (-1: ? Drop)
            # 3: 4?? ??????? (Next Turn)
            # 4: 5?? X??? (Next Turn)
            action = next_actions[index]

            ###############################################
            # ??? ??????
            ###############################################
            # ?????? Drop ??????????? (-1: ?????, ????: ????)
            # third_y, forth_direction, fifth_x
            # self.skip_drop = [-1, -1, -1]
            # ???????
            nextMove["strategy"]["direction"] = action[1]
            # ???
            nextMove["strategy"]["x"] = action[0]
            ###########
            # Drop Down ?????
            if action[2] == -1 and action[3] == -1 and action[4] == -1:
                # Drop Down ??
                nextMove["strategy"]["y_operation"] = 1
                # Move Down ???
                nextMove["strategy"]["y_moveblocknum"] = 1
                # ?????? Drop ??????????? (-1: ?????, ????: ????)
                self.skip_drop = [-1, -1, -1]
            ###########
            # Move Down ?????
            else:
                # Move Down ??
                nextMove["strategy"]["y_operation"] = 0
                # Move Down ?? ?
                nextMove["strategy"]["y_moveblocknum"] = action[2]
                # ?????? Drop ??????????? (-1: ?????, ????: ????)
                # third_y, forth_direction, fifth_x
                self.skip_drop = [action[2], action[3], action[4]]
                # debug
                if self.debug_flag_move_down == 1:
                    print("Move Down: ", "(", action[0], ",", action[2], ")")
        # ????
        if self.time_disp:
            print(datetime.now()-t1)
        # ??
        return nextMove

    ####################################
    # ????????????????????Top num_steps???
    # self:
    # predict_model: ?????
    # is_train: ?????????? (no_grad?????)
    # GameStatus: GameStatus
    # prev_steps: ???????????
    # num_steps: 1???????????????
    # next_order: ????????
    # left: ?????????????
    # index_list: ?????index???
    # index_list_to_q: ?????index????? Q ?????
    ####################################
    def get_predictions(self, predict_model, is_train, GameStatus, prev_steps, num_steps, next_order, left, index_list, index_list_to_q, highest_q):
        # ??????
        next_predictions = []
        # index_list ??
        new_index_list = []

        # ????????
        # next_predict_backboard = []

        # ????????????? action ? states ????states ???
        next_actions, next_states = zip(*prev_steps.items())
        next_states = torch.stack(next_states)
        # ????????
        if is_train:
            # GPU ??????????
            #            if torch.cuda.is_available():
            #                next_states = next_states.cuda()
            # ????????????????
            with torch.no_grad():
                # ???? Q ???? (model ? __call__ ? forward)
                predictions = predict_model(next_states)[:, 0]
        # ????????
        else:
            # ???? Q ???? (model ? __call__ ? forward)
            predictions = predict_model(next_states)[:, 0]

        # num_steps ???? Top ? index ??
        top_indices = torch.topk(predictions, num_steps).indices

        # ????
        if next_order < left:
            # ??????????
            # predict_order = 0
            for index in top_indices:
                # index_list ???
                new_index_list = index_list.copy()
                new_index_list.append(index)
                # Q ???
                now_q = predictions[index].item()
                if now_q > highest_q:
                    # ??????
                    highest_q = now_q

                # ??????? (torch) ????????
                next_state = next_states[index, :]
                # print(next_order, ":", next_state)
                # Numpy ???? int ????1???
                # next_predict_backboard.append(np.ravel(next_state.numpy().astype(int)))
                # print(predict_order,":", next_predict_backboard[predict_order])

                # ????????
                # next_state Numpy ???? int ????1???
                next_steps = self.get_next_func(np.ravel(next_state.numpy().astype(int)),
                                                GameStatus["block_info"]["nextShapeList"]["element"+str(next_order)]["index"],
                                                GameStatus["block_info"]["nextShapeList"]["element"+str(next_order)]["class"])
                # GameStatus["block_info"]["nextShapeList"]["element"+str(1)]["direction_range"]

                # ??????? num_steps ??, next_order ???? left ??????
                new_index_list, index_list_to_q, new_next_actions, new_next_states\
                    = self.get_predictions(predict_model, is_train, GameStatus,
                                           next_steps, num_steps, next_order+1, left, new_index_list, index_list_to_q, highest_q)
                # ??????
                # predict_order += 1
        # ????
        else:
            # Top ?? index_list ???
            new_index_list = index_list.copy()
            new_index_list.append(top_indices[0])
            # Q ???
            now_q = predictions[top_indices[0]].item()
            if now_q > highest_q:
                # ??????
                highest_q = now_q
            # index_list ?? q ?????????
            # print (new_index_list, highest_q, now_q)
            index_list_to_q[tuple(new_index_list)] = highest_q

        # ???????Q?, ?????? action, state ???
        return new_index_list, index_list_to_q, next_actions, next_states

    ####################################
    # ??????????????????????
    # self,
    # Shape_class: ?????????????
    # direction: ??????????
    ####################################
    def getSearchXRange(self, Shape_class, direction):
        #
        # get x range from shape direction.
        #
        # ?????????? x ?????????????????
        # get shape x offsets[minX,maxX] as relative value.
        minX, maxX, _, _ = Shape_class.getBoundingOffsets(direction)
        # ????????
        xMin = -1 * minX
        # ???????????????????
        xMax = self.board_data_width - maxX
        return xMin, xMax

    ####################################
    # direction (????)??????????????????x,y????????2?????????
    ####################################
    def getShapeCoordArray(self, Shape_class, direction, x, y):
        #
        # get coordinate array by given shape.
        # direction (????)??????????????????x,y????????2?????????
        # get array from shape direction, x, y.
        coordArray = Shape_class.getCoords(direction, x, y)
        return coordArray

    ####################################
    # ???????????????????????????????????????y?????
    # board_backboard: ???????
    # Shape_class: ??????/?????
    # direction: ?????????
    # center_x: ?????x??
    # center_y: ?????y??
    ####################################
    def getBoard(self, board_backboard, Shape_class, direction, center_x, center_y):
        #
        # get new board.
        #
        # copy backboard data to make new board.
        # if not, original backboard data will be updated later.
        board = copy.deepcopy(board_backboard)
        # ??????????????????????????????????
        _board, drop_y = self.dropDown(board, Shape_class, direction, center_x, center_y)
        return _board, drop_y

    ####################################
    # ??????????????????????????????????
    # board: ???????
    # Shape_class: ??????/?????
    # direction: ?????????
    # center_x: ?????x??
    # center_y: ?????y?? (-1: Drop ??)
    ####################################
    def dropDown(self, board, Shape_class, direction, center_x, center_y):
        #
        # internal function of getBoard.
        # -- drop down the shape on the board.
        #
        ###############
        # Drop Down ?????
        if center_y == -1:
            center_y = 0

        # ???????????? dy ??
        dy = self.board_data_height - 1
        # direction (????)??????2??????????????x,y???????????????
        coordArray = self.getShapeCoordArray(Shape_class, direction, center_x, center_y)

        # update dy
        # ????????????...
        for _x, _y in coordArray:
            _yy = 0
            # _yy ?????????????????????????
            # _yy+???????y ? ???????????(_yy +???????y???????? ??? ???????_x,_yy+???????_y????????)
            while _yy + _y < self.board_data_height and (_yy + _y < 0 or board[(_y + _yy) * self.board_data_width + _x] == self.ShapeNone_index):
                # _yy ??????(?????)
                _yy += 1
            _yy -= 1
            # ???? dy /???????????(??)?? __yy ??????????
            if _yy < dy:
                dy = _yy
        # dy: ??????????????
        _board = self.dropDownWithDy(board, Shape_class, direction, center_x, dy)

        # debug
        if self.debug_flag_drop_down == 1:
            print("<%%", direction, center_x, center_y, dy, "%%>", end='')
            self.debug_flag_drop_down = 0
        return _board, dy

    ####################################
    # ???????????????
    # board: ???????
    # Shape_class: ??????/?????
    # direction: ?????????
    # center_x: ?????x??
    # center_y: ?????y?????
    ####################################
    def dropDownWithDy(self, board, Shape_class, direction, center_x, center_y):
        #
        # internal function of dropDown.
        #
        # board ???
        _board = board
        # direction (????)??????2??????????????x,y???????????????
        coordArray = self.getShapeCoordArray(Shape_class, direction, center_x, 0)
        # ???????????????
        for _x, _y in coordArray:
            # center_x, center_y ? ?????????????????????????????
            _board[(_y + center_y) * self.board_data_width + _x] = Shape_class.shape
        return _board


BLOCK_CONTROLLER_TRAIN = Block_Controller()