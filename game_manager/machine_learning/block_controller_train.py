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
# �u���b�N����N���X
###################################################
###################################################


class Block_Controller(object):

    ####################################
    # �N����������
    # init parameter
    board_backboard = 0
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0
    CurrentShape_class = 0
    NextShape_class = 0

    # ��2weight
    # �L�����ǂ���
    weight2_available = False
    # �Q�[���r���̐؂�ւ��t���O
    weight2_enable = False
    predict_weight2_enable_index = 0
    predict_weight2_disable_index = 0

    # Debug �o��
    debug_flag_shift_rotation = 0
    debug_flag_shift_rotation_success = 0
    debug_flag_try_move = 0
    debug_flag_drop_down = 0
    debug_flag_move_down = 0

    ####################################
    # �N����������
    ####################################
    def __init__(self):
        # init parameter
        self.mode = None
        # train
        self.init_train_parameter_flag = False
        # predict
        self.init_predict_parameter_flag = False

    ####################################
    # Yaml �p�����[�^�ǂݍ���
    ####################################
    def yaml_read(self, yaml_file):
        with open(yaml_file, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        return cfg

    ####################################
    # ���� parameter ��ݒ�
    ####################################
    def set_parameter(self, yaml_file=None, predict_weight=None):
        self.result_warehouse = "outputs/"
        self.latest_dir = self.result_warehouse+"/latest"
        predict_weight2 = None

        ########
        # Config Yaml �ǂݍ���
        if yaml_file is None:
            raise Exception('Please input train_yaml file.')
        elif not os.path.exists(yaml_file):
            raise Exception(
                'The yaml file {} is not existed.'.format(yaml_file))
        cfg = self.yaml_read(yaml_file)

        ########
        # �w�K�̏ꍇ
        if self.mode == "train" or self.mode == "train_sample" or self.mode == "train_sample2":
            # ouput dir �Ƃ��ē��t�f�B���N�g���쐬
            dt = datetime.now()
            self.output_dir = self.result_warehouse + \
                dt.strftime("%Y-%m-%d-%H-%M-%S")
            os.makedirs(self.output_dir, exist_ok=True)

            # weight_dir �Ƃ��� output_dir ���� trained model �t�H���_�� output_dir �P���ɍ��
            self.weight_dir = self.output_dir+"/trained_model/"
            self.best_weight = self.weight_dir + "best_weight.pt"
            os.makedirs(self.weight_dir, exist_ok=True)
        ########
        # ���_�̏ꍇ
        else:
            # Config Yaml �Ŏw��̏ꍇ
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

            # ��2 model
            self.weight2_available = False
            self.weight2_enable = False
            # config yaml �� weight2_available �� True, ���� predict_weight2 �����肩�� predict_weight ���w��łȂ��ꍇ
            if ('weight2_available' in cfg["common"]) \
                    and cfg["common"]["weight2_available"] \
                    and cfg["common"]["predict_weight2"] != None \
                    and predict_weight_cfg:
                self.weight2_available = True
                predict_weight2 = cfg["common"]["predict_weight2"]
                self.predict_weight2_enable_index = cfg["common"]["predict_weight2_enable_index"]
                self.predict_weight2_disable_index = cfg["common"]["predict_weight2_disable_index"]

        ####################
        # default.yaml �� output_dir �ɃR�s�[���Ă���
        # subprocess.run("cp config/default.yaml %s/"%(self.output_dir), shell=True)
        shutil.copy2(yaml_file, self.output_dir)

        # Tensorboard �o�̓t�H���_�ݒ�
        self.writer = SummaryWriter(
            self.output_dir+"/"+cfg["common"]["log_path"])

        ####################
        # ���O�t�@�C���ݒ�
        ########
        # ���_�̏ꍇ
        if self.mode == "predict" or self.mode == "predict_sample":
            self.log = self.output_dir+"/log_predict.txt"
            self.log_score = self.output_dir+"/score_predict.txt"
            self.log_reward = self.output_dir+"/reward_predict.txt"
        ########
        # �w�K�̏ꍇ
        else:
            self.log = self.output_dir+"/log_train.txt"
            self.log_score = self.output_dir+"/score_train.txt"
            self.log_reward = self.output_dir+"/reward_train.txt"

        # ���O
        with open(self.log, "w") as f:
            print("start...", file=f)

        # �X�R�A���O
        with open(self.log_score, "w") as f:
            print(0, file=f)

        # ��V���O
        with open(self.log_reward, "w") as f:
            print(0, file=f)

        # Move Down �~���L����
        if 'move_down_flag' in cfg["train"]:
            self.move_down_flag = cfg["train"]["move_down_flag"]
        else:
            self.move_down_flag = 0

        # ���̃e�g���~�m�\����
        if cfg["model"]["name"] == "DQN" and ('predict_next_num' in cfg["train"]):
            self.predict_next_num = cfg["train"]["predict_next_num"]
        else:
            self.predict_next_num = 0

        # ���̃e�g���~�m��␔
        if cfg["model"]["name"] == "DQN" and ('predict_next_steps' in cfg["train"]):
            self.predict_next_steps = cfg["train"]["predict_next_steps"]
        else:
            self.predict_next_steps = 0

        # ���̃e�g���~�m�\���� (�w�K��)
        if cfg["model"]["name"] == "DQN" and ('predict_next_num_train' in cfg["train"]):
            self.predict_next_num_train = cfg["train"]["predict_next_num_train"]
        else:
            self.predict_next_num_train = 0

        # ���̃e�g���~�m��␔ (�w�K��)
        if cfg["model"]["name"] == "DQN" and ('predict_next_steps_train' in cfg["train"]):
            self.predict_next_steps_train = cfg["train"]["predict_next_steps_train"]
        else:
            self.predict_next_steps_train = 0

        # �I�������\��
        if 'time_disp' in cfg["common"]:
            self.time_disp = cfg["common"]["time_disp"]
        else:
            self.time_disp = False

        ####################
        # =====Set tetris parameter=====
        # Tetris �Q�[���w��
        # self.board_data_width , self.board_data_height �Ɠ�d�w��A�����K�v
        self.height = cfg["tetris"]["board_height"]
        self.width = cfg["tetris"]["board_width"]

        # �ő�e�g���~�m
        self.max_tetrominoes = cfg["tetris"]["max_tetrominoes"]

        ####################
        # �j���[�����l�b�g���[�N�̓��͐�
        self.state_dim = cfg["state"]["dim"]
        # �w�K+���_����
        print("model name: %s" % (cfg["model"]["name"]))

        # config/default.yaml �őI��
        # MLP �̏ꍇ
        if cfg["model"]["name"] == "MLP":
            # =====load MLP=====
            # model/deepnet.py �� MLP �ǂݍ���
            from machine_learning.model.deepqnet import MLP
            # ���͐��ݒ肵�� MLP ���f���C���X�^���X�쐬
            self.model = MLP(self.state_dim)
            # ������ԋK��
            self.initial_state = torch.FloatTensor(
                [0 for i in range(self.state_dim)])
            # �e�֐��K��
            self.get_next_func = self.get_next_states
            self.reward_func = self.step
            # ��V�֘A�K��
            self.reward_weight = cfg["train"]["reward_weight"]
            # ���̏�̐ςݏグ�y�i���e�B
            self.hole_top_limit = 1
            # ���̏�̐ςݏグ�y�i���e�B������΍���
            self.hole_top_limit_height = -1

        # DQN �̏ꍇ
        elif cfg["model"]["name"] == "DQN":
            # =====load Deep Q Network=====
            from machine_learning.model.deepqnet import DeepQNetwork
            # DQN ���f���C���X�^���X�쐬
            self.model = DeepQNetwork()
            if self.weight2_available:
                self.model2 = DeepQNetwork()

            # ������ԋK��
            self.initial_state = torch.FloatTensor(
                [[[0 for i in range(10)] for j in range(22)]])
            # �e�֐��K��
            self.get_next_func = self.get_next_states_v2
            self.reward_func = self.step_v2
            # ��V�֘A�K��
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

            # ���̏�̐ςݏグ�y�i���e�B
            if 'hole_top_limit' in cfg["train"]:
                self.hole_top_limit = cfg["train"]["hole_top_limit"]
            else:
                self.hole_top_limit = 1
            print("hole_top_limit:", self.hole_top_limit)

            # ���̏�̐ςݏグ�y�i���e�B������΍���
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

        # ���ʕ�V�֘A�K��
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
        # ���_�̏ꍇ ���_�E�F�C�g�� torch�@�œǂݍ��� model �ɓ����B
        if self.mode == "predict" or self.mode == "predict_sample":
            if not predict_weight == "None":
                if os.path.exists(predict_weight):
                    print("Load {}...".format(predict_weight))
                    # ���_�C���X�^���X�쐬
                    self.model = torch.load(predict_weight)
                    # �C���X�^���X�𐄘_���[�h�ɐ؂�ւ�
                    self.model.eval()
                else:
                    print("{} is not existed!!".format(predict_weight))
                    exit()
            else:
                print("Please set predict_weight!!")
                exit()

            # ��2 model
            if self.weight2_available and (not predict_weight2 == "None"):
                if os.path.exists(predict_weight2):
                    print("Load2 {}...".format(predict_weight2))
                    # ���_�C���X�^���X�쐬
                    self.model2 = torch.load(predict_weight2)
                    # �C���X�^���X�𐄘_���[�h�ɐ؂�ւ�
                    self.model2.eval()
                else:
                    print("{} is not existed!!(predict 2)".format(predict_weight))
                    exit()

        ####################
        # finetune �̏ꍇ
        # (�ȑO�̊w�K���ʂ��g���ꍇ�@
        elif cfg["model"]["finetune"]:
            # weight �t�@�C��(�ȑO�̊w�K�t�@�C��)���w��
            self.ft_weight = cfg["common"]["ft_weight"]
            if not self.ft_weight is None:
                # �ǂݍ���ŃC���X�^���X�쐬
                self.model = torch.load(self.ft_weight)
                # ���O�֏o��
                with open(self.log, "a") as f:
                    print("Finetuning mode\nLoad {}...".format(
                        self.ft_weight), file=f)

        # GPU �g�p�ł���Ƃ��͎g��
#        if torch.cuda.is_available():
#            self.model.cuda()

        # =====Set hyper parameter=====
        #  �w�K�o�b�`�T�C�Y(�w�K�̕����P��, �f�[�^�T�C�Y�𕪊����Ă���)
        self.batch_size = cfg["train"]["batch_size"]
        # lr = learning rate�@�w�K��
        self.lr = cfg["train"]["lr"]
        # pytorch �݊����̂���float �ɕϊ�
        if not isinstance(self.lr, float):
            self.lr = float(self.lr)
        # ���v���C�������T�C�Y
        self.replay_memory_size = cfg["train"]["replay_memory_size"]
        self.replay_memory = deque(maxlen=self.replay_memory_size)
        # �ő� Episode �T�C�Y = �ő�e�g���~�m��
        # 1 Episode = 1 �e�g���~�m
        self.max_episode_size = self.max_tetrominoes
        self.episode_memory = deque(maxlen=self.max_episode_size)
        # �w�K���������ʂ��o�� EPOCH ���@(1 EPOCH = 1�Q�[��)
        self.num_decay_epochs = cfg["train"]["num_decay_epochs"]
        # EPOCH ��
        self.num_epochs = cfg["train"]["num_epoch"]
        # epsilon: �ߋ��̊w�K���ʂ���ύX���銄�� initial �͏����l�Afinal �͍ŏI�l
        # Fine Tuning ���� initial �������߂�
        self.initial_epsilon = cfg["train"]["initial_epsilon"]
        self.final_epsilon = cfg["train"]["final_epsilon"]
        # pytorch �݊����̂���float �ɕϊ�
        if not isinstance(self.final_epsilon, float):
            self.final_epsilon = float(self.final_epsilon)

        # �����֐��i�\���l�ƁA���ۂ̐���l�̌덷�j�ƌ��z�@(ADAM or SGD) �̌���
        # =====Set loss function and optimizer=====
        # ADAM �̏ꍇ .... �ړ����ςŐU����}�����郂�[�����^�� �� �w�K���𒲐����ĐU����}������RMSProp ��g�ݍ��킹�Ă���
        if cfg["train"]["optimizer"] == "Adam" or cfg["train"]["optimizer"] == "ADAM":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr)
            self.scheduler = None
        # ADAM �łȂ��ꍇSGD (�m���I���z�~���@�A���[�����^���� STEP SIZE ���w�K�������X�P�W���[�����ݒ�)
        else:
            # ���[�����^���ݒ�@���܂ł̈ړ��Ƃ��ꂩ�瓮���ׂ��ړ��̕��ς��Ƃ�U����h�����߂̊֐�
            self.momentum = cfg["train"]["lr_momentum"]
            # SGD �ɐݒ�
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=self.momentum)
            # �w�K���X�V�^�C�~���O�� EPOCH ��
            self.lr_step_size = cfg["train"]["lr_step_size"]
            # �w�K�����ݒ�@...  Step Size �i�� EPOCH �� gammma ���w�K���ɏ�Z�����
            self.lr_gamma = cfg["train"]["lr_gamma"]
            # �w�K���X�P�W���[��
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)
        # �덷�֐� - MSELoss ���ϓ��덷
        self.criterion = nn.MSELoss()

        # �e�p�����[�^������
        # =====Initialize parameter=====
        # 1EPOCH ... 1���s
        self.epoch = 0
        self.score = 0
        self.max_score = -99999
        self.epoch_reward = 0
        self.cleared_lines = 0
        self.cleared_col = [0, 0, 0, 0, 0]
        self.iter = 0
        # �����X�e�[�g
        self.state = self.initial_state
        # �e�g���~�m0
        self.tetrominoes = 0

        # �O�̃^�[���� Drop ���X�L�b�v���Ă������H (-1: ���Ă��Ȃ�, ����ȊO: ���Ă���)
        # third_y, forth_direction, fifth_x
        self.skip_drop = [-1, -1, -1]

        # �� ������ = �����̉��l���ǂ̒��x�����邩
        self.gamma = cfg["train"]["gamma"]
        # ��V��1 �Ő��K�����邩�ǂ����A������������V�̂݁@
        self.reward_clipping = cfg["train"]["reward_clipping"]

        self.score_list = cfg["tetris"]["score_list"]
        # ��V�ǂݍ���
        self.reward_list = cfg["train"]["reward_list"]  #�������s���ɉ�������V
        # Game Over ��V = Penalty
        self.penalty = self.reward_list[5]  # �Q�[���I�[�o�[���̃y�i���e�B

        ########
        # ��V�� 1 �Ő��K���A������������V�̂�...Q�l�̋}���ȕϓ��}��
        # =====Reward clipping=====
        if self.reward_clipping:
            # ��V���X�g�ƃy�i���e�B(GAMEOVER ��V)���X�g�̐�Βl�̍ő���Ƃ�
            self.norm_num = max(max(self.reward_list), abs(self.penalty))
            # �ő�l�Ŋ������l�����߂ĕ�V���X�g�Ƃ���
            self.reward_list = [r/self.norm_num for r in self.reward_list]
            # �y�i���e�B���X�g�������悤�ɂ���
            self.penalty /= self.norm_num
            # max_penalty �ݒ�� penalty �ݒ�̏���������V���� �y�i���e�B�l�Ƃ���
            self.penalty = min(cfg["train"]["max_penalty"], self.penalty)

        #########
        # =====Double DQN=====
        self.double_dqn = cfg["train"]["double_dqn"]
        self.target_net = cfg["train"]["target_net"]
        if self.double_dqn:
            self.target_net = True

        # Target_net ON �Ȃ��
        if self.target_net:
            print("set target network...")
            # �@�B�w�K���f������
            self.target_model = copy.deepcopy(self.model)
            self.target_copy_intarval = cfg["train"]["target_copy_intarval"]

        ########
        # =====Prioritized Experience Replay=====
        # �D�揇�ʂ��o���w�K�L���Ȃ��
        self.prioritized_replay = cfg["train"]["prioritized_replay"]
        if self.prioritized_replay:
            from machine_learning.qlearning import PRIORITIZED_EXPERIENCE_REPLAY as PER
            # �D�揇�ʂ��o���w�K�ݒ�
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
    # ���Z�b�g���ɃX�R�A�v�Z�� episode memory �� penalty �ǉ�
    # �o���w�K�̂��߂� episode_memory �� replay_memory �ǉ�
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

            # �o���w�K�̂��߂� episode_memory �� replay_memory �ǉ�
            self.replay_memory.extend(self.episode_memory)
            # �e�ʒ�������폜
            self.episode_memory = deque(maxlen=self.max_episode_size)
        else:
            pass

    ####################################
    # Game �� Reset �̎��{ (Game Over��)
    # nextMove["option"]["reset_callback_function_addr"] �֐ݒ�
    ####################################
    def update(self):

        ##############################
        # �w�K�̏ꍇ
        ##############################
        if self.mode == "train" or self.mode == "train_sample" or self.mode == "train_sample2":
            # ���Z�b�g���ɃX�R�A�v�Z�� episode memory �� penalty �ǉ�
            # replay_memory �� episode memory �ǉ�
            self.stack_replay_memory()

            ##############################
            # ���O�\��
            ##############################
            # ���v���C��������1/10���܂��Ă��Ȃ��Ȃ�A
            if len(self.replay_memory) < self.replay_memory_size / 10:
                print("================pass================")
                print("iter: {} ,meory: {}/{} , score: {}, clear line: {}, block: {}, col1-4: {}/{}/{}/{} ".format(self.iter,
                                                                                                                   len(self.replay_memory), self.replay_memory_size / 10, self.score, self.cleared_lines, self.tetrominoes, self.cleared_col[1], self.cleared_col[2], self.cleared_col[3], self.cleared_col[4]))
            # ���v���C�������������ς��Ȃ�
            else:
                print("================update================")
                self.epoch += 1
                # �D�揇�ʂ��o���w�K�L���Ȃ�
                if self.prioritized_replay:
                    # replay batch index �w��
                    batch, replay_batch_index = self.PER.sampling(
                        self.replay_memory, self.batch_size)
                # �����łȂ��Ȃ�
                else:
                    # batch �m���I���z�~���@�ɂ�����A�S�p�����[�^�̂��������_�����o���Č��z�����߂�p�����[�^�̐� batch_size �Ȃ�
                    batch = sample(self.replay_memory, min(
                        len(self.replay_memory), self.batch_size))

                # batch ����e���������o��
                # (episode memory �̕���)
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
                # ���`���� Q �l���擾 (model �� __call__ �� forward)
                ###########################
                # max_next_state_batch = torch.stack(tuple(state for state in max_next_state_batch))
                q_values = self.model(state_batch)

                ###################
                # Traget net �g���ꍇ
                if self.target_net:
                    if self.epoch % self.target_copy_intarval == 0 and self.epoch > 0:
                        print("target_net update...")
                        # self.target_copy_intarval ���Ƃ� best_weight �� target �ɐ؂�ւ�
                        self.target_model = torch.load(self.best_weight)
                        # self.target_model = copy.copy(self.model)
                    # �C���X�^���X�𐄘_���[�h�ɐ؂�ւ�
                    self.target_model.eval()
                    # ======predict Q(S_t+1 max_a Q(s_(t+1),a))======
                    # �e���\���̌��z�̌v�Z��s�Ƃ���
                    with torch.no_grad():
                        # ���̎��̏�� batch ����
                        # �m���I���z�~���@�ɂ����� batch ���� "�^�[�Q�b�g" ���f���ł� q �l�����߂�
                        next_prediction_batch = self.target_model(
                            next_state_batch)
                else:
                    # �C���X�^���X�𐄘_���[�h�ɐ؂�ւ�
                    self.model.eval()
                    # �e���\���̌��z�̌v�Z��s�Ƃ���
                    with torch.no_grad():
                        # �m���I���z�~���@�ɂ����� batch �����`���� Q �l���擾 (model �� __call__ �� forward)
                        next_prediction_batch = self.model(next_state_batch)

                ##########################
                # ���f���̊w�K���{
                ##########################
                self.model.train()

                ##########################
                # Multi Step lerning �̏ꍇ
                if self.multi_step_learning:
                    print("multi step learning update")
                    y_batch = self.MSL.get_y_batch(
                        done_batch, reward_batch, next_prediction_batch)

                # Multi Step lerning �łȂ��ꍇ
                else:
                    # done_batch, reward_bach, next_prediction_batch(Target net �Ȃǔ�r�Ώ� batch)
                    # �����ꂼ��Ƃ肾�� done �� True �Ȃ� reward, False (Gameover �Ȃ� reward + gammma * prediction Q�l)
                    # �� y_batch�Ƃ��� (gamma �͊�����)
                    y_batch = torch.cat(
                        tuple(reward if done[0] else reward + self.gamma * prediction for done, reward, prediction in
                              zip(done_batch, reward_batch, next_prediction_batch)))[:, None]
                # �œK���Ώۂ̂��ׂẴe���\���̌��z�� 0 �ɂ��� (�t�`��backward �O�ɕK�{)
                self.optimizer.zero_grad()
                #########################
                # �w�K���{ - �t�`��
                #########################
                # �D�揇�ʂ��o���w�K�̏ꍇ
                if self.prioritized_replay:
                    # �D��x�̍X�V�Əd�݂Â��擾
                    # ���̏�Ԃ�batch index
                    # ���̏�Ԃ�batch ��V
                    # ���̏�Ԃ�batch �� Q �l
                    # ���̎��̏�Ԃ�batch �� Q �l (Target model �L���̏ꍇ Target model ���Z)
                    loss_weights = self.PER.update_priority(
                        replay_batch_index, reward_batch, q_values, next_prediction_batch)
                    # print(loss_weights *nn.functional.mse_loss(q_values, y_batch))
                    # �덷�֐��Əd�݂Â��v�Z (q_values ������ ���f������, y_batch ����r�Ώ�[Target net])
                    loss = (loss_weights *
                            self.criterion(q_values, y_batch)).mean()
                    # loss = self.criterion(q_values, y_batch)

                    # �t�`��-���z�v�Z
                    loss.backward()
                else:
                    loss = self.criterion(q_values, y_batch)
                    # �t�`��-���z�v�Z
                    loss.backward()
                # weight ���w�K���Ɋ�Â��X�V
                self.optimizer.step()
                # SGD �̏ꍇ
                if self.scheduler != None:
                    # �w�K���X�V
                    self.scheduler.step()

                ###################################
                # ���ʂ̏o��
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

                # TensorBoard �ւ̏o��
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
            # EPOCH �����K�萔�𒴂�����
            if self.epoch > self.num_epochs:
                # ���O�o��
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
                # �I��
                exit()

        ###################################
        # ���_�̏ꍇ
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
        # �Q�[���p�����[�^������
        self.reset_state()

    ####################################
    # �ݐϒl�̏����� (Game Over ��)
    ####################################

    def reset_state(self):
        # �w�K�̏ꍇ
        if self.mode == "train" or self.mode == "train_sample" or self.mode == "train_sample2":
            # �ō��_,500 epoch �����ɕۑ�
            if self.score > self.max_score or self.epoch % 500 == 0:
                torch.save(
                    self.model, "{}/tetris_epoch{}_score{}.pt".format(self.weight_dir, self.epoch, self.score))
                self.max_score = self.score
                torch.save(self.model, self.best_weight)
        # �������X�e�[�g
        self.state = self.initial_state
        self.score = 0
        self.cleared_lines = 0
        self.cleared_col = [0, 0, 0, 0, 0]
        self.epoch_reward = 0
        # �e�g���~�m 0 ��
        self.tetrominoes = 0
        # �O�̃^�[���� Drop ���X�L�b�v���Ă������H (-1: ���Ă��Ȃ�, ����ȊO: ���Ă���)
        # third_y, forth_direction, fifth_x
        self.skip_drop = [-1, -1, -1]

    ####################################
    # �폜�����Line�𐔂���
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
    # �ł��ڂ��x, �������v, �����ő�, �����ŏ������߂�
    ####################################
    def get_bumpiness_and_height(self, reshape_board):
        # �{�[�h��� 0 �łȂ�����(�e�g���~�m�̂���Ƃ���)�𒊏o
        # (0,1,2,3,4,5,6,7) �� �u���b�N���� True, �Ȃ� False �ɕύX
        mask = reshape_board != 0
        # pprint.pprint(mask, width = 61, compact = True)

        # ����� �����u���b�N��������΁A����index��Ԃ�
        # �Ȃ���Ή�ʃ{�[�h�c�T�C�Y��Ԃ�
        # ��L�� ��ʃ{�[�h�̗�ɑ΂��Ď��{�����̔z��(���� width)��Ԃ�
        invert_heights = np.where(
            mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        # �ォ��̋����Ȃ̂Ŕ��] (�z��)
        heights = self.height - invert_heights
        # �����̍��v���Ƃ� (�Ԃ�l�p)
        total_height = np.sum(heights)
        # �ł������Ƃ�����Ƃ� (�Ԃ�l�p)
        max_height = np.max(heights)
        # �ł��Ⴂ�Ƃ�����Ƃ� (�Ԃ�l�p)
        min_height = np.min(heights)
        min_height_l = np.min(heights[1:])    #�� ���[�ȊO�ōŏ�����

        # �E�[�������� �����z��
        # currs = heights[:-1]
        currs = heights[1:-1]

        # ���[��2������������z��
        # nexts = heights[1:]
        nexts = heights[2:]

        # �����̐�Βl���Ƃ�z��ɂ���
        diffs = np.abs(currs - nexts)
        # ���[��� self.bumpiness_left_side_relax �i���܂ŋ��e
        if heights[1] - heights[0] > self.bumpiness_left_side_relax or heights[1] - heights[0] < 0:
            diffs = np.append(abs(heights[1] - heights[0]), diffs)

        # �����̐�Βl�����v���Ăł��ڂ��x�Ƃ���
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height, max_height, min_height, heights[0], min_height_l

    ####################################
    # ���̐�, ���̏�ςݏグ Penalty, �ł��������̈ʒu�����߂�
    # reshape_board: 2������ʃ{�[�h
    # min_height: ���B�\�̍ŉ��w���1�s���̌��̈ʒu���`�F�b�N -1 �Ŗ��� hole_top_penalty ����
    ####################################
    def get_holes(self, reshape_board, min_height):
        # ���̐�
        num_holes = 0
        # ���̏�̐ςݏグ�y�i���e�B
        hole_top_penalty = 0
        # �n�ʂ̍��� list
        highest_grounds = [-1] * self.width
        # �ł��������� list
        highest_holes = [-1] * self.width
        # �񂲂Ƃɐ؂�o��
        for i in range(self.width):
            # ��擾
            col = reshape_board[:, i]
            # print(col)
            ground_level = 0
            # ��̍s���� 0(�u���b�N�Ȃ�) ���݂��Ă���, ground_level �����̗�̍ŏ�w
            while ground_level < self.height and col[ground_level] == 0:
                ground_level += 1
            # ���̍s�ȍ~�̌���list �����
            cols_holes = []
            for y, state in enumerate(col[ground_level + 1:]):
                # ���̂���ꏊ��list�ɂ���, list�l�Ƃ��Č��̈ʒu�������
                if state == 0:
                    # num_holes += 1
                    cols_holes.append(self.height - (ground_level + 1 + y) - 1)
            # �� 1 liner �����̃J�E���g
            # cols_holes = [x for x in col[ground_level + 1:] if x == 0]
            # list ���J�E���g���Č��̐����J�E���g
            num_holes += len(cols_holes)

            # �n�ʂ̍����z��
            highest_grounds[i] = self.height - ground_level - 1

            # �ł��������̈ʒu�z��
            if len(cols_holes) > 0:
                highest_holes[i] = cols_holes[0]
            else:
                highest_holes[i] = -1

        # �ł������������߂�
        max_highest_hole = max(highest_holes)

        # ���B�\�̍ŉ��w���1�s���̌��̈ʒu���`�F�b�N
        if min_height > 0:
            # �ł������Ƃ���ɂ��錊�̐�
            highest_hole_num = 0
            # �񂲂Ƃɐ؂�o��
            for i in range(self.width):
                # �ł������ʒu�̌��̗�̏ꍇ
                if highest_holes[i] == max_highest_hole:
                    highest_hole_num += 1
                    # ���̐�Έʒu��hole_top_limit_height��荂��
                    # ���̏�̒n�ʂ������Ȃ� Penalty
                    if highest_holes[i] > self.hole_top_limit_height and \
                            highest_grounds[i] >= highest_holes[i] + self.hole_top_limit:
                        hole_top_penalty += highest_grounds[i] - \
                            (highest_holes[i])
            # �ł������ʒu�ɂ��錊�̐��Ŋ���E�E�E�����̐�����Ȃ��č����Ƃ��낪�J���Ă����̐����B
            hole_top_penalty /= highest_hole_num
            # debug
            # print(['{:02}'.format(n) for n in highest_grounds])
            # print(['{:02}'.format(n) for n in highest_holes])
            # print(hole_top_penalty, hole_top_penalty*max_highest_hole)
            # print("==")

        return num_holes, hole_top_penalty, max_highest_hole

    ####################################
    # �����Ԃ̊e��p�����[�^�擾 (MLP
    ####################################
    def get_state_properties(self, reshape_board):
        # �폜���ꂽ�s�̕�V
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        # ���̐�
        holes, _, _ = self.get_holes(reshape_board, -1)
        # �ł��ڂ��̐�
        bumpiness, height, max_height, min_height, _, _ = self.get_bumpiness_and_height(reshape_board)

        return torch.FloatTensor([lines_cleared, holes, bumpiness, height])

    ####################################
    # �����Ԃ̊e��p�����[�^�擾�@�����t�� ���͎g���Ă��Ȃ�
    ####################################
    def get_state_properties_v2(self, reshape_board):
        # �폜���ꂽ�s�̕�V
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        # ���̐�
        holes, _, _ = self.get_holes(reshape_board, -1)
        # �ł��ڂ��̐�
        bumpiness, height, max_row, min_height, _, _ = self.get_bumpiness_and_height(
            reshape_board)
        # �ő卂��
        # max_row = self.get_max_height(reshape_board)
        return torch.FloatTensor([lines_cleared, holes, bumpiness, height, max_row])

    ####################################
    # �ő�̍������擾
    # get_bumpiness_and_height �ɂƂ肱�܂ꂽ�̂Ŕp�~
    ####################################
    def get_max_height(self, reshape_board):
        # X ���̃Z���𑫂��Z����
        sum_ = np.sum(reshape_board, axis=1)
        # print(sum_)
        row = 0
        # X ���̍��v��0�ɂȂ� Y ����T��
        while row < self.height and sum_[row] == 0:
            row += 1
        return self.height - row

    ####################################
    # ���[�ȊO���܂��Ă��邩�H
    ####################################
    def get_tetris_fill_reward(self, reshape_board, piece_id):
        # �����̏ꍇ
        if self.tetris_fill_height == 0:
            return 0

        # ��V
        reward = 0
        lines = 0   #�A�����Ė��܂��Ă��郉�C����
        max_reward = self.tetris_fill_height
        # �{�[�h��� 0 �łȂ�����(�e�g���~�m�̂���Ƃ���)�𒊏o
        # (0,1,2,3,4,5,6,7) �� �u���b�N���� True, �Ȃ� False �ɕύX
        mask = reshape_board != 0
        # X ���̃Z���𑫂��Z����
        sum_ = np.sum(mask, axis=1)
        # �z��mask�ŁA��1�̕����ő����Z����B���̍s�Ŗ��܂��Ă��鐔���v�Z
        # print(sum_)

        # line (1 - self.tetris_fill_height)�i�ڂ̍��[�ȊO������Ă��邩
        for i in range(1, self.tetris_fill_height):
            # ������Ă���i���Ƃɕ�V�B�|���Z�ŒႢ�s�ł������񂻂���Ă���̂��V�����B
            if self.get_line_right_fill(reshape_board, sum_, i):
                reward += 1
#                # 1�i�ڂ�2�{    �o�O���Ă��B������ĂȂ��Ă�reward +5�ɂȂ��Ă��B
#                if i == 1:
#                    reward += 5

#        # �z�[���h���Ă���e�g���~�m��I�ł���΁A��V��1�s�ڂƓ���������������
#        if piece_id == 1:  # I
#            reward += 5

        return reward

    ####################################
    # line �i�ڂ����[�ȊO������Ă��邩
    ####################################
    def get_line_right_fill(self, reshape_board, sum_, line):
        # 1�i�ڂ��[�ȊO������Ă���
        if sum_[self.height - line] == self.width - 1 \
                and reshape_board[self.height - line][0] == 0:
            # or reshape_board[self.height-1][self.width-1] == 0 ):
            # print("line:", line)
            return True
        else:
            return False

    ####################################
    # ���̏�ԃ��X�g���擾(2�����p) DQN .... ��ʃ{�[�h�� �e�g���~�m��]��� �ɗ����������Ƃ��̎��̏�Ԉꗗ���쐬
    #  get_next_func �ł�т������
    # curr_backboard �����
    # piece_id �e�g���~�m I L J T O S Z
    # currentshape_class = status["field_info"]["backboard"]
    ####################################
    def get_next_states_v2(self, curr_backboard, piece_id, CurrentShape_class):
        # ���̏�Ԉꗗ
        states = {}

        # �e�g���~�m��]�������Ƃ̔z�u��
        x_range_min = [0] * 4
        x_range_max = [self.width] * 4

        # �ݒu�������X�g drop_y_list[(direction,x)] = height
        drop_y_list = {}
        # ���؍σ��X�g checked_board[(direction0, x0, drop_y)] =True
        checked_board = {}

        # �e�g���~�m���Ƃɉ�]�����ӂ�킯
        if piece_id == 5:  # O piece => 1
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:  # I, S, Z piece => 2
            num_rotations = 2
        else:  # the others => 4
            num_rotations = 4

        ####################
        # Drop Down ���� �̏ꍇ�̈ꗗ�쐬
        # �e�g���~�m��]�������ƂɈꗗ�ǉ�
        for direction0 in range(num_rotations):
            # �e�g���~�m���z�u�ł��鍶�[�ƉE�[�̍��W��Ԃ�
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            (x_range_min[direction0], x_range_max[direction0]) = (x0Min, x0Max)

            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                # ��ʃ{�[�h�f�[�^���R�s�[���Ďw����W�Ƀe�g���~�m��z�u��������������ʃ{�[�h��y���W��Ԃ�
                board, drop_y = self.getBoard(
                    curr_backboard, CurrentShape_class, direction0, x0, -1)
                # ��̂��ߕۑ�
                drop_y_list[(direction0, x0)] = drop_y
                checked_board[(direction0, x0, drop_y)] = True

                # �{�[�h���Q������
                reshape_backboard = self.get_reshape_backboard(board)
                # numpy to tensor (�z���1�����ǉ�)
                reshape_backboard = torch.from_numpy(
                    reshape_backboard[np.newaxis, :, :]).float()
                # ��ʃ{�[�hx0�� �e�g���~�m��]��� direction0 �ɗ����������Ƃ��̎��̏�Ԃ��쐬 �ǉ�
                #  states
                #    Key = Tuple (�e�g���~�m Drop Down ���� ����ʃ{�[�hX���W, �e�g���~�m��]���
                #                 �e�g���~�m Move Down �~�� ��, �e�g���~�m�ǉ��ړ�X���W, �e�g���~�m�ǉ���])
                #                 ... -1 �̏ꍇ ����ΏۊO
                #    Value = ��ʃ{�[�h���
                # (action �p)
                states[(x0, direction0, -1, -1, -1)] = reshape_backboard

        # print(len(states), end='=>')

        # Move Down �~�������̏ꍇ
        if self.move_down_flag == 0:
            return states

        ####################
        # Move Down �~�� �̏ꍇ�̈ꗗ�쐬
        # �ǉ��␳�ړ�
        third_y = -1
        forth_direction = -1
        fifth_x = -1
        sixth_y = -1

        # �{�[�h���Q������
        reshape_curr_backboard = self.get_reshape_backboard(curr_backboard)

        # �{�[�h��� 0 �łȂ�����(�e�g���~�m�̂���Ƃ���)�𒊏o
        # (0,1,2,3,4,5,6,7) �� �u���b�N���� True, �Ȃ� False �ɕύX
        mask_board = reshape_curr_backboard != 0
        # pprint.pprint(mask_board, width = 61, compact = True)

        # ����� �����u���b�N��������΁A����index��Ԃ�
        # �Ȃ���Ή�ʃ{�[�h�c�T�C�Y��Ԃ�
        # ��L�� ��ʃ{�[�h�̗�ɑ΂��Ď��{�����̔z��(���� width)��Ԃ�
        invert_heights = np.where(mask_board.any(
            axis=0), np.argmax(mask_board, axis=0), self.height)
        # �ォ��̋����Ȃ̂Ŕ��] (�z��)
        heights = self.height - invert_heights
        # �ő卂��
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

        # 1 ��ڂ� ��]
        for first_direction in range(num_rotations):
            if self.debug_flag_shift_rotation == 1:
                print(" 1d", first_direction, "/ second_x:",
                      x_range_min[first_direction], " to ", x_range_max[first_direction])
            # 2 ��ڂ� x ���ړ�
            for second_x in range(x_range_min[first_direction], x_range_max[first_direction]):
                # �������ő�̍���-1���傫���ꍇ�����݂��Ȃ��̂Ŏ���
                if drop_y_list[(first_direction, second_x)] < invert_max_height + 1:
                    continue
                # ������ ��ʍő�-2���傫���ꍇ�������݂��Ȃ��̂Ŏ���
                if invert_heights[second_x] < 2:
                    continue
                # y ���W�̉����� �u���b�N�ő�̍���-1 �Ō���
                if self.debug_flag_shift_rotation == 1:
                    print("   2x", second_x, "/ third_y: ", invert_max_height,
                          " to ", drop_y_list[(first_direction, second_x)]+1)

                # 3 ��ڂ� y ���~��
                for third_y in range(invert_max_height, drop_y_list[(first_direction, second_x)]+1):
                    # y ���W�̉����� �u���b�N�ő�̍���-1 �Ō���
                    if self.debug_flag_shift_rotation == 1:
                        print("    3y", third_y, "/ forth_direction: ")

                    # �E��]�Œ�Ȃ̂ŏ�����ς���
                    direction_order = [0] * num_rotations
                    # �ŏ��� first_direction
                    new_direction_order = first_direction
                    #
                    for order_num in range(num_rotations):
                        direction_order[order_num] = new_direction_order
                        new_direction_order += 1
                        if not (new_direction_order < num_rotations):
                            new_direction_order = 0

                    # print(first_direction,"::", direction_order)

                    # 4 ��ڂ� ��] (Turn 2)
                    # first_direction ����E��]���Ă���
                    for forth_direction in direction_order:
                        # y ���W�̉����� �u���b�N�ő�̍���-1 �Ō���
                        if self.debug_flag_shift_rotation == 1:
                            print("     4d", forth_direction, "/ fifth_x: ", 0,
                                  " to ", x_range_max[forth_direction] - second_x, end='')
                            print("//")
                            print("       R:", end='')
                        # 0 ����T��
                        start_point_x = 0
                        # �ŏ��Ɠ�����]��ԂȂ炸�炵���Ƃ��납��T��
                        if first_direction == forth_direction:
                            start_point_x = 1

                        # �E��]����t���O
                        right_rotate = True

                        # 5 ��ڂ� x ���ړ� (Turn 2)
                        # shift_x ���E�ɂ��炵�Ċm�F
                        for shift_x in range(start_point_x, x_range_max[forth_direction] - second_x):
                            fifth_x = second_x + shift_x
                            # ���炵����̕��������[���ꍇ�͒T�����~
                            if not ((forth_direction, fifth_x) in drop_y_list):
                                if self.debug_flag_shift_rotation == 1:
                                    print(shift_x, ": False(OutRange) ", end='/ ')
                                break
                            if third_y <= drop_y_list[(forth_direction, second_x + shift_x)]:
                                if self.debug_flag_shift_rotation == 1:
                                    print(shift_x, ": False(drop) ", end='/ ')
                                break
                            # direction (��]���)�̃e�g���~�m2�������W�z����擾���A�����x,y�ɔz�u�����ꍇ�̍��W�z���Ԃ�
                            coordArray = self.getShapeCoordArray(
                                CurrentShape_class, forth_direction, fifth_x, third_y)
                            # x���W�����Ƀe�g���~�m���������邩�m�F����
                            judge = self.try_move_(curr_backboard, coordArray)
                            if self.debug_flag_shift_rotation == 1:
                                print(shift_x, ":", judge, end='/')
                            # �E�ړ��\
                            if judge:
                                ####
                                # �o�^�ς��m�F���ASTATES �֓����
                                states, checked_board = \
                                    self.second_drop_down(curr_backboard, CurrentShape_class,
                                                          first_direction, second_x, third_y, forth_direction, fifth_x,
                                                          states, checked_board)
                            # �E�ړ��s�Ȃ�I��
                            else:
                                # �ŏ��̈ʒu�ŉE��]�����܂��s���Ȃ��ꍇ�͉�]��Ƃ�������
                                if shift_x == 0 and judge == False:
                                    right_rotate = False
                                break

                        # �ŏ��̈ʒu�ŉE��]�����܂��s���Ȃ��ꍇ�͔�����
                        if right_rotate == False:
                            if self.debug_flag_shift_rotation_success == 1:
                                print(" |||", CurrentShape_class.shape, "-", forth_direction,
                                      "(", second_x, ",", third_y, ")|||<=RotateStop|||")
                            break

                        # �E���炵�I��
                        if self.debug_flag_shift_rotation == 1:
                            print("//")
                            print("       L:", end='')

                        ########
                        # shift_x �����ɂ��炵�Ċm�F
                        for shift_x in range(-1, -second_x - 1, -1):
                            fifth_x = second_x + shift_x
                            # ���炵����̕��������[���ꍇ�͒T�����~
                            if not ((forth_direction, fifth_x) in drop_y_list):
                                break
                            if third_y <= drop_y_list[(forth_direction, fifth_x)]:
                                break
                            # direction (��]���)�̃e�g���~�m2�������W�z����擾���A�����x,y�ɔz�u�����ꍇ�̍��W�z���Ԃ�
                            coordArray = self.getShapeCoordArray(
                                CurrentShape_class, forth_direction, fifth_x, third_y)
                            # x���W�����Ƀe�g���~�m���������邩�m�F����
                            judge = self.try_move_(curr_backboard, coordArray)

                            if self.debug_flag_shift_rotation == 1:
                                print(shift_x, ":", judge, end='/ ')

                            # ���ړ��\
                            if judge:
                                ####
                                # �o�^�ς��m�F���ASTATES �֓����
                                states, checked_board = \
                                    self.second_drop_down(curr_backboard, CurrentShape_class,
                                                          first_direction, second_x, third_y, forth_direction, fifth_x,
                                                          states, checked_board)

                            # ���ړ��s�Ȃ�I��
                            else:
                                break
                        # �����炵�I��
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
        # states (action) ��Ԃ�
        return states

    ####################################
    # ���̏�Ԃ��擾(1�����p) MLP  .... ��ʃ{�[�h�� �e�g���~�m��]��� �ɗ����������Ƃ��̎��̏�Ԉꗗ���쐬
    #  get_next_func �ł�т������
    ####################################

    def get_next_states(self, curr_backboard, piece_id, CurrentShape_class):
        # ���̏�Ԉꗗ
        states = {}

        if piece_id == 5:  # O piece
            num_rotations = 1
        elif piece_id == 1 or piece_id == 6 or piece_id == 7:
            num_rotations = 2
        else:
            num_rotations = 4

        ####################
        # Drop Down ���� �̏ꍇ�̈ꗗ�쐬
        # �e�g���~�m��]�������ƂɈꗗ�ǉ�
        for direction0 in range(num_rotations):
            # �e�g���~�m���z�u�ł��鍶�[�ƉE�[�̍��W��Ԃ�
            x0Min, x0Max = self.getSearchXRange(CurrentShape_class, direction0)
            # �e�g���~�m���[����E�[�܂Ŕz�u
            for x0 in range(x0Min, x0Max):
                # get board data, as if dropdown block
                # ��ʃ{�[�h�f�[�^���R�s�[���Ďw����W�Ƀe�g���~�m��z�u��������������ʃ{�[�h��y���W��Ԃ�
                board, drop_y = self.getBoard(
                    curr_backboard, CurrentShape_class, direction0, x0, -1)
                # �{�[�h���Q������
                reshape_board = self.get_reshape_backboard(board)
                # ��ʃ{�[�hx0�� �e�g���~�m��]��� direction0 �ɗ����������Ƃ��̎��̏�Ԃ��쐬
                #  states
                #    Key = Tuple (�e�g���~�m Drop Down ���� ����ʃ{�[�hX���W, �e�g���~�m��]���
                #                 �e�g���~�m Move Down �~�� ��, �e�g���~�m�ǉ��ړ�X���W, �e�g���~�m�ǉ���])
                #    Value = ��ʃ{�[�h���
                states[(x0, direction0, 0, 0, 0)
                       ] = self.get_state_properties(reshape_board)

        return states

    ####################################
    # �e�g���~�m�z�u���� states �ɓo�^���� (���炵���p)
    ####################################
    def second_drop_down(self, curr_backboard, CurrentShape_class,
                         first_direction, second_x, third_y, forth_direction, fifth_x, states, checked_board):
        # debug
        # self.debug_flag_drop_down = 1

        # ��ʃ{�[�h�f�[�^���R�s�[���Ďw����W�Ƀe�g���~�m��z�u��������������ʃ{�[�h��y���W��Ԃ�
        new_board, drop_y = self.getBoard(
            curr_backboard, CurrentShape_class, forth_direction, fifth_x, third_y)
        # �����w��
        sixth_y = third_y + drop_y

        # debug
        if self.debug_flag_shift_rotation_success == 1:
            print(" ***", CurrentShape_class.shape, "-", forth_direction,
                  "(", fifth_x, ",", third_y, "=>", sixth_y, ")***", end='')

        # �o�^�ςłȂ����m�F
        if not ((forth_direction, fifth_x, sixth_y) in checked_board):
            # debug
            if self.debug_flag_shift_rotation_success == 1:
                print("<=NEW***", end='')
            # �~���㓮��Ƃ��ēo�^ (�d���h�~)
            checked_board[(forth_direction, fifth_x, sixth_y)] = True
            # �{�[�h���Q������
            reshape_backboard = self.get_reshape_backboard(new_board)
            # numpy to tensor (�z���1�����ǉ�)
            reshape_backboard = torch.from_numpy(
                reshape_backboard[np.newaxis, :, :]).float()
            ####################
            # ��ʃ{�[�hx0�� �e�g���~�m�ړ����������̂̏�Ԃ��쐬 �ǉ�
            #  states
            #    Key = Tuple (�e�g���~�m Drop Down ���� ����ʃ{�[�hX���W, �e�g���~�m��]���
            #                 �e�g���~�m Move Down �~�� ��, �e�g���~�m�ǉ��ړ�X���W, �e�g���~�m�ǉ���])
            #                 ... -1 �̏ꍇ ����ΏۊO
            #    Value = ��ʃ{�[�h���
            # (action �p)
            states[(second_x, first_direction, third_y,
                    forth_direction, fifth_x)] = reshape_backboard

        # debug
        if self.debug_flag_shift_rotation_success == 1:
            print("")

        return states, checked_board

    ####################################
    # �z�u�ł��邩�m�F����
    # board: 1�������W
    # coordArray: �e�g���~�m2�������W
    ####################################
    def try_move_(self, board, coordArray):
        # �e�g���~�m���W�z��(�e�}�X)���Ƃ�
        judge = True

        debug_board = [0] * self.width * self.height
        debug_log = ""

        for coord_x, coord_y in coordArray:
            debug_log = debug_log + \
                "==(" + str(coord_x) + "," + str(coord_y) + ") "

            # �e�g���~�m���Wcoord_y �� ��ʉ�������@���@(�e�g���~�m���Wcoord_y����ʏ������
            # �e�g���~�m���Wcoord_x, �e�g���~�m���Wcoord_y�̃u���b�N���Ȃ�)
            if 0 <= coord_x and \
                coord_x < self.width and \
                coord_y < self.height and \
                (coord_y * self.width + coord_x < 0 or
                    board[coord_y * self.width + coord_x] == 0):

                # �͂܂�
                debug_board[coord_y * self.width + coord_x] = 1

            # �͂܂�Ȃ��̂� False
            else:
                judge = False
                # �͂܂�Ȃ�
                # self.debug_flag_try_move = 1
                if 0 <= coord_x and coord_x < self.width \
                   and 0 <= coord_y and coord_y < self.height:
                    debug_board[coord_y * self.width + coord_x] = 8

        # Debug �p
        if self.debug_flag_try_move == 1:
            print(debug_log)
            pprint.pprint(board, width=31, compact=True)
            pprint.pprint(debug_board, width=31, compact=True)
            self.debug_flag_try_move = 0
        return judge

    ####################################
    # �{�[�h���Q������
    ####################################
    def get_reshape_backboard(self, board):
        board = np.array(board)
        # ����, ���� reshape
        reshape_board = board.reshape(self.height, self.width)
        # 1, 0 �ɕύX
        reshape_board = np.where(reshape_board > 0, 1, 0)
        return reshape_board

    ####################################
    # ��V���v�Z(2�����p)
    # reward_func ����Ăяo�����
    # �z�[���h���̃e�g���~�m�`�����V�v�Z�Ɏg�p����B
    ####################################
    def step_v2(self, curr_backboard, action, curr_shape_class, hold_shape_id):
        # ���� action �� index �����Ɍ���
        # 0: 2�Ԗ� X���ړ�
        # 1: 1�Ԗ� �e�g���~�m��]
        # 2: 3�Ԗ� Y���~�� (-1: �� Drop)
        # 3: 4�Ԗ� �e�g���~�m��] (Next Turn)
        # 4: 5�Ԗ� X���ړ� (Next Turn)
        x0, direction0, third_y, forth_direction, fifth_x = action
        # ��ʃ{�[�h�f�[�^���R�s�[���Ďw����W�Ƀe�g���~�m��z�u��������������ʃ{�[�h��y���W��Ԃ�
        board, drop_y = self.getBoard(curr_backboard, curr_shape_class, direction0, x0, -1)
        # �{�[�h���Q������
        reshape_board = self.get_reshape_backboard(board)
        # ��V�v�Z���̒l�擾
        # �ł��ڂ��x, �������v, �����ő�, �����ŏ������߂�
        bampiness, total_height, max_height, min_height, left_side_height, min_height_l = self.get_bumpiness_and_height(reshape_board)
        # max_height = self.get_max_height(reshape_board)
        # ���̐�, ���̏�ςݏグ Penalty, �ł��������̈ʒu�����߂�
        hole_num, hole_top_penalty, max_highest_hole = self.get_holes(reshape_board, min_height)
        # ���[�������`��̕�V�v�Z
        tetris_reward = self.get_tetris_fill_reward(reshape_board, hold_shape_id)
        # ������Z���̊m�F
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        
        ## �z�[���h���Ă���e�g���~�m�v��̕�V�v�Z�@�Ǝv�������A�ێ����Ă�����̂�I�^�̎��̍��[�J�����ꍇ�̕�V���グ��悤�ɂ���B
        
        # ��V�̌v�Z
        reward = self.reward_list[lines_cleared] * (1 + (self.height - max(0, max_height))/self.height_line_reward)

        # I�^�e�g���~�m���N���A�s��3�����̏ꍇ�Ƀz�[���h���Ă���΁A��V�{
        if hold_shape_id == 1 & lines_cleared < 3:
            reward += 0.001

#       ���[���������Œፂ����4�ȉ��̏ꍇ�͒Ⴂ�������y�i���e�B
#        reward += min(0, (min_height_l - 4))/self.height_line_reward
#       ���[���������Œፂ����4�ȉ��̏ꍇ�͕�V�𔼕��ɂ���
#        if min_height_l < 4:
#            reward /= 2
        # �p����V
        # reward += 0.01
        # �`��̔���V
        # �ł��ڂ��x��
        reward -= self.reward_weight[0] * bampiness
        # �ő卂�����@->�@�ő卂���𒴂��������ɔ�Ⴗ��悤�ɕύX
        if max_height > self.max_height_relax:
            reward -= self.reward_weight[1] * max(0, max_height-self.max_height_relax)
        # ���̐���
        reward -= self.reward_weight[2] * hole_num
        # ���̏�̃u���b�N����
        reward -= self.hole_top_limit_reward * hole_top_penalty * max_highest_hole
        # ���[�ȊO���߂Ă����ԕ�V
        reward += tetris_reward * self.tetris_fill_reward
        # ���[����������ꍇ�̔�
        if left_side_height > self.bumpiness_left_side_relax:
            reward -= (left_side_height - self.bumpiness_left_side_relax) * self.left_side_height_penalty

        self.epoch_reward += reward

        # �X�R�A�v�Z
        self.score += self.score_list[lines_cleared]
        # �������C���J�E���g
        self.cleared_lines += lines_cleared
        self.cleared_col[lines_cleared] += 1
        # �e�g���~�m���J�E���g���₷
        self.tetrominoes += 1
        return reward

    ####################################
    # ��V���v�Z(1�����p)
    # reward_func ����Ăяo�����
    ####################################
    def step(self, curr_backboard, action, curr_shape_class):
        x0, direction0, third_y, forth_direction, fifth_x = action
        # ��ʃ{�[�h�f�[�^���R�s�[���Ďw����W�Ƀe�g���~�m��z�u��������������ʃ{�[�h��y���W��Ԃ�
        board, drop_y = self.getBoard(curr_backboard, curr_shape_class, direction0, x0, -1)
        # �{�[�h���Q������
        reshape_board = self.get_reshape_backboard(board)
        # ��V�v�Z���̒l�擾
        bampiness, height, max_height, min_height, _, _ = self.get_bumpiness_and_height(reshape_board)
        # max_height = self.get_max_height(reshape_board)
        hole_num, _, _ = self.get_holes(reshape_board, min_height)
        lines_cleared, reshape_board = self.check_cleared_rows(reshape_board)
        # ��V�̌v�Z
        reward = self.reward_list[lines_cleared]
        # �p����V
        # reward += 0.01
        # ��
        reward -= self.reward_weight[0] * bampiness
        if max_height > self.max_height_relax:
            reward -= self.reward_weight[1] * max(0, max_height)
        reward -= self.reward_weight[2] * hole_num
        self.epoch_reward += reward

        # �X�R�A�v�Z
        self.score += self.score_list[lines_cleared]

        # ���������ǉ�
        self.cleared_lines += lines_cleared
        self.cleared_col[lines_cleared] += 1
        self.tetrominoes += 1
        return reward

    ####################################
    ####################################
    ####################################
    ####################################
    # ���̓���擾: �Q�[���R���g���[�����疈��Ă΂��
    ####################################
    ####################################
    ####################################
    ####################################
    def GetNextMove(self, nextMove, GameStatus, yaml_file=None, weight=None):

        t1 = datetime.now()
        # RESET �֐��ݒ� callback function ��� (Game Over ��)
        nextMove["option"]["reset_callback_function_addr"] = self.update
        # mode �̎擾 (train �ł���)
        self.mode = GameStatus["judge_info"]["mode"]

        ################
        # �����p�����[�^�Ȃ��ꍇ�͏����p�����[�^�ǂݍ���
        if self.init_train_parameter_flag == False:
            self.init_train_parameter_flag = True
            self.set_parameter(yaml_file=yaml_file, predict_weight=weight)

        self.ind = GameStatus["block_info"]["currentShape"]["index"]
        curr_backboard = GameStatus["field_info"]["backboard"]

        ##################
        # default board definition
        # self.width, self.height �Əd��
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
        # Move Down �� �O��̎����z�����삪����ꍇ�@���̓�������ďI��
        if self.skip_drop != [-1, -1, -1]:
            # third_y, forth_direction, fifth_x
            nextMove["strategy"]["direction"] = self.skip_drop[1]
            # ������
            nextMove["strategy"]["x"] = self.skip_drop[2]
            # Move Down �~��
            nextMove["strategy"]["y_operation"] = 1
            # Move Down �~�� ��
            nextMove["strategy"]["y_moveblocknum"] = 1
            # �O�̃^�[���� Drop ���X�L�b�v���Ă������H������ (-1: ���Ă��Ȃ�, ����ȊO: ���Ă���)
            self.skip_drop = [-1, -1, -1]
            # �I������
            if self.time_disp:
                print(datetime.now()-t1)
            # �I��
            return nextMove

        ###################
        # ��ʃ{�[�h�� �e�g���~�m��]��� �ɗ����������Ƃ��̎��̏�Ԉꗗ���쐬
        # next_steps
        #    Key = Tuple (�e�g���~�m��ʃ{�[�hX���W, �e�g���~�m��]���)
        #                 �e�g���~�m Move Down �~�� ��, �e�g���~�m�ǉ��ړ�X���W, �e�g���~�m�ǉ���])
        #    Value = ��ʃ{�[�h���
        next_steps = self.get_next_func(curr_backboard, curr_piece_id, curr_shape_class)

        # hold���g�����ꍇ�̃p�^�[��
        if hold_piece_id == None:
            # ���߂Ă�hold�̏ꍇ
            hold_steps = self.get_next_func(curr_backboard, next_piece_id, next_shape_class)
        else:
            # 2��ڈȍ~�̏ꍇ
            # print(hold_piece_id)
            hold_steps = self.get_next_func(curr_backboard, hold_piece_id, hold_shape_class)
        # print (len(next_steps), end='=>')

        ###############################################
        ###############################################
        # �w�K�̏ꍇ
        ###############################################
        ###############################################
        if self.mode == "train" or self.mode == "train_sample" or self.mode == "train_sample2":
            # init parameter
            # epsilon = �w�K���ʂ��痐���ŕύX���銄���Ώ�
            # num_decay_epochs ���O�܂ł͔��ŏ��� epsilon ���猸�炵�Ă���
            # num_decay_ecpchs �ȍ~�� final_epsilon�Œ�
            epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
                self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
            u = random()
            # epsilon ��藐�� u ���������ꍇ�t���O�����Ă�
            random_action = u <= epsilon

            # ���̃e�g���~�m�\��
            if self.predict_next_num_train > 0:
                ##########################
                # ���f���̊w�K���{
                ##########################
                self.model.train()
                # index_list [1�Ԗ�index, 2�Ԗ�index, 3�Ԗ�index ...] => q
                index_list = []
                hold_index_list = []
                # index_list_to_q (1�Ԗ�index, 2�Ԗ�index, 3�Ԗ�index ...) => q
                index_list_to_q = {}
                hold_index_list_to_q = {}
                ######################
                # ���̗\�������predict_next_steps_train���{, 1�Ԗڂ���predict_next_num_train�Ԗڂ܂ŗ\��
                index_list, index_list_to_q, next_actions, next_states \
                    = self.get_predictions(self.model, True, GameStatus, next_steps, self.predict_next_steps_train, 1, self.predict_next_num_train, index_list, index_list_to_q, -60000)
                # print(index_list_to_q)
                # print("max")
                hold_index_list, hold_index_list_to_q, hold_next_actions, hold_next_states \
                    = self.get_predictions(self.model, True, GameStatus, hold_steps, self.predict_next_steps_train, 1, self.predict_next_num_train, hold_index_list, hold_index_list_to_q, -60000)
#�����hold�����̃����o�͖��Ȃ��̂��낤����
                # �S�\���̍ő� q
                max_index_list = max(index_list_to_q, key=index_list_to_q.get)
                hold_max_index_list = max(hold_index_list_to_q, key=hold_index_list_to_q.get)

                # �z�[���h�����ق���Q�l���ǂ������Ƃ��͕ϐ����z�[���h�̏ꍇ�ɑ�����ւ�����
                # ��x�ڂ̃z�[���h�̏ꍇ��action�͖������ꎟ�̃^�[���ƂȂ�A��x�ڈȍ~�̃z�[���h�̏ꍇ�̓z�[���h����Ă����~�m�̓����action�Ɋi�[����
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
                # ������ epsilon ��菬�����ꍇ��
                if random_action:
                    # index �𗐐��Ƃ���
                    index = randint(0, len(next_steps) - 1)
                else:
                    # 1��ڂ� index ����
                    index = max_index_list[0].item()
            else:
                # ���̏�Ԉꗗ�� action �� states �Ŕz��
                #    next_actions  = Tuple (�e�g���~�m��ʃ{�[�hX���W, �e�g���~�m��]���)�@�ꗗ
                #    next_states = ��ʃ{�[�h��� �ꗗ
                next_actions, next_states = zip(*next_steps.items())
                hold_next_actions, hold_next_states = zip(*hold_steps.items())
                # next_states (��ʃ{�[�h��� �ꗗ) �̃e���\����A�� (��ʃ{�[�h��Ԃ�list �̍ŏ��̗v�f�ɏ�Ԃ��ǉ����ꂽ)
                next_states = torch.stack(next_states)
                hold_next_states = torch.stack(hold_next_states)

                # GPU �g�p�ł���Ƃ��͎g��
#                if torch.cuda.is_available():
#                    next_states = next_states.cuda()

                ##########################
                # ���f���̊w�K���{
                ##########################
                self.model.train()
                # �e���\���̌��z�̌v�Z��s�Ƃ���(Tensor.backward() ���Ăяo���Ȃ����Ƃ��m���ȏꍇ)
                with torch.no_grad():
                    # ���`���� Q �l���擾 (model �� __call__ �� forward)
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

                # ������ epsilon ��菬�����ꍇ��
                if random_action:
                    # index �𗐐��Ƃ���
                    index = randint(0, len(next_steps) - 1)
                else:
                    # index �𐄘_�̍ő�l�Ƃ���
                    index = torch.argmax(predictions).item()

            # ���� action states ����L�� index ���Ɍ���
            next_state = next_states[index, :]

            # index �ɂĎ��� action �̌���
            # action �� list
            # 0: 2�Ԗ� X���ړ�
            # 1: 1�Ԗ� �e�g���~�m��]
            # 2: 3�Ԗ� Y���~�� (-1: �� Drop)
            # 3: 4�Ԗ� �e�g���~�m��] (Next Turn)
            # 4: 5�Ԗ� X���ړ� (Next Turn)
            action = next_actions[index]
            # step, step_v2 �ɂ���V�v�Z
            if nextMove["strategy"]["use_hold_function"] == "y":
              if hold_piece_id == None: # ����hold
                reward = 0  # hold�͑��߂Ɏ��{�����悤max�l��ݒ�
              else:
                reward = self.step_v2(curr_backboard, action, hold_shape_class, curr_piece_id)
            else:
              reward = self.step_v2(curr_backboard, action, curr_shape_class, hold_piece_id)

            done = False  # game over flag

            #####################################
            # Double DQN �L����
            # ======predict max_a Q(s_(t+1),a)======
            # if use double dqn, predicted by main model
            if self.double_dqn:
                # ��ʃ{�[�h�f�[�^���R�s�[���� �w����W�Ƀe�g���~�m��z�u��������������ʃ{�[�h��y���W��Ԃ�
                if nextMove["strategy"]["use_hold_function"] == "y":
                    if hold_piece_id == None:
                        next_backboard = curr_backboard
                        drop_y = 0  # �Ƃ肠��������Ă��邾���B�g��Ȃ��͂�
                    else:
                        next_backboard, drop_y = self.getBoard(curr_backboard, hold_shape_class, action[1], action[0], action[2])
                else:
                    next_backboard, drop_y = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0], action[2])

                # ��ʃ{�[�h�� �e�g���~�m��]��� �ɗ����������Ƃ��̎��̏�Ԉꗗ���쐬
                next2_steps = self.get_next_func(next_backboard, next_piece_id, next_shape_class)
                # �����z�[���h�̏ꍇ�͍��͍l���Ȃ����ƂƂ���
                # if nextMove["strategy"]["use_hold_function"] == "y" and hold_piece_id == None:
                #     next2_steps = self.get_next_func(
                #         next_backboard, next_piece_id, next_shape_class)
                # else:
                #     next2_steps = self.get_next_func(
                #         next_backboard, next_piece_id, next_shape_class)
                #     hold_next2_steps = self.get_next_func(
                #         next_backboard, next_next_piece_id, next_next_shape_class)

                # ���̏�Ԉꗗ�� action �� states �Ŕz��
                next2_actions, next2_states = zip(*next2_steps.items())
                # next_states �̃e���\����A��
                next2_states = torch.stack(next2_states)
                # GPU �g�p�ł���Ƃ��͎g��
#                if torch.cuda.is_available():
#                    next2_states = next2_states.cuda()
                ##########################
                # ���f���̊w�K���{
                ##########################
                self.model.train()
                # �e���\���̌��z�̌v�Z��s�Ƃ���
                with torch.no_grad():
                    # ���`���� Q �l���擾 (model �� __call__ �� forward)
                    next_predictions = self.model(next2_states)[:, 0]
                # ���� index �𐄘_�̍ő�l�Ƃ���
                next_index = torch.argmax(next_predictions).item()
                # ���̏�Ԃ� index �Ŏw�肵�擾
                next2_state = next2_states[next_index, :]

            ################################
            # Target Next �L����
            # if use target net, predicted by target model
#             elif self.target_net:
#                 # ��ʃ{�[�h�f�[�^���R�s�[���� �w����W�Ƀe�g���~�m��z�u��������������ʃ{�[�h��y���W��Ԃ�
#                 next_backboard, drop_y = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0], action[2])
#                 # ��ʃ{�[�h�� �e�g���~�m��]��� �ɗ����������Ƃ��̎��̏�Ԉꗗ���쐬
#                 next2_steps = self.get_next_func(next_backboard, next_piece_id, next_shape_class)
#                 # ���̏�Ԉꗗ�� action �� states �Ŕz��
#                 next2_actions, next2_states = zip(*next2_steps.items())
#                 # next_states �̃e���\����A��
#                 next2_states = torch.stack(next2_states)
#                 # GPU �g�p�ł���Ƃ��͎g��
# #                if torch.cuda.is_available():
# #                    next2_states = next2_states.cuda()
#                 ##########################
#                 # ���f���̊w�K���{
#                 ##########################
#                 self.target_model.train()
#                 # �e���\���̌��z�̌v�Z��s�Ƃ���
#                 with torch.no_grad():
#                     # �w�^�[�Q�b�g���f���x�� Q�l�Z�o
#                     next_predictions = self.target_model(next2_states)[:, 0]
#                 # ���� index �𐄘_�̍ő�l�Ƃ���
#                 next_index = torch.argmax(next_predictions).item()
#                 # ���̏�Ԃ� index �Ŏw�肵�擾
#                 next2_state = next2_states[next_index, :]

#             # if not use target net,predicted by main model
#             else:
#                 # ��ʃ{�[�h�f�[�^���R�s�[���� �w����W�Ƀe�g���~�m��z�u��������������ʃ{�[�h��y���W��Ԃ�
#                 next_backboard, drop_y = self.getBoard(curr_backboard, curr_shape_class, action[1], action[0], action[2])
#                 # ��ʃ{�[�h�� �e�g���~�m��]��� �ɗ����������Ƃ��̎��̏�Ԉꗗ���쐬
#                 next2_steps = self.get_next_func(next_backboard, next_piece_id, next_shape_class)
#                 # ���̏�Ԉꗗ�� action �� states �Ŕz��
#                 next2_actions, next2_states = zip(*next2_steps.items())
#                 # ���̏�Ԃ� index �Ŏw�肵�擾
#                 next2_states = torch.stack(next2_states)

#                 # GPU �g�p�ł���Ƃ��͎g��
# #                if torch.cuda.is_available():
# #                    next2_states = next2_states.cuda()
#                 ##########################
#                 # ���f���̊w�K���{
#                 ##########################
#                 self.model.train()
#                 # �e���\���̌��z�̌v�Z��s�Ƃ���
#                 with torch.no_grad():
#                     # ���`���� Q �l���擾 (model �� __call__ �� forward)
#                     next_predictions = self.model(next2_states)[:, 0]

#                 # epsilon = �w�K���ʂ��痐���ŕύX���銄���Ώ�
#                 # num_decay_epochs ���O�܂ł͔��ŏ��� epsilon ���猸�炵�Ă���
#                 # num_decay_ecpchs �ȍ~�� final_epsilon�Œ�
#                 epsilon = self.final_epsilon + (max(self.num_decay_epochs - self.epoch, 0) * (
#                     self.initial_epsilon - self.final_epsilon) / self.num_decay_epochs)
#                 u = random()
#                 # epsilon ��藐�� u ���������ꍇ�t���O�����Ă�
#                 random_action = u <= epsilon

#                 # ������ epsilon ��菬�����ꍇ��
#                 if random_action:
#                     # index �𗐐��w��
#                     next_index = randint(0, len(next2_steps) - 1)
#                 else:
#                    # ���� index �𐄘_�̍ő�l�Ƃ���
#                     next_index = torch.argmax(next_predictions).item()
#                 # ���̏�Ԃ� index �ɂ��w��
#                 next2_state = next2_states[next_index, :]

            # =======================================
            # Episode Memory ��
            # next_state  ���̌���1�ʎ�
            # reward ��V
            # next2_state ��r�Ώۂ̃��f���ɂ����� (Target net �Ȃ�)
            # done Game Over flag
            # self.replay_memory.append([next_state, reward, next2_state,done])
            self.episode_memory.append([next_state, reward, next2_state, done])
            # �D�揇�ʂ��o���w�K�L���Ȃ��
            if self.prioritized_replay:
                # �L���[�Ƀ��v���C�p�̏����i�[���Ă���
                self.PER.store()

            # self.replay_memory.append([self.state, reward, next_state,done])

            ###############################################
            # �w�K�� ���̓���w��
            ###############################################
            # �O�̃^�[���� Drop ���X�L�b�v���Ă������H (-1: ���Ă��Ȃ�, ����ȊO: ���Ă���)
            # third_y, forth_direction, fifth_x
            # self.skip_drop = [-1, -1, -1]

            # �e�g���~�m��]
            nextMove["strategy"]["direction"] = action[1]
            # ������
            nextMove["strategy"]["x"] = action[0]
            ###########
            # Drop Down �����̏ꍇ
            if action[2] == -1 and action[3] == -1 and action[4] == -1:
                # Drop Down ����
                nextMove["strategy"]["y_operation"] = 1
                # Move Down �~����
                nextMove["strategy"]["y_moveblocknum"] = 1
                # �O�̃^�[���� Drop ���X�L�b�v���Ă������H (-1: ���Ă��Ȃ�, ����ȊO: ���Ă���)
                self.skip_drop = [-1, -1, -1]
            ###########
            # Move Down �~���̏ꍇ
            else:
                # Move Down �~��
                nextMove["strategy"]["y_operation"] = 0
                # Move Down �~�� ��
                nextMove["strategy"]["y_moveblocknum"] = action[2]
                # �O�̃^�[���� Drop ���X�L�b�v���Ă������H (-1: ���Ă��Ȃ�, ����ȊO: ���Ă���)
                # third_y, forth_direction, fifth_x
                self.skip_drop = [action[2], action[3], action[4]]
                # debug
                if self.debug_flag_move_down == 1:
                    print("Move Down: ", "(", action[0], ",", action[2], ")")

            ##########
            # �w�K�I������
            ##########
            # 1�Q�[��(EPOCH)�̏���e�g���~�m���𒴂����烊�Z�b�g�t���O�𗧂Ă�
            if self.tetrominoes > self.max_tetrominoes:
                nextMove["option"]["force_reset_field"] = True
            # STATE = next_state ���
            self.state = next_state

        ###############################################
        ###############################################
        # ���_ �̏ꍇ
        ###############################################
        ###############################################
        elif self.mode == "predict" or self.mode == "predict_sample":
            ##############
            # model �؂�ւ�
            if self.weight2_available:
                # �{�[�h���Q������
                reshape_board = self.get_reshape_backboard(curr_backboard)
                # �ł��������̈ʒu�����߂�
                _, _, max_highest_hole = self.get_holes(reshape_board, -1)
                # model2 �؂�ւ�����
                if max_highest_hole < self.predict_weight2_enable_index:
                    self.weight2_enable = True
                # model1 �؂�ւ�����
                if max_highest_hole > self.predict_weight2_disable_index:
                    self.weight2_enable = False

                # debug
                print(GameStatus["judge_info"]["block_index"], self.weight2_enable, max_highest_hole)

            ##############
            # model �w��
            predict_model = self.model
            if self.weight2_enable:
                predict_model = self.model2

            # ���_���[�h�ɐ؂�ւ�
            predict_model.eval()

            # ���̃e�g���~�m�\��
            if self.predict_next_num > 0:

                # index_list [1�Ԗ�index, 2�Ԗ�index, 3�Ԗ�index ...] => q
                index_list = []
                hold_index_list = []
                # index_list_to_q (1�Ԗ�index, 2�Ԗ�index, 3�Ԗ�index ...) => q
                index_list_to_q = {}
                hold_index_list_to_q = {}
                ######################
                # ���̗\�������predict_next_steps_train���{, 1�Ԗڂ���predict_next_num_train�Ԗڂ܂ŗ\��
#���@                                                   �����ꂪ�Aoriginal��false����true�ɕς���Ă���
                index_list, index_list_to_q, next_actions, next_states \
                    = self.get_predictions(self.model, False, GameStatus, next_steps, self.predict_next_steps_train,
                     1, self.predict_next_num_train, index_list, index_list_to_q, -60000)
                # print(index_list_to_q)
                # print("max")
                hold_index_list, hold_index_list_to_q, hold_next_actions, hold_next_states\
                    = self.get_predictions(self.model, False, GameStatus, hold_steps, self.predict_next_steps_train,
                     1, self.predict_next_num_train, hold_index_list, hold_index_list_to_q, -60000)

                # �S�\���̍ő� q
                max_index_list = max(index_list_to_q, key=index_list_to_q.get)
                hold_max_index_list = max(hold_index_list_to_q, key=hold_index_list_to_q.get)

                # �z�[���h�����ق���Q�l���ǂ������Ƃ��͕ϐ����z�[���h�̏ꍇ�ɑ�����ւ�����
                # ��x�ڂ̃z�[���h�̏ꍇ��action�͖������ꎟ�̃^�[���ƂȂ�A��x�ڈȍ~�̃z�[���h�̏ꍇ�̓z�[���h����Ă����~�m�̓����action�Ɋi�[����
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
                # 1��ڂ� index ����
                index = max_index_list[0].item()

            else:
                # ��ʃ{�[�h�̎��̏�Ԉꗗ�� action �� states �ɂ킯�Astates ��A��
                next_actions, next_states = zip(*next_steps.items())
                hold_next_actions, hold_next_states = zip(*hold_steps.items())
                next_states = torch.stack(next_states)
                hold_next_states = torch.stack(hold_next_states)

                # ���`���� Q �l���擾 (model �� __call__ �� forward)
                predictions = predict_model(next_states)[:, 0]
                hold_predictions = predict_model(hold_next_states)[:, 0]
                ## �ő�l�� index �擾
                index = torch.argmax(predictions).item()
                hold_index = torch.argmax(hold_predictions).item()

                if max(predictions) < max(hold_predictions):
                    nextMove["strategy"]["use_hold_function"] = "y"
                    next_steps = hold_steps
                    next_actions = hold_next_actions
                    next_states = hold_next_states
                    predictions = hold_predictions
                    index = hold_index

            # ���� action �� index �����Ɍ���
            # 0: 2�Ԗ� X���ړ�
            # 1: 1�Ԗ� �e�g���~�m��]
            # 2: 3�Ԗ� Y���~�� (-1: �� Drop)
            # 3: 4�Ԗ� �e�g���~�m��] (Next Turn)
            # 4: 5�Ԗ� X���ړ� (Next Turn)
            action = next_actions[index]

            ###############################################
            # ���_�� ���̓���w��
            ###############################################
            # �O�̃^�[���� Drop ���X�L�b�v���Ă������H (-1: ���Ă��Ȃ�, ����ȊO: ���Ă���)
            # third_y, forth_direction, fifth_x
            # self.skip_drop = [-1, -1, -1]
            # �e�g���~�m��]
            nextMove["strategy"]["direction"] = action[1]
            # ������
            nextMove["strategy"]["x"] = action[0]
            ###########
            # Drop Down �����̏ꍇ
            if action[2] == -1 and action[3] == -1 and action[4] == -1:
                # Drop Down ����
                nextMove["strategy"]["y_operation"] = 1
                # Move Down �~����
                nextMove["strategy"]["y_moveblocknum"] = 1
                # �O�̃^�[���� Drop ���X�L�b�v���Ă������H (-1: ���Ă��Ȃ�, ����ȊO: ���Ă���)
                self.skip_drop = [-1, -1, -1]
            ###########
            # Move Down �~���̏ꍇ
            else:
                # Move Down �~��
                nextMove["strategy"]["y_operation"] = 0
                # Move Down �~�� ��
                nextMove["strategy"]["y_moveblocknum"] = action[2]
                # �O�̃^�[���� Drop ���X�L�b�v���Ă������H (-1: ���Ă��Ȃ�, ����ȊO: ���Ă���)
                # third_y, forth_direction, fifth_x
                self.skip_drop = [action[2], action[3], action[4]]
                # debug
                if self.debug_flag_move_down == 1:
                    print("Move Down: ", "(", action[0], ",", action[2], ")")
        # �I������
        if self.time_disp:
            print(datetime.now()-t1)
        # �I��
        return nextMove

    ####################################
    # �e�g���~�m�̗\���ɑ΂��Ď��̏�ԃ��X�g��Top num_steps�擾
    # self:
    # predict_model: ���f���w��
    # is_train: �w�K���[�h��Ԃ̏ꍇ (no_grad�ɂ��邽��)
    # GameStatus: GameStatus
    # prev_steps: �O�̎�Ԃ̌��胊�X�g
    # num_steps: 1�̎�ԂŌ���������T����
    # next_order: ������̎�Ԃ�
    # left: ���Ԗڂ̎�Ԃ܂ŒT�����邩
    # index_list: ��Ԃ��Ƃ�index���X�g
    # index_list_to_q: ��Ԃ��Ƃ�index���X�g���� Q �l�ւ̕ϊ�
    ####################################
    def get_predictions(self, predict_model, is_train, GameStatus, prev_steps, num_steps, next_order, left, index_list, index_list_to_q, highest_q):
        # ���̗\���ꗗ
        next_predictions = []
        # index_list ����
        new_index_list = []

        # �\���̉�ʃ{�[�h
        # next_predict_backboard = []

        # ��ʃ{�[�h�̎��̏�Ԉꗗ�� action �� states �ɂ킯�Astates ��A��
        next_actions, next_states = zip(*prev_steps.items())
        next_states = torch.stack(next_states)
        # �w�K���[�h�̏ꍇ
        if is_train:
            # GPU �g�p�ł���Ƃ��͎g��
            #            if torch.cuda.is_available():
            #                next_states = next_states.cuda()
            # �e���\���̌��z�̌v�Z��s�Ƃ���
            with torch.no_grad():
                # ���`���� Q �l���擾 (model �� __call__ �� forward)
                predictions = predict_model(next_states)[:, 0]
        # ���_���[�h�̏ꍇ
        else:
            # ���`���� Q �l���擾 (model �� __call__ �� forward)
            predictions = predict_model(next_states)[:, 0]

        # num_steps �Ԗڂ܂� Top �� index �擾
        top_indices = torch.topk(predictions, num_steps).indices

        # �ċA�T��
        if next_order < left:
            # �\���̉�ʃ{�[�h�擾
            # predict_order = 0
            for index in top_indices:
                # index_list �ɒǉ�
                new_index_list = index_list.copy()
                new_index_list.append(index)
                # Q �l��r
                now_q = predictions[index].item()
                if now_q > highest_q:
                    # �ō��l�Ƃ���
                    highest_q = now_q

                # ���̉�ʃ{�[�h (torch) ���Ђ��ς��Ă���
                next_state = next_states[index, :]
                # print(next_order, ":", next_state)
                # Numpy �ɕϊ��� int �ɂ��āA1������
                # next_predict_backboard.append(np.ravel(next_state.numpy().astype(int)))
                # print(predict_order,":", next_predict_backboard[predict_order])

                # ���̗\�z�胊�X�g
                # next_state Numpy �ɕϊ��� int �ɂ��āA1������
                next_steps = self.get_next_func(np.ravel(next_state.numpy().astype(int)),
                                                GameStatus["block_info"]["nextShapeList"]["element"+str(next_order)]["index"],
                                                GameStatus["block_info"]["nextShapeList"]["element"+str(next_order)]["class"])
                # GameStatus["block_info"]["nextShapeList"]["element"+str(1)]["direction_range"]

                # ���̗\������� num_steps ���{, next_order �Ԗڂ��� left �Ԗڂ܂ŗ\��
                new_index_list, index_list_to_q, new_next_actions, new_next_states\
                    = self.get_predictions(predict_model, is_train, GameStatus,
                                           next_steps, num_steps, next_order+1, left, new_index_list, index_list_to_q, highest_q)
                # ���̃J�E���g
                # predict_order += 1
        # �ċA�I��
        else:
            # Top �̂� index_list �ɒǉ�
            new_index_list = index_list.copy()
            new_index_list.append(top_indices[0])
            # Q �l��r
            now_q = predictions[top_indices[0]].item()
            if now_q > highest_q:
                # �ō��l�Ƃ���
                highest_q = now_q
            # index_list ���� q �l�ւ̎���������
            # print (new_index_list, highest_q, now_q)
            index_list_to_q[tuple(new_index_list)] = highest_q

        # ���̗\���ꗗ��Q�l, ����эŏ��� action, state ��Ԃ�
        return new_index_list, index_list_to_q, next_actions, next_states

    ####################################
    # �e�g���~�m���z�u�ł��鍶�[�ƉE�[�̍��W��Ԃ�
    # self,
    # Shape_class: ���݂Ɨ\���e�g���~�m�̔z��
    # direction: ���݂̃e�g���~�m����
    ####################################
    def getSearchXRange(self, Shape_class, direction):
        #
        # get x range from shape direction.
        #
        # �e�g���~�m�����_���� x �������ɍő剽�}�X��L����̂��擾
        # get shape x offsets[minX,maxX] as relative value.
        minX, maxX, _, _ = Shape_class.getBoundingOffsets(direction)
        # �������̃T�C�Y��
        xMin = -1 * minX
        # �E�����̃T�C�Y���i��ʃT�C�Y����Ђ��j
        xMax = self.board_data_width - maxX
        return xMin, xMax

    ####################################
    # direction (��]���)�̃e�g���~�m���W�z����擾���A�����x,y�ɔz�u�����ꍇ��2�������W�z���Ԃ�
    ####################################
    def getShapeCoordArray(self, Shape_class, direction, x, y):
        #
        # get coordinate array by given shape.
        # direction (��]���)�̃e�g���~�m���W�z����擾���A�����x,y�ɔz�u�����ꍇ��2�������W�z���Ԃ�
        # get array from shape direction, x, y.
        coordArray = Shape_class.getCoords(direction, x, y)
        return coordArray

    ####################################
    # ��ʃ{�[�h�f�[�^���R�s�[���Ďw����W�Ƀe�g���~�m��z�u��������������ʃ{�[�h��y���W��Ԃ�
    # board_backboard: �����ʃ{�[�h
    # Shape_class: �e�g���~�m��/�\�����X�g
    # direction: �e�g���~�m��]����
    # center_x: �e�g���~�mx���W
    # center_y: �e�g���~�my���W
    ####################################
    def getBoard(self, board_backboard, Shape_class, direction, center_x, center_y):
        #
        # get new board.
        #
        # copy backboard data to make new board.
        # if not, original backboard data will be updated later.
        board = copy.deepcopy(board_backboard)
        # �w����W���痎���������Ƃ���Ƀe�g���~�m���Œ肵���̉�ʃ{�[�h��Ԃ�
        _board, drop_y = self.dropDown(board, Shape_class, direction, center_x, center_y)
        return _board, drop_y

    ####################################
    # �w����W���痎���������Ƃ���Ƀe�g���~�m���Œ肵���̉�ʃ{�[�h��Ԃ�
    # board: �����ʃ{�[�h
    # Shape_class: �e�g���~�m��/�\�����X�g
    # direction: �e�g���~�m��]����
    # center_x: �e�g���~�mx���W
    # center_y: �e�g���~�my���W (-1: Drop �w��)
    ####################################
    def dropDown(self, board, Shape_class, direction, center_x, center_y):
        #
        # internal function of getBoard.
        # -- drop down the shape on the board.
        #
        ###############
        # Drop Down �����̏ꍇ
        if center_y == -1:
            center_y = 0

        # ��ʃ{�[�h�������W�Ƃ��� dy �ݒ�
        dy = self.board_data_height - 1
        # direction (��]���)�̃e�g���~�m2�������W�z����擾���A�����x,y�ɔz�u�����ꍇ�̍��W�z���Ԃ�
        coordArray = self.getShapeCoordArray(Shape_class, direction, center_x, center_y)

        # update dy
        # �e�g���~�m���W�z�񂲂Ƃ�...
        for _x, _y in coordArray:
            _yy = 0
            # _yy ��������Ƃ����Ƃɂ��u���b�N�̗����������m�F
            # _yy+�e�g���~�m���Wy �� ��ʉ�������@���@(_yy +�e�g���~�m���Wy����ʏ������ �܂��� �e�g���~�m���W_x,_yy+�e�g���~�m���W_y�̃u���b�N���Ȃ�)
            while _yy + _y < self.board_data_height and (_yy + _y < 0 or board[(_y + _yy) * self.board_data_width + _x] == self.ShapeNone_index):
                # _yy �𑫂��Ă���(�����Ă���)
                _yy += 1
            _yy -= 1
            # �������W dy /���܂ł̉�����菬����(����)�Ȃ� __yy �𗎉������Ƃ��Đݒ�
            if _yy < dy:
                dy = _yy
        # dy: �e�g���~�m�����������W���w��
        _board = self.dropDownWithDy(board, Shape_class, direction, center_x, dy)

        # debug
        if self.debug_flag_drop_down == 1:
            print("<%%", direction, center_x, center_y, dy, "%%>", end='')
            self.debug_flag_drop_down = 0
        return _board, dy

    ####################################
    # �w��ʒu�Ƀe�g���~�m���Œ肷��
    # board: �����ʃ{�[�h
    # Shape_class: �e�g���~�m��/�\�����X�g
    # direction: �e�g���~�m��]����
    # center_x: �e�g���~�mx���W
    # center_y: �e�g���~�my���W���w��
    ####################################
    def dropDownWithDy(self, board, Shape_class, direction, center_x, center_y):
        #
        # internal function of dropDown.
        #
        # board �R�s�[
        _board = board
        # direction (��]���)�̃e�g���~�m2�������W�z����擾���A�����x,y�ɔz�u�����ꍇ�̍��W�z���Ԃ�
        coordArray = self.getShapeCoordArray(Shape_class, direction, center_x, 0)
        # �e�g���~�m���W�z������ɐi�߂�
        for _x, _y in coordArray:
            # center_x, center_y �� ��ʃ{�[�h�Ƀu���b�N��z�u���āA���̉�ʃ{�[�h�f�[�^��Ԃ�
            _board[(_y + center_y) * self.board_data_width + _x] = Shape_class.shape
        return _board


BLOCK_CONTROLLER_TRAIN = Block_Controller()