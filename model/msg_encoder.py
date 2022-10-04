# coding=utf-8
# /usr/bin/env python

'''
Author: hongleizhang
Email: hongleizhang1993@gmail.com
Date: 2020/9/12/8:21
Desc:
'''

import torch.nn as nn
import torch
import os


class MsgEncoder(nn.Module):
    '''
    Here, we can expand our basic model to a wider or deeper neural network.
    '''
    def __init__(self, message_length, msg_reduce_len):
        super(MsgEncoder, self).__init__()
        self.msg_in_len = message_length
        self.msg_out_len = msg_reduce_len

        layers = []
        input_len = self.msg_in_len
        for output_len in [self.msg_out_len]:
            layers.append(nn.Linear(input_len, output_len))
            layers.append(nn.LeakyReLU(inplace=False, negative_slope=2.5e-1))


        self.layers = nn.Sequential(*layers)
        msg_enc_path = "./saved_models/msg_enc_"+str(self.msg_out_len)+".h5"

        if os.path.exists(msg_enc_path):
            self.load_model(msg_enc_path)

    def forward(self, x):
        x = self.layers(x)
        return x

    def save_model(self, path):
        dict_to_save = {
            'msg_encoder': self.state_dict(),
        }
        torch.save(dict_to_save, path)

    def load_model(self, path):
        if path == None:
            return
        states = torch.load(path)
        self.load_state_dict(states['msg_encoder'])