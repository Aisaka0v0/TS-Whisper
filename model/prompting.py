#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：TS_Whisper 
@File    ：prompting.py
@IDE     ：PyCharm 
@Author  ：Hao Ma@SDU
@Date    ：2023/12/18 下午3:54 
'''
import torch
from torch import nn


class ResMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn_dim = dim//2
        self.net = nn.Sequential(
            nn.Linear(dim, self.bn_dim),
            nn.ReLU(),
            nn.Linear(self.bn_dim, dim),
            nn.LayerNorm(dim)
        )
        # for module in self.net:
        #     if isinstance(module, nn.Linear):
        #         nn.init.zeros_(module.weight)
        #         nn.init.zeros_(module.bias)

    def forward(self, x):
        return x + self.net(x)


class Prompting(nn.Module):  # basic prompt tuning
    def __init__(self, dim, prompt_length, use_mlp=True, depth=32):
        super(Prompting, self).__init__()
        self.use_mlp = use_mlp
        self.depth = depth
        self.prompt_length = prompt_length
        self.spk_embed_layer = nn.Linear(512, dim)
        self.soft_prompt_extra_encoder = nn.Parameter(torch.Tensor(depth, prompt_length, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.soft_prompt_extra_encoder)
        self.soft_prompt_extra_decoder = nn.Parameter(torch.Tensor(depth, prompt_length, dim), requires_grad=True)
        torch.nn.init.xavier_uniform_(self.soft_prompt_extra_decoder)
        if self.use_mlp:
            self.mlp_encoder = nn.ModuleList([ResMLP(dim) for i in range(depth)])
            self.mlp_decoder = nn.ModuleList([ResMLP(dim) for i in range(depth)])

        # cal params
        total = sum([param.nelement() for param in self.parameters()])
        print('Number of parameter: % .4fM' % (total / 1e6))

    def forward(self, embed_in):
        b = embed_in.size(0)
        embed = self.spk_embed_layer(embed_in)
        prompt_encoder = self.soft_prompt_extra_encoder
        prompt_decoder = self.soft_prompt_extra_decoder
        prompt_encoder, prompt_decoder = map(lambda x: list(x.chunk(self.depth, dim=0)),
                                             (prompt_encoder, prompt_decoder))

        for i in range(self.depth):
            if self.use_mlp:
                prompt_encoder[i] = self.mlp_encoder[i](prompt_encoder[i])
                prompt_decoder[i] = self.mlp_decoder[i](prompt_decoder[i])
            prompt_encoder[i] = prompt_encoder[i].repeat(b, 1, 1)
            prompt_decoder[i] = prompt_decoder[i].repeat(b, 1, 1)

        prompt_encoder[0] = torch.concat((embed.unsqueeze(1), prompt_encoder[0]), dim=1)

        return prompt_encoder + prompt_decoder

    def reparameterization(self):
        if self.use_mlp:
            with torch.no_grad():
                for i in range(self.depth):
                    self.soft_prompt_extra_encoder.data[i, :, :] = self.mlp_encoder[i](self.soft_prompt_extra_encoder[i, :, :])
                    self.soft_prompt_extra_decoder.data[i, :, :] = self.mlp_decoder[i](self.soft_prompt_extra_decoder[i, :, :])
            del self.mlp_encoder, self.mlp_decoder
            self.use_mlp = False
            # cal params
            total = sum([param.nelement() for param in self.parameters()])
            print('Number of parameter: % .4fM' % (total / 1e6))
            return self
        else:
            return self


class Prompting_len0(nn.Module):  # for lora and full fine-tuning
    def __init__(self, dim, depth=32):
        super(Prompting_len0, self).__init__()
        self.depth = depth
        self.spk_embedding1 = nn.Linear(512, dim)

    def forward(self, embed_in):
        embed = self.spk_embedding1(embed_in)
        return embed, None


if __name__ == '__main__':
    xvec = torch.randn((16, 512))
    p_layer = Prompting(dim=1280, prompt_length=8, use_mlp=True, depth=32)(xvec)
    p_layer.reparameterization()
    print()
