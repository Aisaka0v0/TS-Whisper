#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：whisper-main
@File    ：tuning_16.py
@IDE     ：PyCharm
@Author  ：Aisaka/Hao Ma @SDU
@Date    ：2023/7/16 下午10:54
'''

import argparse
import random
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import whisper
from tqdm import tqdm
from whisper.tokenizer import get_tokenizer
from data_utils.dataloader import get_dataloader
from model.prompting import Prompting, Prompting_len0
import os
import loralib as lora
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine-tune a Whisper model for ASR")
    # Dataloader-related arguments
    parser.add_argument(
        "--train-json",
        type=str,
        # required=True,
        default='data_utils/data/train-100.json',
        help="Path to a json file containing training data",
    )
    parser.add_argument(
        "--dev-json",
        type=str,
        # required=True,
        default='data_utils/data/dev_data.json',
        help="Path to a json file containing development data",
    )
    parser.add_argument(
        "--embed_path",
        default="/home/user/202212661/promptEng/whisper-main/whisper_finetuning/embed/",
        type=str,
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--dev-batch-size", type=int, default=1, help="Batch size for validation")
    parser.add_argument(
        "--no-timestamps-training",
        default=True,
        help="Always use the no-timestamps training mode",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use for training",
    )

    parser.add_argument(
        "--model",
        default="large",
        choices=whisper.available_models(),
        help="name of the Whisper model to use",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training")
    parser.add_argument("--exp_name", type=str, required=True, help="exp_name")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    return parser


class System(nn.Module):
    def __init__(self, model, prompt):
        super(System, self).__init__()
        self.prompt = prompt
        self.model = model
        self.emb, self.hook = self.install_forward_hook()

    def forward(self, x, xvec, y_in):
        self.emb[self.model.encoder.conv2], _ = self.prompt(xvec)
        logits = self.model.decoder(y_in, self.model.encoder(x))
        return logits

    def install_forward_hook(self):
        hooks = []
        weight = {}
        layer = self.model.encoder.conv2

        def cln_func(module, _, output):
            output[:, :, 0] = weight[module]
            return output
        hooks.append(layer.register_forward_hook(cln_func))
        return weight, hooks


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def train(
        system,
        train_loader,
        epochs,
        optimizer,
        scheduler,
        exp_name,
        train_loss=None,
        init_epoch=-1,
):
    system.cuda()
    system.train()
    system.model.eval()
    train_loss = [] if train_loss is None else train_loss.tolist()
    for e in range(init_epoch + 1, epochs):
        pbar = tqdm(train_loader)
        for i, (x, xvec, y_in, y_out) in enumerate(pbar):
            x, xvec, y_in, y_out = x.cuda(), xvec.cuda(), y_in.cuda(), y_out.cuda()
            logits = system(x, xvec, y_in)
            loss = F.cross_entropy(logits.transpose(1, 2), y_out)
            loss.backward()
            # if (i + 1) % 16 == 0:
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.detach().cpu().numpy())
            pbar.set_postfix({"loss": train_loss[-1], 'loss_mean': np.sum(train_loss) / len(train_loss)})
        scheduler.step()
        torch.save(lora.lora_state_dict(system, bias='lora_only'), './checkpoint/lora_' + exp_name + str(e))
        torch.save(system.prompt.state_dict(), './checkpoint/spk_adapt_' + exp_name + str(e))


def set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)


def process_model(model):
    load_layer = ['query', 'value']
    for module in model.named_modules():
        if any(c in module[0] for c in load_layer):
            lora_layer = lora.Linear(module[1].in_features, module[1].out_features, r=4,
                                     bias=hasattr(module[1], 'bias'), merge_weights=False)
            lora_layer.weight = module[1].weight
            if hasattr(module[1], 'bias'):
                lora_layer.bias = module[1].bias
            set_module(model, module[0], lora_layer)
    return None


def main():
    args = get_parser().parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task="transcribe")
    model = whisper.load_model(args.model, args.device)
    #  -1 is for the special token `sot_prev` and the other half is for the transcribed tokens
    max_prompt_length = model.dims.n_text_ctx // 2 - 1

    fp16 = False
    train_loader = get_dataloader(
        json=args.train_json,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=1,
        no_timestamps_rate=0.0,
        embed_path=args.embed_path,
        shuffle=True,
        context_len=0,
        n_workers=8,
    )
    prompt_layer = Prompting_len0(dim=model.dims.n_text_state, depth=model.dims.n_audio_layer)

    # freeze the whole whisper model
    for p in model.parameters():
        p.requires_grad = False

    process_model(model)
    lora.mark_only_lora_as_trainable(model, bias='lora_only')
    system = System(model, prompt_layer)
    total = sum([param.nelement() for param in system.parameters() if param.requires_grad])
    print('Number of parameter: % .4fM' % (total / 1e6))
    optimizer = torch.optim.AdamW([param for param in system.parameters() if param.requires_grad], lr=args.lr,
                                  )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, verbose=True)

    train(
        system=system,
        train_loader=train_loader,
        epochs=10,
        optimizer=optimizer,
        scheduler=scheduler,
        exp_name=args.exp_name,
    )

    train_loader = get_dataloader(
        json='data_utils/data/train-100-noisy.json',
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        fp16=fp16,
        no_timestamps_training=args.no_timestamps_training,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=1,
        no_timestamps_rate=0.0,
        embed_path=args.embed_path,
        shuffle=True,
        context_len=0,
        n_workers=8,
    )
    optimizer = torch.optim.AdamW([param for param in system.parameters() if param.requires_grad], lr=args.lr * 0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1, verbose=True)
    train(
        system=system,
        train_loader=train_loader,
        epochs=1,
        optimizer=optimizer,
        scheduler=scheduler,
        exp_name='noisy' + args.exp_name,
    )


if __name__ == "__main__":
    main()
