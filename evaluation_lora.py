#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：whisper-main
@File    ：test_cln.py
@IDE     ：PyCharm
@Author  ：Aisaka/Hao Ma @SDU
@Date    ：2023/9/24 下午1:56
'''
from whisper.decoding import DecodingTask
from model.prompting import Prompting, Prompting_len0
import argparse
import whisper
from whisper.tokenizer import get_tokenizer
import torch
from data_utils.dataloader import get_dataloader
import os
from jiwer import wer
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer
import loralib as lora


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test script")
    # Dataloader-related arguments
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
    parser.add_argument(
        "--embed_path",
        default="/home/user/202212661/promptEng/whisper-main/whisper_finetuning/embed/",
        type=str,
        required=True,
    )
    parser.add_argument("--model_name", type=str, required=True, help="model_name")


    return parser


def install_forward_hook(model):
    hooks = []
    weight = {}
    layer = model.encoder.conv2

    def cln_func(module, _, output):
        output[:, :, 0] = weight[module]
        return output
    hooks.append(layer.register_forward_hook(cln_func))
    return weight, hooks


def test_(loader,
          model,
          prompt_layer,
          mytask,
          normal
          ):
    wer_scores = []
    pbar = tqdm(loader)
    weight, hook = install_forward_hook(model)
    module = model.encoder.conv2
    with torch.no_grad():
        for x, xvec, y_in, y_out, text in pbar:
            x, xvec, y_in = x.to(model.device), xvec.to(model.device), y_in.to(model.device)
            weight[module], _ = prompt_layer(xvec)
            results = mytask.run(x)
            for result, t in zip(results, text):
                wer_ = wer(normal(t), normal(result.text))
                wer_scores.append(wer_)
                pbar.set_postfix({'wer': wer_, 'wer_mean': sum(wer_scores)/len(wer_scores), 'text': result.text})
                print(result.text)
    return sum(wer_scores)/len(wer_scores)


def process_model(model):
    def set_module(model, submodule_key, module):
        tokens = submodule_key.split('.')
        sub_tokens = tokens[:-1]
        cur_mod = model
        for s in sub_tokens:
            cur_mod = getattr(cur_mod, s)
        setattr(cur_mod, tokens[-1], module)
    load_layer = ['query', 'value']
    for module in model.named_modules():
        if any(c in module[0] for c in load_layer):
        # if isinstance(module[1], nn.Linear):
            lora_layer = lora.Linear(module[1].in_features, module[1].out_features, r=4,
                                     bias=hasattr(module[1], 'bias'))
            lora_layer.weight = module[1].weight
            if hasattr(module[1], 'bias'):
                lora_layer.bias = module[1].bias
            set_module(model, module[0], lora_layer)
    return None


if __name__ == '__main__':
    args = get_parser().parse_args()
    tokenizer = get_tokenizer(multilingual=".en" not in args.model, task="transcribe")
    model = whisper.load_model(args.model, args.device)

    max_prompt_length = model.dims.n_text_ctx // 2 - 1

    options = whisper.DecodingOptions(without_timestamps=True, fp16=False)
    task = DecodingTask(model, options)
    normalizer = EnglishTextNormalizer()
    score = []
    model_ = []
    dataloader = get_dataloader(
        json='./data_utils/data/test.json',
        tokenizer=tokenizer,
        batch_size=16,
        fp16=False,
        no_timestamps_training=True,
        max_prompt_length=max_prompt_length,
        prompt_use_rate=1.0,
        no_timestamps_rate=0.0,
        context_len=0,
        embed_path=args.embed_path,
        shuffle=False,
        n_workers=4,
        dev=True,
    )
    prompt_layer = Prompting_len0(dim=model.dims.n_text_state, depth=model.dims.n_audio_layer)

    weight = torch.load('./checkpoint/spk_adapt_' + args.model_name)
    prompt_layer.load_state_dict(weight, strict=True)
    lora_weight = torch.load('./checkpoint/lora_' + args.model_name)
    NewDict = {k.replace('model.', ''): v for k, v in lora_weight.items()}
    process_model(model)
    model.load_state_dict(NewDict, strict=False)
    prompt_layer.eval()
    model.cuda()
    prompt_layer.cuda()
    model.eval()
    score_ = test_(loader=dataloader, model=model, prompt_layer=prompt_layer, mytask=task, normal=normalizer)
    print(score_)
