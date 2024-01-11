# Extending Whisper through Prompt Tuning to Target-speaker ASR
This is the official implementation of Extending Whisper through Prompt Tuning to Target-speaker ASR (ICASSP 2024).
![](https://files.mdnice.com/user/53953/581bda84-f42d-40fb-9826-7c265f2fec18.png)

## Start
Install `Python >= 3.8` and `Pytorch >=2.1.0`. Then, install Whipser through the pip command:
```
pip install -U openai-whisper
```
## Get the Speaker Embedding

Follow this [repo](https://github.com/HuangZiliAndy/SSL_for_multitalker) to get speaker embeddings. Download the pre-extracted x-vector embeddings and put all the `.npy` files into one directory.


## Training

```
Python train_pt.py 
--model large-v2 
--exp_name example 
--embed_path directory_to_embeddings 
--deep 
--use_mlp
--prompt_length 16 
--batch_size 1
```
where the `--deep` flag specifies whether to use deep prompting and the `--use_mlp` flag specifies whether to use re-parameterisation MLP.

## Evaluation

```
Python evaluation_pt.py 
--model large-v2 
--model_name checkpoint_name 
--embed_path directory_to_embeddings 
--deep 
--use_mlp 
--prompt_length 16
```
## LoRA Tuning
We also provide script for LoRA tuning. To use these scripts, install `loralib` first:
```
pip install loralib
```
Then, check `train_lora.py` and `evaluation_lora.py`. The usage is the same as prompt tuning scripts.
# Citation
```
@inproceedings{ts_whisper,
  title={Extending Whisper with prompt tuning to target-speaker ASR},
  author={Ma, Hao and Peng, Zhiyuan and Shao, Mingjie and Li, Jing and Liu, Ju},
  booktitle={IEEE ICASSP},
  year={2024},
}
```
