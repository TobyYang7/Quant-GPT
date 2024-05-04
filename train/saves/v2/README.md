---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /root/autodl-tmp/qwen/Qwen-7B-Chat
model-index:
- name: v2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# v2

This model is a fine-tuned version of [/root/autodl-tmp/qwen/Qwen-7B-Chat](https://huggingface.co//root/autodl-tmp/qwen/Qwen-7B-Chat) on the ft_data_train, the ft_data_test and the sentiment datasets.
It achieves the following results on the evaluation set:
- Loss: 0.6252

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- gradient_accumulation_steps: 4
- total_train_batch_size: 4
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 2.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.6299        | 0.5750 | 500  | 0.6522          |
| 0.6191        | 1.1501 | 1000 | 0.6571          |
| 0.4993        | 1.7251 | 1500 | 0.6252          |


### Framework versions

- PEFT 0.10.0
- Transformers 4.40.1
- Pytorch 2.1.2+cu121
- Datasets 2.18.0
- Tokenizers 0.19.1