---
license: other
library_name: peft
tags:
- llama-factory
- lora
- generated_from_trainer
base_model: /root/autodl-tmp/qwen/Qwen-7B-Chat
model-index:
- name: v1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# v1

This model is a fine-tuned version of [/root/autodl-tmp/qwen/Qwen-7B-Chat](https://huggingface.co//root/autodl-tmp/qwen/Qwen-7B-Chat) on the ft_data_train and the ft_data_test datasets.
It achieves the following results on the evaluation set:
- Loss: 0.6572

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
| 0.7559        | 0.2949 | 200  | 0.7451          |
| 0.7838        | 0.5898 | 400  | 0.7202          |
| 0.9792        | 0.8846 | 600  | 0.7065          |
| 0.5938        | 1.1795 | 800  | 0.6733          |
| 0.602         | 1.4744 | 1000 | 0.6753          |
| 0.5666        | 1.7693 | 1200 | 0.6572          |


### Framework versions

- PEFT 0.10.0
- Transformers 4.40.1
- Pytorch 2.1.2+cu121
- Datasets 2.18.0
- Tokenizers 0.19.1