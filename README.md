# 🐂 Oxen.ai Self-Rewarding Language Models 🔁

This is work done by the [Oxen.ai Community](https://oxen.ai/community), trying to reproduce the [Self-Rewarding Language Model paper](https://arxiv.org/abs/2401.10020) from MetaAI. Thanks to [@raulc0399](https://github.com/raulc0399) for putting in all the original effort reproducing. Check out his repository [here](https://github.com/raulc0399/self-rewarding-language-models).

Every Friday we get together for a paper club called [Arxiv Dives](https://www.oxen.ai/community/arxiv-dives) where we read interesting research papers. We thought the Self-Rewarding Language Models paper felt very approachable and reproducible, so we spent some time implementing it.

<img src="./images/SRLM.png" width="512px"></img>

If you want to learn more about Self-Rewarding Language Models you can find our deep dive on it [here](https://www.oxen.ai/blog/arxiv-dives-self-rewarding-language-models).

## 🤖 Goal

The goal is to have a single script that can take in a base LLM and put it into a Self-Reward loop. The initial experiments were run with `mistralai/Mistral-7B-v0.1` as the base model, but in theory could be run with any model.

```bash
./self-reward.sh scripts mistralai/Mistral-7B-v0.1 M0
```

Currently this script will get you from M0 to M1, but in theory we can wrap it in a loop and kick off a self-reward cycle.

## 🏃‍➡️ Steps

There are 5 main steps in each iteration of the Self-Rewarding loop.

0) [00_sft.py](scripts/00_sft.py) - Supervised Fine-Tuning (SFT) of a base model to give it instruction following and evaluation skills.
1) [01_gen_prompts.py](scripts/01_gen_prompts.py) - Generate new prompts to add to the training set.
2) [02_gen_responses.py](scripts/02_gen_responses.py) - Generate N Responses per prompt, so that we can create preference pairs.
3) [03_gen_scores.py](scripts/03_gen_scores.py) - Score each response from 1-5 for how well it answered the prompt.
4) [04_gen_preferences.py](scripts/04_gen_preferences.py) - Generate preference pairs given the scores to create a DPO dataset
5) [05_dpo.py](scripts/05_dpo.py) - Run Direct Preference Optimization (DPO) to train the next iteration of the model

## TODO
Format MMLU as [ift_eft.jsonl](https://www.oxen.ai/datasets/Self-Rewarding-Language-Models/file/main/M0/train/ift_eft.jsonl)

Add NLL loss to winning preferences.
