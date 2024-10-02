# Text-Fluoroscopy
This is the official implementation of our paper [Text Fluoroscopy: Detecting LLM-Generated Text through Intrinsic Features](https://openreview.net/pdf?id=VvrZGNHg1e)



## step1: download the LLM Alibaba-NLP/gte-Qwen1.5-7B-instruct
for example, use huggingface-cli
```bash
huggingface-cli download --resume-download Alibaba-NLP/gte-Qwen1.5-7B-instruct  --local-dir ../huggingface_model/gte-Qwen1.5-7B-instruct --cache-dir ../huggingface_model/gte-Qwen1.5-7B-instruct --local-dir-use-symlinks False
```


## step2: save extracted feature and KL divergence

```bash
CUDA_VISIBLE_DEVICES=0,1 python gte-qwen/save_KL_with_first_and_last_layer.py
CUDA_VISIBLE_DEVICES=2,3 python gte-qwen/save_embedding.py
```
## step3: train the classifier and test

```bash
python embedding_classify/classify_with_max_KL_layer.py
```

