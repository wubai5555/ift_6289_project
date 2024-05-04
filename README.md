# ift_6289_project

In this project, we aim to use text summerization to enhance the performance of text reranking.

To use gpt to summerize passages

```
python gpt_sum.py --dataset dl19
```

The gpt file already contains summerization results of dl19, covid, and touche.


To use rankgpt, i.e., using gpt to rerank and evaluate the performance, in metric of NDCG@10

```
python rankgpt.py
```

One needs to change the ```file_path``` variable before using.
