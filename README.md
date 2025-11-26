# ChatGPT conversations clustering

This project is about **NLP**, **medium size data** and **clustering**.  


## Data
This project uses the **WildChat** dataset by Zhao et al. (2024), containing 1M ChatGPT interaction logs. It's a great source of random texts to explore with clustering algorithms.  


- **Paper**: [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://openreview.net/forum?id=Bl8u7ZRlbM) 
- **Dataset**: [Hugging Face Link](https://huggingface.co/papers/2405.01470)
- **License**: [License Type] - see `data/LICENSE`

### Citation

```bibtex
@inproceedings{
  zhao2024wildchat,
  title={WildChat: 1M Chat{GPT} Interaction Logs in the Wild},
  author={Wenting Zhao and Xiang Ren and Jack Hessel and Claire Cardie and Yejin Choi and Yuntian Deng},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=Bl8u7ZRlbM}
}
```

### Run marimo's notebook as a python script
Since a marimo notebook is a python file, it can be excecute as such.  
To run the data preparation notebook from the project's root:
```shell
uv run marimo scripts/nb_prep_clean.py
```
This scripts read the in the raw data, manipulates, filter, clean the data and export it to parquet file. It also exports the vocab of the conversations.
