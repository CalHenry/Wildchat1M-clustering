# ChatGPT conversations clustering

This project is about **NLP**, **medium size data** and **clustering**.

## Data
This project uses the **WildChat** dataset by Zhao et al. (2024), containing 1M ChatGPT interaction logs. This rich dataset of real-world AI interactions provides an ideal testbed for clustering algorithms.
- **Paper**: [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://openreview.net/forum?id=Bl8u7ZRlbM) 
- **Dataset**: [Hugging Face Link](https://huggingface.co/papers/2405.01470)
- **License**: - see `data/LICENSE`

## Working with data bigger than the memory  

The entire dataset represents ~15GB of uncompressed data (3GB in Parquet format).  
Traditional in-memory processing creates significant overhead: operations like joins and aggregations usually require 3-5x the data size in RAM due to intermediate copies, while unnesting operations can exceed 10x meaning this 15GB dataset would demand 45-75GB of RAM using conventional approaches.  
Disk-based workflows exist but involve constant read/write operations that severely impact performance. Furthermore, key operations like large joins or groupbys can still exceed available RAM. Even using a database with SQL may not be enough."  
The traditional solution has been **Cloud computing**, launching a cluster in a datacenter that has massive RAM. It's powerful, complex to set up, requires to use different libraries (Spark), expensive and is often overblown for the size of the data.  
For this project with 15GB of data, a disk workflow would probably be fine but it wouldn't be as efficient and not as fast.  

We are in 2025, my computer has 16GB of RAM and that is enough thanks to new libraries like [Polars](https://pola.rs) and [Duckdb](https://duckdb.org/).

### How it works
They integrate 3 game-changing improvements to process small to medium-sized datasets:

- **Modern query engines** integrate an *intelligent query planning* that basically minimize memory usage and maximize speed. Duckdb and Polars also use columnar storage and include parallel processing. A good query engine optimizes the query so that only necessary operations are executed. 

- **Lazy evaluation** defers computation until results are actually needed. For example, if you filter data and then select specific columns, a lazy evaluator can push the column selection down to the read operation, loading only those columns from disk. Instead of executing each operation sequentially and materializing outputs in RAM, we build a query plan that the engine can optimize...

- **Streaming execution** process the data in batches. Instead of loading the full data in memory, the engine load a batch, process it, and frees the memory when it's done processing. This works well with parallelization: the engine can process multiple batches concurrently across CPU cores. Streaming execution is **how** we can process data that is larger than the available memory.

These three concepts work together: lazy evaluation builds an optimized plan, modern query engines execute that plan intelligently, and streaming execution ensures the process scales beyond memory constraints.  

In practice, this means the 15GB WildChat dataset—with complex unnesting operations that would traditionally require 75GB of RAM—processes smoothly on a 16GB laptop in under 2 minutes.
This represents a paradigm shift: workflows that previously required cloud infrastructure now execute on regular laptops. With increasingly powerful hardware and optimized software, single machines can now handle datasets in the hundreds of gigabytes, often well beyond typical needs in machine learning, data analysis, or finance.


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

- Get the excecution time:
```shell
  time uv run scripts/nb_prep_clean.py
```
