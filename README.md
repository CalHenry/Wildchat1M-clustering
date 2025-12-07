# ChatGPT conversations clustering

This project is about **NLP**, **medium size data** and **clustering**.

## Data
This project uses the **WildChat** dataset by Zhao et al. (2024), containing 1M ChatGPT interaction logs. This rich dataset of real-world AI interactions provides an ideal testbed for clustering algorithms.
- **Paper**: [WildChat: 1M ChatGPT Interaction Logs in the Wild](https://openreview.net/forum?id=Bl8u7ZRlbM) 
- **Dataset**: [Hugging Face Link](https://huggingface.co/papers/2405.01470)
- **License**: - see [data/LICENSE](data/LICENSE.md)

## Part1: Data cleaning and preparation for clustering

Our dataset consists of 14 parquet files containing ~1 million conversation records. The raw data uses a nested schema optimized for storage, with each conversation containing multiple messages and their metadata.  
Data Processing Pipeline:
- Restructure nested data into a tidy, conversation-level format
- Clean text by removing stop words, punctuation, and non-semantic elements
- Apply [lemmatization](https://www.ibm.com/think/topics/stemming-lemmatization) for word normalization
- Tokenize messages to build the corpus

The preprocessing reduces the dataset to ~475,000 rows suitable for analysis.

## Part 2: Clustering analysis

We apply unsupervised clustering to reveal hidden patterns in ChatGPT conversations and understand better our dataset. Moving beyond individual data points to understand global structure and cluster-specific characteristics.  

Dataset: We will work on the first message only, this reduce the size and the complexity of the data.
Methods: We compute the [TF-IDF](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) matrix
Exploration work: subset of 50,000 conversations (~10% of the data). The full dataset is too big (expensive RAM operations way beyond my 16GB) to produce 2d representation, compute some clustering scores or algorithms (but clustering on the entire dataset is done at the end of the part2 notebook).   

**Goal**: Identify distinct prompt types, themes, and vocabulary patterns.

**Methodology**:  
Dimensionality Reduction:  
The **TF-IDF matrix** creates high-dimensional sparse representations that challenge direct clustering. We apply manifold learning to reduce dimensions:
- **TruncatedSVD**
- **UMAP**

*(Dimension reduction is mandatory to produce 2D representations, compute scores, run HDBSCAN. High dimensional input is a constraint but not for Kmeans that handles this very efficiently and doesn't require dimension reduction)*

Clustering Algorithms:
- **K-means** (*MiniBatchKMeans*): Try to separate samples in n groups of equal variance clusters, minimize the intertia, requires to provide the cluster count. Best for shperical clusters, suited for massive datasets.
- **HDBSCAN**: Density-based approach suited to find clusters of varying densities, handles noise by leaving outliers unclustered, automatically determines the optimal number of clusters.

Both algorithms are 2 different lenses and will produce different results: K-means provides complete partitioning of the entire dataset, while HDBSCAN reveals density-based groupings and identifies noise (conversations that don't fit in the clusters).   
Comparing the result will highlight which patterns are robust and which are algorithm-dependant.  


## Working with data bigger than the memory  

The entire dataset represents ~15GB of uncompressed data (3GB in Parquet format).  
Traditional in-memory processing creates significant overhead: operations like joins and aggregations usually require 3-5x the data size in RAM due to intermediate copies, while unnesting operations can exceed 10x meaning this 15GB dataset would demand 45-75GB of RAM using conventional approaches.  
Disk-based workflows exist but involve constant read/write operations that severely impact performance. Furthermore, key operations like large joins or groupbys can still exceed available RAM. Even using a database with SQL may not be enough.    
The traditional solution has been **Cloud computing**, launching a cluster in a datacenter that has massive RAM. It's powerful, complex to set up, requires to use different libraries (Spark), expensive and is often overblown for the size of the data.   
For this project with 15GB of data, a disk workflow would probably be fine but it wouldn't be as efficient and not as fast.    

We are in 2025, my computer has 16GB of RAM and that is enough thanks to new libraries like [Polars](https://pola.rs) and [Duckdb](https://duckdb.org/).

### How it works
They integrate 3 key improvements to process medium-sized to huge datasets:

- **[Modern query engines](https://docs.pola.rs/user-guide/lazy/query-plan/#optimized-query-plan)** integrate an *intelligent query planning* that basically minimize memory usage and maximize speed. Duckdb and Polars also use columnar storage and include parallel processing. A good query engine optimizes the query so that only necessary operations are executed. 

- **[Lazy evaluation](https://stackoverflow.com/questions/24704503/lazy-evaluation-why-is-it-faster-advantages-vs-disadvantages-mechanics-why-i/24706028#24706028)** postpone evaluation of something until it's needed. For example, if you filter data and then select specific columns, a lazy evaluator can push the column selection down to the read operation, loading only those columns from disk. Instead of executing each operation sequentially and materializing outputs in RAM, we build a query plan that the engine can optimize...

- **[Streaming execution](https://docs.pola.rs/user-guide/concepts/streaming/)** process the data in batches. Instead of loading the full data in memory, the engine load a batch, process it, and frees the memory when it's done processing. This works well with parallelization: the engine can process multiple batches concurrently across CPU cores. Streaming execution is **how** we can process data that is larger than the available memory.

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

### Run a marimo notebook as a python script
Since a marimo notebook is a python file, it can be excecuted as such.  
To run the data preparation notebook from the project's root:
```shell
uv run marimo scripts/nb_prep_clean.py
```
This scripts read the in the raw data, manipulates, filter, clean the data and export it to parquet file. It also exports the vocab of the conversations.

- Get the execution time:
```shell
  time uv run scripts/nb_prep_clean.py
```
