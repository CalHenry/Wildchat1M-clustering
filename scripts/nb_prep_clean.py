import marimo

__generated_with = "0.17.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import json

    import duckdb
    import marimo as mo
    import polars as pl
    from simplemma import lemmatize

    return duckdb, json, lemmatize, mo, pl


@app.cell
def _():
    # fmt: off
    try:
        from nltk.corpus import stopwords
    except LookupError:
        import nltk
        nltk.download("stopwords")  # should go to "User/nltk_data/corpus"
        from nltk.corpus import stopwords
        print('stopwords downloaded ✅')
    return (stopwords,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The goal of this notebook is to clean and prepare the data for clustering.
    We have to:
    - Filter only the english content
    - Gather the content from all 14 files
    - Manipulate the data to obtain a tidy data with a row for each conversation and a columns for each feature

    This notebook is organized in columns, the first one is the main workflow, the second column contain the helper functions used in the main columns.

    ## Data preparation

    Our data is spread in 14 parquet files, each row is a conversation. The data schema is optimized for storage not for analysis.
    For example, in the raw data, the column 'conversation' is a nested structure that contains a list (one item for each message of the conversation), and all the variables about the messages (ex: content, language).

    Data preparation will include reorganizing the dataset to have a tidy structure at the conversation level.
    We will also clean the data, removing stop words, punctuation... and all the elements that do not hold semantic meaning. We will also [lemmatize](https://www.ibm.com/think/topics/stemming-lemmatization) the words.
    Finally we will tokenize the messages into list of words to build the **corpus** of our dataset.
    The raw dataset has ~1 milion rows and we will end up with ~475 000 rows

    ### Computational challenge

    The entire dataset represents ~15GB of uncompressed data (3GB in Parquet format), which would typically require 45-75GB of RAM using conventional in-memory approaches.
    This notebook processes the data efficiently in a single pipeline using Polars and DuckDB. For details on how **modern query engines**, **lazy** evaluation, and **streaming execution** make this possible, see the README
    The final dataset weigths ~2GB in parquet.


    ### A word on the libraries used:

    **DuckDB** and **Polars** are both excellent choices for this type of work. While the project could be completed entirely with any of the 2, I prefer Polars's expressive syntax for complex, chained DataFrame transformations.
    Both libraries feature lazy execution and are highly optimized. The workflow leverages their complementary strengths:

    - **DuckDB** to import data or filtered subsets from raw files, simple queries can accomplish a lot
    - **Polars** for complex manipulations in the core workflow

    Key benefits of this hybrid approach:
    - Extremely fast data reading (both libraries are excellent)
    - Efficient data import with simple SQL queries
    - Expressive pipeline with polars
    - Complete lazy pipeline allow to optimize from the begining to the end by polars query optimizer

    ➡️ Both libraries are interchangeable, but Duckdb requires that the input is either loaded on RAM or written on disk.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next cell is a great example of seemless integration between the 2 libraries:
    1. We use duckdb for a SQL query
    2. We specify that we want to convert the query's result to Polars df.
    As soon as ```.pl(lazy=True)``` is written, the object is now a polars LazyFrame, to which i can apply polars code.
    We use both libraries in the same sequence.

    We don't actually read the entire dataset from the 14 files. Duckdb can use the parquet file metadata to discover the data's schema without reading it. It can already exclude the variables that we've specified. The same goes for the filter on 'language', duckdb will only discover the rows where language is english. This reduces massively the amount of data to process.
    All of the optimization happens before the query is executed. Here we push it further by asking for a polars lazyframe as the output. We don't request a dataframe but a query plan that can be excecuted. This is why the cell runs in 30ms, we don't load the data, we are writing a query plan and let polars optimize it for us.

    > We use the abbreviation '**lf**' for lazyframes to differentiate with dataframes 'df'
    """)
    return


@app.cell
def _(duckdb):
    # Load data (lazy)
    lf = (
        duckdb.query(
            """
        SELECT * EXCLUDE (toxic, timestamp, redacted, openai_moderation, detoxify_moderation)
        FROM 'data/raw/*.parquet'
        WHERE language = 'English'
        """
        )
        .pl(lazy=True)
        .rename({"language": "lang_conv"})
    )
    return (lf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Few observations on the raw dataset:

    - This dataset has 3 levels:
    1. **User** level: identified by **'hashed_ip'** (based on the user's IP address)
    2. **Conversations** level: identified by **'conversation_hash'**. *Uniquely* identified by **'conversation_hash'** + **'hashed_ip'**. ('conversation_hash' is not a unique identifier. 2 identical conversations will have the hash regardless the user)
    3. **Message** level: identified **'turn_identifier'** (unique across a conversation, not across all the messages)

    1 user can have N conversations of N messages.
    1 message belongs to 1 conversation that belongs to 1 user

    - Messages works in pairs: one **user** message, one **assistant** response. **turn_identifier** + **role** give the order of the messages. The **user** always comes first.
    - **'conversations'** is a complex variable of **List[structs]**. Each element of the list is 1 message. The fields of the **struct** are the message's content and metadata. Some fields are specific to the user, others to the assistant (expected missing values).

    ### Tidy: manipulation, variable selection
    We want to be at the **conversation level** for the analysis.
    To do so we have to:
    - flatten the dataset to message level, each row is a unique message, n rows for the same conversation.
    - aggregate back to conversation level by creating features around the messages. We keep the metadata variables for the conversations and not the messages.
    - Some filtering/ cleaning is easier to do at the message level. Having acces to both levels gives more flexibility
    - Conversation level allow to use different level of context with different aggregations. For example features can be *'Only the first user message'*, *'Only user messages'* or *'full conversation'*.


    ### Order of the tasks:

    1. Import data, shortlist of variables, filter only English conversations
    2. Compact conversation level --> Message level (flatten the dataset)
    3. Unique ID for conversations
    4. Message level --> Conversation level (aggregation)
    """)
    return


@app.cell
def _(lf):
    # Same variables names in the nested var 'conversation' are also present in the dataset. We have to rename them before unnesting otherwise there are conflicts
    # Front vars give the info at the conversation level and the nested ones give the info at the message level
    # We rename the nested vars to make them obvious

    conflicting_cols = ["language", "country", "state", "hashed_ip", "header"]

    struct_fields = [
        field.name for field in lf.collect_schema()["conversation"].inner.fields
    ]

    new_fields = [
        f"{field}_nested" if field in conflicting_cols else field
        for field in struct_fields
    ]
    return (new_fields,)


@app.cell
def _(lf, new_fields, pl, remove_nolang_messages, remove_non_english_mess):
    # no unique id for conversations so we use the hash + the ip to make one
    create_conversation_id = (
        pl.col("conversation_hash") + "_" + pl.col("hashed_ip")
    ).alias("conversation_id")

    # Flatten the data (explode nested variables), 2 levels depth
    # conversation contains many nested vars and a nested var inside the nested vars: list[struct[struct]]

    lf_mess_raw = (
        (
            lf.with_columns(["conversation_hash", "conversation"])
            .explode(  # explode the list of struct (-1 layer)
                "conversation"
            )
            .with_columns(
                pl.col("conversation").struct.rename_fields(
                    new_fields
                )  # var name conflict between original and nested vars
            )
            .unnest(
                "conversation", "header"
            )  # unnest the struct cols (2nd layer, now flat lf)
            .drop(
                "hashed_ip_nested",
                "header_nested",
                "state_nested",
                "country_nested",
                "toxic",
                "user-agent",
                "accept-language",
                "redacted",
                "model",
            )
        )
        .rename({"language_nested": "language_message"})
        .with_columns(create_conversation_id)
    )

    # Clean messages of lf_mess_raw
    lf_mess = lf_mess_raw.pipe(remove_nolang_messages).pipe(remove_non_english_mess)
    return (lf_mess,)


@app.cell
def _(get_stopwords_pattern, lf_mess, pl, process_message_tokens):
    # From the flat message dataset, we aggregate to conversation level
    # (4 different aggregations = 4 new variables)
    lf_conv_init = lf_mess.group_by("conversation_id").agg(
        [
            # First user message content
            pl.col("content")
            .filter(pl.col("role") == "user")
            .first()
            .alias("first_user_content"),
            # All user messages concatenated
            pl.col("content")
            .filter(pl.col("role") == "user")
            .str.join(delimiter="\n\n")
            .alias("all_user_content"),
            # All assistant messages concatenated
            pl.col("content")
            .filter(pl.col("role") == "assistant")
            .str.join(delimiter="\n\n")
            .alias("all_assistant_content"),
            # All messages concatenated (user and assistant)
            pl.col("content").str.join(delimiter="\n\n").alias("all_content"),
            # Keep other columns (take first value)
            pl.exclude(
                [
                    "content",
                    "turn_identifier",
                    "role",
                    "timestamp",
                    "language_message",
                ]
            ).first(),
        ]
    )

    # We can now clean and tokenize the messages:
    stop_pattern = get_stopwords_pattern()  # from NLTK as a regex string

    # Clean aggregated messages of lf_conv_lvl_init
    lf_conv, vocab = process_message_tokens(
        lf_conv_init, "first_user_content", stop_pattern
    )
    return lf_conv, vocab


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Takeway:

    Most conversations are length 2 (1 user prompt, 1 assistant response). Longer conversations go up to 22 messages and the second most common number of turns is 6.

    *Summary statistics and visualization are available in the exploration notebook 'explo_nb.py'*.

    We focus on the English conversations and english messages.
    We try to do as many operations as possible at the conversation level to improve simplicity and speed.
    We created some functions (lemmatization) and choose some libraries over others to have better performances (**simplemma** over **spacy** <-> speed over quality)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(json, vocab):
    # save vocab to disk

    with open("data/processed/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("✅ Saved vocab to: data/processed/vocab.json")
    return


@app.cell
def _(lf_conv):
    # save lf_conv to parquet
    # ! memory intensive excecution, it's here that the data passes trough the RAM
    lf_conv.sink_parquet("data/clean/df_conv.parquet")
    print("✅ Saved lf_conv to: data/clean/df_conv.parquet")
    return


@app.cell
def _():
    # lf_conv.select(pl.col(['conversation_id', 'first_user_content', 'first_user_content_tokens'])).sink_parquet("data/clean/df_conv_fuc.parquet")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Takeway on the workflow

    - We went full polars except the import part at the begining with duckdb
    - We kept a full lazy workflow, only declaring lazyframes
    - We collected data once during the pipeline: to build the vocab of the document
    - We never collected the dataset we were building
    - The code actually processes the data when we save the last lazy frame (lf_conv) to parquet
    - Polars can see the full query plan, from the moment duckdb returns a LazyFrame to the last step that creates lf_conv. It can optimize it, don't materialize the intermediate lazyframe defined (like lf_mess) for maximum performances
    - We took full advantages of the query engine, lazy excecution and streaming engine
    - Most of the code executes in milliseconds: Lazyframes are just query plans and use basically no RAM. The parts of the code that take time are when data is loaded in the RAM, for the **stopwords**, to build the **vocab**, to save the data to a parquet file.
    - The entire notebook executes in **1m29** (1m06 for the **sink_parquet()**) *if the stopwords are already downloaded on disk from nltk*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```shell
    time uv run scripts/nb_prep_clean.py

    ✅ Saved vocab to: data/processed/vocab.json
    ✅ Saved lf_conv to: data/clean/df_conv.parquet
    uv run scripts/nb_prep_clean.py  167.72s user 42.57s system 236% cpu 1:28.82 total
    ```
    """)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data cleaning

    Cleaning the content means removing elements that don't have a semantic meaning or one with poor semantic value. We want to keep only the informative elements and remove content that could reduce the quality of the clustering.
    We want to clean the data to optimize the expensive computational steps (Tokenizations, Lemmatization)

    Clean messages: (here the content is type **str**)
    - remove emails, url and punctuation
    - remove stopwords from the NLTK list of stopwords
    - tokenize

    Clean tokens: (here the content is type **list[str]**)
    - remove empty tokens
    - remove numbers, email addresses, URLs, punctuation
    - remove single characters tokens
    - lemmatization (done last to be done on fewer tokens as possible, because takes longer)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Messages cleaning
    """)
    return


@app.cell
def _(pl):
    def remove_nolang_messages(lf_mess: pl.LazyFrame) -> pl.LazyFrame:
        """
        1. Removes first turns (user + assistant) where user message is empty (Nolang)
        2. Removes any other remaining Nolang messages
        """
        first_turn_empty = (
            lf_mess.filter(
                (pl.col("turn") == 1) & (pl.col("language_message") == "Nolang")
            )
            .select(["conversation_id", "turn_identifier"])
            .unique()
        )

        if first_turn_empty.head(1).collect().height > 0:
            lf_mess = lf_mess.join(  # 1.
                first_turn_empty,
                on=["conversation_id", "turn_identifier"],
                how="anti",
            )

        return lf_mess.remove(pl.col("language_message") == "Nolang")  # 2.

    return (remove_nolang_messages,)


@app.cell
def _(pl):
    def remove_non_english_mess(lf_mess: pl.LazyFrame) -> pl.LazyFrame:
        """Keep only messages that are English or Latin"""
        return lf_mess.filter(
            (pl.col("language_message") == "English")
            | (pl.col("language_message") == "Latin")
        )

    return (remove_non_english_mess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conversations cleaning
    """)
    return


@app.cell
def _(stopwords):
    # get stopwords list for nltk
    def get_stopwords_pattern() -> str:
        """
        Returns regex pattern of nltk's english stopwords.

        To customize:
        - Add or remove words from `STOP_WORDS` with a dict
        - To exclude words, replace '= set()' by a dict like so:
            REMOVE_FROM_STOPWORDS = {"i", "who",...}
        """
        STOP_WORDS = set(stopwords.words("english"))

        REMOVE_FROM_STOPWORDS = set()
        ADD_TO_STOPWORDS = set()
        STOP_WORDS.update(ADD_TO_STOPWORDS)  # add words
        STOP_WORDS.difference_update(REMOVE_FROM_STOPWORDS)  # remove words

        stop_pattern = "|".join(f"\\b{word}\\b" for word in STOP_WORDS)

        return stop_pattern

    return (get_stopwords_pattern,)


@app.cell
def clean_messages(pl):
    def clean_and_tokenize(
        lf_conv: pl.LazyFrame, columns: str | list[str], stop_pattern: str
    ) -> pl.LazyFrame:
        """Clean and tokenize the aggregated messages"""

        if isinstance(columns, str):  # single string to list for consistency
            columns = [columns]

        return (
            lf_conv.with_columns(
                pl.col(columns)
                .str.to_lowercase()
                .str.replace_all(r"https?://\S+|www\.\S+", "")  # remove URLs
                .str.replace_all(r"\S+@\S+", "")  # remove emails
                .str.replace_all(r"[[:punct:]]", "")  # remove punctuation
                .str.replace_all(stop_pattern, "")  # remove stop words
                .str.replace_all(r"\d+", "")  # remove numbers
                .str.replace_all(
                    r"[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]",
                    "",
                )  # remove non latin letters (breakdown below)
                .str.replace_all(r"\s+", " ")  # many whitespaces to a single one
                .str.strip_chars()
                .str.split(" ")  # tokenize
                .list.eval(
                    pl.element()
                    .filter(pl.element() != "")
                    .filter(pl.element().str.len_chars() > 1)
                )  # remove empty tokens and single characters tokens
                .name.suffix("_tokens")
            ).drop_nulls(columns)  # remove rows with empty messages
        )

    # [^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]
    return (clean_and_tokenize,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]**

    This pattern removes:
    - Greek letters (α, β, γ)
    - Cyrillic (а, б, в)
    - Chinese/Japanese/Korean characters
    - Arabic, Hebrew, Thai, etc.

    While keeping:

    - All basic Latin letters and extended Latin with diacritics
    - Spaces, punctuation, numbers (they're in the ASCII range)

    1. **\x00-\x7F** (0-127): Basic ASCII
    Includes A-Z, a-z, digits, punctuation, spaces, control characters

    2. **\x80-\xFF** (128-255): Latin-1 Supplement
    Includes: é, ñ, ü, ç, à, ô, etc.
    Common accented characters in Western European languages

    3. **\u0100-\u017F**: Latin Extended-A
    Includes: ā, ē, ī, ő, ű, ł, ń, ś, etc.
    Used in Central/Eastern European languages (Polish, Czech, etc.)

    4. **\u0180-\u024F**: Latin Extended-B
    Includes: ƒ, ǎ, ǐ, ǒ, ǔ, Ș, Ț, etc.
    Less common Latin characters, some African languages

    5. **\u1E00-\u1EFF**: Latin Extended Additional
    Includes: Ṁ, ṁ, Ẁ, ẁ, Ỳ, ỳ, Vietnamese characters, etc.
    Specialized diacritics and Vietnamese
    """)
    return


@app.cell
def _(lemmatize, pl):
    def get_lemma_dict_and_vocab(
        lf_conv: pl.LazyFrame, token_vars: str | list[str]
    ) -> tuple[dict, list]:
        """
        lemmatize the tokens
        - use a dict to maps each unique tokens to lemmatizer equivalent
        - lemmatize all tokens at once with polars list.eval(). Super fast
        """

        if isinstance(token_vars, str):  # convert to list[str] for consistency
            token_vars = [token_vars]

        # ! memory intensive action. The whole data pass through here
        unique_tokens = (
            lf_conv.select(pl.col("first_user_content_tokens").list.explode())
            .unique()
            .drop_nulls()
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )

        lemma_dict = {token: lemmatize(token, lang="en") for token in unique_tokens}
        vocab = list(set(lemma_dict.values()))

        return lemma_dict, vocab

    def fast_lemmatize(
        lf_conv: pl.LazyFrame, token_vars: str | list[str], lemma_dict: dict
    ) -> pl.lazyframe:
        """lemmatize the tokens using a dictionnary and list.eval(). Super fast"""

        if isinstance(token_vars, str):  # convert str to list[str] for consistency
            token_vars = [token_vars]

        return lf_conv.with_columns(
            pl.col(token_vars).list.eval(
                pl.element().replace_strict(lemma_dict, default=pl.element())
            )
        )

    # .list.eval() has parallel processing across the rows AND within each row, the modifications are applied to a batch of elements. It's why it's really efficient and fast
    return fast_lemmatize, get_lemma_dict_and_vocab


@app.cell
def _(clean_and_tokenize, fast_lemmatize, get_lemma_dict_and_vocab, pl):
    # Wrapped the 3 functions that cleans lf_conv into one to reduce intermediates and for ease of use
    def process_message_tokens(
        lf_conv_init: pl.LazyFrame,
        mess_cols: str | list[str],
        stop_pattern: str,
    ) -> tuple[pl.LazyFrame, list]:
        """
        from lf_conv_init to 'lf_conv', modifies the messages content only:
        - clean the messages content and tokenize
        - get a list of the unique tokens -> vocab
        - lemmatize the tokens
        """

        if isinstance(mess_cols, str):  # convert to list[str] for consistency
            mess_cols = [mess_cols]

        lf_conv_clean = clean_and_tokenize(lf_conv_init, mess_cols, stop_pattern)

        # functions below works with tokenize version of the input cols, quick update to 'mess_cols' var names
        token_cols = [f"{col}_tokens" for col in mess_cols]

        lemma_dict, vocab = get_lemma_dict_and_vocab(lf_conv_clean, token_cols)
        lf_conv_lema = fast_lemmatize(lf_conv_clean, token_cols, lemma_dict)

        return lf_conv_lema, vocab

    return (process_message_tokens,)


if __name__ == "__main__":
    app.run()
