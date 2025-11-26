import marimo

__generated_with = "0.17.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    import numpy as np
    from simplemma import lemmatize
    return duckdb, lemmatize, mo, pl


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
    ## Data preparation

    ### A word on the libraries used:

    **Duckdb** is great and **Polars** is awsome.

    I could use Duckdb only and write SQL queries but I am more confortable with **Polars's** synthax for complex and chainned modifications to dataframes.
    They both have lazy excecution and are very fast.

    I can use:
    - SQL at the begining, to import data or a filtered version from raw files. Simple queries can do a lot.
    - Polars for the complex manipulations in the workflow.
    - SQL and Duckdb for big aggregations and joins.

    Benefits of using duckdb in my workflow:
    - reading data is extremly fast (although polars is very fast as well)
    - can import in a  super efficient way with just a simple SQL query
    - SQL relations are similar to polars's expressions and can be very handy
    - SQL views are also very handy


    ➡️ Both libraries are interchangeable here and I can choose the one I prefer for each part of my work and keep a seamless workflow.
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
    I used both libraries in the same sequence.

    > I'll use the abbreviation '**lf**' for lazyframes to differentiate with dataframes
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
    2. **Conversations** level: identified by **'conversation_hash'**. *Uniquely* identified by **'conversation_hash'** + **'hashed_ip'**. ('conversation_hash' is not a unique identifier. 2 identical conversations will have the hash regarless the user, messages inside the conv will always have unique id)
    3. **Message** level: identified **'turn_identifier'**

    1 user can have N conversations of N messages.
    1 message belong to 1 conversation that belong to 1 user

    - Messages works in pairs: one **user** message, one **assistant** response. **turn_identifier** + **role** give the order of the messages. The **user** always comes before the **assistant**.
    - **'conversations'** is a complex variable of **List[structs]**. Each element of the list is 1 message. The fields of the **struct** are the message's content and it's metadata. Some fields are specific to the user, others to the assistant (expected missing values).

    ### Tidy: manipulation, variable selection

    We want to be at the **conversation level**, but we don't want a complex schema of **List[structs]** inside a single variable like we have in the raw dataset (the list can be +100 elements for long conversations).
    Logic:
    - flatten the dataset to message level, each row is a unique message, n rows for the same conversation.
    - aggregate back to conversation level by creating features around the messages. We keep metadata variables for the conversations and not the messages.
    - Final tidy dataset at the conversation level with several variables for different aggregations of the  messages's content (see section for details).


    ### Order of the tasks:

    1. Import data, shortlist of variables, filter only English conversations
    2. User level --> Message level (flatten the dataset)
    3. Unique ID for conversations
    4. Message level --> Conversation level
    """)
    return


@app.cell
def _(lf):
    # Same variables names in the nested vars of 'conversation'. We have to rename them before unnesting
    # We do it his way because we want to keep the names on the front vars and remove some nested vars bc their info is already in the front vars

    conflicting_cols = ["language", "country", "state", "hashed_ip", "header"]

    struct_fields = (
        lf.select("conversation")
        .head(0)  # don't care about the data, only the var names
        .explode("conversation")
        .collect()  # need eager to extract the list
        .to_series()
        .struct.fields
    )

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


    # Flatten the data (explode nested variables)
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
            .unnest("conversation", "header")  # unnest the struct (now flat)
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

    # Clean messages of lf_mess_lvl_raw)
    lf_mess = lf_mess_raw.pipe(remove_nolang_messages).pipe(
        remove_non_english_mess
    )
    return (lf_mess,)


@app.cell
def _(get_stopwords_pattern, lf_mess, pl, process_message_tokens):
    # From the flat message dataset, we can aggregate to conversation level
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

    # We can now clean and tokenize the messages
    # get stopwords from NLTK as regex string
    stop_pattern = get_stopwords_pattern()

    # Clean aggregated messages of lf_conv_lvl_init
    lf_conv, vocab = process_message_tokens(
        lf_conv_init, "first_user_content", stop_pattern
    )
    return (lf_conv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    Takeway:

    3 levels in the data:
    - User (top level, raw data)
    - Conversation (user can have [1:n] conversations)
    - Message

    Most conversations are lenght 2 (1 user prompt, 1 assistant response). If the conversation is longer, it's most common to have a 6 turns conversations.

    We focus on the english conversations and english messages.
    Some cleanings had to be done at the message level, but most of them are done at the conversation because it's faster.
    We designed some functions (lemmatization) and choose some libraries over others to have better performances (**simplemma** over **spacy** <-> speed over quality)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(tokens):
    # save vocab to disk

    with open("data/processed/vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(tokens))
    print("✅ Saved vocab to data/processed/vocab.txt")
    return


@app.cell
def _(lf_conv):
    # save to parquet
    lf_conv.sink_parquet("data/clean/df_conv.parquet")
    print("✅ Saved to to data/clean/df_conv.parquet")
    return


@app.cell
def _():
    # lf_conv.select(pl.col(['conversation_id', 'first_user_content', 'first_user_content_tokens'])).sink_parquet("data/clean/df_conv_fuc.parquet")
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data cleaning

    Now that we have prepare the dataset by moving variables around, we can dig into the content and prepare the messages.

    The idea is to remove any content that would worsen the clustering (REFAIREPHRASE) and increase the computation time.
    We want to clean the data to optimize the computationnal steps (Tokenizations, Lemmatization)

    Clean messages: (here the rows are of type **str**)
    - remove emails, url and punctuation
    - remove stopwords from the NLTK list of stopwords
    - tokenize

    Clean tokens: (here the rows are of type **list[str]**)
    - remove empty tokens
    - normalize numbers
    - remove single characters tokens except a shortlist of word (like pronouns '**I**')
    - maybe special characters if some found
    - keep programming tokens ? (x, y , i)

    - lemmatization (done last because takes longer, so done on fewer tokens as possible)
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


@app.cell(hide_code=True)
def fast_lemmatize(mo):
    mo.md(r"""
    def _fast_lemmatize(
        lf_conv: pl.LazyFrame, token_vars: str | list[str]
    ) -> pl.LazyFrame:
        "\"\"
        lemmatize the tokens
        - use dict to maps each unique tokens to lemmatizer equivalent
        - lemmatize all tokens at once with polars list.eval. Super fast
        "\"\"

        if isinstance(token_vars, str):  # convert to list[str] for consistency
            token_vars = [token_vars]

        unique_tokens = (
            lf_conv.select(pl.col("first_user_content_tokens").list.explode())
            .unique()
            .drop_nulls()
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )

        lemma_dict = {
            token: lemmatize(token, lang="en") for token in unique_tokens
        }
        vocab = list(lemma_dict.values())

        return lf_conv.with_columns(
            pl.col(token_vars).list.eval(
                pl.element().replace_strict(lemma_dict, default=pl.element())
            )
        ), unique_tokens
    """)
    return


@app.cell
def _(lemmatize, pl):
    def get_lemma_dict_and_vocab(
        lf_conv: pl.LazyFrame, token_vars: str | list[str]
    ) -> tuple[dict, list]:
        """
        lemmatize the tokens
        - use dict to maps each unique tokens to lemmatizer equivalent
        - lemmatize all tokens at once with polars list.eval(). Super fast
        """

        if isinstance(token_vars, str):  # convert to list[str] for consistency
            token_vars = [token_vars]
        """ v1
        unique_tokens = (
            lf_conv.select(pl.col("first_user_content_tokens"))
            .collect(engine="streaming")
            .select(pl.col("first_user_content_tokens").list.explode().unique())
            .drop_nulls()
            .to_series()
            .to_list()
        )
        """
        # v2
        unique_tokens = (
            lf_conv.select(pl.col("first_user_content_tokens").list.explode())
            .unique()
            .drop_nulls()
            .collect(engine="streaming")
            .to_series()
            .to_list()
        )

        lemma_dict = {
            token: lemmatize(token, lang="en") for token in unique_tokens
        }
        vocab = list(lemma_dict.keys())  # keys are unique, values arn't

        return lemma_dict, vocab


    def fast_lemmatize(
        lf_conv: pl.lazyframe, token_vars: str | list[str], lemma_dict: dict
    ) -> pl.lazyframe:
        """lemmatize the tokens using a dictionnary and list.eval(). Super fast"""

        if isinstance(token_vars, str):  # convert str to list[str] for consistency
            token_vars = [token_vars]

        return lf_conv.with_columns(
            pl.col(token_vars).list.eval(
                pl.element().replace_strict(lemma_dict, default=pl.element())
            )
        )
    return fast_lemmatize, get_lemma_dict_and_vocab


@app.cell
def _(clean_and_tokenize, fast_lemmatize, get_lemma_dict_and_vocab, pl):
    # Wrapped the functions into one to reduce intermediate lf overhead and for ease of use with the columns to process
    def process_message_tokens(
        lf_conv_init: pl.LazyFrame,
        mess_cols: str | list[str],
        stop_pattern: str,
    ) -> tuple[pl.LazyFrame, list]:
        """
        from init to final form of 'lf_conv', modifies the messages content only:
        - clean the messages content and tokenize
        - get a list of the unique tokens
        - lemmatize the tokens
        """

        if isinstance(mess_cols, str):  # convert to list[str] for consistency
            mess_cols = [mess_cols]

        lf_conv_clean = clean_and_tokenize(lf_conv_init, mess_cols, stop_pattern)

        # functions below works with tokenize version of the input cols, quick update to the var names
        token_cols = [f"{col}_tokens" for col in mess_cols]

        lemma_dict, vocab = get_lemma_dict_and_vocab(lf_conv_clean, token_cols)
        lf_conv_lema = fast_lemmatize(lf_conv_clean, token_cols, lemma_dict)

        return lf_conv_lema, vocab
    return (process_message_tokens,)


if __name__ == "__main__":
    app.run()
