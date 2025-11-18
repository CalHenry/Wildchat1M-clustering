import marimo

__generated_with = "0.17.7"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import duckdb
    import polars as pl
    return duckdb, mo, pl


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
    """)
    return


@app.cell
def _(duckdb):
    # Load data (lazy)
    df = (
        duckdb.query(
            """
        SELECT * EXCLUDE (toxic, timestamp, redacted, openai_moderation, detoxify_moderation)
        FROM 'data/raw/train-00000-of-00014.parquet' 
        WHERE language = 'English'
        """
        )
        .pl(lazy=True)
        .rename({"language": "lang_conv"})
    )
    return (df,)


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
def _(df):
    # Same variables names in the nested vars of 'conversation'. We have to rename them before unnesting
    # We do it his way because we want to keep the names on the front vars and remove some nested vars bc their info is already in the front vars

    conflicting_cols = ["language", "country", "state", "hashed_ip", "header"]

    struct_fields = (
        df.select("conversation")
        .explode("conversation")
        .head(0)  # don't care about the data, only the var names
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
def _(df, new_fields, pl, remove_nolang_messages):
    # no unique id for conversations so we use the hash + the ip to make one
    create_conversation_id = (
        pl.col("conversation_hash") + "_" + pl.col("hashed_ip")
    ).alias("conversation_id")


    # Flatten the data (explode nested variables)
    df_mess_lvl_raw = (
        (
            df.with_columns(["conversation_hash", "conversation"])
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

    # clean df_mess_lvl_raw
    df_mess_lvl = remove_nolang_messages(df_mess_lvl_raw)
    return (df_mess_lvl,)


@app.cell
def _(df_mess_lvl, pl):
    # From the flat dataset, we can aggregate to conversation level
    df_conv_lvl = df_mess_lvl.group_by("conversation_id").agg(
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
    return (df_conv_lvl,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(df, df_conv_lvl, df_mess_lvl, mo):
    mo.md(rf"""
    Takeway: 

    3 levels in the data:
    - User (top level, raw data)
    - Conversation (1 user can have [1:n] conversations)
    - Message

    Manipulation:   
    **User level** *{df.collect().height} rows* ➡️ **Message level** *{df_mess_lvl.collect().height} rows* ➡️ **Conversation level** *{df_conv_lvl.collect().height} rows*

    Most of the users have a single conversation. There is only 71 more conversations than users
    """)
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data cleaning

    Now that we have prepare the dataset by moving variables around, we can dig into the content and prepare the messages.

    The idea is to remove any content that would worsen the clustering (REFAIREPHRASE) and increase the computation time.
    We want to clean the data to optimize the computationnal steps (Tokenizations, Lemmatization)

    Specifically, we will clean the messages variables: (here the rows are of type **str**)
    - remove emails, url and punctuation
    - remove stopwords from the NLTK list of stopwords
    - tokenize

    Clean tokens: (here the rows are of type
    **list[str]**)
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
    def remove_nolang_messages(df: pl.LazyFrame) -> pl.LazyFrame:
        """
        1. Removes first turns (user + assistant) where user message is empty (Nolang)
        2. Removes any other remaining Nolang messages
        """
        first_turn_empty = (
            df.filter(
                (pl.col("turn") == 1) & (pl.col("language_message") == "Nolang")
            )
            .select(["conversation_id", "turn_identifier"])
            .unique()
        )

        if first_turn_empty.head(1).collect().height > 0:
            df = df.join(  # 1
                first_turn_empty,
                on=["conversation_id", "turn_identifier"],
                how="anti",
            )

        return df.remove(pl.col("language_message") == "Nolang")  # 2
    return (remove_nolang_messages,)


@app.cell
def _():
    # TOADD: function to remove non english messages
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Conversations cleaning
    """)
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Exploration

    Conversations are a set of messages between the user and the assistant (AI).
    What are baasic elements about the conversations and messages ?

    - Number of turns (1 turn is a set of 1 message from the user and 1 message (the response) from the assistant)
    - Number of message per conversations
    - user's messages to assistant's messages ratio
    - Lenght of messages/ conversation in tokens
    """)
    return


@app.cell
def _():
    import altair as alt
    return (alt,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import seaborn as sns
    return (plt,)


@app.cell
def _(alt, df_conv_lvl, mo):
    df_plots = df_conv_lvl.head(500).collect()  # 500 sample

    plot_nbr_turns = (
        alt.Chart(df_plots)
        .mark_bar()
        .encode(x=alt.X("turn:N", labelAngle=0), y=alt.Y("count()", title=""))
        .properties(title="Count of turns")
    )


    mo.ui.altair_chart(plot_nbr_turns)

    """ seaborn version of the plot
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_conv_lvl.collect(), x="turn")
    plt.xticks(rotation=0)  # Keep x-axis labels horizontal
    plt.title("Count of turns")
    plt.ylabel("Count")
    plt.xlabel("Turn")
    plt.show()
    """
    return


@app.cell
def _(alt, df_mess_lvl, mo):
    _test = (
        df_mess_lvl.collect()
        .head(100)
        .group_by("conversation_hash")
        .len(name="nbr_mess_per_conv")
    )


    nbr_mess_per_conv = (
        alt.Chart(_test)
        .mark_bar()
        .encode(
            x=alt.X("nbr_mess_per_conv:N", title="Number of messages"),
            y=alt.Y("count()", title="Number of conversations"),
        )
        .properties(title="Number of conversations with n messages")
    )

    mo.ui.altair_chart(nbr_mess_per_conv)


    """ seaborn version of the plot
    _test = (
        df_mess_lvl.collect()
        .head(100)
        .group_by("conversation_hash")
        .len(name="nbr_mess_per_conv")
    )

    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=_test,
        x='nbr_mess_per_conv',
        hue='nbr_mess_per_conv',
        palette='colorblind'
    )
    plt.title("Number of conversations with n messages")
    plt.xlabel("Number of messages")
    plt.ylabel("Number of conversations")
    plt.xticks(rotation=45)  # Rotate x-axis labels if needed
    plt.tight_layout()
    plt.show()
    """
    return


@app.cell
def _(alt, df_mess_lvl, mo):
    _test = df_mess_lvl.collect().sample(100).group_by("role").len()

    ratio_role = (
        alt.Chart(_test)
        .mark_arc(innerRadius=50)
        .encode(
            theta=alt.Theta("len:Q"),
            color=alt.Color(
                "role:N",
            ),
        )
        .properties(
            title="User vs Assistant Message Distribution", width=200, height=150
        )
    )


    _left_label = (
        ratio_role.transform_filter(alt.datum.role == "user")
        .mark_text(dx=-95, fontSize=30, fontWeight="bold")
        .encode(text="len:Q", color=alt.value("black"))
    )
    _right_label = (
        ratio_role.transform_filter(alt.datum.role == "assistant")
        .mark_text(dx=95, fontSize=30, fontWeight="bold")
        .encode(text="len:Q", color=alt.value("black"))
    )
    _percent_label = ratio_role.mark_text(
        fontSize=60,
        text="%",
    ).encode(color=alt.value("black"))

    mo.ui.altair_chart(ratio_role + _left_label + _right_label + _percent_label)
    return


@app.cell
def _(clean_messages, df_conv_lvl, pl, plt, stop_pattern):
    from wordcloud import WordCloud

    _df_wc = df_conv_lvl.head(500).collect().pipe(clean_messages, stop_pattern)


    text = _df_wc.select(
        pl.col("cleaned_content").list.join(" ").str.join(" ")
    ).item()


    wordcloud = WordCloud().generate(text)

    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    return


@app.cell(column=3, hide_code=True)
def _(mo):
    mo.md(r"""
    # Test zone
    """)
    return


@app.cell
def _(df):
    dff = df.head(30)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    """'conversation' schema:

    List(Struct({
    'content': String,
    'country': String,
    'hashed_ip': String,
    'header': Struct({
        'accept-language': String,
        'user-agent': String}),
    'language': String,
    'redacted': Boolean,
    'role': String,
    'state': String,
    'timestamp': Datetime(time_unit='us', time_zone='Europe/Paris'),
    'toxic': Boolean,
    'turn_identifier': Int64}))
    """
    return


@app.cell
def _():
    # how many non english messages ?
    # ~10% on file 0014
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _():
    import nltk

    try:
        nltk.data.find("corpora/stopwords")
        from nltk.corpus import stopwords
    except LookupError:
        nltk.download("stopwords")

    # corpus should be in "User/nltk_data/corpus"
    return (stopwords,)


@app.cell
def _(STOP_WORDS):
    STOP_WORDS
    return


@app.cell
def _(stopwords):
    STOP_WORDS = set(stopwords.words("english"))

    remove_from_stopwords = {"i", "who", 'how', 'you'}  # fmt: skip
    STOP_WORDS.difference_update(remove_from_stopwords)


    stop_pattern = "|".join(f"\\b{word}\\b" for word in STOP_WORDS)
    return STOP_WORDS, stop_pattern


@app.cell
def _(df_conv_lvl):
    dft = df_conv_lvl.head(100).collect()
    return


@app.cell
def _():
    # removed at least 1 message composed with stop words only ('how are you?')
    return


@app.cell
def _(pl):
    def clean_messages(lf: pl.LazyFrame, stop_pattern: str) -> pl.LazyFrame:
        return lf.with_columns(
            pl.col("first_user_content")
            .str.to_lowercase()
            .str.replace_all(r"https?://\S+|www\.\S+", "")  # remove URLs
            .str.replace_all(r"\S+@\S+", "")  # remove emails
            .str.replace_all(r"[[:punct:]]", "")  # remove punctuation
            .str.replace_all(stop_pattern, "")  # remove stop words
            .str.replace_all(r"\s+", " ")  # many whitespaces to a single one
            .str.strip_chars()
            .str.split(" ")  # tokenize
            .alias("cleaned_content")
        )
    return (clean_messages,)


@app.cell
def _(clean_messages, df_conv_lvl, stop_pattern):
    _df_tokens = clean_messages(df_conv_lvl, stop_pattern)
    return


@app.cell
def _(df_conv_lvl, pl, stop_pattern):
    def clean_messages_vars(
        lf: pl.LazyFrame, columns: str | list[str], stop_pattern: str
    ) -> pl.LazyFrame:
        """
        Clean text columns and tokenize
        """
        if isinstance(columns, str):
            columns = [columns]

        cleaned_cols = [
            pl.col(col)
            .str.to_lowercase()
            .str.replace_all(r"https?://\S+|www\.\S+", "")  # remove URLs
            .str.replace_all(r"\S+@\S+", "")  # remove emails
            .str.replace_all(r"[[:punct:]]", "")  # remove punctuation
            .str.replace_all(stop_pattern, "")  # remove stop words
            .str.replace_all(r"\s+", " ")  # normalize whitespace
            .str.strip_chars()
            .str.split(" ")  # tokenize
            for col in columns
        ]

        return lf.with_columns(cleaned_cols).rename(
            {col: f"{col}_tokens" for col in columns}
        )


    vars_to_tokenize = ["first_user_content"]
    df_tokens = clean_messages_vars(df_conv_lvl, vars_to_tokenize, stop_pattern)
    return (df_tokens,)


@app.cell
def _(df_conv_lvl, pl, stop_pattern):
    df_conv_lvl.with_columns(
        pl.col(["first_user_content", "all_user_content"])
        .str.to_lowercase()
        .str.replace_all(r"https?://\S+|www\.\S+", "")  # remove URLs
        .str.replace_all(r"\S+@\S+", "")  # remove emails
        .str.replace_all(r"[[:punct:]]", "")  # remove punctuation
        .str.replace_all(stop_pattern, "")  # remove stop words
        .str.replace_all(r"\s+", " ")  # many whitespaces to a single one
        .str.strip_chars()
        .str.split(" ")  # tokenize
    ).rename({"first_user_content": "first_user_content_tokens"})
    return


@app.cell
def _():
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    # nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "morphologizer"])
    return (nlp,)


@app.cell
def _(df_tokens):
    df_test = (
        df_tokens.head(1000).collect().lazy()
    )  # keep the same head content, otherwise randomly pulls
    return (df_test,)


@app.cell
def _(df, pl):
    empty_strings = df.filter(pl.col("text_column") == "")
    return


@app.cell
def _(df_tokens, pl):
    df_tokens.collect().with_columns(
        pl.col("first_user_content_tokens")
        .list.eval(pl.element().filter(pl.element() == ""))
        .list.len()
        .alias("empty_count")
    )
    return


@app.cell
def _(df_conv_lvl):
    df_conv_lvl.select("conversation_id", "first_user_content").collect()
    return


@app.cell
def _(df_test, nlp, pl):
    def lemmatize_tokens_batch(token_lists: list[list[str]]) -> list[list[str]]:
        """
        Lemmatize batches of token lists using spaCy's pipe.
        """

        texts = [" ".join(tokens) for tokens in token_lists]

        # docs = [Doc(nlp.vocab, words=token_list) for token_list in texts]

        docs = nlp.pipe(texts)

        # Extract lemmas
        lemmatized = [[token.lemma_ for token in doc] for doc in docs]

        return lemmatized


    # Apply with map_batches for parallel processing
    df_result = (
        df_test.with_columns(
            pl.col("first_user_content_tokens")
            .map_batches(
                lambda series: pl.Series(
                    [lemmatize_tokens_batch(series.to_list())][0]
                ),
                return_dtype=pl.List(pl.String),
            )
            .alias("lemmatized_tokens")
        )
    ).collect(engine="streaming")
    return (df_result,)


@app.cell
def _(df_result):
    df_result["conversation_id", "first_user_content_tokens", "lemmatized_tokens"]
    return


@app.cell
def _(df_tokens, pl):
    df_tokens.filter(
        pl.col("conversation_id")
        == "a5aae6199e4cbb97f020337bce9dbf66_ff02a68ec3eb845852b0d02da7403246f4ae93b9a12a0cde19304bd21337b358"
    )
    return


@app.cell
def _(df_test, pl):
    df_test.with_columns(pl.col("first_user_content_tokens")).collect_schema()
    return


@app.cell
def _(df_test, pl):
    # working nicely
    from simplemma import lemmatize


    def fast_lemmatize(token_lists: list[str]) -> list[str]:
        """Ultra-fast lemmatization without NLP pipeline."""
        return [lemmatize(token, lang="en") for token in token_lists]


    _df_result = df_test.with_columns(
        pl.col("first_user_content_tokens")
        .map_elements(fast_lemmatize, return_dtype=pl.List(pl.String))
        .alias("lemmatized_tokens")
    ).collect(engine="streaming")

    _df_result["first_user_content_tokens", "lemmatized_tokens"]
    return


@app.cell
def _(df_test, pl, simplemma):
    def _lemmatize_tokens_batch(token_lists: list[list[str]]) -> list[list[str]]:
        """
        Lemmatize batches of token lists using simplemma.
        """
        lemmatized = [
            [simplemma.lemmatize(token, lang="en") for token in token_lists]
            for token_list in token_lists
        ]
        return lemmatized


    # Apply with map_batches for parallel processing
    _df_result = df_test.with_columns(
        pl.col("first_user_content_tokens")
        .map_batches(
            lambda series: pl.Series(
                [_lemmatize_tokens_batch(series.to_list())][0]
            )
        )
        .alias("lemmatized_tokens")
    ).collect(engine="streaming")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


if __name__ == "__main__":
    app.run()
