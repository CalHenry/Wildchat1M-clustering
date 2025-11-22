import duckdb
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt

con = duckdb.connect()

#duckdb.read_parquet("data/raw/train-00000-of-00014.parquet")


# See what's in one file
con.execute("DESCRIBE SELECT * FROM 'data/raw/train-00000-of-00014.parquet'").pl()

# Query all files together (assuming they have the same structure)
con.execute("SELECT * FROM 'data/raw/*.parquet' LIMIT 10").pl()

# Get row counts
con.execute("SELECT COUNT(*) FROM 'data/raw/*.parquet'").pl()

# Explore columns
con.execute("SELECT header FROM 'data/raw/train-00000-of-00014.parquet' LIMIT 5").pl()


# combine into a single parquet file
con.execute("""
    COPY (SELECT * FROM 'data/raw/*.parquet')
    TO 'data/processedcombined_data.parquet' (FORMAT PARQUET)
""")

# Better to keep the data in separate files for quick summary and exploration (ddb parallel processing)
con.execute("SUMMARIZE SELECT * FROM 'data/raw/*.parquet'").pl()

# Messages over time
con.execute("""
    SELECT 
        DATE_TRUNC('day', timestamp) as date,
        COUNT(*) as messages
    FROM 'data/raw/*.parquet'
    GROUP BY date
    ORDER BY date
""").pl()

con.execute("""
    SELECT 
        EXTRACT(HOUR FROM timestamp) as hour_of_day,
        COUNT(*) as message_count
    FROM 'data/raw/*.parquet'
    GROUP BY hour_of_day
    ORDER BY hour_of_day
""").pl()

# Day of week patterns
con.execute("""
    SELECT 
        DAYNAME(timestamp) as day_of_week,
        COUNT(*) as messages
    FROM 'data/raw/*.parquet'
    GROUP BY day_of_week
    ORDER BY DAYOFWEEK(timestamp)
""").pl()

# Monthly trends
con.execute("""
    SELECT 
        DATE_TRUNC('month', timestamp) as month,
        COUNT(DISTINCT conversation_id) as conversations,
        COUNT(*) as messages
    FROM 'data/raw/*.parquet'
    GROUP BY month
    ORDER BY month
""").pl()

con.execute("""
    SELECT 
        conversation_hash,
        COUNT(*) as turns,
        language,
        MIN(timestamp) as started_at
    FROM 'data/raw/*.parquet'
    GROUP BY conversation_hash, language
    ORDER BY turns DESC
    LIMIT 10
""").pl()

#### first visu

messages_overtime = con.execute("""
    SELECT 
        EXTRACT(HOUR FROM timestamp) as hour_of_day,
        COUNT(*) as message_count
    FROM 'data/raw/*.parquet'
    GROUP BY hour_of_day
    ORDER BY hour_of_day
""").pl()

sns.lineplot(
    data=messages_overtime,
    x="hour_of_day",
    y="message_count"
)
plt.title("ChatGPT Messages Over Time")
plt.tight_layout()
plt.show()


con.execute("""
    SELECT
        EXTRACT(HOUR FROM timestamp) as hour_of_day,
        COUNT(*) as message_count
    FROM 'data/raw/*.parquet'
    GROUP BY hour_of_day
    ORDER BY hour_of_day
""").pl().plot.line(
    x="hour_of_day",
    y="message_count",
    figsize=(8, 4),
    title="ChatGPT Messages Over Time"
)
plt.tight_layout()
plt.show()
