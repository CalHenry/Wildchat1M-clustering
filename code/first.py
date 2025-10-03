import duckdb


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
