

# import

    import pandas as pd
    import numpy as np

# Quick data inspection

    df.head()
    df.tail()
    df.shape
    df.columns
    df.dtypes
    df.isna().sum()
    df.describe()
    df = df.drop_duplicates()

# Dedupe on keys (keep first/last)
    df = df.sort_values("date").drop_duplicates(["customer_id"], keep="last")
    df = df.drop_duplicates(["transaction_id"], keep="first")


# Datetime

    # Convert
    df["date"] = pd.to_datetime(df["date"])

    # Get day, year, month, DoW
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dow"] = df["date"].dt.dayofweek

    # Date Difference
    df["days_since"] = (df["date"] - df["date"].min()).dt.days

# Type conversion + Missing Value

    # Set to numerical
    df["dollars"] = df["dollars"].astype(float)
    df["tenure"] = df["tenure"].astype(int)

    # Fill missing
    df["dollars"] = df["dollars"].fillna(0)
    df["industry"] = df["industry"].fillna("Unknown")

    # Drop missing
    df = df.dropna(subset=["customer_id","date"])


# Column manipulation: Select / Create / Rename / Drop

    #select
    df2 = df[["a","b","c"]]
    df2 = df.loc[:, ["a","b","c"]]

    #Create/ Over write columne
    df["x"] = df["a"] + df["b"]
    df["flag"] = (df["dollars"] >= 1000).astype(int)

    # Create column with other column's condition
    df["same_industry"] = (df["industry_l"] == df["industry_r"]).astype(int)
    df["mismatch_tier"] = (df["tier_l"] != df["tier_r"]).astype(int)=

    #Rename col
    df = df.rename(columns={"old":"new"})

    #Drop Col
    df = df.drop(columns=["col_to_drop"])

# Filtering - Rows

    df2 = df[df["dollars"] >= 1000]

    df2 = df[(df["industry"]=="Tech") & (df["pricing_tier"]=="A")]
    df2 = df[(df["dollars"]>0) | (df["tenure"]>=12)]

    #is in / not in
    df2 = df[df["industry"].isin(["Tech","Retail"])]
    df2 = df[~df["industry"].isin(["Unknown"])]

    # Between
    df2 = df[df["dollars"].between(100, 500)]

    # Sort
    df = df.sort_values(["customer_id","date"])
    df = df.sort_values("dollars", ascending=False)

# Groupby, Aggregations

    df.groupby("customer_id")["dollars"].sum()
    df.groupby("customer_id")["transaction_id"].count()
    df.groupby("week_start")["customer_id"].nunique()

    # multi-key groupby
    out = df.groupby(["year","week_start","customer_id"])["dollars"].sum().reset_index()

    # multiple type of aggregation
    out = df.groupby("customer_id").agg(
        total_spend=("dollars","sum"),
        avg_spend=("dollars","mean"),
        n_txn=("transaction_id","count"),
        last_date=("date","max")
    ).reset_index()

    # Group size (count rows)
    out = df.groupby(["week_start"]).size().reset_index(name="n_rows")

    # Value count per group
    out = df.groupby("industry")["pricing_tier"].value_counts().reset_index(name="cnt")

    # Add group mean or sum directly to each row
    df["cust_total_spend"] = df.groupby("customer_id")["dollars"].transform("sum")
    df["cust_avg_spend"] = df.groupby("customer_id")["dollars"].transform("mean")

    # Ratio inside each group added to each row
    df["share_in_week"] = df["dollars"] / df.groupby("week_start")["dollars"].transform("sum")

# Window Functions (Sql -> Pandas)

    # Row number
    df = df.sort_values(["customer_id","date"])
    df["row_number"] = df.groupby("customer_id").cumcount() + 1

    # Running sum / cumulative count
    df = df.sort_values(["customer_id","date"])
    df["running_spend"] = df.groupby("customer_id")["dollars"].cumsum()
    df["running_cnt"] = df.groupby("customer_id").cumcount() + 1

    # Lag / lead
    df = df.sort_values(["customer_id","date"])
    df["prev_dollars"] = df.groupby("customer_id")["dollars"].shift(1)
    df["next_dollars"] = df.groupby("customer_id")["dollars"].shift(-1)

    # Cumulative within year (YTD-style)
    df["year"] = df["date"].dt.year
    df = df.sort_values(["customer_id","year","date"])
    df["ytd_spend"] = df.groupby(["customer_id","year"])["dollars"].cumsum()

    # Rolling window per group (fixed window size by rows)
    df = df.sort_values(["customer_id","date"])
    df["roll_7_txn"] = (
        df.groupby("customer_id")["dollars"]
        .rolling(7).sum()
        .reset_index(level=0, drop=True)
        )
# Join / Merge and Union

    # Basic Joins
    df3 = df1.merge(df2, on="customer_id", how="left")
    df3 = df1.merge(df2, on=["a","b"], how="inner")

    # Keep only matches / non-matches
    matched = df1.merge(df2[["customer_id"]], on="customer_id", how="inner")
    only_df1 = df1[~df1["customer_id"].isin(df2["customer_id"])]

    # Cross join (pairing / similarity / combinations)
    pairs = left_df.merge(right_df, how="cross", suffixes=("_l","_r"))

    # Concatenate (union all)
    df_all = pd.concat([df1, df2], ignore_index=True)

# One-hot Encoding

    df_enc = pd.get_dummies(df, columns=["industry","pricing_tier"])
