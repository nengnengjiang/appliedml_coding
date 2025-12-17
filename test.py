import pandas as pd

def detect_high_engagement(df: pd.DataFrame, threshold: float = 2.0) -> pd.DataFrame:
    out = df.copy()
    out["view_timestamp"] = pd.to_datetime(out["view_timestamp"])

    # sort so "last 5 pages" is well-defined per user
    out = out.sort_values(["user_id", "view_timestamp"])

    # rolling mean of previous 5 pages (shift(1) excludes current page)
    prev5_avg = (
        out.groupby("user_id")["time_on_page_seconds"]
           .apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
           .reset_index(level=0, drop=True)
    )
    out["avg_time_last_5_pages"] = prev5_avg

    # if no history, avg_time_last_5_pages is NaN -> treat as not high engagement
    ratio = out["time_on_page_seconds"] / out["avg_time_last_5_pages"]
    out["is_high_engagement"] = ratio.ge(threshold) & out["avg_time_last_5_pages"].notna()

    return out



def build_engagement_bursts(df_with_flags: pd.DataFrame, gap_minutes: int = 30) -> pd.DataFrame:
    x = df_with_flags.copy()
    x["view_timestamp"] = pd.to_datetime(x["view_timestamp"])

    # keep only high engagement page views
    x = x[x["is_high_engagement"]].sort_values(["user_id", "view_timestamp"])
    if x.empty:
        return pd.DataFrame(columns=[
            "user_id", "burst_id", "start_time", "end_time",
            "duration_minutes", "max_time_on_page", "page_count"
        ])

    # compute gaps between consecutive high-engagement views per user
    prev_ts = x.groupby("user_id")["view_timestamp"].shift(1)
    gap = x["view_timestamp"] - prev_ts

    # new burst starts if first event OR gap > threshold
    new_burst = prev_ts.isna() | (gap > pd.Timedelta(minutes=gap_minutes))

    # burst_id increments within each user
    x["burst_id"] = new_burst.groupby(x["user_id"]).cumsum().astype(int)

    bursts = (
        x.groupby(["user_id", "burst_id"])
         .agg(
             start_time=("view_timestamp", "min"),
             end_time=("view_timestamp", "max"),
             max_time_on_page=("time_on_page_seconds", "max"),
             page_count=("page_id", "count"),
         )
         .reset_index()
    )

    bursts["duration_minutes"] = (
        (bursts["end_time"] - bursts["start_time"]).dt.total_seconds() / 60.0
    )

    return bursts[[
        "user_id", "burst_id", "start_time", "end_time",
        "duration_minutes", "max_time_on_page", "page_count"
    ]]
