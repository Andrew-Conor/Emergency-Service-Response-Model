import pandas as pd


def match_data(sf_df, meteo_df):

    return pd.merge(
        sf_df,
        meteo_df,
        how="left",
        left_on=pd.to_datetime(sf_df["received_dttm"].round("h")),
        right_on="time",
    )
