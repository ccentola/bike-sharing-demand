import os
from typing import List
import zipfile
import numpy as np
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

RAW_DIR_PATH = "/Users/ccentola/Documents/projects/bike-sharing-demand/src/data/raw"
PROCESSED_DIR_PATH = "/Users/ccentola/Documents/projects/bike-sharing-demand/src/data/processed"

# get data from kaggle if not currently in directory
def get_competition_data():
    """
    uses the Kaggle API to download competition files
    """
    # instantiate and authenticate
    api = KaggleApi()
    api.authenticate()

    # check if we already downloaded
    if len(os.listdir(RAW_DIR_PATH)) == 0:
        api.competition_download_files("bike-sharing-demand", path=RAW_DIR_PATH)

        # unzip file
        with zipfile.ZipFile(f"{RAW_DIR_PATH}/bike-sharing-demand.zip", "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(f"{RAW_DIR_PATH}/bike-sharing-demand.zip"))

        os.remove(f"{RAW_DIR_PATH}/bike-sharing-demand.zip")
    else:
        print("files already downloaded.")


# check for nulls
def has_nulls(df: pd.DataFrame) -> bool:
    return df.isnull().any()


# identify and drop independent vars missing from test set
def diff_features(
    df_train: pd.DataFrame, df_test: pd.DataFrame, y: str
) -> pd.DataFrame:
    """
    remove features in our training set that are missing from our test set
    """
    missing_features = df_train.columns.difference(df_test.columns)
    df_train.drop([x for x in missing_features if y not in x], axis=1, inplace=True)
    return df_train


# convert features to categorical
def make_categorical(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    for f in features:
        df[f] = df[f].astype("category")
    return df


# breakout datetime
def datetime_breakout(df: pd.DataFrame, dt_feature: str) -> pd.DataFrame:
    df["day"] = df[dt_feature].dt.day
    df["month"] = df[dt_feature].dt.month
    df["hour"] = df[dt_feature].dt.hour
    df["year"] = df[dt_feature].dt.year
    df["day_of_week"] = df[dt_feature].dt.day_of_week
    return df


# define and implement 'is_weekday' and 'demand' algorithms
# 'is_weekday' feature engineering
def is_weekday(df: pd.DataFrame, dt_feature: str) -> pd.DataFrame:
    """
    classifies 'day of week' attribute of a datetime as weekday: 1 or weekend: 0
    """
    df["is_weekday"] = (df[dt_feature].dt.day_of_week).astype(int)
    # df["is_weekday"] = df["is_weekday"].astype("category")
    return df

# classify demand
def categorize_demand(df: pd.DataFrame) -> pd.DataFrame:
    """
    categorize demand based on day of week and time of day
    """
    conditions = [
        (df["hour"] <= 6),  # 0
        (df["hour"] > 6) & (df["hour"] <= 9) & (df["is_weekday"] == 1),  # 3
        (df["hour"] > 6) & (df["hour"] <= 9) & (df["is_weekday"] == 0),  # 1
        (df["hour"] > 9) & (df["hour"] <= 15) & (df["is_weekday"] == 1),  # 1,
        (df["hour"] > 9) & (df["hour"] <= 15) & (df["is_weekday"] == 0),  # 2,
        (df["hour"] > 15) & (df["hour"] <= 19) & (df["is_weekday"] == 1),  # 3
        (df["hour"] > 15) & (df["hour"] <= 19) & (df["is_weekday"] == 0),  # 1
        (df["hour"] > 19) & (df["hour"] <= 21) & (df["is_weekday"] == 1),  # 2 ,
        (df["hour"] > 19) & (df["hour"] <= 21) & (df["is_weekday"] == 0),  # 1 ,
        (df["hour"] > 21),  # 1
    ]
    # create a list of the values we want to assign for each condition
    values = [0, 3, 1, 1, 2, 3, 1, 2, 1, 1]

    # create a new column and use np.select to assign values to it using our lists as arguments
    df["demand"] = np.select(conditions, values)

    return df


# output preprocessed training and testing data to 'processed' folder
def main() -> None:
    """
    run preprocessing steps on data
    """
    # get data from api
    get_competition_data()

    # load data
    train = pd.read_csv(f"{RAW_DIR_PATH}/train.csv", parse_dates=["datetime"])
    test = pd.read_csv(f"{RAW_DIR_PATH}/test.csv", parse_dates=["datetime"])

    # check for nulls
    has_nulls(train)
    has_nulls(test)

    # breakout datetime
    train = datetime_breakout(train, "datetime")
    test = datetime_breakout(test, "datetime")

    train = diff_features(train, test, "count")

    train = is_weekday(train, "datetime")
    test = is_weekday(test, "datetime")

    train = categorize_demand(train)
    test = categorize_demand(test)

     # convert features to categorical
    make_categorical(train, ["season", "weather", 'is_weekday', 'demand'])
    make_categorical(test, ["season", "weather", 'is_weekday', 'demand'])

    # write data to processed folder
    train.to_csv(f'{PROCESSED_DIR_PATH}/train_preprocessed.csv', index=False)
    test.to_csv(f'{PROCESSED_DIR_PATH}/test_preprocessed.csv', index=False)


if __name__ == "__main__":
    main()
    
