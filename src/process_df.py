# %% 
%load_ext autotime

import pandas as pd
import numpy as np
try:
    import pycaret
except:
    pass

asset_df = pd.read_csv('/media/gyasis/Drive 2/Data/crypto/asset_details.csv')
asset_df.head()


# %%
supplemental_df = pd.read_csv('/media/gyasis/Drive 2/Data/crypto/supplemental_train.csv')
supplemental_df.head(20)



# %%

# select new dataframe with assestid = 6 
s_df = supplemental_df[supplemental_df.Asset_ID == 6]

# select new dataframe with minute = 0,15,30,45
s_df = s_df[s_df.minute.isin([0,15,30,45])]
# %%
from pycaret.regression import *
grid = setup(supplemental_df,
             target = 'Target',
             verbose = True,
             silent = True,
             use_gpu=True)
# %%
print(supplemental_df.columns)
# %%
import datetime
ex = datetime.datetime.fromtimestamp(supplemental_df.timestamp[0])
print(ex)
# %%
def convert_timestame(df):
    df['timestamp'] = df.timestamp.apply(lambda x: datetime.datetime.fromtimestamp(x))
    return df
# %%
df = convert_timestame(supplemental_df)
# %%
df.head()
# %%
df.tail()
# %%
#return df row by index
df.loc[1813600]
# %%
#extract month, day, year, hour, minute from timestamp

def split_timestamp(df):
    df['month'] = df.timestamp.apply(lambda x: x.month)
    df['day'] = df.timestamp.apply(lambda x: x.day)
    df['weekday'] = df.timestamp.apply(lambda x: x.weekday())
    df['year'] = df.timestamp.apply(lambda x: x.year)
    df['hour'] = df.timestamp.apply(lambda x: x.hour)
    df['minute'] = df.timestamp.apply(lambda x: x.minute)
    return df
# %%
df = split_timestamp(df)
# %%
df.head()

x = df.loc[1813600]

# %%
# select new dataframe with minute = 0,15,30,45
s_df = s_df[s_df.minute.isin([0,15,30,45])]
# %%
x[0]
# %%
type(x[0])
# %%
train = df[df['timestamp'] < x[0]]
test = df[df['timestamp'] >= x[0]]
# %%
train.head()
# %%
test.head()
# %%
#extract weekday from timestamp
def split_timestamp(df):
    df['weekday'] = df.timestamp.apply(lambda x: x.weekday())
    return df

#onehot encode dataframe columne and return new dataframe

def onehot_encode(dataframe, column):
    dataframe = pd.get_dummies(dataframe, columns=[column])
    return dataframe

#change numbers to days of the week mapping
weekday_map = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}


def convert_weekday(df):
    df['weekday'] = df.weekday.apply(lambda x: weekday_map[x])
    return df

train = convert_weekday(df)
train = onehot_encode(train, 'weekday')
train.head()
# %%
#strip "weekday_" from column names of a dataframe
def strip_weekday(df):
    df.columns = df.columns.str.replace('weekday_', '')
    return df
# %%
# train = train.drop(columns=['Weekday'])
train = strip_weekday(train)
train.head()
# %%
# create timeseries plt with seaborn columns timestamp and open
def time_line_plot(df, x, y):
    import seaborn as sns
    import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(figsize=(15,5))
    sns.set(style="darkgrid")
    sns.lineplot(x=x, y=y, data=df)
    plt.show()
    return
# %%
# lets extract features using Kats
import numpy
from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures

test_TF = TimeSeriesData(s_df)
# %%
import kats
# %%
from numba import *
# %%
# %%
#pringt numba version
print(numba.__version__)
# %%
print(numpy.__version__)
pip in# %%
