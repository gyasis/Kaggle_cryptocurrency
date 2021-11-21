# look at database for competition 
# using kaggle to get database
# kaggle competitions download -c g-research-crypto-forecasting
# %%
#
import pandas as pd
import numpy as np

# %%
asset_df = pd.read_csv('/media/gyasis/Drive 2/Data/crypto/asset_details.csv')
# %%
asset_df.head()
# %%
asset_df.describe()
# %%
supplemental_df = pd.read_csv('/media/gyasis/Drive 2/Data/crypto/supplemental_train.csv')
# %%
supplemental_df.head(20)
# %%
#create pivot tables of supplemental_df by Asset_ID
def create_pivoted_df(dataframe, column_name):
        pivoted_df = dataframe.pivot_table(index=column_name, columns='Asset_ID', values='Close')
        return pivoted_df

# %%
import matplotlib.pyplot as plt
# %%
tree = create_pivoted_df(supplemental_df, supplemental_df.timestamp)
# %%
#create combined time series line plot of each asset from pivot table, each column is a seperate line  plot
def create_time_series_plot(dataframe):
        dataframe.plot()
        plt.show()
# %%
timeseries_plot(tree)
# %%
import seaborn as sns
sns.set_styple('whitegrid')

# %%
tree[0].head(1000).plot(legend=True, figsize=(12,5))
# %%
print(tree[1])
# %%
print(tree.columns)
# %%
test_df = supplemental_df[supplemental_df.Asset_ID == 3]
# %%
test_df.head()
# %%
# create filtered tables from Asset_ID set index to timestamp and create spaghetti plot for all Asset_ID
def create_filtered_df(dataframe, column_name, column_value):
        filtered_df = dataframe[dataframe[column_name] == column_value]
        filtered_df.set_index('timestamp', inplace=True)
        return filtered_df
    
#create spaghetti plot for all Asset_ID
def create_spaghetti_plot(dataframe):
        dataframe.plot()
        plt.show()
# %%
import seaborn as sns
# %%
sns.lineplot(x=supplemental_df.timestamp, y=supplemental_df.Open, hue=supplemental_df.Asset_ID)
# %%
