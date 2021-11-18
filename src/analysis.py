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
