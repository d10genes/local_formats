
# coding: utf-8

# In[ ]:

# import datetime as dt
import itertools as it

import numpy.random as nr
from pandas import pandas as pd, DataFrame
get_ipython().magic('matplotlib inline')
pd.options.display.notebook_repr_html = False
pd.options.display.width = 120


# # Load and set up df
# 
# tps = DataFrame({'Nu': dforig.apply(pd.Series.nunique), 'Tp': dforig.dtypes})

# In[ ]:

# http://archive.ics.uci.edu/ml/datasets/Census-Income+%28KDD%29
with open('data/census_col_name_desc_mod.txt', 'r') as f:
    csd = f.read().splitlines()


dforig = pd.read_csv('data/census-income.data', header=None).iloc[:, :-2]
dforig.columns = [l.split()[-1] for l in it.takewhile(bool, csd)]


# In[ ]:

# Get a few of each dtype
# ctypes = [c for dtp, gdf in dforig.dtypes.reset_index(drop=0).groupby(0) for c in gdf['index'][:4]]
ctypes = 'HHDFMX PEMNTVTY GRINST AREORGN MARSUPWT DIVVAL CAPLOSS'.split()
dfs = dforig[ctypes].copy()


# In[ ]:

get_dtypes(dfs, object)


# In[ ]:

# Nullify some elems from 3 of the object columns
nr.seed(0)
get_dtypes = lambda df, tp='category': df.columns[df.dtypes == tp]

for c in get_dtypes(dfs, object)[:-1]:
    rand_ixs = nr.choice(dfs.index, size=100, replace=False)
    dfs.loc[rand_ixs, c] = None


# dfs['HHDFMX'] = dfs['HHDFMX'].astype('category')

# In[ ]:

DataFrame({'Nulls': dfs.isnull().sum(), 'Dtypes': dfs.dtypes})


# In[ ]:




# In[ ]:

dforig[:2]

