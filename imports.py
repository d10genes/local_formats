from collections import OrderedDict
from functools import partial
from importlib import reload
import itertools as it
import os
import shutil
import string
import uuid

import numpy.random as nr
import numpy as np
import bcolz
import fastparquet
import feather
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.compat import lmap
from pandas import pandas as pd, DataFrame
pd.options.display.notebook_repr_html = False
pd.options.display.width = 120
