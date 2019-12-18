# %%
import pandas as pd
import numpy as np
from IPython.display import display

res = pd.DataFrame([], columns=['Date', 'Close', 'Open', 'High'])
res = res.append({'Date' : 1, 'Close' : 2}, ignore_index=True)
res[['Open', 'High']][-1:] = np.repeat(999, 2)
res = res.append({'Date' : 1, 'Close' : 3}, ignore_index=True)
res[['Open', 'High']][-1:] = np.repeat(999, 2)
res = res.append({'Date' : 1, 'Close' : 4}, ignore_index=True)
res[['Open', 'High']][-1:] = np.repeat(999, 2)
res = res.append({'Date' : 1, 'Close' : 5}, ignore_index=True)
res[['Open', 'High']][-1:] = np.repeat(999, 2)
res = res.append({'Date' : 1, 'Close' : 6}, ignore_index=True)
res[['Open', 'High']][-1:] = np.repeat(999, 2)

display(res)
# %%
