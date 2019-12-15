# %%
import pandas as pd
from IPython.display import display
data = [['tom', 10], ['nick', 15], ['juli', 14]] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Name', 'Age'])

display(df.columns.values)

columns = np.array(['Age']) 

display(columns)
import numpy as np
np.intersect1d(df.columns.values, columns, assume_unique=True)