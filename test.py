# %%
import pandas as pd
import numpy as np
from IPython.display import display
data = [['tom', 10], ['nick', 15], ['juli', 14]] 
  
# Create the pandas DataFrame 
df = pd.DataFrame(data, columns = ['Name', 'Age'])

display(df[-1:])

# %%
