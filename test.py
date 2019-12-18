# %%

import os
from IPython.core.display import display
file_list = []
for file in os.listdir("data"):
    if file.endswith(".csv"):
        file_list.append(file.partition('.')[0])

display(file_list)
# %%
