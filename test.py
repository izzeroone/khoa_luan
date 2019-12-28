# %%

import os
from IPython.core.display import display
import json
# a = {'a' : 1, 'n' : 2}

with open('data.txt', 'w') as outfile:
     json.dump(a, outfile)

with open('data.txt', 'r') as outfile:
    b = json.load(outfile)

display(b)

# %%
