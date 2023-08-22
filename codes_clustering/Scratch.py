import pandas as pd

asdf1 = pd.DataFrame([[1, 2, 0, 11], [30, 4, 20, float('NaN')], [18, 39, 11, 0]])
asdf1 = asdf1.set_index(0)
asdf1 = asdf1.rename(columns={1: 'col2', 2: 'col1', 3: 'col3'})
asdf2 = pd.DataFrame([[1, 2, 10, 11], [30, 4, 20, 21], [118, 319, 111, -10]])
asdf2 = asdf2.set_index(0)
asdf2 = asdf2.rename(columns={1: 'col1', 2: 'col4', 3: 'col3'})
ser1 = pd.Series([1, 2, 3, 4])
print(asdf1)
