
from PCA_and_ETC import *

df=pd.read_csv('../files/firm_list.csv')



df=pd.DataFrame(df.drop_duplicates())
df = df.reset_index(drop=True)

df.to_csv('../files/firm_lists.csv')

print(df)