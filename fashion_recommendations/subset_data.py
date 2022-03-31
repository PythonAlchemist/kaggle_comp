import dask.array as da
import pandas as pd
import numpy as np


customers = pd.read_csv('data/customers.csv').reset_index().set_index('customer_id')
articles = pd.read_csv('data/articles.csv').reset_index().set_index('article_id')
n_customers = len(customers.index)
n_articles = len(articles.index)
# sales = pd.read_csv('data/transactions_train.csv')
# cus_subset = customers.index.tolist()[:50000]
# customers = customers[customers.index.isin(cus_subset)]
# sales = sales[sales['customer_id'].isin(cus_subset)]

# customers.to_pickle('calculations/customers.pkl')
# articles.to_pickle('calculations/articles.pkl')
# sales.to_pickle('calculations/sales.pkl')

import csv
import scipy.sparse as sp

ct = 0
s = sp.dok_matrix((n_customers, n_articles), dtype=np.byte)
f = open('data/transactions_train.csv')
reader = csv.reader(f)
next(reader)
for line in reader:
    ct += 1
    if ct % 100_000 == 0:
        print(100 * ct/ 31e6)
    s[int(customers.loc[line[1]]['index']), int(articles.loc[int(line[2])]['index'])] = 1
s = s.tocoo()
sp.save_npz('calculations/full_coo.npz', s)
print(s)