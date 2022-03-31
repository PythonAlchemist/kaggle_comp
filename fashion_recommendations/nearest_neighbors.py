import numpy as np
import pandas as pd
import os
import time
import scipy.sparse
import dask
import dask.dataframe as dd
import dask.array as da

# customers = pd.read_csv('data/customers.csv').reset_index().set_index('customer_id')
# articles = pd.read_csv('data/articles.csv').reset_index().set_index('article_id')
# sales = pd.read_csv('data/transactions_train.csv').head(10_000)

customers = pd.read_pickle('calculations/customers.pkl')
articles = pd.read_pickle('calculations/articles.pkl')
sales = pd.read_pickle('calculations/sales.pkl').head(100_000)

n_customers = len(customers.index)
n_articles = len(articles.index)
n_sales = max(sales.index)
batch_size = 60_000

# purchase = da.zeros((n_customers, n_articles), dtype=np.byte, chunks=(n_customers, batch_size))
purchase = da.zeros((n_customers, n_articles), dtype=np.byte)

def update(x):
    cus_ix = int(customers.loc[x['customer_id']]['index'])
    art_ix = int(articles.loc[x['article_id']]['index'])
    purchase[cus_ix, art_ix] = 1
    if x.name % 10_000 == 0:
        print(f"Percent: {100*x.name/n_sales}")

sales.apply(lambda x: update(x), axis=1)

print("finished")

cus = int(customers.loc['000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318']['index'])
art = int(articles.loc[663713001]['index'])
print(cus, art, purchase[cus, art].compute())
# da.to_npy_stack('calculations/', purchase)
# purchase = purchase.sum(axis=1).compute()
# print(purchase.shape)
# print(purchase.max())
comp = purchase.dot(purchase.T)
user_1 = comp[0].argtopk(10).compute()
print(user_1)
vals = comp[0, user_1].compute()
print(vals)
