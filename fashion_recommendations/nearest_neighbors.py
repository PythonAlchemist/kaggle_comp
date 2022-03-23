import numpy as np
import pandas as pd
import os
import time
from functools import partial
import scipy.sparse
import dask
import dask.dataframe as dd
import dask.array as da
import multiprocessing

customers = pd.read_csv('data/customers.csv')
articles = pd.read_csv('data/articles.csv')
sales = pd.read_csv('data/transactions_train.csv').head(100_000)

customers = customers.reset_index().set_index('customer_id')
articles = articles.reset_index().set_index('article_id')
n_customers = len(customers.index)
n_articles = len(articles.index)
batch_size = 60_000

# print(customers.loc['000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318']['index'])
# print(articles.loc[663713001]['index'])

# purchase = da.zeros((n_customers, n_articles), dtype=np.byte, chunks=(n_customers, batch_size))
purchase = da.zeros((n_customers, n_articles), dtype=np.byte)
# dask_df = ddf.from_pandas(sales, npartitions=24)
# print(dask_df)

# def run_process(df, start):
#     for key in df:
#         cus_ix = int(customers.loc[df[key]['customer_id']]['index'])
#         art_ix = int(articles.loc[df[key]['article_id']]['index'])
#         purchase[cus_ix, art_ix] = 1
#         if key % 1000 == 0:
#             print(f"Row: {key} Percent: {100*key/100000}")
bucket_size = int(len(sales.index)/24) + 1

def update(x):
    cus_ix = int(customers.loc[x['customer_id']]['index'])
    art_ix = int(articles.loc[x['article_id']]['index'])
    purchase[cus_ix, art_ix] = 1

def run_process(sales, start):
    df = sales[start:start+bucket_size]
    df.apply(lambda x: update(x), axis=1)

chunks  = [x for x in range(0,sales.shape[0], bucket_size)]
pool = multiprocessing.Pool()
func = partial(run_process, sales)
temp = pool.map(func,chunks)
pool.close()
pool.join()

print("finished")
# dask_df.map_partitions(updateArray).compute()
cus = int(customers.loc['000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318']['index'])
art = int(articles.loc[663713001]['index'])
print(cus, art, purchase[cus, art].compute())
# da.to_npy_stack('calculations/', purchase)
purchase = purchase.sum(axis=1).compute()
print(purchase.shape)
print(purchase.max())


# sales = sales.set_index('article_id')

# batch_size = 10_000
# n_batches = int(len(articles.index) / 10_000) # plus partial set

# item_group = articles[articles['index'] < batch_size-1]['index']
# print(item_group)

# purchase = np.zeros((batch_size, len(customers.index)), dtype=np.bool_)

# print(sales.loc[sales.index[item_group]])

# cust = sales.index.get_level_values('customer_id').unique()
# c_ref = cust[0]
# top = dict()
# t0 = time.time()
# for c_comp in cust:
#     score = len(sales.loc[c_ref].index.intersection(sales.loc[c_comp].index))
#     if score > 0:
#         top[c_comp] = score
# t1 = time.time()
# print(top)
# print(f"Compute Time: {t1-t0}")