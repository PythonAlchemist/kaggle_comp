import numpy as np
import pandas as pd
import os
import time
import scipy.sparse
import dask
import dask.dataframe as dd
import dask.array as da

# customers = pd.read_csv('data/customers.csv')
# articles = pd.read_csv('data/articles.csv')
# sales = pd.read_csv('data/transactions_train.csv').head(10_000)

# customers = customers.reset_index().set_index('customer_id')
# articles = articles.reset_index().set_index('article_id')

customers = pd.read_pickle('calculations/customers.pkl')
articles = pd.read_pickle('calculations/articles.pkl')
sales = pd.read_pickle('calculations/sales.pkl')

n_customers = len(customers.index)
n_articles = len(articles.index)
batch_size = 60_000

# print(customers.loc['000058a12d5b43e67d225668fa1f8d618c13dc232df0cad8ffe7ad4a1091e318']['index'])
# print(articles.loc[663713001]['index'])

# purchase = da.zeros((n_customers, n_articles), dtype=np.byte, chunks=(n_customers, batch_size))
purchase = da.zeros((n_customers, n_articles), dtype=np.byte)



def update(x):
    cus_ix = int(customers.loc[x['customer_id']]['index'])
    art_ix = int(articles.loc[x['article_id']]['index'])
    purchase[cus_ix, art_ix] = 1


# chunks  = [x for x in range(0,sales.shape[0], bucket_size)]
# pool = multiprocessing.Pool()
# func = partial(run_process, sales)
# temp = pool.map(func,chunks)
# pool.close()
# pool.join()

sales.apply(lambda x: update(x), axis=1)

print("finished")
# dask_df.map_partitions(updateArray).compute()
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