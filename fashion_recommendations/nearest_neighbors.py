import numpy as np
import pandas as pd
import os
import time

customers = pd.read_csv('data/customers.csv')
articles = pd.read_csv('data/articles.csv')
sales = pd.read_csv('data/transactions_train.csv')

customers = customers.reset_index().set_index('customer_id')
articles = articles.reset_index().set_index('article_id')
sales = sales.set_index('article_id')

batch_size = 10_000
n_batches = int(len(articles.index) / 10_000) # plus partial set

item_group = articles[articles['index'] < batch_size-1]['index']
print(item_group)

purchase = np.zeros((batch_size, len(customers.index)), dtype=np.bool_)

print(sales.loc[sales.index[item_group]])

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