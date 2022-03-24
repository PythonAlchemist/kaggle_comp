import dask.array as da
import pandas as pd
import numpy as np


customers = pd.read_csv('data/customers.csv').reset_index().set_index('customer_id')
articles = pd.read_csv('data/articles.csv').reset_index().set_index('article_id')
sales = pd.read_csv('data/transactions_train.csv')
cus_subset = customers.index.tolist()[:50000]
customers = customers[customers.index.isin(cus_subset)]
sales = sales[sales['customer_id'].isin(cus_subset)]

customers.to_pickle('calculations/customers.pkl')
articles.to_pickle('calculations/articles.pkl')
sales.to_pickle('calculations/sales.pkl')