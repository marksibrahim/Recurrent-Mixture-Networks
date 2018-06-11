import pandas as pd


PATH = "~/Downloads/instacart_2017_05_01/"


aisles_df = pd.read_csv(PATH + "aisles.csv",)
departments_df = pd.read_csv(PATH + "departments.csv")
# general info. about an order
orders_df = pd.read_csv(PATH + "orders.csv")
products_df = pd.read_csv(PATH + "products.csv")
# main table for customer orders excluding most recent purchase
order_products_prior_df = pd.read_csv(PATH + "order_products__prior.csv")


transactions_df = pd.merge(order_products_prior_df, orders_df, on="order_id")
transactions_df = pd.merge(transactions_df, products_df, on="product_id")
transactions_df = pd.merge(transactions_df, aisles_df, on="aisle_id")
transactions_df = pd.merge(transactions_df, departments_df, on="department_id")

transactions_df.to_csv("transactions.csv")
