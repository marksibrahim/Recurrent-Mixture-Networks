# Recurrent-Mixture-Networks


# Instacart Data Processing
1. Download [instacart dataset](https://www.instacart.com/datasets/grocery-shopping-2017)
2. Edit PATH in `merge_instacart_data.py` to location of data
3. Install pandas and run script `$ python merge_instacart_data.py`

This should generate a ~3GB csv called `transactions` (it took ~10 minutes to run on a Macbook Pro). 

## Open Source Transaction Data

* https://arxiv.org/abs/1511.05957 (3 months ~1 million customers transactions, published in Nature/Science)
  * also appears in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4180893/ (although I'm not certain)

* https://data.world/raghu543/credit-card-fraud-data (two days of transaction data from Europe, published in IEEE)
  - another link to the dataset https://www.openml.org/d/1597 and https://www.kaggle.com/mlg-ulb/creditcardfraud
  - http://www.ulb.ac.be/di/map/adalpozz/pdf/Dalpozzolo2015PhD.pdf (PhD thesis on dataset)
  - ~300k transactions (160MB), .2% fraud rate, and 30 normalized numerical features (anonomous obtained using PCA), no customer id
* https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data) (German Credit Approval data)
* https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients (UCI dataset from Businesses in Taiwan)

### Transactions Published by Governments
* https://catalog.data.gov/dataset/purchase-card-pcard-fiscal-year-2014 (400k transactions from 2014)
* https://data.nashville.gov/Budget-Finance/Metro-Credit-Card-Transactions/ucyr-mx6r (Nashville Governmental Agency Purchases)


### Without a Target
* https://www.instacart.com/datasets/grocery-shopping-2017 (3 million Instacart customer transactions over 2017)
* http://archive.ics.uci.edu/ml/datasets/online+retail (transactions for a UK online retailer, 500k rows no target)
* https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data (22GB of a retailer's transactions by customer)

### Synthetic 
* https://www.kaggle.com/ntnu-testimon/paysim1 (Project called PaySim, synthetically generated fraud dataset based on real aggregated anonymized data)

### Unable to Download, but promising
* http://www.cs.cmu.edu/~awm/15781/project/data.html (Credit Card Section, although wasn't able to download this)
* https://web.archive.org/web/20101116004911/http://mill.ucsd.edu:80/index.php?page=Datasets&subpage=Download (UCSD + FICO credit card data from 2009, can't download this though)
