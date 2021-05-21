# Timeseries Forecasting

We show how to create a model that learns how to forecast the next N observations of
a timeseries.

In this example, we will be creating a model that predicts future cryptocurrency values.

## Step 1: Create Your Dataset

Our dataset is quite simple: it's a CSV file with the following structure (each colum is self explanatory):

```csv
time_idx,Symbol,Date,High,Low,Open,Close,Volume,Marketcap
1,ADA,2017-10-02 23:59:59,0.0300877001136541,0.0199692994356155,0.0246070008724927,0.0259317997843027,57641300.0,628899051.78
2,ADA,2017-10-03 23:59:59,0.0274251997470855,0.0206898991018533,0.025756599381566,0.0208158008754253,16997800.0,539692714.905
```

We will training a series of models on Grid. Now, in order to make the process of updating
the dataset easier we will be creating a [Grid Datastore](https://docs.grid.ai/products/add-data-to-grid-datastores). Datstores are collections of files that are versioned and can
be mounted anywhere in the experiment context.

We'll be creating a new Datastore using the Grid CLI with the following command:

```shell
$ grid datastores create --name crypto_prices --source data/
upload ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0%
✔ Finished uploading datastore.
```

Then check that your datsatore is ready to use by calling `grid datastores list`:

```shell
$ grid datstores list
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Credential Id ┃                Name ┃ Version ┃     Size ┃          Created ┃    Status ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ cc-grv4f      │       crypto_prices │       1 │  12.6 MB │ 2021-05-20 01:17 │ Succeeded │
└───────────────┴─────────────────────┴─────────┴──────────┴──────────────────┴───────────┘
```

Whenever your datastore has `Status` of `Succeeded` you are ready to go on training.

## Step 2: Train Your Model on Grid AI

You are now ready to train your model on Grid.

We'll be using the CLI but you can do the same thing by using the web UI. We have placed a configuration file 
locally (`.grid/config.yml`) that you can use as reference instead of passing all the parameters to
the CLI manually -- or just click on Grid badge:

```shell
$ grid run --grid_config .grid/config.yml \
           train.py \
           --max_epochs 100 \
           --data_path /dataset/cryptocurrency_prices.csv \
           --learning_rate "uniform(0,0.03,5)" \
           --hidden_size "[16,32,64]"

No --grid_name passed, naming your run glossy-manatee-255
Using default cloud credentials cc-bwhth to run on AWS.

                Run submitted!
                `grid status` to list all runs
                `grid status glossy-manatee-255` to see all experiments for this run

                ----------------------
                Submission summary
                ----------------------
                script:                  train.py
                instance_type:           g4dn.xlarge
                distributed:             False
                use_spot:                True
                cloud_provider:          aws
                cloud_credentials:       cc-bwhth
                grid_name:               glossy-manatee-255
                datastore_name:          crypto_prices
                datastore_version:       1
                datastore_mount_dir:     /dataset
```

### Bonus: Run a Hyperparameter Sweep

Grid AI makes it trivial to run a [hyperparameter sweep](https://docs.grid.ai/products/global-cli-configs/cli-api/grid-train#hyperparameter-sweeps)
without having to change anything in your scripts. Let's experiment with a number of different learning rates for our model:

```shell
$ grid run --grid_config .grid/config.yml \
           train.py --max_epochs 100 \
           --data_path /dataset/cryptocurrency_prices.csv \
           --learning_rate "uniform(0,0.03,5)" \
           --hidden_size "[16,32,64]"
```

That will generate 15 experimentst with different learning rate combinations.

## Attribution

This project relies heavily on the [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/en/latest/) package. The implementation herein adapts
from their documentation and tutorials.

The dataset used in this demo comes from [CoinMarketCap](https://coinmarketcap.com/), a cryptocurrency price-tracking service. We have downloaded a processed version of the data
available in this [Kaggle page](https://www.kaggle.com/sudalairajkumar/cryptocurrencypricehistory).
