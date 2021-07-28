import os
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from model import CryptocurrencyForecast
import wget

if __name__ == '__main__':

    # define CLI arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default=Path().cwd() / Path("/dataset/cryptocurrency_prices.csv"), help="data path")
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=0)
    parser.add_argument('--data_src', type=str, default="https://raw.githubusercontent.com/gridai/gridai-timeseries-forecasting-demo/main/data/cryptocurrency_prices.csv")
    args = parser.parse_args()

    # download if data is not prepared
    if not Path(args.data_path).is_file():
        filename = wget.download(args.data_src)
        args.data_path=os.getcwd() + "/" + filename
    
    formatter_class=ArgumentDefaultsHelpFormatter

    # start training
    model = CryptocurrencyForecast(**vars(args))
    model.train()
