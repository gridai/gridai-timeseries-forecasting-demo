from pathlib import Path
from argparse import ArgumentParser
from model import CryptocurrencyForecast


if __name__ == '__main__':

    # define CLI arguments
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default=Path().cwd() / Path("/dataset/cryptocurrency_prices.csv"))
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--hidden-size', type=int, default=16)
    parser.add_argument('--gpus', type=int, default=None)
    args = parser.parse_args()

    # start training
    model = CryptocurrencyForecast(**vars(args))
    model.train()
