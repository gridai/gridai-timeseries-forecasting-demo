from pathlib import Path
from argparse import ArgumentParser
from model import CryptocurrencyForecast


if __name__ == '__main__':

    # define CLI arguments
    parser = ArgumentParser()
    parser.add_argument('--data_path', type=str, default=Path().cwd() / Path("dataset"))
    args = parser.parse_args()

    # plot prediction
    model = CryptocurrencyForecast(**vars(args))
    model.plot_predictions()
