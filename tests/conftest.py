from pathlib import Path
import pytest

from swapscribe.demo import generate_demo
from swapscribe.market import load_rates
from swapscribe.risk import load_portfolio

@pytest.fixture(scope="session")
def sample_data(tmp_path_factory):
    folder = tmp_path_factory.mktemp("sample")
    generate_demo(folder)
    return folder

@pytest.fixture(scope="session")
def rates(sample_data):
    return load_rates(sample_data / "market_rates.csv")

@pytest.fixture(scope="session")
def portfolio(sample_data):
    return load_portfolio(sample_data / "portfolio.csv")
