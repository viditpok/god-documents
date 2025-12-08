from __future__ import annotations

import argparse

from src.data.build_panel import build_fair_value_panel
from src.data.factors import prepare_ff_factors
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="1987-03-31")
    parser.add_argument("--end", default="2012-12-31")
    args = parser.parse_args()

    logger.info("building fair-value panel for %s–%s", args.start, args.end)
    build_fair_value_panel(start=args.start, end=args.end)

    logger.info("preparing FF factor file for %s–%s", args.start, args.end)
    prepare_ff_factors(start=args.start, end=args.end)


if __name__ == "__main__":
    main()
