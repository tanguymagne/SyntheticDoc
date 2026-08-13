"""Command line entry point generating a batch of arcsim configs for a given scenario."""

from config_generators.curve_by_pull import CurveByPullConfigGenerator
from config_generators.fall_on_ball import FallOnBallConfigGenerator
from config_generators.fall_on_many import FallOnManyConfigGenerator
from config_generators.fall_on_roller import FallOnRollerConfigGenerator
from config_generators.fall_on_roof import FallOnRoofConfigGenerator
from config_generators.fold_by_pull import FoldByPullConfigGenerator


def get_generator(config_type):
    """Return the config generator matching `config_type`."""
    if config_type == "curve_by_pull":
        return CurveByPullConfigGenerator()
    elif config_type == "fall_on_ball":
        return FallOnBallConfigGenerator()
    elif config_type == "fall_on_many":
        return FallOnManyConfigGenerator()
    elif config_type == "fall_on_roller":
        return FallOnRollerConfigGenerator()
    elif config_type == "fall_on_roof":
        return FallOnRoofConfigGenerator()
    elif config_type == "fold_by_pull":
        return FoldByPullConfigGenerator()
    else:
        raise ValueError(f"Unknown config type: {config_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-type",
        type=str,
        choices=[
            "curve_by_pull",
            "fall_on_ball",
            "fall_on_many",
            "fall_on_roller",
            "fall_on_roof",
            "fold_by_pull",
        ],
        required=True,
        help="Type of configuration to generate",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--num", type=int, default=1, help="Number of configurations to generate")
    args = parser.parse_args()

    generator = get_generator(args.config_type)
    generator.batch_generate(seed=args.seed, num=args.num, save_dir=f"configs/{args.config_type}")
