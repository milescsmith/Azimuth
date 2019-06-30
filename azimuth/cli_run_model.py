import os

import click
from dill import load, dump

from .model_comparison import run_models


@click.command()
@click.option(
    "model", help="model e.g. L1, or GP", default=None, required=False, type=str
)
@click.option("--test", type=bool, default=False, is_flag=True)
@click.option("--order", type=int, default=1)
@click.option("--likelihood", type=str, default="gaussian")  # applies only to GP:
@click.option("--weighted_degree", type=int, default=3)  # applies only to GP:
@click.option("--adaboost_learning_rate", type=float, default=0.1)
@click.option("--adaboost_max_depth", type=int, default=3)
@click.option("--adaboost_num_estimators", type=int, default=100)
@click.option("--adaboost_CV", dest="adaboost_CV", is_flag=True, default=False)
@click.option("--output_dir", type=str, default="./")
@click.option("--exp_name", type=str, default=None)
@click.help_option()
def main(
    model,
    test,
    order,
    likelihood,
    weighted_degree,
    adaboost_learning_rate,
    adaboost_max_depth,
    adaboost_num_estimators,
    adaboost_CV,
    output_dir,
    exp_name,
):
    """command-line version of model_comparison.py
    \f
    (see that file for more options?)
    """

    # store current directory
    cur_dir = os.getcwd()
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    # change directory to script directory so that relative paths work
    os.chdir(dname)

    with open(output_dir + "/learn_pickle", "rb") as f:
        learn_options = load(f)

    results, all_learn_options = run_models(
        [model],
        learn_options_set=learn_options,
        orders=[order],
        test=test,
        GP_likelihoods=[likelihood],
        WD_kernel_degrees=[weighted_degree],
        adaboost_num_estimators=[adaboost_num_estimators],
        adaboost_max_depths=[adaboost_max_depth],
        adaboost_learning_rates=[adaboost_learning_rate],
        adaboost_CV=adaboost_CV,
    )

    if exp_name is None:
        exp_name = list(results.keys())[0]

    os.chdir(cur_dir)

    with open(output_dir + "/" + exp_name + ".pickle", "wb") as f:
        dump((results, all_learn_options), f)


if __name__ == "__main__":
    main()
