import argparse


def get_model_id(framework, args):
    model_id = "_".join([framework, args.DS, args.aug, str(args.num_gc_layers), str(args.batch_size),str(args.epochs),str(args.seed)])
    return model_id

def arg_parse():
    parser = argparse.ArgumentParser(description="GcnInformax Arguments.")
    parser.add_argument("--DS", dest="DS", help="Dataset")
    parser.add_argument(
        "--local", dest="local", action="store_const", const=True, default=False
    )
    parser.add_argument(
        "--glob", dest="glob", action="store_const", const=True, default=False
    )
    parser.add_argument(
        "--prior", dest="prior", action="store_const", const=True, default=False
    )

    parser.add_argument("--lr", dest="lr", type=float, help="Learning rate.")
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        type=int,
        default=5,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--hidden-dim", dest="hidden_dim", type=int, default=32, help=""
    )

    parser.add_argument(
        "--epochs", dest="epochs", type=int, default=20, help=""
    )
    parser.add_argument(
        "--batch_size", dest="batch_size", type=int, default=20, help=""
    )
    parser.add_argument("--aug", type=str, default="dnodes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bottleneck_dim", type=int, default=64)
    return parser.parse_args()
