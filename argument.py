from argparse import ArgumentParser


def argument():
    parser = ArgumentParser(description="Generator")
    parser.add_argument('--test-size', type=float,
                        metavar='TS', help='test split ratio')
    parser.add_argument('--train-gen', type=int,
                        metavar='TRG', help='Train set size')
    parser.add_argument('--test-gen', type=int,
                        metavar='TEG', help='Test set size')
    argument_object = parser.parse_args()
    return argument_object
