from argparse import ArgumentParser
from src.data import extract, prepare_train_test
from context_based_fault_localization import fault_localization


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("example", type=str, required=True)
    args = parser.parse_args()
    dest_dir = extract(args.example)
    train_file, test_file = prepare_train_test(dest_dir)
    fault_localization(train_file, test_file)