import decision_tree
import scan
import utils


def main():
    binary_label = True

    data = scan.scan('foods.txt', binary_label)
    length = len(data)

    train_data = data[:int(length*.8)]
    test_data = data[int(length*.8):]

    decision_tree = dt.train(train_data)
    test_results = dt.test(decision_tree, test_data)

    print test_results

if __name__ == '__main__':
    main()
