import numpy as np


def main():
    (X_train, X_val, X_test,
     X_train_tfidf, X_val_tfidf, X_test_tfidf,
     X_train_pos, X_val_pos, X_test_pos,
     X_train_ner, X_val_ner, X_test_ner,
     y_train, y_val, y_test) = load_npz('data.npz')


def load_npz(file='data.npz'):
    data = np.load(file)[0].item()

    X_train = data[()]['X_train']
    X_val = data[()]['X_val']
    X_test = data[()]['X_test']

    X_train_tfidf = data[()]['X_train_tfidf']
    X_val_tfidf = data[()]['X_val_tfidf']
    X_test_tfidf = data[()]['X_test_tfidf']

    X_train_pos = data[()]['X_train_pos']
    X_val_pos = data[()]['X_val_pos']
    X_test_pos = data[()]['X_test_pos']

    y_train = data[()]['y_train']
    y_val = data[()]['y_val']
    y_test = data[()]['y_test']

    X_train_ner = data[()]['X_train_ner']
    X_val_ner = data[()]['X_val_ner']
    X_test_ner = data[()]['X_test_ner']

    print(X_train.head())

    return (X_train, X_val, X_test,
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            X_train_pos, X_val_pos, X_test_pos,
            X_train_ner, X_val_ner, X_test_ner,
            y_train, y_val, y_test)


if __name__ == '__main__':
    main()
