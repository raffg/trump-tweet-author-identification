import numpy as np


def main():
    if ner:
        (X_train, X_val, X_test,
         X_train_tfidf, X_val_tfidf, X_test_tfidf,
         X_train_pos, X_val_pos, X_test_pos,
         X_train_ner, X_val_ner, X_test_ner,
         y_train, y_val, y_test) = load_npy('data.npy', ner=False)
    else:
        (X_train, X_val, X_test,
         X_train_tfidf, X_val_tfidf, X_test_tfidf,
         X_train_pos, X_val_pos, X_test_pos,
         y_train, y_val, y_test) = load_npy('data.npy', ner=False)


def load_npy(file='data.npy', ner=False):
    data = np.load(file)

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

    if ner:
        X_train_ner = data[()]['X_train_ner']
        X_val_ner = data[()]['X_val_ner']
        X_test_ner = data[()]['X_test_ner']

        return (X_train, X_val, X_test,
                X_train_tfidf, X_val_tfidf, X_test_tfidf,
                X_train_pos, X_val_pos, X_test_pos,
                X_train_ner, X_val_ner, X_test_ner,
                y_train, y_val, y_test)

    return (X_train, X_val, X_test,
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            X_train_pos, X_val_pos, X_test_pos,
            y_train, y_val, y_test)


if __name__ == '__main__':
    main()
