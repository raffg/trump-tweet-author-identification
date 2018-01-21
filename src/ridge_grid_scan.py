from sklearn.linear_model import RidgeClassifier


def ridge_grid_scan(X_train, y_train, n=100):
    '''
    Recursively performs ridge regression to sort all features in order of
    importance and return the top n features
    INPUT: X DataFrame, y DataFrame, int
    OUTPUT: list of feature importances
    '''

    scan = GridScan(X_train, y_train, n)
    return scan.feature_importances


class GridScan(object):
    '''
    Grid scan object to track alpha levels of ridge regression and features
    which are driven out of the model at each alpha levels
    '''

    def __init__(self, X_train, y_train, n):
        self.X_train = X_train
        self.y_train = y_train
        self.n = n
        alpha_min = 1e-8
        alpha_max = 1e24
        self.alpha_levels = {}
        self.feature_importances = []

        self.ridge(alpha_min)
        self.ridge(alpha_max)
        while len(self.alpha_levels[alpha_max]) < len(self.X_train.columns):
            print('alpha too low; increasing value')
            alpha_max *= 2
            self.ridge(alpha_max)

        self.scan(alpha_min, alpha_max)

        self.feature_importances.sort(key=lambda feature_alpha:
                                      -feature_alpha[1])

    def scan(self, lower, upper):
        '''
        Takes a lower and upper bound for alpha levels and recursively runs
        ridge regression until only one feature is eliminated from the model
        INPUT: int, int
        OUTPUT:
        '''

        mid = (lower + upper) / 2

        if len(self.alpha_levels[upper]) <= (len(self.X_train.columns) -
                                             self.n):
            return

        diff = self.alpha_levels[upper] - self.alpha_levels[lower]
        if not diff:
            return

        if len(diff) == 1:
            for feature in diff:
                self.feature_importances.append((feature, mid))
                print('========')
                print(len(self.feature_importances), feature)
                print('{:0.1f}% complete'.format((
                      len(self.feature_importances) / self.n) * 100))
                print('========')
            return

        self.ridge(mid)

        self.scan(lower, mid)
        self.scan(mid, upper)

    def ridge(self, alpha):
        '''
        Takes an alpha level and runs ridge regression
        INPUT: float
        OUTPUT:
        '''

        print(alpha)
        model = RidgeClassifier(alpha=alpha)

        model.fit(self.X_train, self.y_train)

        feat_coef = list(zip(self.X_train.columns, model.coef_[0]))

        features = set()
        for element in feat_coef:
            if abs(element[1]) < 1e-24:
                features.add(element[0])

        self.alpha_levels[alpha] = features
