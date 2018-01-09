import pickle

pkl_file = open('data.pkl', 'rb')

X_train = pickle.load(pkl_file)
X_val = pickle.load(pkl_file)
X_test = pickle.load(pkl_file)
y_train = pickle.load(pkl_file)
y_val = pickle.load(pkl_file)
y_test = pickle.load(pkl_file)
features = pickle.load(pkl_file)

pkl_file.close()
