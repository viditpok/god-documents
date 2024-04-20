import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def get_housing_dataset():
    dataset = fetch_california_housing()  # load the dataset
    x, y = dataset.data, dataset.target
    y = y.reshape(-1, 1)
    perm = np.random.RandomState(seed=3).permutation(x.shape[0])[:500]
    x = x[perm]
    y = y[perm]

    index_array = np.argsort(y.flatten())
    x, y = x[index_array], y[index_array]

    values_per_list = len(y) // 3
    list1 = y[:values_per_list]
    list2 = y[values_per_list : 2 * values_per_list]
    list3 = y[2 * values_per_list :]
    label_mapping = {
        tuple(value): label
        for label, value_list in enumerate([list1, list2, list3])
        for value in value_list
    }
    updated_values = [label_mapping[tuple(value)] for value in y]
    num_classes = len(set(updated_values))
    one_hot_encoded = np.eye(num_classes)[updated_values]
    y = np.array(one_hot_encoded)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, random_state=1
    )  # split data

    x_scale = MinMaxScaler()
    x_train = x_scale.fit_transform(x_train)  # normalize data
    x_test = x_scale.transform(x_test)

    return x_train, y_train, x_test, y_test
