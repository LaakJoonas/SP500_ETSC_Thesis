import numpy as np

from calimera import CALIMERA
from sktime.datasets import load_from_tsfile_to_dataframe
from sklearn.metrics import accuracy_score


def load_example_data():
    # example data from https://timeseriesclassification.com/
    X_train, y_train = load_from_tsfile_to_dataframe(
        'SP500_VIX_MACRO_TRAIN.ts'
    )
    X_train = np.asarray([[[v for v in channel] for channel in sample] for sample in X_train.to_numpy()])
    X_test, y_test = load_from_tsfile_to_dataframe(
        'SP500_VIX_MACRO_TEST.ts'
    )
    X_test = np.asarray([[[v for v in channel] for channel in sample] for sample in X_test.to_numpy()])
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_example_data()



    print(f"Using subset: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

    delay_penalty = 0.6  #Adjust this value to focus on accuracy or earliness, lower values prioritize accuracy
    print(f"Using delay penalty: {delay_penalty}")
    model = CALIMERA(delay_penalty=delay_penalty)
    model.fit(X_train, y_train)

    stop_timestamps, y_pred = model.test(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    earliness = sum(stop_timestamps) / (X_test.shape[-1] * X_test.shape[0])
    cost = 1.0 - accuracy + delay_penalty * earliness
    print(f'Accuracy: {accuracy}\nEarliness: {earliness}\nCost: {cost}')
