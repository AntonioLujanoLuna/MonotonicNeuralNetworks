import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union


def generate_one_hot_mat(mat: np.ndarray) -> np.ndarray:
    upper_bound = np.max(mat)
    mat_one_hot = np.zeros((mat.shape[0], int(upper_bound + 1)))
    for j in range(mat.shape[0]):
        mat_one_hot[j, int(mat[j])] = 1.
    return mat_one_hot

def generate_normalize_numerical_mat(mat: np.ndarray) -> np.ndarray:
    if np.max(mat) == np.min(mat):
        return mat
    return (mat - np.min(mat)) / (np.max(mat) - np.min(mat))


def normalize_data(data_train: np.ndarray, data_test: np.ndarray, mono_list: List[int], class_list: List[int]) -> Tuple[
    np.ndarray, np.ndarray, List[int], List[int]]:
    n_train, n_test = data_train.shape[0], data_test.shape[0]
    data_feature = np.concatenate((data_train, data_test), axis=0)

    data_feature_normalized = np.zeros((n_train + n_test, 0))  # Start with empty array
    start_index, cat_length = [], []

    # Process all features
    for i in range(data_feature.shape[1]):
        if i in class_list:
            # Don't normalize monotonic or categorical features
            mat = data_feature[:, i][:, np.newaxis]
        else:
            # Normalize non-monotonic numerical features
            mat = generate_normalize_numerical_mat(data_feature[:, i])[:, np.newaxis]

        data_feature_normalized = np.concatenate((data_feature_normalized, mat), axis=1)

    # Process categorical features
    for i in class_list:
        mat = generate_one_hot_mat(data_feature[:, i])
        start_index.append(data_feature_normalized.shape[1])
        cat_length.append(mat.shape[1])
        data_feature_normalized = np.concatenate((data_feature_normalized, mat), axis=1)

    return data_feature_normalized[:n_train, :], data_feature_normalized[n_train:, :], start_index, cat_length

def train_test_splitter(data: pd.DataFrame, test_size: float, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)  # Create a random number generator with the given seed
    shuffled_indices = rng.permutation(len(data))
    test_set_size = int(len(data) * test_size)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

def load_data(
        path: Union[str, Tuple[str, str]],
        mono_inc_list: List[int],
        mono_dec_list: List[int],
        class_list: List[int],
        target_column: str,
        train_test_split: float = 0.8,
        constant: float = 1.0,
        normalize_target: bool = False,
        preprocess_func: Optional[callable] = None,
        seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if isinstance(path, tuple):
        # Load pre-split train and test data
        data_train = pd.read_csv(path[0])
        data_test = pd.read_csv(path[1])
    else:
        # Load single file and split into train and test
        data = pd.read_csv(path)
        data_train, data_test = train_test_splitter(data, test_size=1-train_test_split, random_state=seed)

    if preprocess_func:
        data_train = preprocess_func(data_train)
        data_test = preprocess_func(data_test)

    # Split features and target for train and test sets
    X_train = data_train.drop(columns=[target_column]).values
    y_train = data_train[target_column].values
    X_test = data_test.drop(columns=[target_column]).values
    y_test = data_test[target_column].values

    # Normalize data
    mono_list = mono_inc_list + mono_dec_list
    X_train, X_test, start_index, cat_length = normalize_data(X_train, X_test, mono_list, class_list)

    # Apply constant transformation to decreasing monotonic features
    for i, col in enumerate(mono_dec_list):
        X_train[:, col] = constant - X_train[:, col]
        X_test[:, col] = constant - X_test[:, col]

    # Normalize target variable if specified
    if normalize_target:
        y = generate_normalize_numerical_mat(np.concatenate([y_train, y_test], axis=0))
        y_train, y_test = y[:len(y_train)], y[len(y_train):]

    # Reorder cols so that monotonic features come first
    X_train = np.concatenate((X_train[:, mono_list], X_train[:, [i for i in range(X_train.shape[1]) if i not in mono_list]]), axis=1)
    X_test = np.concatenate((X_test[:, mono_list], X_test[:, [i for i in range(X_test.shape[1]) if i not in mono_list]]), axis=1)

    return X_train, y_train, X_test, y_test

def preprocess_abalone(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data['Sex'] = pd.Categorical(data['Sex']).codes
    return data

def preprocess_auto_mpg(data: pd.DataFrame) -> pd.DataFrame:
    data = data.drop('car name', axis=1)
    data = data.dropna()
    return data

def preprocess_blog_feedback(data: pd.DataFrame) -> pd.DataFrame:
    data_array = np.array(data.values)
    X_array = data_array[:, :280].astype(np.float64)
    y_array = data_array[:, 280].astype(np.float64)
    # Calculate the 90th percentile of the target variable
    q = np.percentile(y_array, 90)
    # Create a mask for rows to keep (where y <= q)
    mask = y_array <= q
    # Apply the mask to both X and y
    X_array = X_array[mask]
    y_array = y_array[mask]
    # Convert y back to uint8 if needed
    y_array = y_array.astype(np.uint8)
    data_array = np.column_stack((X_array, y_array))
    columns = [f'col_{i}' for i in range(1, data_array.shape[1])]
    columns.append('target')
    data = pd.DataFrame(data_array, columns=columns)
    return data


def preprocess_compas(data: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    data = data.copy()
    # Data cleaning as performed by ProPublica
    data = data[(data['days_b_screening_arrest'] <= 30) & (data['days_b_screening_arrest'] >= -30)]
    data = data[data['is_recid'] != -1]
    data = data[data['c_charge_degree'] <= "O"]
    data = data[data['score_text'] != 'N/A']
    race_replace = ['African-American', 'Hispanic', 'Asian', 'Caucasian', 'Native American', 'Other']
    sex_replace = ['Male', 'Female']
    # Use pd.Categorical to avoid downcasting warnings
    data['race'] = pd.Categorical(data['race'], categories=race_replace).codes
    data['sex'] = pd.Categorical(data['sex'], categories=sex_replace).codes
    data = data[
        ['priors_count', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'age', 'race', 'sex', 'two_year_recid']]
    return data


def preprocess_loan(data: pd.DataFrame) -> pd.DataFrame:
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    data = data.copy()
    # Use pd.Categorical for grade to avoid downcasting warning
    data['grade'] = pd.Categorical(data['grade'], categories=['A', 'B', 'C', 'D', 'E', 'F', 'G'], ordered=True).codes
    grade = data['grade']
    data = data.drop('grade', axis=1)
    data.insert(1, 'grade', grade)
    # Use pd.Categorical for emp_length to avoid downcasting warning
    emp_length_categories = ['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years',
                             '6 years', '7 years', '8 years', '9 years', '10+ years']
    data['emp_length'] = pd.Categorical(data['emp_length'], categories=emp_length_categories, ordered=True).codes
    data = data.dropna(subset=[data.columns[1], data.columns[2]])
    return data

def load_abalone(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path="../datasets/abalone.csv",
        mono_inc_list=[4, 5, 6, 7],
        mono_dec_list=[],
        class_list=[],
        target_column="Rings",
        normalize_target=True,
        preprocess_func=preprocess_abalone,
        seed=seed
    )

def load_auto_mpg(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path="../datasets/auto-mpg.csv",
        mono_inc_list=[4, 5, 6],
        mono_dec_list=[0, 1, 2, 3],
        class_list=[],
        target_column="mpg",
        normalize_target=True,
        preprocess_func=preprocess_auto_mpg,
        seed=seed
    )

def load_blog_feedback(seed:int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path=("../datasets/blogfeedback_train.csv", "../datasets/blogfeedback_test.csv"),
        mono_inc_list=list(range(50, 59)),  # col51 to col59
        mono_dec_list=[],
        class_list=[],
        target_column="target",
        normalize_target=True,
        preprocess_func=preprocess_blog_feedback,
        seed=seed
    )

def load_boston_housing(seed:int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path="../datasets/BostonHousing.csv",
        mono_inc_list=[5],
        mono_dec_list=[0],
        class_list=[],
        target_column="MEDV",
        normalize_target=True,
        seed=seed
    )

def load_compas(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path="../datasets/compas_scores_two_years.csv",
        mono_inc_list=[0, 1, 2, 3],
        mono_dec_list=[],
        class_list=[],
        target_column="two_year_recid",
        normalize_target=False,
        preprocess_func=preprocess_compas,
        seed=seed
    )

def load_era(seed:int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path="../datasets/era.csv",
        mono_inc_list=[0, 1, 2, 3],
        mono_dec_list=[],
        class_list=[],
        target_column="out1",
        normalize_target=True,
        seed=seed
    )

def load_esl(seed:int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path="../datasets/esl.csv",
        mono_inc_list=[0, 1, 2, 3],
        mono_dec_list=[],
        class_list=[],
        target_column="out1",
        normalize_target=True,
        seed=seed
    )

def load_heart(seed:int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path="../datasets/heart.csv",
        mono_inc_list=[3, 4],
        mono_dec_list=[],
        class_list=[],
        target_column="target",
        normalize_target=False,
        seed=seed
    )

def load_lev(seed:int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path="../datasets/lev.csv",
        mono_inc_list=[0, 1, 2, 3],
        mono_dec_list=[],
        class_list=[],
        target_column="Out1",
        normalize_target=True,
        seed=seed
    )

def load_loan(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path="../datasets/loan.csv",
        mono_inc_list=[1, 4],  # feature_1, feature_4
        mono_dec_list=[0, 2, 3],  # feature_0, feature_2, feature_3
        class_list=[],
        target_column="loan_status",
        preprocess_func=preprocess_loan,
        normalize_target=False,
        seed=seed
    )

def load_swd(seed:int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return load_data(
        path="../datasets/swd.csv",
        mono_inc_list=[0, 1, 2, 4, 6, 8, 9],  # In1, In2, In3, In5, In7, In9, In10
        mono_dec_list=[],
        class_list=[],
        target_column="Out1",
        normalize_target=True,
        seed=seed
    )

def save_preprocessed_data(
        dataset_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        save_split: bool = True
):
    # Create directory if it doesn't exist
    os.makedirs("./preprocessed_data", exist_ok=True)

    # Combine features and target
    train_data = np.column_stack((X_train, y_train))
    test_data = np.column_stack((X_test, y_test))

    # Create column names
    feature_cols = [f"feature_{i}" for i in range(X_train.shape[1])]
    target_col = ["target"]
    columns = feature_cols + target_col

    if save_split:
        # Save train and test separately
        pd.DataFrame(train_data, columns=columns).to_csv(f"./preprocessed_data/{dataset_name}_train_preprocessed.csv",
                                                         index=False)
        pd.DataFrame(test_data, columns=columns).to_csv(f"./preprocessed_data/{dataset_name}_test_preprocessed.csv",
                                                        index=False)
    else:
        # Combine train and test, then save
        all_data = np.vstack((train_data, test_data))
        pd.DataFrame(all_data, columns=columns).to_csv(f"./preprocessed_data/{dataset_name}_preprocessed.csv",
                                                       index=False)


def save_all_preprocessed_datasets():
    datasets = [
        ("abalone", load_abalone),
        ("auto_mpg", load_auto_mpg),
        ("blog_feedback", load_blog_feedback),
        ("boston_housing", load_boston_housing),
        ("compas", load_compas),
        ("era", load_era),
        ("esl", load_esl),
        ("heart", load_heart),
        ("lev", load_lev),
        ("loan", load_loan),
        ("swd", load_swd)
    ]

    for dataset_name, load_func in datasets:
        print(f"Processing {dataset_name}...")
        X_train, y_train, X_test, y_test = load_func()
        save_preprocessed_data(dataset_name, X_train, y_train, X_test, y_test, False)
        print(f"{dataset_name} processed and saved.")


# Run the script to save all preprocessed datasets
if __name__ == "__main__":
    save_all_preprocessed_datasets()