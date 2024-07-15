import numpy as np
import pandas as pd
import os

# Set a global seed for reproducibility
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)


def generate_one_hot_mat(column):
    unique_values = np.unique(column)
    if len(unique_values) == 2:
        # For binary columns, ensure values are 0 and 1
        return np.where(column == unique_values[0], 0, 1)
    one_hot_dict = {val: idx for idx, val in enumerate(unique_values)}
    mat_one_hot = np.zeros((column.shape[0], len(unique_values)))
    for i, val in enumerate(column):
        mat_one_hot[i, one_hot_dict[val]] = 1
    return mat_one_hot


def generate_normalize_numerical_mat(train_column, test_column):
    train_min = np.min(train_column)
    train_max = np.max(train_column)
    if train_min == train_max:
        return np.zeros_like(train_column), np.zeros_like(test_column)
    train_normalized = (train_column - train_min) / (train_max - train_min)
    test_normalized = (test_column - train_min) / (train_max - train_min)
    return train_normalized, test_normalized


def handle_missing_values(data):
    # Set numpy and pandas random seeds
    np.random.seed(GLOBAL_SEED)
    pd.reset_option('mode.chained_assignment')
    pd.options.mode.chained_assignment = None

    # For numerical columns, fill with mean
    num_cols = data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        mean_value = data[col].mean()
        data[col] = data[col].fillna(mean_value)

    # For categorical columns, fill with mode
    cat_cols = data.select_dtypes(exclude=[np.number]).columns
    for col in cat_cols:
        mode_value = data[col].mode().iloc[0]
        data[col] = data[col].fillna(mode_value)

    return data


def preprocess_data(data_train, data_test, target_column, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    data_train_normalized = np.zeros((data_train.shape[0], 0))
    data_test_normalized = np.zeros((data_test.shape[0], 0))
    new_column_names = []  # To keep track of new column names

    for col in data_train.columns:
        if col in exclude_columns or col == target_column:
            continue
        if pd.api.types.is_numeric_dtype(data_train[col]):
            train_mat, test_mat = generate_normalize_numerical_mat(
                data_train[col].values, data_test[col].values)
            data_train_normalized = np.hstack((data_train_normalized, train_mat.reshape(-1, 1)))
            data_test_normalized = np.hstack((data_test_normalized, test_mat.reshape(-1, 1)))
            new_column_names.append(col)
        else:
            train_mat = generate_one_hot_mat(data_train[col].values)
            data_train_normalized = np.hstack((data_train_normalized, train_mat))
            new_column_names.extend([col + '_' + str(int(i)) for i in range(train_mat.shape[1])])

            # Use the same one-hot encoding for test data
            test_mat = np.zeros((data_test.shape[0], train_mat.shape[1]))
            for i, val in enumerate(data_test[col].values):
                if val in data_train[col].values:
                    test_mat[i] = train_mat[list(data_train[col].values).index(val)]
            data_test_normalized = np.hstack((data_test_normalized, test_mat))

    return data_train_normalized, data_test_normalized, new_column_names


def prepare_data(file_path, target_column, drop_columns=None, exclude_columns=None, test_size=0.2, is_presplit=False):
    if is_presplit:
        train_path = file_path.replace('.csv', '_train.csv')
        test_path = file_path.replace('.csv', '_test.csv')
        data_train = pd.read_csv(train_path)
        data_test = pd.read_csv(test_path)
    else:
        data = pd.read_csv(file_path)
        data = data.sample(frac=1, random_state=GLOBAL_SEED).reset_index(drop=True)
        n_train = int(len(data) * (1 - test_size))
        data_train, data_test = data.iloc[:n_train], data.iloc[n_train:]

    if drop_columns:
        data_train = data_train.drop(columns=drop_columns)
        data_test = data_test.drop(columns=drop_columns)

    data_train = handle_missing_values(data_train)
    data_test = handle_missing_values(data_test)

    X_train = data_train.drop(columns=[target_column])
    y_train = data_train[target_column]
    X_test = data_test.drop(columns=[target_column])
    y_test = data_test[target_column]

    X_train_preprocessed, X_test_preprocessed, new_column_names = preprocess_data(X_train, X_test, target_column,
                                                                                  exclude_columns)

    # Combining preprocessed data for saving
    combined_data = np.vstack((X_train_preprocessed, X_test_preprocessed))
    combined_data = pd.DataFrame(combined_data, columns=new_column_names)
    combined_data[target_column] = np.concatenate([y_train.values, y_test.values])

    return X_train_preprocessed, y_train.values, X_test_preprocessed, y_test.values, combined_data


def process_multiple_datasets(datasets, output_dir='preprocessed_data', test_size=0.2):
    os.makedirs(output_dir, exist_ok=True)

    for file_path, target_column, drop_columns, exclude_columns in datasets:
        print(f"Processing {file_path} with target column '{target_column}'...")

        is_presplit = os.path.exists(file_path.replace('.csv', '_train.csv'))
        X_train, y_train, X_test, y_test, combined_data = prepare_data(
            file_path, target_column, drop_columns, exclude_columns, test_size, is_presplit)

        print(f"Training Data Shape: {X_train.shape}, Training Labels Shape: {y_train.shape}")
        print(f"Test Data Shape: {X_test.shape}, Test Labels Shape: {y_test.shape}")

        # Save preprocessed data
        dataset_name = os.path.splitext(os.path.basename(file_path))[0]
        output_path = os.path.join(output_dir, f"{dataset_name}_preprocessed.csv")
        combined_data.to_csv(output_path, index=False)
        print(f"Preprocessed data saved as {output_path}")
        print()


if __name__ == "__main__":
    datasets = [
        ("../datasets/abalone.csv", 'Rings', None, None), # Monotonic is Shell, shucked, viscera, and whole weights
        ("../datasets/auto-mpg.csv", 'mpg', ['car name'], None),
        # ("../datasets/blogFeedback.csv", 'target', None, None), Already processed
        ("../datasets/BostonHousing.csv", 'MEDV', None, None),
        # ("../datasets/compas.csv", 'ground_truth', None, ['priors_count','juv_fel_count','juv_misd_count',
        #    'juv_other_count','age', 'race_0', 'race_1', 'race_2', 'race_3', 'race_4', 'race_5', 'sex_0',
        #    'sex_1']),
        ("../datasets/era.csv", 'out1', None, None),
        ("../datasets/esl.csv", 'out1', None, None),
        ("../datasets/heart.csv", 'target', None, None),
        ("../datasets/lev.csv", 'Out1', None, None),
        # ("../datasets/loan.csv", 'target1', ['id_col'], None), Already processed
        ("../datasets/swd.csv", 'Out1', None, None),
    ]


    process_multiple_datasets(datasets)

    """
    Abalone dataset monotonicity
        { 'Sex_0': 0,
            'Sex_1': 0,
            'Sex_2': 0,
            'Length': 0,
            'Diameter': 0,
            'Height': 0,
            'Whole weight': 1,
            'Shucked weight': 1,
            'Viscera weight': 1,
            'Shell weight': 1}  
        }

    Auto-mpg dataset monotonicity
    {   "cylinders": -1,
        "displacement": -1,
        "horsepower": -1,
        "weight": -1,
        "acceleration": 1,
        "model_Year": 1,
        "origin": 1,
    }
    
    BlogFeedback dataset monotonicity
    
    {
    ...
    col51: 1,
    col52: 1,
    col53: 1,
    col54: 1,
    col55: 1,
    col56: 1,
    col57: 1,
    col58: 1,
    col59: 1,
    ...
    }
    
    Boston Housing dataset monotonicity
    {
        'CRIM': -1,
         'ZN': 0,
         'INDUS': 0,
         'CHAS': 0,
         'NOX': 0,
         'RM': 1,
         'AGE': 0,
         'DIS': 0,
         'RAD': 0,
         'TAX': 0,
         'PTRATIO': 0,
         'B': 0,
         'LSTAT': 0
     }
     
    Compas dataset monotonicity
    
    {
        'prior_count': 1,
        'juv_fel_count': 1,
        'juv_misd_count': 1,
        'juv_other_count': 1,
        'age': 0,
        'race_0': 0,
        'race_1': 0,
        'race_2': 0,
        'race_3': 0,
        'race_4': 0,
        'race_5': 0,
        'sex_0': 0,
        'sex_1': 0
    }
    
    Era dataset monotonicity
    {
        'in1': 1,
        'in2': 1,
        'in3': 1,
        'in4': 1,
    }
    
    ESL dataset monotonicity
    
    {
        'in1': 1,
        'in2': 1,
        'in3': 1,
        'in4': 1,
    }
    
    Heart dataset monotonicity
    
    {
        "age": 0,
        "sex": 0,
        "cp": 0,
        "trestbps": 1,
        "chol": 1,
        "fbs": 0,
        "restecg": 0,
        "thalach": 0,
        "exang": 0,
        "oldpeak": 0,
        "slope": 0,
        "ca": 0,
        "thal": 0,
    }

    Lev dataset monotonicity
    
    {
        'In1': 1,
        'In2': 1,
        'In3': 1,
        'In4': 1,
    }
    
    Loan dataset monotonicity
        {'feature_0': -1,
         'feature_1': 1,
         'feature_2': -1,
         'feature_3': -1,
         'feature_4': 1,
         'feature_5': 0,
         'feature_6': 0,
         'feature_7': 0,
         'feature_8': 0,
         'feature_9': 0,
         'feature_10': 0,
         'feature_11': 0,
         'feature_12': 0,
         'feature_13': 0,
         'feature_14': 0,
         'feature_15': 0,
         'feature_16': 0,
         'feature_17': 0,
         'feature_18': 0,
         'feature_19': 0,
         'feature_20': 0,
         'feature_21': 0,
         'feature_22': 0,
         'feature_23': 0,
         'feature_24': 0,
         'feature_25': 0,
         'feature_26': 0,
         'feature_27': 0     
    }
    
    Swd dataset monotonicity
    
    {
        'In1': 1,
        'In2': 1,
        'In3': 1,
        'In4': 0,
        'In5': 1,
        'In6': 0,
        'In7': 1,
        'In8': 0,
        'In9': 1,
        'In10': 1,
    }
    
    """
