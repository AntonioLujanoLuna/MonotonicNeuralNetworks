import pandas as pd
import os
import numpy as np
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt
from scikit_posthocs import critical_difference_diagram


def read_csv_files(file_list, directory):
    data_dict = {}
    for file in file_list:
        method_name = os.path.splitext(file)[0]
        full_path = os.path.join(directory, file)
        df = pd.read_csv(full_path)
        data_dict[method_name] = df
    return data_dict


def create_performance_df(data_dict):
    performance_data = {}
    performance_std_data = {}
    for method, df in data_dict.items():
        if 'Metric Value' in df.columns and 'Metric Std Dev' in df.columns:
            performance_data[method] = df.set_index('Dataset')['Metric Value']
            performance_std_data[method] = df.set_index('Dataset')['Metric Std Dev']
        else:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                performance_data[method] = df.set_index('Dataset')[numeric_cols[0]]
                performance_std_data[method] = pd.Series(np.nan, index=df['Dataset'])
            else:
                print(f"Warning: No suitable numeric column found for method {method}")

    performance_df = pd.concat(performance_data, axis=1)
    performance_std_df = pd.concat(performance_std_data, axis=1)
    return performance_df.sort_index(), performance_std_df.sort_index()


def create_params_df(data_dict):
    params_data = {}
    for method, df in data_dict.items():
        params_data[method] = df.set_index('Dataset')['NumOfParameters']
    params_df = pd.concat(params_data, axis=1)
    return params_df.sort_index()


def create_mono_dfs(data_dict):
    mono_metrics = ['Mono Random', 'Mono Train', 'Mono Val']
    mono_dfs = {}
    for metric in mono_metrics:
        mono_data = {}
        mono_std_data = {}
        for method, df in data_dict.items():
            mono_data[method] = df.set_index('Dataset')[f'{metric} Mean']
            mono_std_data[method] = df.set_index('Dataset')[f'{metric} Std']
        mono_df = pd.concat(mono_data, axis=1)
        mono_std_df = pd.concat(mono_std_data, axis=1)
        mono_dfs[metric] = (mono_df.sort_index(), mono_std_df.sort_index())
    return mono_dfs


def format_value_with_std(value, std):
    return f"{value:.4f} $\\pm$ {std:.4f}"

def extract_value(formatted_string):
    return float(formatted_string.split()[0])

def bold_min_value(series):
    values = series.apply(extract_value)
    min_idx = values.idxmin()
    return series.apply(lambda x: f"\\textbf{{{x}}}" if x == series[min_idx] else x)

def bold_max_value(series):
    values = series.apply(extract_value)
    max_idx = values.idxmax()
    return series.apply(lambda x: f"\\textbf{{{x}}}" if x == series[max_idx] else x)

def df_to_latex(df, std_df, caption, label, bold_func):
    formatted_df = pd.DataFrame()
    for method in df.columns:
        formatted_values = df[method].combine(std_df[method], format_value_with_std)
        formatted_df[method] = formatted_values

    for idx in formatted_df.index:
        formatted_df.loc[idx] = bold_func(formatted_df.loc[idx])

    latex_table = formatted_df.to_latex(escape=False, caption=caption, label=label)
    return latex_table

def params_df_to_latex(df, caption, label):
    formatted_df = df.astype(str)
    for idx in formatted_df.index:
        formatted_df.loc[idx] = bold_min_value(formatted_df.loc[idx])

    latex_table = formatted_df.to_latex(escape=False, caption=caption, label=label)
    return latex_table

def perform_wilcoxon_tests(df):
    methods = df.columns
    n_methods = len(methods)
    results = pd.DataFrame(index=methods, columns=methods)

    for i in range(n_methods):
        for j in range(i+1, n_methods):
            method1 = methods[i]
            method2 = methods[j]
            try:
                _, p_value = wilcoxon(df[method1], df[method2])
                results.loc[method1, method2] = p_value
                results.loc[method2, method1] = p_value
            except Exception as e:
                print(f"Error performing Wilcoxon test between {method1} and {method2}: {str(e)}")
                results.loc[method1, method2] = np.nan
                results.loc[method2, method1] = np.nan

    return results

def perform_friedman_test(df):
    try:
        chi2, p_value = friedmanchisquare(*[df[col] for col in df.columns])
        return chi2, p_value
    except Exception as e:
        print(f"Error performing Friedman test: {str(e)}")
        return np.nan, np.nan

def perform_nemenyi_test(df):
    try:
        return posthoc_nemenyi_friedman(df)
    except Exception as e:
        print(f"Error performing Nemenyi test: {str(e)}")
        return None

def create_latex_table(df, caption, label):
    latex_table = df.to_latex(float_format="%.4f", caption=caption, label=label, escape=False)
    return latex_table

def create_critical_difference_diagram(performance_df, filename='critical_difference_diagram.png'):
    # Compute average ranks
    ranks = performance_df.rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()

    # Prepare data for the diagram
    ranks_dict = avg_ranks.to_dict()

    # Perform Nemenyi test to get the p-value matrix
    nemenyi_results = posthoc_nemenyi_friedman(performance_df)

    # Create the diagram
    fig, ax = plt.subplots(figsize=(10, 5))

    critical_difference_diagram(
        ranks=ranks_dict,
        sig_matrix=nemenyi_results,
        ax=ax,
        label_fmt_left='{label} ({rank:.2f})',
        label_fmt_right='({rank:.2f}) {label}',
    )

    plt.title("Critical Difference Diagram")
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Critical Difference Diagram saved as {filename}")


def main():
    csv_directory = '/home/antoniolujano/MonotonicNeuralNetworks/src/resultsExps'
    csv_files = [
        'expsMLP.csv', 'expsMLP_weightsconstrained.csv', 'exps_minmax.csv',
        'exps_minmax_aux.csv', 'exps_smooth_minmax.csv', 'exps_smooth_minmax_aux.csv',
        'expsCoMNN_type1.csv', 'expsLMNN_lip1.csv', 'expsSMNN.csv'
    ]

    csv_files = [
        'expsMLP.csv', 'expsMLP_weightsconstrained.csv', 'exps_minmax.csv',
        'exps_smooth_minmax.csv', 'expsCoMNN_type1.csv', 'expsLMNN_lip1.csv', 'expsSMNN.csv'
    ]

    csv_files = [
        'exps_minmax.csv','exps_minmax_aux.csv', 'exps_smooth_minmax.csv', 'exps_smooth_minmax_aux.csv'
    ]

    data_dict = read_csv_files(csv_files, csv_directory)

    performance_df, performance_std_df = create_performance_df(data_dict)
    params_df = create_params_df(data_dict)
    mono_dfs = create_mono_dfs(data_dict)

    try:
        performance_latex = df_to_latex(performance_df, performance_std_df, "Performance Metrics", "tab:performance",
                                        bold_min_value)
        print("Performance Table created successfully")
        print(performance_latex)
    except Exception as e:
        print(f"Error creating Performance Table: {str(e)}")

    try:
        params_latex = params_df_to_latex(params_df, "Number of Parameters", "tab:params")
        print("Parameters Table created successfully")
        print(params_latex)
    except Exception as e:
        print(f"Error creating Parameters Table: {str(e)}")

    """
    for metric, (df, std_df) in mono_dfs.items():
        try:
            mono_latex = df_to_latex(df, std_df, f"{metric} Metrics", f"tab:{metric.lower().replace(' ', '_')}",
                                     bold_max_value)
            print(f"{metric} Table created successfully")
            print(mono_latex)
        except Exception as e:
            print(f"Error creating {metric} Table: {str(e)}")
    """

    print("Performing Friedman test...")
    chi2, p_value = perform_friedman_test(performance_df)
    print(f"Friedman test statistic: {chi2}")
    print(f"Friedman test p-value: {p_value}")

    if p_value < 0.05:
        print("\nSignificant differences found. Performing Wilcoxon and Nemenyi tests...")

        print("\nPerforming Wilcoxon signed-rank tests...")
        wilcoxon_results = perform_wilcoxon_tests(performance_df)
        print("Wilcoxon test results:")
        print(wilcoxon_results)

        print("\nPerforming Nemenyi post-hoc test...")
        nemenyi_results = perform_nemenyi_test(performance_df)
        print("Nemenyi test results:")
        print(nemenyi_results)

        #print("\nCreating Critical Difference Diagram...")
        #create_critical_difference_diagram(performance_df)

        # Create LaTeX tables
        wilcoxon_latex = create_latex_table(wilcoxon_results, "Wilcoxon Test Results", "tab:wilcoxon")
        nemenyi_latex = create_latex_table(nemenyi_results, "Nemenyi Test Results", "tab:nemenyi")

        print("\nWilcoxon Test Results (LaTeX):")
        print(wilcoxon_latex)
        print("\nNemenyi Test Results (LaTeX):")
        print(nemenyi_latex)
    else:
        print("\nNo significant differences found. Skipping Wilcoxon and Nemenyi tests.")


if __name__ == "__main__":
    main()