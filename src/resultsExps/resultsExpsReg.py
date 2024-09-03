import pandas as pd
import os
import numpy as np
from scikit_posthocs import posthoc_nemenyi_friedman
from scipy.stats import wilcoxon, friedmanchisquare
import matplotlib.pyplot as plt
from scikit_posthocs import critical_difference_diagram
import seaborn as sns
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.stats")

def read_csv_files(file_list, directory):
    data_dict = {}
    for file in file_list:
        method_name = os.path.splitext(file)[0]
        full_path = os.path.join(directory, file)
        df = pd.read_csv(full_path)
        data_dict[method_name] = df
    return data_dict


def safe_float_convert(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        print(f"Warning: Could not convert '{value}' to float. Returning original value.")
        return value

import json

def extract_monotonicity_weight(config_str):
    try:
        config = json.loads(config_str.replace("'", '"'))
        return config.get('monotonicity_weight', 'N/A')
    except json.JSONDecodeError:
        print(f"Error parsing JSON: {config_str}")
        return 'N/A'

def create_performance_df(data_dict):
    performance_data = []
    for method, df in data_dict.items():
        for _, row in df.iterrows():
            dataset = row['Dataset']
            weight = extract_monotonicity_weight(row['Best Configuration'])
            performance_data.append({
                'Dataset': dataset,
                'Method': method,
                'Weight': weight,
                'Metric Value': safe_float_convert(row['Metric Value']),
                'Metric Std Dev': safe_float_convert(row['Metric Std Dev'])
            })
    return pd.DataFrame(performance_data)

def create_mono_dfs(data_dict):
    mono_metrics = ['Mono Random', 'Mono Train', 'Mono Val']
    mono_data = {metric: [] for metric in mono_metrics}

    for method, df in data_dict.items():
        for _, row in df.iterrows():
            dataset = row['Dataset']
            weight = extract_monotonicity_weight(row['Best Configuration'])
            for metric in mono_metrics:
                mono_data[metric].append({
                    'Dataset': dataset,
                    'Method': method,
                    'Weight': weight,
                    'Metric Value': safe_float_convert(row[f'{metric} Mean']),
                    'Metric Std Dev': safe_float_convert(row[f'{metric} Std'])
                })
    return {metric: pd.DataFrame(data) for metric, data in mono_data.items()}


def format_value_with_std(value, std, include_std=True):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)):
        if include_std:
            if pd.isna(std):
                return f"{value:.4f}"
            return f"{value:.4f} $\\pm$ {std:.4f}"
        else:
            return f"{value:.4f}"
    else:
        return str(value)  # Return as string if it's not a number


def df_to_latex(df, caption, label, bold_func, include_std=False):
    methods = sorted(df['Method'].unique())
    weights = sorted([w for w in df['Weight'].unique() if w != 'N/A'])

    # Filter out datasets with only one method
    datasets = df.groupby('Dataset').filter(lambda x: len(x['Method'].unique()) > 1)['Dataset'].unique()
    datasets = sorted(datasets)

    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{" + caption + "}")
    latex_lines.append("\\label{" + label + "}")
    latex_lines.append("\\resizebox{\\textwidth}{!}{")
    latex_lines.append(
        "\\begin{tabular}{l" + "c" * (len(methods) * (1 if 'expsMLP' in methods else len(weights))) + "}")
    latex_lines.append("\\toprule")

    # Top-level header (methods)
    header_top = ["\\multirow{2}{*}{Dataset}"]
    for method in methods:
        if method == 'expsMLP':
            header_top.append(f"\\multicolumn{{1}}{{c}}{{expsMLP}}")
        else:
            header_top.append(f"\\multicolumn{{{len(weights)}}}{{c}}{{{method}}}")
    latex_lines.append(" & ".join(header_top) + " \\\\")

    # Bottom-level header (weights)
    header_bottom = [""]
    for method in methods:
        if method == 'expsMLP':
            header_bottom.append("")
        else:
            header_bottom.extend([f"w={w}" for w in weights])
    latex_lines.append(" & ".join(header_bottom) + " \\\\")
    latex_lines.append("\\midrule")

    # Data rows
    for dataset in datasets:
        values = [dataset]
        for method in methods:
            if method == 'expsMLP':
                method_data = df[(df['Dataset'] == dataset) & (df['Method'] == method)]
                if not method_data.empty:
                    value = format_value_with_std(method_data['Metric Value'].iloc[0],
                                                  method_data['Metric Std Dev'].iloc[0], include_std)
                    values.append(bold_func(df[df['Dataset'] == dataset], method, value))
                else:
                    values.append("--")
            else:
                for weight in weights:
                    method_data = df[(df['Dataset'] == dataset) & (df['Method'] == method) & (df['Weight'] == weight)]
                    if not method_data.empty:
                        value = format_value_with_std(method_data['Metric Value'].iloc[0],
                                                      method_data['Metric Std Dev'].iloc[0], include_std)
                        values.append(bold_func(df[df['Dataset'] == dataset], method, value))
                    else:
                        values.append("--")
        latex_lines.append(" & ".join(values) + " \\\\")

    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)

def create_params_df(data_dict):
    all_datasets = set()
    for df in data_dict.values():
        all_datasets.update(df['Dataset'])
    all_datasets = sorted(list(all_datasets))

    params_data = []
    for dataset in all_datasets:
        row = {'Dataset': dataset}
        for method, df in data_dict.items():
            dataset_df = df[df['Dataset'] == dataset]
            if 'monotonicity_weight' in df.columns:
                # PWL case: find the params corresponding to the best Metric Value
                best_idx = dataset_df['Metric Value'].idxmin()
                row[f'{method}_params'] = dataset_df.loc[best_idx, 'NumOfParameters']
                row[f'{method}_weight'] = dataset_df.loc[best_idx, 'monotonicity_weight']
            else:
                # Non-PWL case
                if not dataset_df.empty:
                    row[f'{method}_params'] = dataset_df['NumOfParameters'].iloc[0]
                else:
                    row[f'{method}_params'] = np.nan
        params_data.append(row)

    return pd.DataFrame(params_data)



def bold_min_value(df, method, value):
    min_value = df['Metric Value'].min()
    return f"\\textbf{{{value}}}" if float(value.split()[0]) == min_value else value

def bold_max_value(df, method, value):
    max_value = df['Metric Value'].max()
    return f"\\textbf{{{value}}}" if float(value.split()[0]) == max_value else value


def params_df_to_latex(df, caption, label):
    methods = sorted(set([col.split('_')[0] for col in df.columns if col.endswith('_params')]))

    latex_lines = []
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{" + caption + "}")
    latex_lines.append("\\label{" + label + "}")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(methods) + "}")
    latex_lines.append("\\hline")

    # Header
    header = ["Dataset"] + methods
    latex_lines.append(" & ".join(header) + " \\\\")
    latex_lines.append("\\hline")

    # Data rows
    for _, row in df.iterrows():
        dataset = row['Dataset']
        values = []
        for method in methods:
            value = f"{row[f'{method}_params']:.0f}"
            if f'{method}_weight' in row:
                value += f" \\\\ ({row[f'{method}_weight']:.1f})"
            values.append(value)
        latex_lines.append(f"{dataset} & " + " & ".join(values) + " \\\\")
        latex_lines.append("\\hline")

    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")

    return "\n".join(latex_lines)


def perform_wilcoxon_tests(data):
    methods = list(data.keys())
    n_methods = len(methods)
    results = pd.DataFrame(index=methods, columns=methods)

    for i in range(n_methods):
        for j in range(i + 1, n_methods):
            method1 = methods[i]
            method2 = methods[j]
            try:
                _, p_value = wilcoxon(data[method1], data[method2])
                results.loc[method1, method2] = p_value
                results.loc[method2, method1] = p_value
            except Exception as e:
                print(f"Error performing Wilcoxon test between {method1} and {method2}: {str(e)}")
                results.loc[method1, method2] = np.nan
                results.loc[method2, method1] = np.nan

    return results


def perform_friedman_test(df):
    methods = [col.split('_')[0] for col in df.columns if col.endswith('_value')]
    try:
        chi2, p_value = friedmanchisquare(*[df[f'{method}_value'] for method in methods])
        return chi2, p_value
    except Exception as e:
        print(f"Error performing Friedman test: {str(e)}")
        return np.nan, np.nan


def perform_nemenyi_test(data):
    try:
        # Convert the data to the format expected by posthoc_nemenyi_friedman
        df = pd.DataFrame(data)
        return posthoc_nemenyi_friedman(df)
    except Exception as e:
        print(f"Error performing Nemenyi test: {str(e)}")
        print(f"Data shape: {df.shape}")
        print(f"Data columns: {df.columns}")
        print(f"Data types: {df.dtypes}")
        return None


def create_latex_table(df, caption, label):
    latex_table = df.to_latex(float_format="%.4f", caption=caption, label=label, escape=False)
    return latex_table


def create_critical_difference_diagram(df, filename='critical_difference_diagram.png'):
    methods = [col.split('_')[0] for col in df.columns if col.endswith('_value')]

    # Compute average ranks
    ranks = df[[f'{method}_value' for method in methods]].rank(axis=1, ascending=False)
    avg_ranks = ranks.mean().sort_values()

    # Prepare data for the diagram
    ranks_dict = {method: avg_ranks[f'{method}_value'] for method in methods}

    # Perform Nemenyi test to get the p-value matrix
    nemenyi_results = perform_nemenyi_test(df)

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


def perform_statistical_tests(df, metric_name):
    print(f"\nPerforming statistical tests for {metric_name}...")

    # Prepare data for Friedman test
    datasets = df['Dataset'].unique()
    methods = df['Method'].unique()
    data = {method: [] for method in methods}

    for dataset in datasets:
        for method in methods:
            method_data = df[(df['Dataset'] == dataset) & (df['Method'] == method)]
            if not method_data.empty:
                best_value = method_data['Metric Value'].min()  # Always use min for both performance and mono metrics
                data[method].append(best_value)
            else:
                data[method].append(np.nan)

    # Remove NaN values
    data = {method: [v for v in values if not np.isnan(v)] for method, values in data.items()}

    # Ensure all methods have the same number of values
    min_length = min(len(values) for values in data.values())
    data = {method: values[:min_length] for method, values in data.items()}

    try:
        chi2, p_value = friedmanchisquare(*data.values())
        print(f"Friedman test statistic: {chi2}")
        print(f"Friedman test p-value: {p_value}")

        if p_value < 0.05:
            print("\nSignificant differences found. Performing Wilcoxon and Nemenyi tests...")

            # Wilcoxon test
            wilcoxon_results = perform_wilcoxon_tests(pd.DataFrame(data))
            print("Wilcoxon test results:")
            print(wilcoxon_results)

            # Nemenyi test
            nemenyi_results = perform_nemenyi_test(data)
            if nemenyi_results is not None:
                print("Nemenyi test results:")
                print(nemenyi_results)

                # Create LaTeX tables
                wilcoxon_latex = create_latex_table(wilcoxon_results, f"Wilcoxon Test Results for {metric_name}",
                                                    f"tab:wilcoxon_{metric_name.lower().replace(' ', '_')}")
                nemenyi_latex = create_latex_table(nemenyi_results, f"Nemenyi Test Results for {metric_name}",
                                                   f"tab:nemenyi_{metric_name.lower().replace(' ', '_')}")

                print(f"\nWilcoxon Test Results for {metric_name} (LaTeX):")
                print(wilcoxon_latex)
                print(f"\nNemenyi Test Results for {metric_name} (LaTeX):")
                print(nemenyi_latex)
            else:
                print("Nemenyi test could not be performed due to an error.")
        else:
            print("\nNo significant differences found. Skipping Wilcoxon and Nemenyi tests.")
    except Exception as e:
        print(f"Error performing statistical tests: {str(e)}")


def is_pareto_efficient(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient


def dominance_analysis(df, methods, datasets, weights):
    dominance_counts = {method: 0 for method in methods}
    total_comparisons = len(datasets) * len(weights)

    # Identify the performance and monotonicity columns
    performance_col = 'Performance'
    monotonicity_col = 'Monotonicity'

    if performance_col not in df.columns or monotonicity_col not in df.columns:
        print(f"Error: Required columns not found. Available columns: {df.columns}")
        return None

    for dataset in datasets:
        for weight in weights:
            current_data = df[(df['Dataset'] == dataset) & (df['Weight'] == weight)]

            if len(current_data) > 1:  # Only perform analysis if we have more than one method for this combination
                costs = current_data[[performance_col, monotonicity_col]].values

                pareto_optimal = is_pareto_efficient(costs)

                for method, is_optimal in zip(current_data['Method'], pareto_optimal):
                    if is_optimal:
                        dominance_counts[method] += 1

    dominance_percentages = {method: count / total_comparisons * 100
                             for method, count in dominance_counts.items()}

    return dominance_percentages


def perform_dominance_analysis(df, metric_name):
    print(f"\nPerforming dominance analysis for {metric_name}...")

    methods = df['Method'].unique()
    datasets = df['Dataset'].unique()
    weights = df['Weight'].unique()

    dominance_results = dominance_analysis(df, methods, datasets, weights)

    if dominance_results is not None:
        print(f"Dominance analysis results for {metric_name}:")
        for method, percentage in dominance_results.items():
            print(f"{method}: {percentage:.2f}% Pareto-optimal solutions")
    else:
        print("Dominance analysis could not be performed due to missing columns.")

    return dominance_results


def plot_performance_monotonicity_tradeoff(performance_df, mono_df, output_file='tradeoff_plot.png'):
    # Merge performance and monotonicity dataframes
    combined_df = pd.merge(performance_df, mono_df, on=['Dataset', 'Method', 'Weight'])
    combined_df = combined_df.rename(columns={'Metric Value_x': 'Performance', 'Metric Value_y': 'Monotonicity'})

    plt.figure(figsize=(12, 8))
    methods = combined_df['Method'].unique()
    colors = sns.color_palette("husl", len(methods))

    for method, color in zip(methods, colors):
        method_data = combined_df[combined_df['Method'] == method]
        plt.scatter(method_data['Performance'], method_data['Monotonicity'],
                    label=method, color=color, alpha=0.7)

    plt.xlabel('Performance (Error Rate)')
    plt.ylabel('Monotonicity Violation Rate')
    plt.title('Performance vs. Monotonicity Trade-off')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim((0,1))
    plt.show()
    #plt.savefig(output_file, dpi=300, bbox_inches='tight')
    #plt.close()
    #print(f"Trade-off plot saved as {output_file}")


def plot_regularization_impact(performance_df, mono_df, output_file='regularization_impact.png'):
    combined_df = pd.merge(performance_df, mono_df, on=['Dataset', 'Method', 'Weight'])
    combined_df = combined_df.rename(columns={'Metric Value_x': 'Performance', 'Metric Value_y': 'Monotonicity'})

    plt.figure(figsize=(15, 10))

    # Performance subplot
    plt.subplot(2, 1, 1)
    for method in combined_df['Method'].unique():
        method_data = combined_df[combined_df['Method'] == method]
        plt.plot(method_data['Weight'], method_data['Performance'], marker='o', label=method)

    plt.xscale('log')
    plt.xlabel('Regularization Strength (λ)')
    plt.ylabel('Performance (Error Rate)')
    plt.title('Impact of Regularization Strength on Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Monotonicity subplot
    plt.subplot(2, 1, 2)
    for method in combined_df['Method'].unique():
        method_data = combined_df[combined_df['Method'] == method]
        plt.plot(method_data['Weight'], method_data['Monotonicity'], marker='o', label=method)

    plt.xscale('log')
    plt.xlabel('Regularization Strength (λ)')
    plt.ylabel('Monotonicity Violation Rate')
    plt.title('Impact of Regularization Strength on Monotonicity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    #plt.savefig(output_file, dpi=300, bbox_inches='tight')
    #plt.close()
    #print(f"Regularization impact plot saved as {output_file}")


def plot_regularization_impact_single_dataset(performance_df, mono_df, chosen_dataset=None, output_file='regularization_impact_dataset.png'):
    # Merge performance and monotonicity dataframes
    combined_df = pd.merge(performance_df, mono_df, on=['Dataset', 'Method', 'Weight'])
    combined_df = combined_df.rename(columns={'Metric Value_x': 'Performance', 'Metric Value_y': 'Monotonicity'})

    if chosen_dataset is None:
        print("No suitable dataset found.")
        return

    print(f"Chosen dataset: {chosen_dataset}")

    # Filter for the chosen dataset
    dataset_df = combined_df[combined_df['Dataset'] == chosen_dataset]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    methods = dataset_df['Method'].unique()
    colors = sns.color_palette("husl", len(methods))

    # Get MLP values
    mlp_data = dataset_df[dataset_df['Method'] == 'expsMLP']
    mlp_performance = mlp_data['Performance'].values[0] if not mlp_data.empty else None
    mlp_monotonicity = mlp_data['Monotonicity'].values[0] if not mlp_data.empty else None

    for method, color in zip(methods, colors):
        method_data = dataset_df[dataset_df['Method'] == method]

        if method != 'expsMLP':
            # Ensure Weight is numeric and sorted
            method_data['Weight'] = pd.to_numeric(method_data['Weight'], errors='coerce')
            method_data = method_data.sort_values('Weight')

            if not method_data.empty:
                # Performance plot
                ax1.plot(method_data['Weight'], method_data['Performance'], marker='o', color=color, label=method)

                # Monotonicity plot
                ax2.plot(method_data['Weight'], method_data['Monotonicity'], marker='o', color=color, label=method)
            else:
                print(f"No data for method: {method}")

    # Add MLP as horizontal lines
    if mlp_performance is not None:
        ax1.axhline(y=mlp_performance, color='red', linestyle='--', label='expsMLP')
    if mlp_monotonicity is not None:
        ax2.axhline(y=mlp_monotonicity, color='red', linestyle='--', label='expsMLP')

    ax1.set_xscale('log')
    ax1.set_xlabel('Regularization Strength (λ)')
    ax1.set_ylabel('Performance (Error Rate)')
    ax1.set_title(f'Impact of Regularization Strength on Performance ({chosen_dataset})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xscale('log')
    ax2.set_xlabel('Regularization Strength (λ)')
    ax2.set_ylabel('Monotonicity Violation Rate')
    ax2.set_title(f'Impact of Regularization Strength on Monotonicity ({chosen_dataset})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    # plt.savefig(output_file, dpi=300, bbox_inches='tight')
    # plt.close()
    # print(f"Regularization impact plot for {chosen_dataset} dataset saved as {output_file}")


def plot_combined_regularization_impact(performance_df, mono_dfs, chosen_dataset,
                                        output_file='combined_regularization_impact.png'):
    # Merge all dataframes
    combined_df = performance_df[performance_df['Dataset'] == chosen_dataset]
    for metric, df in mono_dfs.items():
        combined_df = pd.merge(combined_df, df[df['Dataset'] == chosen_dataset],
                               on=['Dataset', 'Method', 'Weight'], suffixes=('', f'_{metric}'))

    combined_df = combined_df.rename(columns={'Metric Value': 'Performance'})

    fig, axes = plt.subplots(4, 1, figsize=(12, 20), sharex=True)

    methods = combined_df['Method'].unique()
    colors = sns.color_palette("husl", len(methods))

    # Get MLP values
    mlp_data = combined_df[combined_df['Method'] == 'expsMLP']
    mlp_values = {
        'Performance': mlp_data['Performance'].values[0] if not mlp_data.empty else None,
        'Mono Random': mlp_data['Metric Value_Mono Random'].values[0] if not mlp_data.empty else None,
        'Mono Train': mlp_data['Metric Value_Mono Train'].values[0] if not mlp_data.empty else None,
        'Mono Val': mlp_data['Metric Value_Mono Val'].values[0] if not mlp_data.empty else None
    }

    metrics = ['Performance', 'Mono Random', 'Mono Train', 'Mono Val']

    for ax, metric in zip(axes, metrics):
        for method, color in zip(methods, colors):
            method_data = combined_df[combined_df['Method'] == method]

            if method != 'expsMLP':
                method_data['Weight'] = pd.to_numeric(method_data['Weight'], errors='coerce')
                method_data = method_data.sort_values('Weight')

                if not method_data.empty:
                    y_values = method_data['Performance'] if metric == 'Performance' else method_data[
                        f'Metric Value_{metric}']
                    ax.plot(method_data['Weight'], y_values, marker='o', color=color, label=method)

        # Add MLP as horizontal line
        if mlp_values[metric] is not None:
            ax.axhline(y=mlp_values[metric], color='red', linestyle='--', label='expsMLP')

        ax.set_xscale('log')
        ax.set_ylabel(f'{metric}\n(Error Rate)' if metric == 'Performance' else f'{metric}\n(Violation Rate)')
        ax.set_title(f'Impact of Regularization Strength on {metric} ({chosen_dataset})')
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Regularization Strength (λ)')

    plt.tight_layout()
    plt.show()
    #plt.savefig(output_file, dpi=300, bbox_inches='tight')
    #plt.close()
    #print(f"Combined regularization impact plot for {chosen_dataset} dataset saved as {output_file}")


def plot_dataset_characteristic_correlation(performance_df, dataset_characteristics, characteristic_name,
                                            output_file='dataset_correlation.png'):
    # Merge performance data with dataset characteristics
    combined_df = pd.merge(performance_df, dataset_characteristics, on='Dataset')

    plt.figure(figsize=(12, 8))
    methods = combined_df['Method'].unique()
    colors = sns.color_palette("husl", len(methods))

    for method, color in zip(methods, colors):
        method_data = combined_df[combined_df['Method'] == method]
        plt.scatter(method_data[characteristic_name], method_data['Metric Value'],
                    label=method, color=color, alpha=0.7)

        # Add trendline
        z = np.polyfit(method_data[characteristic_name], method_data['Metric Value'], 1)
        p = np.poly1d(z)
        plt.plot(method_data[characteristic_name], p(method_data[characteristic_name]), color=color, linestyle='--')

    plt.xlabel(characteristic_name)
    plt.ylabel('Performance (Error Rate)')
    plt.title(f'Dataset {characteristic_name} vs. Performance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    #plt.savefig(output_file, dpi=300, bbox_inches='tight')
    #plt.close()
    #print(f"Dataset characteristic correlation plot saved as {output_file}")

def main():
    csv_directory = '/home/antoniolujano/MonotonicNeuralNetworks/src/resultsExps'
    csv_files = [
        'expsMLP.csv', 'expsPWL.csv', 'expsUniformPWL.csv', 'expsMixupPWL.csv'
    ]

    data_dict = read_csv_files(csv_files, csv_directory)

    performance_df = create_performance_df(data_dict)
    mono_dfs = create_mono_dfs(data_dict)

    # Generate LaTeX tables
    performance_latex = df_to_latex(performance_df, "Performance Metrics", "tab:performance", bold_min_value, include_std=False)
    print("Performance Table:")
    print(performance_latex)

    for metric, df in mono_dfs.items():
        mono_latex = df_to_latex(df, f"{metric} Metrics", f"tab:{metric.lower().replace(' ', '_')}", bold_min_value, include_std=False)
        print(f"\n{metric} Table:")
        print(mono_latex)

    # Perform statistical tests
    #print("\nPerforming statistical tests on performance metrics...")
    #perform_statistical_tests(performance_df, "Performance Metrics")

    # Mono metrics
    for metric, df in mono_dfs.items():
        #print(f"\nPerforming statistical tests on {metric}...")
        #perform_statistical_tests(df, metric)

        # Perform dominance analysis
        combined_df = pd.merge(performance_df, df, on=['Dataset', 'Method', 'Weight'])
        combined_df = combined_df.rename(columns={'Metric Value_x': 'Performance', 'Metric Value_y': 'Monotonicity'})

        print("Combined DataFrame columns:", combined_df.columns)  # Debug print

        dominance_results = perform_dominance_analysis(combined_df, metric)

        if dominance_results is not None:
            # Generate LaTeX table for dominance analysis results
            dominance_df = pd.DataFrame(list(dominance_results.items()), columns=['Method', 'Percentage'])
            dominance_latex = create_latex_table(dominance_df, f"Dominance Analysis Results for {metric}",
                                                 f"tab:dominance_{metric.lower().replace(' ', '_')}")
            print(f"\nDominance Analysis Results for {metric} (LaTeX):")
            print(dominance_latex)

    print("\nAnalysis complete. LaTeX tables and statistical test results have been generated.")

if __name__ == "__main__":
    main()