import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ttest_ind
from scipy.stats import chi2, ks_2samp

## Import datasets

# import the following relationships between survey users and political influencers
follow_merge_2 = pd.read_csv('./follow_merge_2.csv', index_col=0)
# import the posts from survey users' followees with incivility detected
influencer_prediction = pd.read_csv('Influencer_Incivility_Predicted_timestamped.csv')
# only include the political influencers in the dataset
influencer_list = follow_merge_2['Target'].unique().tolist()
influencer_prediction = influencer_prediction[influencer_prediction['Username'].isin(influencer_list)]
# import the identity annotations for influencers
influencer_with_identity = pd.read_csv('influencer_annotation.csv')

## filter the predictions in which the positive probabilities are no smaller than 0.7
IMP_influencer = influencer_prediction[(influencer_prediction['labels_IMP'] ==1) & (influencer_prediction['probas_class_1_IMP'] >= 0.7)]
PHAVPR_influencer = influencer_prediction[(influencer_prediction['labels_PHAVPR'] ==1) & (influencer_prediction['probas_class_1_PHAVPR'] >= 0.7)]
HSST_influencer = influencer_prediction[(influencer_prediction['labels_HSST'] ==1) & (influencer_prediction['probas_class_1_HSST'] >= 0.7)]
THREAT_influencer = influencer_prediction[(influencer_prediction['labels_THREAT'] ==1) & (influencer_prediction['probas_class_1_THREAT'] >= 0.7)]

IMP_texts_freq = IMP_influencer.groupby('Username')['Text'].count().reset_index()
IMP_texts_freq.columns = ['Username', 'IMP Texts Frequency']
PHAVPR_texts_freq = PHAVPR_influencer.groupby('Username')['Text'].count().reset_index()
PHAVPR_texts_freq.columns = ['Username', 'PHAVPR Texts Frequency']
HSST_texts_freq = HSST_influencer.groupby('Username')['Text'].count().reset_index()
HSST_texts_freq.columns = ['Username', 'HSST Texts Frequency']
THREAT_texts_freq = THREAT_influencer.groupby('Username')['Text'].count().reset_index()
THREAT_texts_freq.columns = ['Username', 'THREAT Texts Frequency']
overall_texts_freq = influencer_prediction.groupby('Username')['Text'].count().reset_index()
overall_texts_freq.columns = ['Username', 'Overall Texts Frequency']

# Merge overall text frequencies into each uncivil dataset for density calculation
IMP_texts_freq = IMP_texts_freq.merge(overall_texts_freq, on='Username', how='left')
IMP_texts_freq['Probability_of_IMP_Text'] = IMP_texts_freq['IMP Texts Frequency'] / IMP_texts_freq['Overall Texts Frequency']
IMP_texts_freq = IMP_texts_freq[['Username', 'Probability_of_IMP_Text']]

PHAVPR_texts_freq = PHAVPR_texts_freq.merge(overall_texts_freq, on='Username', how='left')
PHAVPR_texts_freq['Probability_of_PHAVPR_Text'] = PHAVPR_texts_freq['PHAVPR Texts Frequency'] / PHAVPR_texts_freq['Overall Texts Frequency']
PHAVPR_texts_freq = PHAVPR_texts_freq[['Username', 'Probability_of_PHAVPR_Text']]

HSST_texts_freq = HSST_texts_freq.merge(overall_texts_freq, on='Username', how='left')
HSST_texts_freq['Probability_of_HSST_Text'] = HSST_texts_freq['HSST Texts Frequency'] / HSST_texts_freq['Overall Texts Frequency']
HSST_texts_freq = HSST_texts_freq[['Username', 'Probability_of_HSST_Text']]

THREAT_texts_freq = THREAT_texts_freq.merge(overall_texts_freq, on='Username', how='left')
THREAT_texts_freq['Probability_of_THREAT_Text'] = THREAT_texts_freq['THREAT Texts Frequency'] / THREAT_texts_freq['Overall Texts Frequency']
THREAT_texts_freq = THREAT_texts_freq[['Username', 'Probability_of_THREAT_Text']]

# Sort the uncivil influencers based on the probability of uncivil posts
IMP_sorted = IMP_texts_freq.sort_values('Probability_of_IMP_Text', ascending=True)
PHAVPR_sorted = PHAVPR_texts_freq.sort_values('Probability_of_PHAVPR_Text', ascending=True)
HSST_sorted  = HSST_texts_freq.sort_values('Probability_of_HSST_Text', ascending=True)
THREAT_sorted = THREAT_texts_freq.sort_values('Probability_of_THREAT_Text', ascending=True)


###### 1. Plot account types of uncivil influencers across density levels and dimensions

influencer_with_identity['Identity Exposure'] = influencer_with_identity.apply(
    lambda row: ' + '.join(
        filter(pd.notna, [row['Political ideology'], row['Personal support'], row['Social identity']])
    ) if row['Account type'] == 'Individual' and not (
        pd.isna(row['Political ideology']) and pd.isna(row['Personal support']) and pd.isna(row['Social identity'])
    ) else None,
    axis=1
)

# Function to divide list into four parts
def divide_list_into_four(df):
    column_name = df.columns[1]
    q1_threshold = df[column_name].quantile(0.25)
    q2_threshold = df[column_name].quantile(0.50)
    q3_threshold = df[column_name].quantile(0.75)
    q1 = df[df[column_name] <= q1_threshold]
    q2 = df[(df[column_name] > q1_threshold) & (df[column_name] <= q2_threshold)]
    q3 = df[(df[column_name] > q2_threshold) & (df[column_name] <= q3_threshold)]
    q4 = df[df[column_name] > q3_threshold]
    return q1, q2, q3, q4

# Divide influencer lists into four parts
IMP_q1, IMP_q2, IMP_q3, IMP_q4 = divide_list_into_four(IMP_sorted)
PHAVPR_q1, PHAVPR_q2, PHAVPR_q3, PHAVPR_q4 = divide_list_into_four(PHAVPR_sorted)
HSST_q1, HSST_q2, HSST_q3, HSST_q4 = divide_list_into_four(HSST_sorted)
THREAT_q1, THREAT_q2, THREAT_q3, THREAT_q4 = divide_list_into_four(THREAT_sorted)

# Function to calculate Account Type densities
def get_account_type_density(influencer_list, group_name):
    # Filter data for the given influencer list
    filtered_df = influencer_with_identity[influencer_with_identity['Username'].isin(influencer_list['Username'].tolist())]
    # Get the count of Account Types
    account_type_counts = filtered_df['Account type'].value_counts()
    total = account_type_counts.sum()
    account_type_density = account_type_counts / total
    return account_type_density.rename_axis('Account Type').rename(group_name)

# Calculate densities for all parts
densities = {
    'IMP_q1': get_account_type_density(IMP_q1, 'IMP_q1'),
    'IMP_q2': get_account_type_density(IMP_q2, 'IMP_q2'),
    'IMP_q3': get_account_type_density(IMP_q3, 'IMP_q3'),
    'IMP_q4': get_account_type_density(IMP_q4, 'IMP_q4'),
    'PHAVPR_q1': get_account_type_density(PHAVPR_q1, 'PHAVPR_q1'),
    'PHAVPR_q2': get_account_type_density(PHAVPR_q2, 'PHAVPR_q2'),
    'PHAVPR_q3': get_account_type_density(PHAVPR_q3, 'PHAVPR_q3'),
    'PHAVPR_q4': get_account_type_density(PHAVPR_q4, 'PHAVPR_q4'),
    'HSST_q1': get_account_type_density(HSST_q1, 'HSST_q1'),
    'HSST_q2': get_account_type_density(HSST_q2, 'HSST_q2'),
    'HSST_q3': get_account_type_density(HSST_q3, 'HSST_q3'),
    'HSST_q4': get_account_type_density(HSST_q4, 'HSST_q4'),
    'THREAT_q1': get_account_type_density(THREAT_q1, 'THREAT_q1'),
    'THREAT_q2': get_account_type_density(THREAT_q2, 'THREAT_q2'),
    'THREAT_q3': get_account_type_density(THREAT_q3, 'THREAT_q3'),
    'THREAT_q4': get_account_type_density(THREAT_q4, 'THREAT_q4')
}
# Combine all densities into a single DataFrame
density_df = pd.concat(densities.values(), axis=1).fillna(0)
density_df = density_df.apply(pd.to_numeric, errors='coerce')

# Plot the stacked bar chart
density_df = density_df.loc[['Individual', 'Media', 'Politician'], :]

category_colors = {
    'Individual': '#ADD8E6',
    'Media': '#4682B4',
    'Politician': '#00008B'
}

colors = [category_colors[category] for category in density_df.index]

group_boundaries = [4, 8, 12]

density_df.index = [tuple(x) if isinstance(x, list) else x for x in density_df.index]

fig, ax = plt.subplots(figsize=(20, 12))

category_colors = {
    'Individual': '#ADD8E6',
    'Media': '#4682B4',
    'Politician': '#00008B'
}

group_padding = 1
bar_positions = []
current_position = 0

for i in range(len(density_df.columns)):
    bar_positions.append(current_position)
    if (i + 1) % 4 == 0:
        current_position += group_padding
    current_position += 1

# Plot each category separately
bottom = pd.DataFrame(0, index=density_df.columns, columns=['bottom'])
for category in density_df.index:
    ax.bar(bar_positions, density_df.loc[category], bottom=bottom['bottom'],
           width=0.8, label=category, color=category_colors[category])
    bottom['bottom'] += density_df.loc[category]

ax.set_xticks([(bar_positions[i] + bar_positions[i + 3]) / 2 for i in range(0, len(bar_positions), 4)])

for tick, label in zip(ax.get_xticks(), ["IMP", "PHAVPR", "HSST", "THREAT"]):
    ax.text(tick, -0.05, f"{label}\n(q1 -q4)", ha='center', va='top', fontsize=40, transform=ax.get_xaxis_transform())

plt.ylabel('Proportion', fontsize=45)
plt.yticks(fontsize=35)
ax.legend(title='Account Type', loc='upper center', bbox_to_anchor=(0.5, -0.35),
          fontsize=40, title_fontsize=45, ncol=3)

plt.tight_layout()
plt.subplots_adjust(bottom=0.45)  # Adjust bottom margin for labels
plt.savefig("ICWSM_Fig_6_1.png", dpi=300, bbox_inches='tight')
plt.show()



## Perform statistical tests to examine the significance of difference of account type distribution
def get_account_type_data(influencer_list, group_name):
    filtered_df = influencer_with_identity[
        influencer_with_identity['Username'].isin(influencer_list['Username'].tolist())]
    return filtered_df['Account type'].rename(group_name)

dataframes = {
    'IMP_q1': get_account_type_data(IMP_q1, 'IMP_q1'),
    'IMP_q2': get_account_type_data(IMP_q2, 'IMP_q2'),
    'IMP_q3': get_account_type_data(IMP_q3, 'IMP_q3'),
    'IMP_q4': get_account_type_data(IMP_q4, 'IMP_q4'),
    'PHAVPR_q1': get_account_type_data(PHAVPR_q1, 'PHAVPR_q1'),
    'PHAVPR_q2': get_account_type_data(PHAVPR_q2, 'PHAVPR_q2'),
    'PHAVPR_q3': get_account_type_data(PHAVPR_q3, 'PHAVPR_q3'),
    'PHAVPR_q4': get_account_type_data(PHAVPR_q4, 'PHAVPR_q4'),
    'HSST_q1': get_account_type_data(HSST_q1, 'HSST_q1'),
    'HSST_q2': get_account_type_data(HSST_q2, 'HSST_q2'),
    'HSST_q3': get_account_type_data(HSST_q3, 'HSST_q3'),
    'HSST_q4': get_account_type_data(HSST_q4, 'HSST_q4'),
    'THREAT_q1': get_account_type_data(THREAT_q1, 'THREAT_q1'),
    'THREAT_q2': get_account_type_data(THREAT_q2, 'THREAT_q2'),
    'THREAT_q3': get_account_type_data(THREAT_q3, 'THREAT_q3'),
    'THREAT_q4': get_account_type_data(THREAT_q4, 'THREAT_q4'),
}

combined_df = pd.concat(dataframes, axis=1)

# Perform G-test for each group
results = {}
for group_name in ['IMP', 'PHAVPR', 'HSST', 'THREAT']:
    group_columns = [col for col in combined_df.columns if col.startswith(group_name)]
    group_data = combined_df[group_columns]

    # Prepare contingency table for G-test
    contingency_table = pd.concat([group_data[col].value_counts() for col in group_columns], axis=1).fillna(0)
    observed = contingency_table.values

    # Perform G-test
    chi2, p_val, dof, expected = chi2_contingency(observed, lambda_="log-likelihood")
    results[group_name] = {'G-test Chi2': chi2, 'G-test P-value': p_val}

# Perform K-S test for neighboring pairs within each group
def ks_test_neighboring_pairs(data, group_name):
    group_columns = [col for col in data.columns if col.startswith(group_name)]
    ks_test_results = []

    for i in range(len(group_columns) - 1):
        col1 = data[group_columns[i]].astype('category').cat.codes.dropna()
        col2 = data[group_columns[i + 1]].astype('category').cat.codes.dropna()
        ks_stat, p_val = ks_2samp(col1, col2)
        ks_test_results.append({
            'Group': group_columns[i],
            'Comparison': group_columns[i + 1],
            'K-S Statistic': ks_stat,
            'P-value': p_val
        })
    return pd.DataFrame(ks_test_results)

# Add K-S test results to the existing G-test results
for group_name in ['IMP', 'PHAVPR', 'HSST', 'THREAT']:
    ks_test_results = ks_test_neighboring_pairs(combined_df, group_name)
    results[group_name]['K-S Test Results'] = ks_test_results

# Print G-test and K-S test results
for group, stats in results.items():
    print(f"Results for {group}:")
    print(f"  G-test Chi2: {stats['G-test Chi2']:.4f}, P-value: {stats['G-test P-value']:.4f}")
    print("  K-S Test Results:")
    print(stats['K-S Test Results'])
    print("\n")

    # Save K-S test results to CSV
    stats['K-S Test Results'].to_csv(f"AT_K-S_Test_Results_{group}.csv", index=False)

# Create a summary of G-test results
g_test_summary = pd.DataFrame(
    [(group, stats['G-test Chi2'], stats['G-test P-value']) for group, stats in results.items()],
    columns=['Group', 'G-test Chi2', 'G-test P-value']
)

# Print G-test summary and save as CSV
print("G-test Summary:")
print(g_test_summary)
g_test_summary.to_csv("AT_G-test_Summary.csv", index=False)


######### 2. Plot exposed identities of uncivil influencers across density levels and dimensions
def get_individual_identity_density(influencer_list, group_name):

    filtered_df = influencer_with_identity[(influencer_with_identity['Username'].isin(influencer_list['Username'].tolist())) & (influencer_with_identity['Account type'] == 'Individual')]

    account_type_counts = filtered_df['Identity Exposure'].value_counts()

    total = account_type_counts.sum()
    account_type_density = account_type_counts / total

    account_type_density_sorted = account_type_density.sort_values(ascending=False)
    cumulative_density = 0
    filtered_density = {}

    for category, density in account_type_density_sorted.items():
        cumulative_density += density
        filtered_density[category] = density
        if cumulative_density >= 0.8:
            break
    filtered_density_series = pd.Series(filtered_density)

    warm_colors = {
        "L": "#E69F00",
        "Lula camp": "#F28E2B",
        "L + Lula camp": "#FCB97D",
        "L + Woman": "#FAA275"
    }

    cold_colors = {
        "R": "#56B4E9",
        "Bolsonaro camp": "#4C72B0",
        "R + Bolsonaro camp": "#377EB8",
        "R + Religious": "#2A5D90"
    }

    neutral_colors = {
        "Woman": "#999999",
        "Religious": "#A6A6A6",
        "LGBTQ": "#117733"
    }

    color_map = {**warm_colors, **cold_colors, **neutral_colors}

    category_colors = {category: color_map.get(category, "#000000") for category in filtered_density_series.index}

    return filtered_density_series.rename_axis('Identity Exposure').rename(group_name), category_colors

densities = {
    'IMP_q1': get_individual_identity_density(IMP_q1, 'IMP_q1'),
    'IMP_q2': get_individual_identity_density(IMP_q2, 'IMP_q2'),
    'IMP_q3': get_individual_identity_density(IMP_q3, 'IMP_q3'),
    'IMP_q4': get_individual_identity_density(IMP_q4, 'IMP_q4'),
    'PHAVPR_q1': get_individual_identity_density(PHAVPR_q1, 'PHAVPR_q1'),
    'PHAVPR_q2': get_individual_identity_density(PHAVPR_q2, 'PHAVPR_q2'),
    'PHAVPR_q3': get_individual_identity_density(PHAVPR_q3, 'PHAVPR_q3'),
    'PHAVPR_q4': get_individual_identity_density(PHAVPR_q4, 'PHAVPR_q4'),
    'HSST_q1': get_individual_identity_density(HSST_q1, 'HSST_q1'),
    'HSST_q2': get_individual_identity_density(HSST_q2, 'HSST_q2'),
    'HSST_q3': get_individual_identity_density(HSST_q3, 'HSST_q3'),
    'HSST_q4': get_individual_identity_density(HSST_q4, 'HSST_q4'),
    'THREAT_q1': get_individual_identity_density(THREAT_q1, 'THREAT_q1'),
    'THREAT_q2': get_individual_identity_density(THREAT_q2, 'THREAT_q2'),
    'THREAT_q3': get_individual_identity_density(THREAT_q3, 'THREAT_q3'),
    'THREAT_q4': get_individual_identity_density(THREAT_q4, 'THREAT_q4')}

density_df = pd.concat([density[0] for density in densities.values()], axis=1).fillna(0)
density_df = density_df.apply(pd.to_numeric, errors='coerce')

desired_order = [
    "L", "Lula camp", "L + Lula camp", "L + Woman",
    "R", "Bolsonaro camp", "R + Bolsonaro camp", "R + Religious",
    "Woman", "Religious", "LGBTQ"
]

density_df = density_df.reindex(desired_order)

group_boundaries = [4, 8, 12, 16]

all_category_colors = {}
for _, category_colors in densities.values():
    all_category_colors.update(category_colors)

color_order = {category: all_category_colors.get(category, "#000000") for category in density_df.index}

density_df.index = [tuple(x) if isinstance(x, list) else x for x in density_df.index]

fig, ax = plt.subplots(figsize=(20, 12))

group_padding = 1
bar_positions = []
current_position = 0

for i in range(len(density_df.columns)):
    bar_positions.append(current_position)
    if (i + 1) % 4 == 0:
        current_position += group_padding
    current_position += 1

# Plot each category separately
bottom = pd.DataFrame(0, index=density_df.columns, columns=['bottom'])
for category in density_df.index:
    if density_df.loc[category].isnull().any():
        print(f"Skipping {category} due to NaN values")
        continue
    ax.bar(bar_positions, density_df.loc[category], bottom=bottom['bottom'],
           width=0.8, label=category, color=color_order[category])
    bottom['bottom'] += density_df.loc[category]

ax.set_xticks([(bar_positions[i] + bar_positions[i + 3]) / 2 for i in range(0, len(bar_positions), 4)])

for tick, label in zip(ax.get_xticks(), ["IMP", "PHAVPR", "HSST", "THREAT"]):
    ax.text(tick, -0.05, f"{label}\n(q1 -q4)", ha='center', va='top', fontsize=35, transform=ax.get_xaxis_transform())

plt.ylabel('Proportion', fontsize=40)
plt.yticks(fontsize=30)
legend_labels = list(density_df.index)
handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in list(color_order.values())]

legend = plt.legend(handles, legend_labels, title='Identity Exposure',
                    loc='upper center', bbox_to_anchor=(0.5, -0.35),
                    fontsize=35, title_fontsize=40, ncol=3)
plt.gca().add_artist(legend)

plt.tight_layout()
plt.subplots_adjust(bottom=0.55, right=0.9) 
plt.savefig("ICWSM_Fig_6_2.png", dpi=300, bbox_inches='tight', bbox_extra_artists=[plt.gca().get_legend()])
plt.show()


## Perform statistical tests to examine the significance of difference of exposed identity distribution
def get_identity_data(influencer_list, group_name):
    filtered_df = influencer_with_identity[
        influencer_with_identity['Username'].isin(influencer_list['Username'].tolist())
    ]
    return filtered_df['Identity Exposure'].rename(group_name)

identity_exposure_dataframes = {
    'IMP_q1': get_identity_data(IMP_q1, 'IMP_q1'),
    'IMP_q2': get_identity_data(IMP_q2, 'IMP_q2'),
    'IMP_q3': get_identity_data(IMP_q3, 'IMP_q3'),
    'IMP_q4': get_identity_data(IMP_q4, 'IMP_q4'),
    'PHAVPR_q1': get_identity_data(PHAVPR_q1, 'PHAVPR_q1'),
    'PHAVPR_q2': get_identity_data(PHAVPR_q2, 'PHAVPR_q2'),
    'PHAVPR_q3': get_identity_data(PHAVPR_q3, 'PHAVPR_q3'),
    'PHAVPR_q4': get_identity_data(PHAVPR_q4, 'PHAVPR_q4'),
    'HSST_q1': get_identity_data(HSST_q1, 'HSST_q1'),
    'HSST_q2': get_identity_data(HSST_q2, 'HSST_q2'),
    'HSST_q3': get_identity_data(HSST_q3, 'HSST_q3'),
    'HSST_q4': get_identity_data(HSST_q4, 'HSST_q4'),
    'THREAT_q1': get_identity_data(THREAT_q1, 'THREAT_q1'),
    'THREAT_q2': get_identity_data(THREAT_q2, 'THREAT_q2'),
    'THREAT_q3': get_identity_data(THREAT_q3, 'THREAT_q3'),
    'THREAT_q4': get_identity_data(THREAT_q4, 'THREAT_q4'),
}

combined_df = pd.concat(identity_exposure_dataframes, axis=1)

results = {}
for group_name in ['IMP', 'PHAVPR', 'HSST', 'THREAT']:
    group_columns = [col for col in combined_df.columns if col.startswith(group_name)]
    group_data = combined_df[group_columns]

    # Prepare contingency table for G-test
    contingency_table = pd.concat([group_data[col].value_counts() for col in group_columns], axis=1).fillna(0)
    observed = contingency_table.values

    # Perform G-test
    chi2, p_val, dof, expected = chi2_contingency(observed, lambda_="log-likelihood")
    results[group_name] = {'G-test Chi2': chi2, 'G-test P-value': p_val}

# Add K-S test results to the existing G-test workflow
for group_name in ['IMP', 'PHAVPR', 'HSST', 'THREAT']:
    # Perform K-S test for neighboring pairs
    ks_test_results = ks_test_neighboring_pairs(combined_df, group_name)
    results[group_name]['K-S Test Results'] = ks_test_results

# Print G-test and K-S test results
for group, stats in results.items():
    print(f"Results for {group}:")
    print(f"  G-test Chi2: {stats['G-test Chi2']:.4f}, P-value: {stats['G-test P-value']:.4f}")
    print("  K-S Test Results:")
    print(stats['K-S Test Results'])
    print("\n")

    # Save K-S test results to CSV
    stats['K-S Test Results'].to_csv(f"IE_K-S_Test_Results_{group}.csv", index=False)

# Create a summary of G-test results
g_test_summary = pd.DataFrame(
    [(group, stats['G-test Chi2'], stats['G-test P-value']) for group, stats in results.items()],
    columns=['Group', 'G-test Chi2', 'G-test P-value']
)

# Print G-test summary and save as CSV
print("G-test Summary:")
print(g_test_summary)
g_test_summary.to_csv("IE_G-test_Summary.csv", index=False)




