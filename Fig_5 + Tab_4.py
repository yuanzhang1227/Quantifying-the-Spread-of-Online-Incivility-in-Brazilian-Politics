import pandas as pd
import numpy as np
import re
import pickle
from matplotlib.gridspec import GridSpec
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

## Import datasets

# import the following relationships between survey users and political influencers
followship = pd.read_csv('./followship.csv', index_col=0)
# import the posts from survey users' followees with incivility detected
influencer_prediction = pd.read_csv('./Influencer_Incivility_Predicted_timestamped.csv')
# only include the political influencers in the dataset
influencer_list = followship['Target'].unique().tolist()
influencer_prediction = influencer_prediction[influencer_prediction['Username'].isin(influencer_list)]

## filter the predictions in which the positive probabilities are no smaller than 0.7
IMP_influencer = influencer_prediction[(influencer_prediction['labels_IMP'] ==1) & (influencer_prediction['probas_class_1_IMP'] >= 0.7)]
PHAVPR_influencer = influencer_prediction[(influencer_prediction['labels_PHAVPR'] ==1) & (influencer_prediction['probas_class_1_PHAVPR'] >= 0.7)]
HSST_influencer = influencer_prediction[(influencer_prediction['labels_HSST'] ==1) & (influencer_prediction['probas_class_1_HSST'] >= 0.7)]
THREAT_influencer = influencer_prediction[(influencer_prediction['labels_THREAT'] ==1) & (influencer_prediction['probas_class_1_THREAT'] >= 0.7)]

############ 1. Plot the distribution of exposed audience at different uncivil densities
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

# Calculate the number of exposed audience for each uncivil influencer
def extract_retweeted_account(text):
    match = re.search(r"RT @(\w+):", text)
    return match.group(1) if match else None

influencer_prediction['retweeted_account'] = influencer_prediction['Text'].apply(extract_retweeted_account)
influencer_prediction_retweets = influencer_prediction.dropna(subset=['retweeted_account'])
merged = followship.merge(influencer_prediction_retweets[['Username', 'retweeted_account']],
                              left_on='Target', right_on='Username', how='left').dropna(subset=['retweeted_account'])
new_rows = merged[['Source', 'retweeted_account']].rename(columns={'retweeted_account': 'Target'})
followship_updated = pd.concat([followship, new_rows], ignore_index=True).drop_duplicates()
followers_count = followship_updated.groupby('Target')['Source'].nunique().rename('Level_of_Exposure')

# Define data preparation function by merging the incivility density data and audience data
def prepare_histogram_data(sorted_data, followers):
    probability_column = sorted_data.columns[1]
    matching_data = pd.merge(
        sorted_data,
        followers.reset_index(),
        left_on='Username',
        right_on='Target',
        how='left'
    )
    matching_data['Level_of_Exposure'] = matching_data['Level_of_Exposure'].fillna(0)
    matching_data = matching_data.rename(columns={probability_column: 'Probability'})
    return matching_data[['Username', 'Target', 'Level_of_Exposure', 'Probability']]

def fit_quantile_regression_and_plot_combined(plot_list, followers_count):
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig)

    for i, (sorted_data, column_name, dataset_name) in enumerate(plot_list):
        followers_data = prepare_histogram_data(sorted_data, followers_count)

        # Fit quantile regression for multiple quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        models = {}
        coefficients = []

        for q in quantiles:
            model = smf.quantreg('Level_of_Exposure ~ Probability', followers_data).fit(q=q)
            models[q] = model
            coef = model.params['Probability']
            sig = '***' if model.pvalues['Probability'] < 0.01 else '**' if model.pvalues[
                                                                                'Probability'] < 0.05 else '*' if \
                model.pvalues['Probability'] < 0.1 else ''
            coefficients.append((q, coef, sig))
        print(f"Quantile Regression Results of:{dataset_name}")
        print(coefficients)

        ax_main = fig.add_subplot(gs[i // 2, i % 2])

        sns.kdeplot(
            x=followers_data['Probability'],
            y=followers_data['Level_of_Exposure'],
            cmap='coolwarm', fill=True,
            thresh=0,
            levels=20,
            bw_adjust=0.5,
            ax=ax_main,
            alpha=0.6
        )

        sns.kdeplot(
            x=followers_data['Probability'],
            y=followers_data['Level_of_Exposure'],
            cmap='coolwarm',
            ax=ax_main,
            contour=True,
            levels=10,
            linewidths=1.5
        )

        # Add histogram on the top (x-axis)
        ax_top = ax_main.inset_axes([0.1, 1.15, 1, 0.2])

        ax_top.hist(
            followers_data['Probability'], bins=30, color='grey', alpha=0.7, density=True
        )
        ax_top.set_ylabel(r'$d$', fontsize=35)
        ax_top.spines['right'].set_visible(False)
        ax_top.spines['top'].set_visible(False)
        ax_top.tick_params(axis='x', labelsize=30)
        ax_top.tick_params(axis='y', labelsize=30)

        # Histogram on the right (y-axis)
        ax_right = ax_main.inset_axes([1.15, 0, 0.2, 1])

        ax_right.hist(
            followers_data['Level_of_Exposure'], bins=30, color='grey', alpha=0.7, density=True,
            orientation='horizontal'
        )
        ax_right.set_xlabel(r'$d$', fontsize=35)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['top'].set_visible(False)
        ax_right.tick_params(axis='x', labelsize=30)
        ax_right.tick_params(axis='y', labelsize=30)

        ax_main.spines['top'].set_visible(False)
        ax_main.spines['right'].set_visible(False)
        ax_main.grid(which="both", linestyle="--", linewidth=0.5, alpha=0.7)

        ax_main.set_xlabel(column_name, fontsize=42)
        ax_main.set_ylabel("Exposure Count", fontsize=42)
        ax_main.tick_params(axis='x', labelsize=30)
        ax_main.tick_params(axis='y', labelsize=30)

    plt.tight_layout()
    plt.savefig('./ICWSM_Fig_5.png', dpi=300, bbox_inches='tight')
    plt.show()


plot_list = [
    (IMP_sorted, 'Ratio of IMP', 'IMP Dataset'),
    (PHAVPR_sorted, 'Ratio of PHAVPR', 'PHAVPR Dataset'),
    (HSST_sorted, 'Ratio of HSST', 'HSST Dataset'),
    (THREAT_sorted, 'Ratio of THREAT', 'THREAT Dataset')
]

fit_quantile_regression_and_plot_combined(plot_list, followers_count)


## Plot the similarity of uncivil influencers across density levels and dimensions

# Function to divide dataframe into four parts
def divide_into_four(df):
    column_name = df.columns[1]
    q1_threshold = df[column_name].quantile(0.25)
    q2_threshold = df[column_name].quantile(0.50)
    q3_threshold = df[column_name].quantile(0.75)

    q1 = df[df[column_name] <= q1_threshold]['Username'].tolist()
    q2 = df[(df[column_name] > q1_threshold) & (df[column_name] <= q2_threshold)]['Username'].tolist()
    q3 = df[(df[column_name] > q2_threshold) & (df[column_name] <= q3_threshold)]['Username'].tolist()
    q4 = df[df[column_name] > q3_threshold]['Username'].tolist()
    return q1, q2, q3, q4


# Divide each sorted dataframe into four sections
IMP_q1, IMP_q2, IMP_q3, IMP_q4 = divide_into_four(IMP_sorted)
PHAVPR_q1, PHAVPR_q2, PHAVPR_q3, PHAVPR_q4 = divide_into_four(PHAVPR_sorted)
HSST_q1, HSST_q2, HSST_q3, HSST_q4 = divide_into_four(HSST_sorted)
THREAT_q1, THREAT_q2, THREAT_q3, THREAT_q4 = divide_into_four(THREAT_sorted)

# Combine the four parts into a dictionary
list_dict = {
    "IMP_q1": set(IMP_q1), "PHAVPR_q1": set(PHAVPR_q1), "HSST_q1": set(HSST_q1), "THREAT_q1": set(THREAT_q1),
    "IMP_q2": set(IMP_q2), "PHAVPR_q2": set(PHAVPR_q2), "HSST_q2": set(HSST_q2), "THREAT_q2": set(THREAT_q2),
    "IMP_q3": set(IMP_q3), "PHAVPR_q3": set(PHAVPR_q3), "HSST_q3": set(HSST_q3), "THREAT_q3": set(THREAT_q3),
    "IMP_q4": set(IMP_q4), "PHAVPR_q4": set(PHAVPR_q4), "HSST_q4": set(HSST_q4), "THREAT_q4": set(THREAT_q4),
}


## Plot the similarity of exposed audience across density levels and dimensions
with open("network_username_to_id.pkl", 'rb') as file:
    username_to_id = pickle.load(file)
def prepare_follower_data(sorted_data, followers):
    matching_data = pd.merge(
        sorted_data,
        followers.reset_index(),
        left_on='Username',
        right_on='Target',
        how='left'
    )
    matching_data['follower_list'] = matching_data['follower_list'].fillna(0)
    return matching_data[['Username', 'Target', 'follower_list', sorted_data.columns[1]]]

follower_list = followship_updated.groupby('Target')['Source'].apply(lambda x: list(x.unique())).reset_index(name='follower_list')

# IMP
IMP_data = prepare_follower_data(IMP_sorted, follower_list)
unique_IMP_followers = list(set(follower for sublist in IMP_data['follower_list'] for follower in sublist))
unique_IMP_followers_ids = [username_to_id[user] for user in unique_IMP_followers if user in username_to_id]
print("Number of followers:")
print(len(unique_IMP_followers_ids))

# PHAVPR
PHAVPR_data = prepare_follower_data(PHAVPR_sorted, follower_list)
unique_PHAVPR_followers = list(set(follower for sublist in PHAVPR_data['follower_list'] for follower in sublist))
unique_PHAVPR_followers_ids = [username_to_id[user] for user in unique_PHAVPR_followers if user in username_to_id]
print(len(unique_PHAVPR_followers_ids))

# HSST
HSST_data = prepare_follower_data(HSST_sorted, follower_list)
unique_HSST_followers = list(set(follower for sublist in HSST_data['follower_list'] for follower in sublist))
unique_HSST_followers_ids = [username_to_id[user] for user in unique_HSST_followers if user in username_to_id]
print(len(unique_HSST_followers_ids))

# THREAT
THREAT_data = prepare_follower_data(THREAT_sorted, follower_list)
unique_THREAT_followers = list(set(follower for sublist in THREAT_data['follower_list'] for follower in sublist))
unique_THREAT_followers_ids = [username_to_id[user] for user in unique_THREAT_followers if user in username_to_id]
print(len(unique_THREAT_followers_ids))

# Divide each unique followers list into four sections
def divide_follower_list_into_four(df):
    column_name = df.columns[3]
    q1_threshold = df[column_name].quantile(0.25)
    q2_threshold = df[column_name].quantile(0.50)
    q3_threshold = df[column_name].quantile(0.75)
    q1 = df[df[column_name] <= q1_threshold]
    q2 = df[(df[column_name] > q1_threshold) & (df[column_name] <= q2_threshold)]
    q3 = df[(df[column_name] > q2_threshold) & (df[column_name] <= q3_threshold)]
    q4 = df[df[column_name] > q3_threshold]
    return q1, q2, q3, q4

IMP_followers_q1, IMP_followers_q2, IMP_followers_q3, IMP_followers_q4 = divide_follower_list_into_four(IMP_data)
PHAVPR_followers_q1, PHAVPR_followers_q2, PHAVPR_followers_q3, PHAVPR_followers_q4 = divide_follower_list_into_four(PHAVPR_data)
HSST_followers_q1, HSST_followers_q2, HSST_followers_q3, HSST_followers_q4 = divide_follower_list_into_four(HSST_data)
THREAT_followers_q1, THREAT_followers_q2, THREAT_followers_q3, THREAT_followers_q4 = divide_follower_list_into_four(THREAT_data)

# Combine the divided parts into a dictionary for Jaccard similarity calculation
followers_dict = {
    "IMP_q1": set(follower for sublist in IMP_followers_q1['follower_list'] for follower in sublist),
    "PHAVPR_q1": set(follower for sublist in PHAVPR_followers_q1['follower_list'] for follower in sublist),
    "HSST_q1": set(follower for sublist in HSST_followers_q1['follower_list'] for follower in sublist),
    "THREAT_q1": set(follower for sublist in THREAT_followers_q1['follower_list'] for follower in sublist),
    "IMP_q2": set(follower for sublist in IMP_followers_q2['follower_list'] for follower in sublist),
    "PHAVPR_q2": set(follower for sublist in PHAVPR_followers_q2['follower_list'] for follower in sublist),
    "HSST_q2": set(follower for sublist in HSST_followers_q2['follower_list'] for follower in sublist),
    "THREAT_q2": set(follower for sublist in THREAT_followers_q2['follower_list'] for follower in sublist),
    "IMP_q3": set(follower for sublist in IMP_followers_q3['follower_list'] for follower in sublist),
    "PHAVPR_q3": set(follower for sublist in PHAVPR_followers_q3['follower_list'] for follower in sublist),
    "HSST_q3": set(follower for sublist in HSST_followers_q3['follower_list'] for follower in sublist),
    "THREAT_q3": set(follower for sublist in THREAT_followers_q3['follower_list'] for follower in sublist),
    "IMP_q4": set(follower for sublist in IMP_followers_q4['follower_list'] for follower in sublist),
    "PHAVPR_q4": set(follower for sublist in PHAVPR_followers_q4['follower_list'] for follower in sublist),
    "HSST_q4": set(follower for sublist in HSST_followers_q4['follower_list'] for follower in sublist),
    "THREAT_q4": set(follower for sublist in THREAT_followers_q4['follower_list'] for follower in sublist),
}


# Function to compute Jaccard similarity
def jaccard_similarity(set1, set2):
    return len(set1.intersection(set2)) / len(set1.union(set2)) if len(set1.union(set2)) > 0 else 0

# Compute Jaccard similarity for influencers
categories_influencers = list(list_dict.keys())
n_influencers = len(categories_influencers)
jaccard_matrix_influencers = np.zeros((n_influencers, n_influencers))

for i, cat1 in enumerate(categories_influencers):
    for j, cat2 in enumerate(categories_influencers):
        jaccard_matrix_influencers[i, j] = jaccard_similarity(list_dict[cat1], list_dict[cat2])

jaccard_df_influencers = pd.DataFrame(jaccard_matrix_influencers, index=categories_influencers, columns=categories_influencers)
mask = ~np.eye(jaccard_df_influencers.shape[0], dtype=bool)
off_diagonal_values = jaccard_df_influencers.values[mask]
average_jaccard = np.mean(off_diagonal_values)
std_jaccard = np.std(off_diagonal_values)
print(f"Average Jaccard similarity (off-diagonal) of influencers: {average_jaccard:.4f}")
print(f"Standard deviation of Jaccard similarity (off-diagonal): {std_jaccard:.4f}")


# Compute Jaccard similarity for audience
categories_audience = list(followers_dict.keys())
n_audience = len(categories_audience)
jaccard_matrix_audience = np.zeros((n_audience, n_audience))

for i, cat1 in enumerate(categories_audience):
    for j, cat2 in enumerate(categories_audience):
        jaccard_matrix_audience[i, j] = jaccard_similarity(followers_dict[cat1], followers_dict[cat2])

jaccard_df_audience = pd.DataFrame(jaccard_matrix_audience, index=categories_audience, columns=categories_audience)

mask = ~np.eye(jaccard_df_audience.shape[0], dtype=bool)
off_diagonal_values = jaccard_df_audience.values[mask]
average_jaccard = np.mean(off_diagonal_values)
std_jaccard = np.std(off_diagonal_values)
print(f"Average Jaccard similarity (off-diagonal) of audience: {average_jaccard:.4f}")
print(f"Standard deviation of Jaccard similarity (off-diagonal): {std_jaccard:.4f}")


fig, axes = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

vmin, vmax = 0, 1

# Plot the similarity of uncivil influencers
heatmap1 = sns.heatmap(
    jaccard_df_influencers,
    annot=True,
    fmt=".2f",
    cmap="Blues",  #
    cbar_kws={'label': 'Jaccard Similarity'},
    ax=axes[0],
    vmin=vmin,
    vmax=vmax,
    annot_kws={'fontsize': 10}
)

colorbar1 = heatmap1.collections[0].colorbar
colorbar1.ax.yaxis.label.set_size(20)
colorbar1.ax.tick_params(labelsize=16)

axes[0].tick_params(axis='x', rotation=90, labelsize=18)
axes[0].tick_params(axis='y', labelsize=18)

# Plot the similarity of exposed audience
heatmap2 = sns.heatmap(
    jaccard_df_audience,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    cbar_kws={'label': 'Jaccard Similarity'},
    ax=axes[1],
    vmin=vmin,
    vmax=vmax,
    annot_kws={'fontsize': 10}
)

colorbar2 = heatmap2.collections[0].colorbar
colorbar2.ax.yaxis.label.set_size(20)
colorbar2.ax.tick_params(labelsize=16)

axes[1].tick_params(axis='x', rotation=90, labelsize=18)
axes[1].tick_params(axis='y', labelsize=18)

plt.show()