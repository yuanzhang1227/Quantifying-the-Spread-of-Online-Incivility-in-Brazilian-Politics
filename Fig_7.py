import pandas as pd
import re
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
## Import datasets

# import the following relationships between survey users and political influencers
follow_merge_2 = pd.read_csv('./followship.csv', index_col=0)
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


## Calculate Communication Motifs
# Function to extract retweeted account from "Text"
def extract_retweeted_account(text):
    match = re.match(r"RT @(\w+):", text)
    return match.group(1) if match else None


def create_multi_edge_network(influencer_data, follow_data, influencer_list):
    # Create bipartite network
    B = nx.Graph()
    follow_subset = follow_data[follow_data['Target'].isin(influencer_list)]

    sources = follow_subset['Source'].unique()
    targets = follow_subset['Target'].unique()

    B.add_nodes_from(sources, bipartite=0)
    B.add_nodes_from(targets, bipartite=1)
    B.add_edges_from(follow_subset[['Source', 'Target']].itertuples(index=False, name=None))

    # Project network onto the "Target" side
    target_projection = nx.Graph()
    target_projection.add_nodes_from(targets)  # otherwise you will not have nodes with zero degree
    for u, v, weight in nx.bipartite.weighted_projected_graph(B, targets).edges(data='weight'):
        target_projection.add_edge(u, v, weight_follow=weight, relationship='follow')

    # Extract retweeted account
    influencer_data['retweeted_account'] = influencer_data['Text'].apply(extract_retweeted_account)

    # Filter retweet edges
    retweet_edges = influencer_data[
        (influencer_data['retweeted_account'].notnull()) &
        (influencer_data['Username'].isin(targets))  # we only want retweets from the political influencers
        ][['retweeted_account', 'Username', ]]

    # Create retweet edges with weights
    retweet_graph = nx.DiGraph()
    weighted_retweet_edges = retweet_edges.groupby(['Username', 'retweeted_account']).size().reset_index(
        name='weight_retweet')
    for _, row in weighted_retweet_edges.iterrows():
        retweet_graph.add_edge(
            row['retweeted_account'], row['Username'],
            weight_retweet=row['weight_retweet'], relationship='retweet'
        )

    # Calculate and add 'n_follower' attribute to each target node
    follower_counts = follow_subset.groupby('Target')['Source'].nunique()
    for target in targets:
        if target in target_projection.nodes:
            target_projection.nodes[target]['n_follower'] = follower_counts.get(target, 0)

    # Calculate and add 'n_original' attribute to each target node
    original_counts = influencer_data[influencer_data['retweeted_account'].isna()].groupby('Username').size()
    for target in targets:
        if target in target_projection.nodes:
            target_projection.nodes[target]['n_original'] = original_counts.get(target, 0)

    return target_projection, retweet_graph


# Create and analyze networks
IMP_network_follow, IMP_network_retweet = create_multi_edge_network(IMP_influencer, follow_merge_2,
                                                                    IMP_influencer['Username'].unique().tolist())
PHAVPR_network_follow, PHAVPR_network_retweet = create_multi_edge_network(PHAVPR_influencer, follow_merge_2,
                                                                          PHAVPR_influencer[
                                                                              'Username'].unique().tolist())
HSST_network_follow, HSST_network_retweet = create_multi_edge_network(HSST_influencer, follow_merge_2,
                                                                      HSST_influencer['Username'].unique().tolist())
THREAT_network_follow, THREAT_network_retweet = create_multi_edge_network(THREAT_influencer, follow_merge_2,
                                                                          THREAT_influencer[
                                                                              'Username'].unique().tolist())


influencer_with_identity['Identity Exposure'] = influencer_with_identity.apply(
    lambda row: ' + '.join(
        filter(pd.notna, [row['Political ideology'], row['Personal support'], row['Social identity']])
    ) if row['Account type'] == 'Individual' and not (
        pd.isna(row['Political ideology']) and pd.isna(row['Personal support']) and pd.isna(row['Social identity'])
    ) else None,
    axis=1
)
print(influencer_with_identity)
def collect_motif_pairs(follow_graph, retweet_graph):
    motif_1_weight = 0
    motif_2_weight = 0
    motif_3_weight = 0

    motif_pairs = {"Motif 1": [], "Motif 2": [], "Motif 3": []}

    # motifs 1
    for node in follow_graph.nodes():
        n_follower = follow_graph.nodes[node].get('n_follower', 0)
        n_original = follow_graph.nodes[node].get('n_original', 0)
        motif_1_weight += n_follower * n_original

        if n_follower * n_original > 0:
            motif_pairs["Motif 1"].append(node)

    # we also add self retweets:
    for u, v, data in nx.selfloop_edges(retweet_graph, data=True):
        n_follower = follow_graph.nodes[u].get('n_follower', 0)
        motif_1_weight += n_follower * data['weight_retweet']

        if n_follower * data['weight_retweet'] > 0:
            motif_pairs["Motif 1"].append(u)

    # loop over each retweet: here v retweets u
    for u, v, retweet_data in tqdm(retweet_graph.edges(data=True)):
        if u != v:  # we don't count self retweets

            # do u and v have common followers:
            follow_edge_data = follow_graph.get_edge_data(u, v)

            if not follow_edge_data:
                # the number of motif 2 created by this retweet edge: v retweets u
                motif_2_weight += retweet_data['weight_retweet'] * follow_graph.nodes[v].get('n_follower', 0)

                if retweet_data['weight_retweet'] * follow_graph.nodes[v].get('n_follower', 0) > 0:
                    motif_pairs["Motif 2"].append((u, v))

            if follow_edge_data:
                # if they follow each other, the common followers will see a motif 3:
                motif_3_weight += retweet_data['weight_retweet'] * follow_edge_data['weight_follow']

                if retweet_data['weight_retweet'] * follow_edge_data['weight_follow'] > 0:
                    motif_pairs["Motif 3"].append((u, v))

                # and the rest of the followers of v will see a motif 2
                rest = follow_graph.nodes[v].get('n_follower', 0) - follow_edge_data['weight_follow']
                assert rest >= 0
                motif_2_weight += retweet_data['weight_retweet'] * rest

                if retweet_data['weight_retweet'] * rest > 0:
                    motif_pairs["Motif 2"].append((u, v))

    # Remove nodes already in motif 3 from motif 1
    motif_1_nodes = set(motif_pairs["Motif 1"])
    motif_3_nodes = {node for pair in motif_pairs["Motif 3"] for node in pair}
    motif_pairs["Motif 1"] = list(motif_1_nodes - motif_3_nodes)

    return motif_pairs

# Collect motif pairs for each graph
graphs = {
    "IMP": (IMP_network_follow, IMP_network_retweet),
    "PHAVPR": (PHAVPR_network_follow, PHAVPR_network_retweet),
    "HSST": (HSST_network_follow, HSST_network_retweet),
    "THREAT": (THREAT_network_follow, THREAT_network_retweet)
}

all_motif_pairs = {graph_name: collect_motif_pairs(graph[0], graph[1]) for graph_name, graph in graphs.items()}

# Create a DataFrame for analysis
results = []

for graph_name, motif_pairs in all_motif_pairs.items():
    for motif, items in motif_pairs.items():
        if motif == "Motif 1":
            for node in items:
                node_type = influencer_with_identity[influencer_with_identity['Username'] == node][
                    'Account type'].values
                if len(node_type) > 0 and not pd.isna(node_type[0]):
                    results.append({
                        'Graph': graph_name,
                        'Motif': motif,
                        'Identity': node_type[0]
                    })
        else:
            for u, v in items:
                u_type = influencer_with_identity[influencer_with_identity['Username'] == u]['Account type'].values
                v_type = influencer_with_identity[influencer_with_identity['Username'] == v]['Account type'].values
                if len(u_type) > 0 and len(v_type) > 0 and not pd.isna(u_type[0]) and not pd.isna(v_type[0]):
                    results.append({
                        'Graph': graph_name,
                        'Motif': motif,
                        'Identity Pair': f"({u_type[0]}, {v_type[0]})"
                    })

motif_df = pd.DataFrame(results)

# Create histogram plots
fig, axes = plt.subplots(3, 4, figsize=(25, 18), sharey=False)
fig.subplots_adjust(hspace=0.8, wspace=0.6)

motifs = ["Motif 1", "Motif 2", "Motif 3"]
graph_names = list(graphs.keys())

for i, motif in enumerate(motifs):
    for j, graph_name in enumerate(graph_names):
        ax = axes[i, j]
        motif_specific_df = motif_df[(motif_df['Motif'] == motif) & (motif_df['Graph'] == graph_name)]

        if motif == "Motif 1":
            sns.countplot(
                data=motif_specific_df,
                x='Identity',
                order=motif_specific_df['Identity'].value_counts().index,
                palette='coolwarm',
                ax=ax
            )
        else:
            sns.countplot(
                data=motif_specific_df,
                x='Identity Pair',
                order=motif_specific_df['Identity Pair'].value_counts().index,
                palette='coolwarm',
                ax=ax
            )

        ax.set_title(f"{graph_name} ({motif})", fontsize=50)
        ax.set_xlabel("", fontsize=20)
        ax.set_ylabel("Count", fontsize=52)


        if motif == "Motif 1":
            labels = [item.get_text() for item in ax.get_xticklabels()]
            new_labels = [
                'I' if label == 'Individual' else 'P' if label == 'Politician' else 'M' if label == 'Media' else label
                for label in labels]
            ax.set_xticklabels(new_labels)
        elif motif in ["Motif 2", "Motif 3"]:
            labels = [item.get_text() for item in ax.get_xticklabels()]
            new_labels = [label.replace('Individual', 'I').replace('Politician', 'P').replace('Media', 'M') for label in
                          labels]
            ax.set_xticklabels(new_labels)


        plt.setp(ax.xaxis.get_majorticklabels(), rotation=90, ha="right", fontsize=18)  # Use plt.setp for alignment
        ax.tick_params(axis='y', labelsize=35)
        ax.tick_params(axis='x', labelsize=38)

        if motif == "Motif 1":
            y_max = motif_specific_df['Identity'].value_counts().max() if not motif_specific_df.empty else 1
        else:
            y_max = motif_specific_df['Identity Pair'].value_counts().max() if not motif_specific_df.empty else 1
        ax.set_ylim(0, y_max + 2)

plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.savefig("ICWSM_Fig_6.png", dpi=300, bbox_inches='tight')
plt.show()

