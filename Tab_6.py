import pandas as pd
import re
import numpy as np
import networkx as nx
from multiprocessing import Pool, cpu_count
import itertools
from tqdm import tqdm

# raise Exception
#%%

# Function to extract retweeted account from "Text"
def extract_retweeted_account(text):
    match = re.match(r"RT @(\w+):", text)
    return match.group(1) if match else None

# Function to create multi-edge network for following and retweeting
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
    target_projection.add_nodes_from(targets) # otherwise you will not have nodes with zero degree
    for u, v, weight in nx.bipartite.weighted_projected_graph(B, targets).edges(data='weight'):
        target_projection.add_edge(u, v, weight_follow=weight, relationship='follow')

    # Extract retweeted account
    influencer_data['retweeted_account'] = influencer_data['Text'].apply(extract_retweeted_account)
    
    # Filter retweet edges
    retweet_edges = influencer_data[
        (influencer_data['retweeted_account'].notnull()) &
        (influencer_data['Username'].isin(targets)) # we only want retweets from the political influencers
        ][['retweeted_account', 'Username',]]

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


# Function to count motifs in the network
def count_motifs(follow_graph, retweet_graph):
    motif_1_weight = 0
    motif_2_weight = 0
    motif_3_weight = 0
    # motif_4_weight = 0

    # motifs 1
    for node in follow_graph.nodes():
        n_follower = follow_graph.nodes[node].get('n_follower', 0)
        n_original = follow_graph.nodes[node].get('n_original', 0)
        motif_1_weight += n_follower * n_original
        
    # we also add self retweets:
    for u, v, data in nx.selfloop_edges(retweet_graph, data=True):
        n_follower = follow_graph.nodes[u].get('n_follower', 0)
        motif_1_weight += n_follower * data['weight_retweet']        

    
    #loop over each retweet: here v retweets u
    for u, v , retweet_data in tqdm(retweet_graph.edges(data=True)):
        if u != v: # we don't count self retweets
            
            # do u and v have common followers:
            follow_edge_data = follow_graph.get_edge_data(u,v)
            
            if not follow_edge_data:
                # the number of motif 2 created by this retweet edge: v retweets u
                motif_2_weight += retweet_data['weight_retweet'] * follow_graph.nodes[v].get('n_follower', 0)
            
            if follow_edge_data:
                # if they follow each others, the common followers will see a motif 3:
                motif_3_weight += retweet_data['weight_retweet'] * follow_edge_data['weight_follow']
    
                # and the rest of the followers of v will see a motif 2
                rest = follow_graph.nodes[v].get('n_follower', 0) - follow_edge_data['weight_follow']
                assert rest >= 0
                motif_2_weight += retweet_data['weight_retweet'] * rest

    # for each motif 3 we already counted a motif 1 from the original tweet of u
    # so we need to remove it
    motif_1_weight -= motif_3_weight

    return {"Motif 1": motif_1_weight, "Motif 2": motif_2_weight, "Motif 3": motif_3_weight}


# Generate randomized network using Weighted Directed Configuration Model
def generate_randomized_retweet_network(retweet_graph, seed=None):
    """
    Generate a randomized retweet network using the Weighted Directed Configuration Model.
    Keeps the follow edges unchanged and replaces retweet edges with configuration model edges,
    while using out-strengths and in-strengths directly as the degree sequences.
    """
    
    # create a fixed node ordering to keep all lists in the same order
    nodelist = list(retweet_graph.nodes())
    
    # # remove the self-edges for the randomization:
    copy = retweet_graph.copy()
    # copy.remove_edges_from(nx.selfloop_edges(copy))
    
    # out_strenghts
    out_strengths = [sum(d['weight_retweet'] for _, _, d in copy.out_edges(u, data=True)) \
                     for u in nodelist]

    in_strengths = [sum(d['weight_retweet'] for _, _, d in copy.in_edges(u, data=True)) \
                     for u in nodelist]


    # Generate a randomized graph using the configuration model
    random_state = np.random.RandomState(seed) # make sure each parallel process gets a difference seed
    config_model = nx.directed_configuration_model(in_strengths, out_strengths, seed=random_state)

    # Map the generated graph back to the original node labels
    randomized_graph = nx.DiGraph()
    randomized_graph.add_nodes_from(nodelist)
    # merge the parallel edges into weighted edges:
    for u, v in config_model.edges():
        if (nodelist[u], nodelist[v]) in randomized_graph.edges:
            randomized_graph[nodelist[u]][nodelist[v]]['weight_retweet'] += 1
        else:
            randomized_graph.add_edge(nodelist[u], nodelist[v], weight_retweet=1)

    return randomized_graph


# Parallel simulation of randomized networks
def parallel_motif_simulations(follow_graph, retweet_graph, num_simulations):
    with Pool(cpu_count()) as pool:
        results = pool.map(generate_randomized_retweet_network, [retweet_graph] * num_simulations)
        return [count_motifs(follow_graph, result) for result in results]


# Function to analyze motifs for a given network
def analyze_motifs(follow_graph, retweet_graph, num_simulations=1000):
    observed_motifs = count_motifs(follow_graph, retweet_graph)
    randomized_results = parallel_motif_simulations(follow_graph, retweet_graph, num_simulations)
    randomized_means = {motif: np.mean([result[motif] for result in randomized_results]) for motif in observed_motifs}
    randomized_stds = {motif: np.std([result[motif] for result in randomized_results]) for motif in observed_motifs}
    
    z_scores = {motif: (observed_motifs[motif] - randomized_means[motif]) / randomized_stds[motif] for motif in observed_motifs}

    print(f"Observed Motif Counts: {observed_motifs}")
    print(f"Randomized Motif Means: {randomized_means}")
    print(f"Randomized Motif STDs: {randomized_stds}")
    print(f"Z-Scores for Motif Prevalence:")
    for motif, z in z_scores.items():
        print(f"{motif}: Z-score = {z:.2f}")

#%%
# Main Workflow
if __name__ == "__main__":
    # Import datasets
    followship = pd.read_csv('./followship.csv', index_col=0)
    influencer_prediction = pd.read_csv('Influencer_Incivility_Predicted_timestamped.csv')
    influencer_list = followship['Target'].unique().tolist()
    influencer_prediction = influencer_prediction[influencer_prediction['Username'].isin(influencer_list)]

    # Filter influencers
    IMP_influencer = influencer_prediction[
        (influencer_prediction['labels_IMP'] == 1) & (influencer_prediction['probas_class_1_IMP'] >= 0.7)]
    PHAVPR_influencer = influencer_prediction[
        (influencer_prediction['labels_PHAVPR'] == 1) & (influencer_prediction['probas_class_1_PHAVPR'] >= 0.7)]
    HSST_influencer = influencer_prediction[
        (influencer_prediction['labels_HSST'] == 1) & (influencer_prediction['probas_class_1_HSST'] >= 0.7)]
    THREAT_influencer = influencer_prediction[
        (influencer_prediction['labels_THREAT'] == 1) & (influencer_prediction['probas_class_1_THREAT'] >= 0.7)]

    # Create and analyze networks
    IMP_network_follow, IMP_network_retweet = create_multi_edge_network(IMP_influencer, followship,
                                            IMP_influencer['Username'].unique().tolist())
    analyze_motifs(IMP_network_follow, IMP_network_retweet)

    PHAVPR_network_follow, PHAVPR_network_retweet = create_multi_edge_network(PHAVPR_influencer, followship,
                                            PHAVPR_influencer['Username'].unique().tolist())
    analyze_motifs(PHAVPR_network_follow, PHAVPR_network_retweet)

    HSST_network_follow, HSST_network_retweet = create_multi_edge_network(HSST_influencer, followship,
                                            HSST_influencer['Username'].unique().tolist())
    analyze_motifs(HSST_network_follow, HSST_network_retweet)

    THREAT_network_follow, THREAT_network_retweet = create_multi_edge_network(THREAT_influencer, followship,
                                            THREAT_influencer['Username'].unique().tolist())
    analyze_motifs(THREAT_network_follow, THREAT_network_retweet)

