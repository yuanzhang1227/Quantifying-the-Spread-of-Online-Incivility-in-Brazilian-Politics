import networkx as nx
import pandas as pd
import re

followship = pd.read_csv('./followship.csv', index_col=0)
influencer_list = followship['Target'].unique().tolist()
influencer_prediction = pd.read_csv('Influencer_Incivility_Predicted_timestamped.csv')
influencer_prediction = influencer_prediction.dropna(subset=['Text']).reset_index(drop=True)

def extract_retweeted_account(text):
    match = re.search(r"RT @(\w+):", text)
    return match.group(1) if match else None

influencer_prediction['Retweet'] = influencer_prediction['Text'].apply(extract_retweeted_account)

## Create multilayer csv file

IMP_influencer = influencer_prediction[(influencer_prediction['labels_IMP']==1) & (influencer_prediction['probas_class_1_IMP'] >= 0.7) & (influencer_prediction['Retweet'].isin(influencer_list)) & (influencer_prediction['Username'] != influencer_prediction['Retweet'])][['Username', 'Retweet']]
IMP_influencer_renamed = IMP_influencer.rename(columns={'Username': 'Source', 'Retweet': 'Target'})
IMP_influencer_grouped = IMP_influencer_renamed.groupby(['Source', 'Target']).size().reset_index(name='Count')

PHAVPR_influencer = influencer_prediction[(influencer_prediction['labels_PHAVPR']==1) & (influencer_prediction['probas_class_1_PHAVPR'] >= 0.7) & (influencer_prediction['Retweet'].isin(influencer_list)) & (influencer_prediction['Username'] != influencer_prediction['Retweet'])][['Username', 'Retweet']]
PHAVPR_influencer_renamed = PHAVPR_influencer.rename(columns={'Username': 'Source', 'Retweet': 'Target'})
PHAVPR_influencer_grouped = PHAVPR_influencer_renamed.groupby(['Source', 'Target']).size().reset_index(name='Count')

HSST_influencer = influencer_prediction[(influencer_prediction['labels_HSST']==1) & (influencer_prediction['probas_class_1_HSST'] >= 0.7) & (influencer_prediction['Retweet'].isin(influencer_list)) & (influencer_prediction['Username'] != influencer_prediction['Retweet'])][['Username', 'Retweet']]
HSST_influencer_renamed = HSST_influencer.rename(columns={'Username': 'Source', 'Retweet': 'Target'})
HSST_influencer_grouped = HSST_influencer_renamed.groupby(['Source', 'Target']).size().reset_index(name='Count')

THREAT_influencer = influencer_prediction[(influencer_prediction['labels_THREAT']==1) & (influencer_prediction['probas_class_1_THREAT'] >= 0.7) & (influencer_prediction['Retweet'].isin(influencer_list)) & (influencer_prediction['Username'] != influencer_prediction['Retweet'])][['Username', 'Retweet']]
THREAT_influencer_renamed = THREAT_influencer.rename(columns={'Username': 'Source', 'Retweet': 'Target'})
THREAT_influencer_grouped = THREAT_influencer_renamed.groupby(['Source', 'Target']).size().reset_index(name='Count')

IMP_influencer_grouped['Layer'] = ["IMP"]*len(IMP_influencer_grouped)
PHAVPR_influencer_grouped['Layer'] = ["PHAVPR"]*len(PHAVPR_influencer_grouped)
HSST_influencer_grouped['Layer'] = ["HSST"]*len(HSST_influencer_grouped)
THREAT_influencer_grouped['Layer'] = ["THREAT"]*len(THREAT_influencer_grouped)  # Fix the mismatch here

multilayer_incivility = pd.concat([IMP_influencer_grouped, PHAVPR_influencer_grouped, HSST_influencer_grouped, THREAT_influencer_grouped], ignore_index=True)

for layer in multilayer_incivility['Layer'].unique():
    layer_edges = multilayer_incivility[multilayer_incivility['Layer'] == layer]
    G = nx.DiGraph()
    for _, row in layer_edges.iterrows():
        source = row['Source']
        target = row['Target']
        weight = row['Count']
        G.add_edge(source, target, weight=weight)

    # Calculate PageRank centrality
    centrality_dict = nx.pagerank(G, weight='weight')

    # Assign PageRank centrality to each node as 'pr_centrality' attribute
    for node in G.nodes():
        if node in centrality_dict:
            G.nodes[node]['pr_centrality'] = int(centrality_dict[node] * 100)
        else:
            print(f"Warning: Node {node} not found in centrality_dict for layer {layer}")

    # Save the layer graph with centrality as a GraphML file
    filename = f"graph_with_centrality_{layer}.graphml"
    nx.write_graphml(G, filename)


# Dictionary to store top 10 nodes for each layer
top_10_nodes_per_layer = {}

for layer in multilayer_incivility['Layer'].unique():
    layer_edges = multilayer_incivility[multilayer_incivility['Layer'] == layer]
    G = nx.DiGraph()
    for _, row in layer_edges.iterrows():
        source = row['Source']
        target = row['Target']
        weight = row['Count']
        G.add_edge(source, target, weight=weight)

    # Calculate PageRank centrality
    centrality_dict = nx.pagerank(G, weight='weight')

    # Assign PageRank centrality to each node
    nx.set_node_attributes(G, centrality_dict, 'pr_centrality')

    # Get the top 10 nodes based on 'pr_centrality'
    top_10_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_node_names = [node for node, _ in top_10_nodes if node in influencer_list]
    
    # Store the top 10 nodes for the layer
    top_10_nodes_per_layer[layer] = top_10_node_names

# Display the top 10 nodes for each layer
for layer, nodes in top_10_nodes_per_layer.items():
    print(f"Top 10 nodes in layer {layer}: {nodes}")




