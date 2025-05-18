import spacy
from nltk import sent_tokenize
import os
import sys
import networkx as nx
from pyvis.network import Network

import pandas as pd

class Networks:
    def __init__(self):
        pass

    def ner(self,df):
        entity_relationship = []
        for row in df["ners"]:
            relationship = []
            for sentence in row:
                relationship.append(list(sentence))
                relationship = relationship[-10:]
                relationship_flat = sum(relationship, [])

                for entity in sentence:
                    for entity_in_flat in relationship_flat:
                        if entity != entity_in_flat:
                            entity_relationship.append(sorted([entity, entity_in_flat]))
        relationship_df = pd.DataFrame({'value': entity_relationship})
        relationship_df["source"] = relationship_df["value"].apply(lambda x: x[0])
        relationship_df["target"] = relationship_df["value"].apply(lambda x: x[1])
        relationship_df = relationship_df.groupby(["source", "target"]).count().reset_index()
        relationship_df = relationship_df.sort_values(by="value", ascending=False)

        return relationship_df

    def draw_network(self,relationship_df):
        relationship_df = relationship_df.sort_values("value",ascending=False)
        relationship_df = relationship_df.head(200)
        G = nx.from_pandas_edgelist(relationship_df,source='source',target='target',edge_attr='value',create_using=nx.Graph())
        net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white",
                      cdn_resources='remote')
        node_degree = dict(G.degree)
        nx.set_node_attributes(G, node_degree, 'size')

        net.from_nx(G)
        net.show("network.html")


