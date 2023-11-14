import numpy as np
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def net_analysis(G):
    
    # Number of nodes
    num_nodes = G.number_of_nodes()
    print(num_nodes)
    # Number of edges
    num_edges = G.number_of_edges()
    print(num_edges)
    degrees = [d for n, d in G.degree()]
    # Mean node degree
    #mean_degree = np.mean(list(nx.average_degree_connectivity(G).values()))
    mean_degree = np.mean(degrees)
    print(mean_degree)
    # Median node degree
    median_degree = np.median(degrees)
    print(median_degree)
    # Largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    # Number of nodes in the largest connected component
    num_nodes_largest_cc = len(largest_cc)
    print(num_nodes_largest_cc)
    # Number of edges in the largest connected component
    num_edges_largest_cc = G.subgraph(largest_cc).number_of_edges()
    print(num_edges_largest_cc)
    # Mean clustering coefficient
    #mean_clustering_coefficient = nx.average_clustering(G)
    #print(mean_clustering_coefficient)
    # Mean shortest path
    #mean_shortest_path = nx.average_shortest_path_length(G)
    #print(mean_shortest_path)
    # Diameter
    #diameter = nx.diameter(G.subgraph(largest_cc))
    #print(diameter)

    data = pd.Series({
        "N nodes": int(num_nodes),
        "N edges": format(int(num_edges),'.1E'),
        "N nodes in the LCC": int(num_nodes_largest_cc),
        "N edges in the LCC": format(int(num_edges_largest_cc),'.1E'),
        "Mean node degree": round(mean_degree, 2),
        "Median node degree": int(median_degree),
        #"Mean clustering coefficient": mean_clustering_coefficient,
        #"Mean shortest path": mean_shortest_path,
        #"Diameter": diameter
    })
    return data


def table_stats(exp_list):
    stats = pd.DataFrame(index=["N nodes","N edges","N nodes in the LCC","N edges in the LCC","Mean node degree","Median node degree"])
    for t in exp_list:
        tissue = t.split("_")[2].split(".")[0]
        df = pd.read_csv("weighted_funcoup_%s.csv"%(tissue))
        for i in [None,0.5,1]:
            if i:df = df.loc[df["weight"]>=i]
            print("Creating graph")
            G=nx.from_pandas_edgelist(df, "gene1", 'gene2', ['weight'])
            print("Done, initiating analysis")
            stats["%s_th_%s" %(tissue,str(i)) ]=net_analysis(G)
    stats.to_csv("FC_Network_HClinks_analysis.csv")


def scale_free_plot(G,tissue,i,gs):
    # Get the degree distribution of the network
    degree_sequence = [d for n, d in G.degree()]
    unique_degrees, degree_counts = np.unique(degree_sequence, return_counts=True)
    degree_frequency = degree_counts / G.number_of_nodes()
    # Plot the degree distribution as a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(unique_degrees, degree_frequency, marker='o',color='goldenrod')#,edgecolors="black")
    plt.xlabel('Degree k')
    plt.ylabel('Frequency p(k)')
    plt.title('Degree Distribution for %s network (%s)' %(tissue,gs))
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig("scale_free_%s_%s_%sth.png" %(gs,tissue,str(i)))


def net_similarity(networks,heat=False,th=0.5):
    vertex_jaccard = []
    edge_jaccard = []
    vertex_ss=[]
    edge_ss=[]
    stats = pd.DataFrame(index=["N nodes","N edges","N nodes in the LCC","N edges in the LCC","Mean node degree","Median node degree"])
    for i in range(len(networks)):
        print("Loading",(networks[i]))
        df = pd.read_csv(networks[i],usecols=range(0,3))#,nrows=1000)
        df = df.loc[df["weight"]>=th]
        print("Creating network with %s cutoff"%(str(th)))
        G1=nx.from_pandas_edgelist(df, "gene1", 'gene2', ['weight'])
        if i != len(networks)-1 and heat:
            for j in range(i + 1, len(networks)):
                df = pd.read_csv(networks[j],usecols=range(0,3))#,nrows=1000)
                df = df.loc[df["weight"]>=th]
                G2=nx.from_pandas_edgelist(df, "gene1", 'gene2', ['weight'])

                print("Intersection and union")
                intersection = nx.intersection(G1,G2)
                union = nx.compose(G1,G2)
                # Calculate J and SS indexes for vertex similarity
                jaccard = intersection.number_of_nodes()/union.number_of_nodes()
                #except ZeroDivisionError:jaccard=0
                ss = intersection.number_of_nodes() / min(G1.number_of_nodes(),G2.number_of_nodes())
                #except ZeroDivisionError:ss=0
                vertex_jaccard.append(jaccard)
                vertex_ss.append(ss)

                # Calculate J and SS indexes for edge similarity
                jaccard = intersection.number_of_edges()/union.number_of_edges()
                #except ZeroDivisionError:jaccard=0
                ss = intersection.number_of_nodes() / min(G1.number_of_edges(),G2.number_of_edges())
                #except ZeroDivisionError:ss=0
                edge_jaccard.append(jaccard)
                edge_ss.append(ss)
                print(vertex_jaccard,edge_jaccard,vertex_ss,edge_ss)
        tissue = networks[i].split("_")[2]
        gs = networks[i].split("_")[3].split(".")[0]
        print("Computing stats")   
        stats["%s_%s" %(tissue,gs)]=net_analysis(G1)
        print("Scale free")
        scale_free_plot(G1,tissue,th,gs)
    if heat:heatmap([vertex_jaccard,edge_jaccard,vertex_ss,edge_ss],networks,th)
    stats.to_csv("FC_Network_HClinks_analysis_%s.csv"%(str(th)))
    return 

def heatmap(scores,networks,th):
    networks = [tissue.split("_")[2] for tissue in networks]
    types = ["Vertex_Jaccard Index","Edge_Jaccard Index","Vertex_SS","Edge_SS"]
    #types.sort()
    for score,type in zip(scores,types):
        
        similarity_matrix = pd.DataFrame(0, index=networks, columns=networks)
        for i in range(len(networks)):
            for j in range(i + 1, len(networks)):
                similarity_matrix.loc[networks[i], networks[j]] = score.pop(0)
        similarity_matrix.to_csv('%s_%s_%sth_similarity.csv'%(type.split("_")[0],type.split("_")[1].split(" ")[0],str(th)))
        

        #similarity_matrix=pd.read_csv("./plots/"+score,usecols=range(1,4))
        similarity_matrix=similarity_matrix.iloc[:-1,1:]
        similarity_matrix.iloc[1,0] = 0
        # Plot the triangular heatmap
        plt.figure(figsize=(15, 15))
        plt.imshow(similarity_matrix, cmap='YlGn', vmin=0, vmax=1)#,extent=[0, len(networks), 0, len(networks)])
        plt.xticks(np.arange(len(networks)-1), networks[1:], rotation='vertical',fontsize=17)
        plt.yticks(np.arange(len(networks)-1), networks[:-1],fontsize=16)
        plt.title('%s %s Similarity Heatmap'%(type.split("_")[0],type.split("_")[1]),fontsize=25)
        plt.colorbar(label="%s overlap"%type.split("_")[1])
        plt.savefig('%s_%s_%sth_similarity.png'%(type.split("_")[0],type.split("_")[1].split(" ")[0],str(th)))


if __name__=="__main__":
    networks = [f for f in os.listdir(".") if f[-4:]==".csv"]

    net_similarity(networks,heat=True)


