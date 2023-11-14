import numpy as np
import pandas as pd
import networkx as nx
import pickle
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import seaborn as sns


def hist_plot(networks):
    d = {}
    for net in networks:
        tissue = net.split("_")[1].split(".")[0]
        df = pd.read_csv(net)
        d[tissue] = (df.loc[df.p_value<=0.05].shape[0])
    d = dict(sorted(d.items(), key=lambda item: item[1],reverse = True))
    plt.rcParams["figure.figsize"] = [15, 12]
    plt.rcParams["figure.autolayout"] = True
    plt.bar(x=d.keys(),height=d.values(), color="darkolivegreen")
    #sns.displot(x=d.Tissues,height=d.Counts,binwidth = 0.3,color='darkolivegreen',bins=20)
    #plt.xlabel("Tissues",fontsize=20)
    plt.ylabel('n significant LCC',fontsize=25)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=25)
    plt.title(f"Significant pathways",fontsize=30)
    plt.savefig(f"pvalue_distribution_fc5.png")      
    return


def heatmap(networks):
    tissues = [net.split("_")[1].split(".")[0] for net in networks]
    overlap_matrix = pd.DataFrame(0, index=tissues, columns=tissues)
    sig_pathways = {}
    for net,tissue in zip(networks,tissues):
        df = pd.read_csv(net,index_col=0)
        sig_pathways[tissue] = df.loc[df.p_value<=0.05].index

    pathways = list(set(pathway for pathways in sig_pathways.values() for pathway in pathways))

    # Create the pathway overlap matrix
    overlap_matrix = np.zeros((len(tissues), len(tissues)), dtype=int)
    for i in range(len(tissues)):
        for j in range(i+1,len(tissues)):
            pathways_i = set(sig_pathways[tissues[i]])
            pathways_j = set(sig_pathways[tissues[j]])
            overlap_matrix[i, j] = len(pathways_i.intersection(pathways_j))
    
    plt.figure(figsize=(15, 15))
    plt.imshow(overlap_matrix[:-1,1:], cmap='YlGn', vmin=0, vmax=overlap_matrix.max())#,extent=[0, len(networks), 0, len(networks)])
    plt.xticks(np.arange(len(tissues)-1), tissues[1:], rotation='vertical',fontsize=17)
    plt.yticks(np.arange(len(tissues)-1), tissues[:-1],fontsize=16)
    plt.title('Significant Pathways overlap',fontsize=25)
    plt.colorbar(label="n pathways overlap")
    plt.savefig('sig_pathways_overlap.png')
    

networks = [f for f in os.listdir(".") if "csv" in f]
hist_plot(networks)
heatmap(networks)