import numpy as np
import pandas as pd
import networkx as nx
import argparse
import random
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests


def gene_map(net,map):
    G=nx.from_pandas_edgelist(net, "gene1", 'gene2', ['weight'])
    map = dict(zip(map['UniProtID(primaryID)'], map['mappedID']))
    G = nx.relabel_nodes(G, map)
    return G

def null_model(allgenes,G,n_genes,path_seed,n_samples=10000):
    LCC_exp = []
    random.seed(path_seed)
    iter_seeds = random.sample(range(0,100000), n_samples)
    for seed in iter_seeds:
        random.seed(seed)
        nodes = random.sample(allgenes,n_genes)
        lcc = len(max(nx.connected_components(G.subgraph(nodes)), key=len))
        LCC_exp.append(lcc)
    return LCC_exp

def compute_pvalue(LCC_obs,LCC_exp,n_samples=10000):
    pvalue = (sum(i >=LCC_obs for i in LCC_exp)+1)/(n_samples+1)
    return round(pvalue, 4)


def pvalue_LCC(allgenes,paths,G,path_seeds,split_n,tissue):
    d = {}
    for path,path_seed in zip(paths.pathway_name.unique(),path_seeds):
        path_df = paths[paths.pathway_name==path]
        genes = list(path_df.hgnc_symbol)
        #print("Numeber of genes in the pathway %s: "%(path),len(genes))
        Gsub = G.subgraph(genes)
        n_genes=len(genes)
        #print("Number of nodes: ",Gsub.number_of_nodes(), "Number of edges: ",Gsub.number_of_edges())
        try:
            LCC_obs = len(max(nx.connected_components(Gsub), key=len))
            #print("LCC: ",LCC_obs)
            LCC_exp = null_model(allgenes,G,n_genes,path_seed)
            pval = compute_pvalue(LCC_obs,LCC_exp)
            d[path] = {"LCC_observed":LCC_obs,"Mean_expected_LCC":np.mean(LCC_exp),"p_value":pval,"Nr_genes_pathway":n_genes}
        except ValueError:
            d[path] = {"LCC_observed":0,"Mean_expected_LCC":0,"p_value":1,"Nr_genes_pathway":n_genes}
    pd.DataFrame(d).transpose().to_csv("pvalue_core_%s_%s.csv"%(split_n,tissue))
    return 
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('net', help='The csv file containing the weighted funcoup network')
    parser.add_argument('mapping', help='mapping file')
    parser.add_argument('paths', help= 'Tsv file with kegg pathways')
    parser.add_argument('-c',help='n of cores',default=8,type=int)
    parser.add_argument('-th',help='LLR threshold',default=0.5,type=float)


    args = parser.parse_args()
    cores_n = args.c
    map = pd.read_csv(args.mapping)
    map = map[map.taxID==9606]
    map = map[map["# Mapping_type"]=="Gene_Symbol"]

    name = args.net
    tissue = name.split("_")[2]
    gs = name.split("_")[3].split(".")[0]  
    
    net = pd.read_csv(name)
    net = net.loc[net.weight>=args.th]
    
    paths = pd.read_table(args.paths)
    mask = paths.pathway_name.value_counts()
    paths = paths[paths["pathway_name"].isin(mask[mask >= 15].index)]
    allgenes = list(paths.hgnc_symbol)
    print("Creating network")
    G = gene_map(net,map)

    main_seed = 12345
    random.seed(main_seed)
    path_seeds = random.sample(range(0,100000), len(paths.pathway_name.unique()))
    
    
    path_split = np.array_split(paths.pathway_name.unique(),cores_n)
    path_split = [paths[paths.pathway_name.isin(split_i)] for split_i in path_split]
    path_seeds_split = np.array_split(path_seeds,cores_n)


    print('##Start up parallelization')
    Parallel(n_jobs=cores_n, verbose=10)\
        (delayed(pvalue_LCC)(allgenes,path_split[split_n], G, path_seeds_split[split_n], split_n,tissue) \
            for split_n in range(cores_n))
    
    print("Merging results")
    results=pd.DataFrame(columns=["LCC_observed","Mean_expected_LCC","p_value","Nr_genes_pathway"])
    for split_n in range(cores_n):
        df = pd.read_csv("pvalue_core_%s_%s.csv"%(split_n,tissue),index_col=0)
        results=results.append(df)
    

    results = results.sort_values(by=['p_value','LCC_observed','Nr_genes_pathway'], ascending=[True, False,False])
    results.p_value=multipletests(results.p_value,alpha=0.05,method='fdr_bh',is_sorted=True,returnsorted=True)[1]

    print(results)
    print(sum(results.p_value<=0.05))
    
    results.to_csv("pvalue_%s.csv"%tissue)    