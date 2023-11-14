import numpy as np
import pandas as pd
import networkx as nx
import argparse
import os
from sys import argv
from tqdm import tqdm


def net_merger(dfs):
    df = pd.read_csv(dfs[0],usecols=range(0,2))
    for i in range(1,len(dfs)):
        print("Merging")
        df2 = pd.read_csv(dfs[i],usecols=range(0,2))
        df["weight"] = df["weight"].where(df["weight"]>=df2["weight"],df2["weight"])
    df.to_csv("merged.csv",index=False)
    return

def special(dfs,tissue):
    df = pd.read_csv(dfs[0])
    print(f"Starting with {dfs[0]}")
    for data in tqdm(dfs[1:]):
        if "heart" in data:df1= pd.read_csv(data,skiprows=1,names=["weight2","gene1","gene2"])
        else:df1= pd.read_csv(data,skiprows=1,names=["gene1","gene2","weight2"])
        df = pd.merge(df, df1, on=['gene1', 'gene2'], how='outer')
        df.fillna(-999, inplace=True)
        df['weight'] = df[['weight', 'weight2']].max(axis=1)
        df.drop(['weight2'], axis=1, inplace=True)
        print(df.shape,data,"merged")
    df.to_csv(f"weighted_funcoup_merged.csv",index=False)


if __name__=="__main__":
    tissue = argv[1]
    dfs=[f for f in os.listdir(".") if tissue in f]#f[-4:]==".csv"]
    #net_merger(dfs)
    special(dfs,tissue)



    #df2 = pd.read_csv(f"weighted_funcoup_{tissue}_signaling.csv",usecols=range(1,4))
    #df2["weight"] = pd.read_csv("merged.csv",usecols=range(0,1))
    #df2.to_csv(f"weighted_funcoup_{tissue}_full.csv",index=False)