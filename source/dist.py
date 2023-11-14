import numpy as np
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv


def dist_plot(fc,tissue):
        plt.rcParams["figure.figsize"] = [18, 18]
        plt.rcParams["figure.autolayout"] = True
        sns.displot(fc, x="weight",binwidth = 0.2,color='slateblue',bins=10)
        plt.xlabel("LLRs",fontsize=10)
        plt.ylabel('log(Counts)',fontsize=10)
        plt.yscale('log')
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8)
        plt.title(f"LLR distribution - {tissue}",fontsize=13)
        plt.savefig(f"weighted_funcoup_{tissue}_final.png") 

df = pd.read_csv(argv[1])
dist_plot(df,argv[1].split("_")[2])