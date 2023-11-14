#! /home/pmastro/miniconda3/envs/dataexp/bin/python

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import anndata as ad
from sys import argv




def sp_correlation(adata,norm=True,periods=100):
        names = list(adata.var.UniprotID)
        if norm:
                try:
                        matrix = pd.DataFrame(adata.X.toarray())
                except AttributeError:
                        matrix = pd.DataFrame(adata.X)
        else:
                matrix = pd.DataFrame(adata.layers["raw_counts"].toarray())
        matrix = matrix.replace(0,np.nan)
        matrix.columns = names

        corr = pd.DataFrame.corr(matrix,method="spearman",min_periods=periods)\
                        .where(np.triu(np.ones((matrix.shape[1],matrix.shape[1])), k=1).astype(bool))\
                        .stack().reset_index()
        corr.columns =  ["protein1","protein2","correlations"]

        sortDF = corr[["protein1","protein2"]].apply(lambda r: sorted(r), axis = 1)
        sortDF = pd.DataFrame(sortDF.to_list(),columns=["protein1","protein2"])
        sortDF["correlations"] = corr.correlations   

        return sortDF


def corr_plot(corr,output):
        plt.rcParams["figure.figsize"] = [18, 18]
        plt.rcParams["figure.autolayout"] = True
        sns.displot(corr, x="correlations",binwidth = 0.1,color='slateblue',bins=10)
        plt.xlabel("Spearman Correlations",fontsize=10)
        plt.ylabel('log(Counts)',fontsize=10)
        plt.yscale('log')
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8)
        plt.title(f"Correlations computed on {output}",fontsize=13)
        plt.savefig(f"{output}_corr_distribution.png")


if __name__=="__main__":
        #start = time.time()
        filename = argv[1]
        periods = argv[2]
        output = filename[:-11]
        
        adata = ad.read(filename)
        print(adata)
        if "raw" in output:
                corr = sp_correlation(adata,norm=False,periods=periods)
        else:
                corr = sp_correlation(adata, norm=True,periods=int(periods))
        corr.to_csv(f"{output}_correlation.csv",index=False)

        #corr = pd.read_csv(filename)
        corr_plot(corr,output)