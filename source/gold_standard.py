#! /home/pmastro/miniconda3/envs/dataexp/bin/python

import pandas as pd
import argparse


def table_mapping(gs,corr,sort):
    
    print("Sorting gold standard")
    gs = gs.apply(lambda r: sorted(r), axis = 1)
    gs = pd.DataFrame(gs.to_list(),columns=["gene1","gene2"])

    if sort:
        print("Sorting correlation dataframe")
        sortDF = corr[["protein1","protein2"]].apply(lambda r: sorted(r), axis = 1)
        sortDF = pd.DataFrame(sortDF.to_list(),columns=["protein1","protein2"])
        sortDF["correlations"] = corr.correlations
    else:
        sortDF = corr

    print("Merging")
    print(gs.shape)
    gs = pd.merge(gs,sortDF,how="inner",left_on = ["gene1","gene2"],right_on=["protein1","protein2"])
    gs = gs.drop(["protein1","protein2"],axis=1)
    gs = gs[~gs.isnull().any(axis=1)]
    print(gs.shape)

    return gs



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('gs', help='Gold standard to use')
    parser.add_argument('correlations', help='file storing the correlations between pairs of genes')
    parser.add_argument('-s','--sort',default=False,type=bool,help="Say if the sorting of the correlation dataframe is required or not")
    parser.add_argument('-l','--len',default = 0,type=int, help="Lenght of the gold standard dataframe to store")
    args = parser.parse_args()

    gs = pd.read_table(args.gs,names=["gene1","gene2"])
    corr = pd.read_csv(args.correlations,names=["protein1","protein2","correlations"],skiprows=1)

    output = args.gs.split(sep="/")[-1].split(sep="_")[0] + "_" +  args.correlations.split(sep="/")[-1].split(sep=".")[0]

    gs=table_mapping(gs,corr,sort = args.sort)

    if args.len > 0 and args.len < gs.shape[0]:
        gs = gs.sample(n=args.len)
    
    print("Storing gold standard")
    if "neg" in args.gs:
        gs.to_csv(f"{output}_neg_gs.csv")
    else:
        gs.to_csv(f"{output}_gs.csv")
