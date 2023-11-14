#! /home/pmastro/miniconda3/envs/dataexp/bin/python
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import anndata as ad
import scanpy as sc
import argparse
import re
import ddqc
import pegasus as pg
from scipy.stats import median_abs_deviation


def calculate_ribo(adata,ids):
        """Function that calculates percent ribo for Pegasus object"""
        import re
        ribo_prefix = "^RP[SL][[:digit:]]|^RPLP[[:digit:]]|^RPSA"
        ribo_genes = \
        adata.var[ids].map(lambda x: re.match(ribo_prefix, x, flags=re.IGNORECASE) is not None)#.values.nonzero()[0]  # get all genes that match the pattern
        adata.var["ribo"] = ribo_genes.astype(bool)
        return adata

def mitoribo_filter(adata,sp):
        if sp == 9606:
                adata = adata[adata.obs["pct_counts_mito"]<10]
                adata = adata[adata.obs["pct_counts_ribo"]<10]
        elif sp == 10090:
                adata = adata[adata.obs["pct_counts_ribo"]<5]
                adata = adata[adata.obs["pct_counts_mito"]<5]
        return adata

def is_outlier(ddqc, metric: str, nmads: int):
        M = ddqc[metric]
        outlier = (M < np.median(M) - nmads * median_abs_deviation(M)) | (
                np.median(M) + nmads * median_abs_deviation(M) < M
        )
        return outlier

def raw_prep(adata):
        if adata.layers:
                for layer in list(adata.layers):
                        if "raw" in layer or "count" in layer:
                                adata.X = adata.layers[layer]
                                print("Data is now row")
                                break
        
        return adata

def col_finder(var,map):
        '''
        This function takes a variable matrix from an anndata object and a mapping file, and returns the name of the column storing the gene symbols
        '''
        types = ["Gene_Symbol","Ensembl","Ensembl_Protein","Gene_ID","KEGG"]
        for type in types:
                map_gs = map[map.Mapping_type == type]
                name = (None,0)
                idx = var.index[0]

                #Only the columns from which the first element matches the regexp will be intersected with the mapping file
                for col in var.columns.values:
                        f=re.findall("[A-Z]+[0-9]+[A-Z]*",str(var[col][0]).upper())    #this would match all types of IDs
                        if f:
                                n=len(np.intersect1d(var[col].astype(str), map_gs.ID.astype(str)))              #the column having the highest number of elements in common with the gene symbol column from the mapping file is the one i need
                                if n > name[1]: name = (col,n)
                
                #if no column was found, there is the possibility that the symbols are in the index
                if not name[0]:
                        if re.findall("[A-Z]+[0-9]+[A-Z]*",str(idx).upper()): name = ("index",0)

                if name[0]: return name[0]

        raise Exception("The given dataset does not have any valid ID, hence mapping can't be performed.")

        

def ambiguities(var,gene_col):
        '''
        This function takes a variable matrix from an anndata object, and returns a dictionary having uniprotIDs as key,
        and genes they were mapped to as values (ambiguity)
        '''
        #this dictionary stores all the unique uniprot IDs associated to their genes
        unique = {}
        for index,row in var.iterrows():
                gene = row[gene_col]
                id = row.UniprotID

                #If the id was already stored for another gene, we just append the new one, along with the original index of it
                if id in unique.keys():
                        unique[id].append((row.OG_i,gene))
                else:
                        unique[id] = [(row.OG_i,gene)]
        
        #this dictionary stores only those IDs that were associated to multiple genes
        sameID = {}
        for key,value in zip(unique.keys(),unique.values()):
                if len(value) > 1:
                        sameID[key] = value

        return sameID


def mapping(adata,mappings,gene_col,species_id):
        '''
        This function maps the gene symbols of the anndata dataset with the uniprotIDs stored in the mapping file. It also handles ambiguities.
        '''
        #first i store the original index, which i need to slice the anndata object
        adata.var["OG_i"] = range(adata.var.shape[0])    


        print("Mapping started")
        #Open the mapping file and take only the entries form the given species
        mappings = mappings[mappings.SpeciesID==species_id]
        
        #if the gene column storing the symbols was not specified, a function is called to find it
        if not gene_col: 
                gene_col = col_finder(adata.var,mappings)
                print(f"Mapping is being performed on the '{gene_col}' column")
        
        #merge the mapping file with the variables matrix of the anndata object, to remove all genes that are not coding for a protein
        print("Starting shape: ",adata.var.shape)
        data=pd.merge(adata.var,mappings, how = "inner",left_on=gene_col,right_on="ID")
        #print(data.shape)

        #Remove duplicated genes to handle the first type of ambiguities (genes mapped to multiple uniprot IDs)
        print("Handling ambiguities")
        data = data.drop_duplicates(subset=["OG_i"], keep='first')
        #print(data.shape)

        #second type of ambiguities: same uniprot ID for multiple genes, probably isoforms
        data = data.drop_duplicates(subset=["UniprotID"],keep="first")
        data.reset_index(inplace=True)
        data.drop(["index","Mapping_type","ID","SpeciesID"],axis=1,inplace=True)
        #in both cases i keep the first one having the ID


        print("Final shape: ",data.shape)
        #Slicing the anndata matrix using the indexes and storing it in a new anndata object, since otherwise there might be problems related to the indexing of the other matrices stored in the ad
        adata = adata[:,list(data["OG_i"])]
        #store only what we need from the original one
        obs = adata.obs
        #layers = adata.layers

        adata = ad.AnnData(adata.X)
        adata.var = data
        adata.obs = obs
        #adata.layers = layers
        print(adata)

        return adata


def cqc_plots(cellQC,reads_f,genes_f,filename):
        plt.rcParams["figure.figsize"] = [20, 13]
        plt.rcParams["figure.autolayout"] = True
        plt.suptitle(f"Quality control on the {filename} dataset",fontsize=25)
        ax = plt.GridSpec(2,2)
        ax.update(hspace=0.5,wspace=0.2)

        ax1 = plt.subplot(ax[0, 0])
        plt.hist(cellQC['n_counts'], bins=40,color='goldenrod',ec='black')
        plt.xlabel('Total counts',fontsize=15)
        plt.ylabel('log(N cells)',fontsize=15)
        plt.yscale('log')
        plt.axvline(reads_f, color='red',linewidth=3,label=f'Filter at {reads_f} reads',linestyle='--')
        plt.legend(loc='upper right')
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.title("Read counts per cell",fontsize=20)

        ax2 = plt.subplot(ax[1, 0])
        plt.hist(cellQC['n_genes'], bins=40,ec='black',color='dodgerblue')
        plt.xlabel('N genes',fontsize=15)
        plt.ylabel('log(N cells)',fontsize=15)
        plt.yscale('log')
        plt.axvline(genes_f, color='red',linewidth=3,label=f'Filter at {genes_f} genes',linestyle='--')
        plt.legend(loc='upper right')
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.title("Number of genes per cell",fontsize=20)

        pct = 10
        ax3 = plt.subplot(ax[0,1])
        plt.hist(cellQC["percent_mito"],color='mediumseagreen',bins=30,ec='black')
        plt.xlabel('Percentage of mitochondrial genes',fontsize=15)
        plt.ylabel('log(N cells)',fontsize=15)
        plt.yscale('log')
        plt.axvline(pct, color='red',linewidth=3,label=f'Filter at {pct}%',linestyle='--')
        plt.legend(loc='upper right')
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.title("Mitochondrial genes per cell",fontsize=20)

        ax4 = plt.subplot(ax[1,1])
        plt.hist(cellQC["percent_ribo"],color='tomato',bins=30,ec='black')
        plt.xlabel('Percentage of ribosomal genes',fontsize=15)
        plt.ylabel('log(N cells)',fontsize=15)
        plt.yscale('log')
        plt.axvline(pct, color='red',linewidth=3,label=f'Filter at {pct}%',linestyle='--')
        plt.legend(loc='upper right')
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.title("Ribosomal genes per cell",fontsize=20)

        plt.savefig(f"cell_qcPlots_{filename}.png")


def gqc_plots(geneQC):
        plt.rcParams["figure.figsize"] = [15, 15]
        plt.rcParams["figure.autolayout"] = True
        plt.tight_layout()
        ax = plt.GridSpec(2,1)
        ax.update(hspace=0.5)

        ax1 = plt.subplot(ax[0, 0])
        plt.hist(geneQC['total_counts'], bins=50,ec='black',color='tomato')
        plt.xlabel('Total counts',fontsize=15)
        plt.ylabel('log(N genes)',fontsize=15)
        plt.yscale('log')
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.title("Read counts per gene",fontsize=20)

        ax2 = plt.subplot(ax[1, 0])
        plt.hist(geneQC['n_cells_by_counts'], bins=50,ec='black',color='mediumturquoise')
        plt.xlabel('N cells expressing > 0',fontsize=15)
        plt.ylabel('log(N genes)',fontsize=15)
        plt.yscale('log')
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)
        plt.title("Number of cells per gene",fontsize=20)

        plt.savefig(f"gene_qcPlots_{filename}.png")



if __name__=="__main__":
        parser = argparse.ArgumentParser(description='...')
        parser.add_argument('-ad', help='The adata file containing the count matrix')
        parser.add_argument('-map',help = 'Mapping file')
        parser.add_argument('-fl','--filter',default = "Standard", help="Type of filtering, default is 'Standard', the other option is 'ddqc'")
        parser.add_argument('-sp','--specimen',dest='speciesID',default=9606,type=int,help="Species ID")
        parser.add_argument('-rf','--reads_filter',default=500,type=int,help='Minimum amount of read counts per cell, used for filtering. Default is 500.')
        parser.add_argument('-gf','--genes_filter',default=200,type=int,help='Minimum amount of genes per cell, used for filtering. Default is 200.')
        parser.add_argument('-sqc','--store_qc',default=False,type=bool,help="Type True if you want the script to store the quality control files. Default is False")
        parser.add_argument('-col','--column',dest='column',default = None, help="Column name for gene symbol")

        args = parser.parse_args()

        filename = args.ad[:-5]
        reads_f = args.reads_filter
        genes_f = args.genes_filter
        filter = args.filter
        sp = args.speciesID
        ids = args.column
        map = pd.read_csv(args.map,names=["Mapping_type","ID","UniprotID","SpeciesID"],skiprows=1)

        if filter == "ddqc":
                adata = pg.read_input(args.ad)
                if not ids: ids = col_finder(adata.var,map)
                adata.var_names = adata.var[ids]
                print(adata.var_names)

                cellQC = ddqc.ddqc_metrics(filename,adata,return_df_qc=True,threshold=0.1,basic_n_counts=reads_f)

                pg.filter_data(adata)
                
        else:
                adata = ad.read(args.ad)
                if not ids: ids = col_finder(adata.var,map)
                adata = raw_prep(adata)
                adata.var["mito"] = adata.var[ids].str.startswith("MT-").astype(bool)
                adata = calculate_ribo(adata,ids)

                cellQC = sc.pp.calculate_qc_metrics(adata,qc_vars=("mito","ribo"))[0]
                
                adata.obs["pct_counts_mito"] = cellQC["pct_counts_mito"]
                adata.obs["pct_counts_ribo"] = cellQC["pct_counts_ribo"]

                print("Initial shape: ",adata.shape)
                adata = mitoribo_filter(adata,sp)
                print("After mito and ribo filtering: ",adata.shape)
                
                sc.pp.filter_cells(adata, min_genes = genes_f)
                sc.pp.filter_cells(adata, min_counts = reads_f)

                print("After genes and reads filtering: ",adata.shape)

                cellQC= cellQC.rename(columns={"total_counts":"n_counts","pct_counts_mito":"percent_mito","n_genes_by_counts":"n_genes","pct_counts_ribo":"percent_ribo"})

        if args.store_qc:
                cellQC.to_csv(f"{filename}_qc_df.csv")
        cqc_plots(cellQC,reads_f,genes_f,filename)

        adata = mapping(adata,map,ids,sp)
        print("After mapping: ",adata.shape)
        adata.write(f"{filename}_final.h5ad")
