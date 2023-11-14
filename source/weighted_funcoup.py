import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse, numpy as np, pandas as pd, os, pickle
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import seaborn as sns


def LLR_table(tissue,gs,save=False):
    LLRs = pd.DataFrame()
    for exp in tqdm(exp_list,desc='Encoding'):
        # Read the dataframe
        df = pd.read_table(exp)
        exp_name = exp.split("_")[2]  # Extract the name from the file
        df=df.set_index("pair")
        df.drop("correlations",axis=1,inplace=True)
        df.rename({"llr":exp_name},inplace=True,axis=1)
        if LLRs.shape[0] == 0: LLRs = df
        else: LLRs = pd.merge(LLRs,df,on="pair",how="outer")
        print(LLRs.shape)
    
    if save:
        pq.write_table(pa.Table.from_pandas(LLRs), 'LLRs_%s_%s.parquet' %(gs,tissue))

    return LLRs



def fc_computation(correlations,llr_split,alpha,split_n,gs):
    fc ={}
    for pair,row in llr_split.iterrows():
        #for each pair, the llr are sorted in a decreasing order
        row = pd.DataFrame(row).dropna() #remove null values
        exps = [(exp,llr) for exp,llr in zip(row.index,row[pair])]
        exps.sort(key=lambda tup: abs(tup[1]), reverse=True)
        fc[pair] = fc.get(pair,0)

        for i,exp1 in enumerate(exps):
            #for each evidence, we grab all the rows containing the correlations with the experiments
            r_eks = correlations.loc[(correlations["exp1"] == exp1[0]) | (correlations["exp2"]==exp1[0])]
            j=i+1
            d_ek = 1.   #prepare the distance
            while j < len(exps):
                #for every experiment with a lower llr, we multiply the r_eks to the distance
                exp2 = exps[j][0]
                weight = float(r_eks.loc[(correlations["exp1"] == exp2) | (correlations["exp2"]==exp2)]["corr"])
                d_ek *= alpha*(1-max(0,weight))
                j+=1
            #we add the llr*d_ek to the funcoup dictionary for the pair. This will be the score
            fc[pair] += (exp1[1]*d_ek) 
    with open('weighted_funcoup_%s_%s_core%s.pkl' %(tissue,gs,split_n), 'wb') as fout:
        pickle.dump(fc, fout)



def weighted_funcoup(tissue,gs,save_LLR=False, save_corr=False, save_fc=False, alpha=0.7,cores_n = 8,nx=True,merge=False):
    '''
    FunCoup4.1 implements in its Bayesian framework a weighting procedure when integrates 
    LLRs for the same Experiment Target (same type of evidence, different experiment)
    to account for redundancy within the evidences dataset.
    
    Here, for every interaction between genes X and Y:
        * tup(Experiment, LLR) are sorted in decreasing order by calling the function sort_LLR()
        * the Spearman correlation coefficient is computed between every tup(Experiment, LLR) 
          by calling the function compute_correlation() on:
            - normalized TIP raw.scores by calling the function normalize_scores()
            - padded TIP raw.scores with the max score (when normalized, by adding 1s)
        * LLR(X-Y) for the specific evidence type t (scRNAseq data in this case) is calculated 
          as the weighted sum of the individual LLR(X-Y)e for each experiment e of type t as follows:
            LLR(a,b)_{t} = \sum_{e} LLR(a,b)_{e} \prod_{k<e} d_{ek}  with  d_{ek} = α(1-max(0,r_{ek}))
    '''
    
    ## sort LLR session
    if save_LLR: LLRs = LLR_table(tissue,gs,save=True)
    else:
        table = pq.read_table('LLRs_%s_%s.parquet' %(gs,tissue))
        LLRs = table.to_pandas()

    ## Spearman correlation session
    if save_corr:
        print("Computing correlations")
        correlations = pd.DataFrame.corr(LLRs[~LLRs.isnull().any(axis=1)],method="spearman")\
                        .where(np.triu(np.ones((LLRs.shape[1],LLRs.shape[1])), k=1).astype(bool))\
                        .stack().reset_index()
        correlations.columns=['exp1','exp2','corr']
        print("Correlations computed")
        print(correlations)
        correlations.to_csv('SPcorrLLR_%s_%s.csv' %(gs,tissue))
    else:
        correlations = pd.read_csv('SPcorrLLR_%s_%s.csv' %(gs,tissue))
        correlations.drop("Unnamed: 0",axis=1,inplace=True)
    # Now we compute LLR of every TF-gene by taking into account the Spearman 
    # correlation coefficient r_ek as in the formula: d_ek = α(1-max(0,r_ek))
    if not merge:
        llr_splits = np.array_split(LLRs,cores_n)

        print('##Start up parallelization')
        Parallel(n_jobs=cores_n, verbose=10)\
            (delayed(fc_computation)(correlations, llr_splits[split_n], alpha, split_n,gs) \
                 for split_n in range(cores_n))
    if save_fc:
        fc = {}
        for core in range(cores_n):
            with open('weighted_funcoup_%s_%s_core%s.pkl' %(tissue,gs,core), 'rb') as f:
                print("Adding split n ",core)
                fc.update(pickle.load(f))
                print(len(fc))

        if nx:
            df = pd.DataFrame({"pairs":fc.keys(),"weight":fc.values()})
            df[['gene1', 'gene2']] = df['pairs'].str.split('__', expand=True)
            df.drop("pairs",axis=1,inplace=True)
            print("Storing Dataframe")
            df.to_csv("weighted_funcoup_%s_%s.csv" %(tissue,gs),index=False)
        else:
            print("Storing final dictionary")
            with open('weighted_funcoup_%s_%s.pkl' %(tissue,gs), 'wb') as fout:
                pickle.dump(fc, fout)
    dist_plot(fc,tissue,gs)
    return(fc)

def dist_plot(fc,tissue,gs):
        plt.rcParams["figure.figsize"] = [18, 18]
        plt.rcParams["figure.autolayout"] = True
        sns.displot(fc.values(),binwidth = 0.3,color='slateblue',bins=20)
        plt.xlabel("LLRs",fontsize=10)
        plt.ylabel('log(Counts)',fontsize=10)
        plt.yscale('log')
        plt.yticks(fontsize=8)
        plt.xticks(fontsize=8)
        plt.title(f"LLR distribution - {gs} - {tissue}",fontsize=13)
        plt.savefig(f"weighted_funcoup_{tissue}_{gs}.png")  

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute and assign LLR to TF-gene links')
    parser.add_argument('-og','--organism', dest='organism', default='human',
                        help='provide the common name of the organism under study (by default human, alternately mouse)')
    parser.add_argument('-fdr', dest='fdr', default='01',
                        help='provide the value alpha of FDR (by default 01, alternately 05)')
    parser.add_argument('-tr','--threshold', dest='threshold', default=False, action='store_true',
                        help='activate threshold mode and save just links with LLR > X (by default False)')
    parser.add_argument('-fl','--filter', dest='filter', default=False, action='store_true',
                        help='activate filter mode (by default False)')
    parser.add_argument('-nx','--netx', dest='netx', default=True, action='store_true',
                        help='activate networkX mode to build a proper input (by default False)')
    parser.add_argument('-grn','--CancerGRN', dest='CancerGRN', default=False, action='store_true',
                        help='activate CancerGRN mode to build a proper input (by default False)')
    parser.add_argument('-ens','--ensembl', dest='ensembl', default=False, action='store_true',
                        help='activate ensembl mode to map gene symbol to Ensembl identifiers (by default False)')
    parser.add_argument('-Nens','--NOTensembl', dest='NOTensembl', default=False, action='store_true',
                        help='activate NOTensembl mode to retrieve unmapped gene symbol to Ensembl identifiers as #tf-link (by default False)')
    parser.add_argument('-c','--cores',type=int,default=8,help='Number of cores for parallelization, default is 8')
    parser.add_argument('-m','--merge', default='False', help='Use the script to merge previously computed parts of the network; default is False')

    args = parser.parse_args() 
    ## This session is dedicated to build from scratch ENCODE and FC dataset
    exp_list = [f for f in os.listdir(".") if "tsv" in f]
    tissue = exp_list[0].split("_")[3].split(".")[0]
    gs = exp_list[0].split("_")[1]
    if args.merge:
        FC = weighted_funcoup(tissue,gs,save_LLR=False, save_corr=False, save_fc=True,cores_n=args.cores,merge=args.merge)
    else:
        FC = weighted_funcoup(tissue,gs,save_LLR=True, save_corr=True, save_fc=True,cores_n=args.cores,merge=False)
