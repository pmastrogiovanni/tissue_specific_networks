#! /home/pmastro/miniconda3/envs/dataexp/bin/python

'''
Script that takes as input 
1) a dataframe from RNA-seq data where each gene pair is associated to its correlation 
2)the GS to use. 
Compute the KDE and return a df where each gene pair is associated with a LLR
'''
import numpy as np
import matplotlib.pyplot as plt
from KDEpy import FFTKDE
import argparse
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time
pd.set_option('display.max_rows', None)
pd.options.display.float_format = '{:.10f}'.format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)



def KDE(gs,cores):               #perform cross validation on GS to obtain the best bandwidth parameter
    data = gs['correlations'].sort_values()
    data = data.to_list()
    data = [[e] for e in data]
    print('grid search started')
    grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.001, 0.2, 20)}, cv=5, n_jobs= cores, refit=True) # 3-fold cross-validation
    print('fitting started')
    grid.fit(data)
    print(grid.best_params_)
    bw = grid.best_params_['bandwidth']
    print('cross val done')

    #Compute KDE using the best bandwidth, return x and y where x are metric's values (corr, jacc, ...) and y the probability density function(pdf) values associated
    kde = FFTKDE(kernel='gaussian', bw=bw).fit(data)
    x, y = kde.evaluate(50000)
    print('dist done')

    return bw,x,y,data



def interpolation(x, y, df, gs):    #Assign to every metric's score value of the dataframe the associated pdf value from the GS and store them in a new df column
    f = interp1d(x, y, kind='slinear', assume_sorted=True, bounds_error=False)
    df = df.sort_values(by=['correlations'])
    y_df = f(df['correlations'])
    df['pdf_%s'%gs] = y_df
    return df



def compute_llr(df):                        #Compute LLR, defined as log ratio between pos and neg GS. Values outside the GS range are filled with the last LLR described by the models
    llr = np.log(df['pdf_pos']/df['pdf_neg'])
    df['llr'] = llr
    df['llr'] = df['llr'].ffill().bfill()
    #first, last = df.llr[~df.llr.isna()].values[[0, -1]]
    #nan = df[df.isna().any(axis=1)]
    #outliers = len(nan)/len(df)
    #nan_neg = nan[nan['corr']<0].copy()
    #nan_pos = nan[nan['corr']>0].copy()
    #nan_neg['llr'] = nan_neg['llr'].fillna(first)
    #nan_pos['llr'] = nan_pos['llr'].fillna(last)
    #pos_values = len(nan_pos)
    #neg_values = len(nan_neg)
    #df.drop(df.tail(pos_values).index,inplace=True)
    #df.drop(df.head(neg_values).index,inplace=True)
    #df = df.append(nan_pos)
    #df = pd.concat([nan_neg,df])
    df.drop(columns=['pdf_pos', 'pdf_neg'], inplace=True)
    print('LLR computed')
    return df



def chunking(df,x_pos,y_pos,x_neg,y_neg,name,gs):      #take chunks from the correlation table and compute the llrs

    print('starting chunking')
    head = pd.DataFrame(columns=['pair', 'correlations', 'llr'])
    head = head.set_index('pair')
    head.to_csv("llr_%s_%s.tsv" % (gs,name), sep="\t", mode='w')
    for chunk in pd.read_csv(df, chunksize=10000000):
        chunk.drop(["Unnamed: 0"],axis=1,inplace=True)
        chunk["pair"] = chunk.protein1.str.cat(chunk.protein2,sep="__")

        #print(chunk.head())
        chunk = interpolation(x_pos, y_pos, chunk, 'pos')
        chunk = interpolation(x_neg, y_neg, chunk, 'neg')
        print('interpolations done')
        chunk = compute_llr(chunk)
        chunk = chunk.set_index('pair')
        chunk.drop(["protein1","protein2"],axis=1,inplace=True)
        chunk.to_csv("llr_%s_%s.tsv" % (gs,name), sep="\t", mode='a', header=False)
        print(chunk.head())



def plot_kde(x_pos, y_pos, x_neg, y_neg, data_pos, data_neg, name):
    df_name = name.split(sep="_")[0]
    gs_name= name.split("_")[1]
    plt.figure(1)           #figure showing the KDE models and the GS values
    plt.plot(x_pos, y_pos ,zorder=10, color='blue', label='KDE pos')
    plt.plot(x_neg, y_neg ,zorder=10, color='red', label='KDE neg')
    plt.scatter(data_pos, np.full_like(data_pos, -0.1), marker='|', c='b')
    plt.scatter(data_neg, np.full_like(data_neg, -0.5), marker='|', c='r')
    plt.legend(loc='best')
    plt.xlabel('corr')
    plt.ylabel('pdf')
    plt.xlim(-1,1)
    plt.title('%s - %s (%i+,%i-)' %(df_name,gs_name,len(data_pos),len(data_neg)))
    plt.savefig('KDE_%s_%s.png' % (gs_name,name))


def plot_log_kde(x_pos, y_pos, x_neg, y_neg, data_pos, data_neg, name):
    df_name = name.split(sep="_")[0]
    gs_name= name.split("_")[1]
    y_pos_log = np.log(y_pos)
    y_neg_log = np.log(y_neg)

    plt.figure(2)           #figure showing the KDE models in log scale and the GS values
    plt.plot(x_pos, y_pos_log ,zorder=10, color='blue', label='KDE pos')
    plt.plot(x_neg, y_neg_log ,zorder=10, color='red', label='KDE neg')
    plt.scatter(data_pos, np.full_like(data_pos, 2), marker='|', c='b')
    plt.scatter(data_neg, np.full_like(data_neg, 3), marker='|', c='r')
    plt.legend(loc='best')
    plt.xlabel('corr')
    plt.ylabel('pdf')
    plt.xlim(-1,1)
    plt.title('%s - %s (%i+,%i-)' %(df_name,gs_name,len(data_pos),len(data_neg)))
    plt.savefig('log_KDE_%s_%s.png' % (gs_name,name))



def plot_llr(gs,df_name):
    chunk_plot = pd.read_csv("llr_%s_%s.tsv" % (gs,df_name), sep="\t", nrows=20000000)

    fig = plt.figure(3)     #figure showing the LLR associated
    ax1 = fig.add_subplot(111)
    ax1.scatter(chunk_plot['correlations'], chunk_plot['llr'], c='blue', label='evidence_df', s=2)
    plt.xlabel('corr')
    plt.ylabel('llr')
    plt.legend(loc='best')
    plt.axhline(0, color='black')
    plt.savefig("llr_%s_%s.png" % (gs,df_name))



if __name__ == '__main__':

    start = time.time()

    parser = argparse.ArgumentParser(description='...')
    parser.add_argument('-d', help='correlations')
    parser.add_argument('-pgs', help='positive gold standard')
    parser.add_argument('-ngs', help='negative gold standard')
    parser.add_argument('-c',default=8,type=int, help="Number of cores to be used")
    parser.add_argument('-do',default="plot",type=str,help="Specify the action to perform. Options are 'plot','llr' and both")

    args = parser.parse_args()
    df = args.d
    cores = args.c
    do = args.do
    name = args.d.split(sep=".")[0]
    gs = args.ngs.split(".")[0].split("_")[0]


    print("Working on positive gs")
    pos_gs = pd.read_csv(args.pgs)
    min_pos,max_pos = min(pos_gs['correlations']), max(pos_gs['correlations'])
    bw_pos,x_pos, y_pos, data_pos = KDE(pos_gs,cores)

    print("Working on negative gs")
    neg_gs = pd.read_csv(args.ngs)
    min_neg,max_neg = min(neg_gs['correlations']), max(neg_gs['correlations'])
    bw_neg,x_neg, y_neg, data_neg = KDE(neg_gs,cores)
 
    #Returning to the original value ranges
    x_y_pos = pd.DataFrame({'x':pd.Series(x_pos), 'y':pd.Series(y_pos)})
    x_y_neg = pd.DataFrame({'x':pd.Series(x_neg), 'y':pd.Series(y_neg)})
    x_y_pos = x_y_pos[(x_y_pos['x'] >= min_pos) & (x_y_pos['x'] <= max_pos)]
    x_y_neg = x_y_neg[(x_y_neg['x'] >= min_neg) & (x_y_neg['x'] <= max_neg)]
    x_pos,y_pos = x_y_pos['x'].to_numpy(), x_y_pos['y'].to_numpy()
    x_neg, y_neg = x_y_neg['x'].to_numpy(), x_y_neg['y'].to_numpy()


    if do == "plot" or do == "both":
        plot_kde(x_pos, y_pos, x_neg, y_neg, data_pos, data_neg,name)
        plot_log_kde(x_pos, y_pos, x_neg, y_neg, data_pos, data_neg,name)
    if do == "llr" or do == "both":
        print("LLR computation started")
        chunking(df,x_pos,y_pos,x_neg,y_neg,name,gs)

        plot_llr(gs,df_name=name)

    end = time.time()
    print(end - start)