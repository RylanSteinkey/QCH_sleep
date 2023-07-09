import pandas as pd
import numpy as np
import os, sys

from statistics import mode
from sklearn.metrics import cohen_kappa_score

import seaborn as sns
import matplotlib.pyplot as plt

def is_equal(l,r,rule):
    """
    Determines if 2 strings are equal,
    Acceptable: W,N1,N2,N3,R
    """
    if rule == 'pure':
        if l==r:
            return True
        else:
            return False
    elif rule == 'non_REM_merge':
        if l==r or (l in ['N1','N2','N3'] and r in ['N1','N2','N3']):
            return True
        else:
            return False
    elif rule == 'wake_sleep':
        if l==r or (l in ['N1','N2','N3','R'] and r in ['N1','N2','N3','R']):
            return True
        else:
            return False
    else:
        raise Exception("Rule {} not supported".format(rule))
def load_excel(file):
    """
    Load file into 2 df's; metadata and predictions
    """
    # df is dict(sheet_name, sheet_data)
    df = pd.read_excel(file, sheet_name = None)

    sheets = list(df.keys())
    for sheet in sheets:
        df[sheet] = [df[sheet].iloc[:10], df[sheet].iloc[10:]]
    return sheets, df

def add_gold_stds(sheets, dfs):
    """
    Calculates the gold standards, adds it as a new row
    """
    for sheet in sheets:
        train_status = dfs[sheet][0].values[1]
        trained_cols = [i for i,j in zip(dfs[sheet][0].columns, train_status) if j=='Trained']

        preds = dfs[sheet][1]
        trained_only_preds = preds[trained_cols].values

        gold_column = [mode(i) for i in trained_only_preds]
        dfs[sheet][1]['gold_std'] = gold_column

    return dfs

def get_perc_ck_per_class(dfs,sheets):
    """
    Calculates % simularity and cohens kappa for each predictor
    as well as means across predictor classes (trained/untrained)

    Returns 2xN array
                {scorer1..scorer13}, Trained, Untrained, C4, F4, O2
    % Simular
    Cohens_k
    """
    rules = ['pure','non_REM_merge','wake_sleep']
    sum_ars = {}
    for rule in rules:
        sum_ars[rule] = {}

    for sheet in sheets:
        train_status = dfs[sheet][0].values[1]
        trained_cols = [i for i,j in zip(dfs[sheet][0].columns, train_status) if j=='Trained']

        for rule in rules:
            # row0 = % sim, row1 = ck
            sums = pd.DataFrame()
            gold_stds = dfs[sheet][1]['gold_std']
            for col_name in dfs[sheet][0].columns:
                if col_name == 'Sleep' or col_name == 'Unnamed: 23':
                    continue
                #compare column to gold std, get %sim and ck

                col = dfs[sheet][1][col_name]
                perc = np.sum([i==j for i,j in zip(col,gold_stds)])/len(gold_stds)

                #ck_fake_left = [is_equal(i,j,rule) for i,j in zip(col,gold_stds)]
                #ck = cohen_kappa_score(ck_fake_left,[True for i in ck_fake_left]) #might not be quite accurate, using below method instead

                ck = cohen_kappa_score(gold_stds,[i if is_equal(i,j,rule) else j for i,j in zip(gold_stds,col)])
                sums[col_name] = pd.Series([perc,ck])

            #average categories into new column # TODO NEXT
            for group in ['Trained', 'In Training']:
                mean_perc = np.mean(sums[[i for i,j in zip(dfs[sheet][0].columns, train_status) if (j==group and 'U-Sleep' not in i)]].values[0])
                mean_ck = np.mean(sums[[i for i,j in zip(dfs[sheet][0].columns, train_status) if (j==group and 'U-Sleep' not in i)]].values[1])
                sums[group+" Average"] = pd.Series([mean_perc,mean_ck])

            sum_ars[rule][sheet] = sums
    return sum_ars

def acc_by_title(summary_arrays, sheets):
    """
    Takes the summary arrays and in a single dataframe,
    returns the accuracy for each type of trainer

    rows = trained status
    cols = study number
    """
    sim_dfs = {}
    ck_dfs = {}
    types = ['Trained Average', 'In Training Average', 'U-Sleep_F4', 'U-Sleep_C4', 'U-Sleep_O2']
    for rule in ['pure','non_REM_merge','wake_sleep']:
        ck_df = pd.DataFrame(index=types)
        sim_df = pd.DataFrame(index=types)
        for sheet in sheets:
            sim_data = []
            ck_data = []
            sdf = summary_arrays[rule][sheet]
            for col_name in types:
                sim_data.append(sdf[col_name].values[0])
                ck_data.append(sdf[col_name].values[1])

            ck_df[sheet] = pd.Series(ck_data,index=types)
            sim_df[sheet] = pd.Series(sim_data,index=types)


        sim_dfs[rule] = sim_df
        ck_dfs[rule] = ck_df
    return sim_dfs, ck_dfs

def print_figs(ck_sum):
    """
    Quick and dirty figure generation for a 5x50 array
    """
    ax = sns.barplot(data=ck_sum.T, errorbar='ci')
    ax.set(xlabel = "Predictor", ylabel="Cohen's Kappa (n=50)")
    plt.show()


def main():
    sheets, dfs = load_excel("data/QSleep_U-Sleep_Data.xlsx")
    #print(dfs[0].columns,dfs[1].columns)
    dfs = add_gold_stds(sheets, dfs)
    summary_arrays = get_perc_ck_per_class(dfs,sheets)
    sim_sum, ck_sum = acc_by_title(summary_arrays, sheets)
    print_figs(ck_sum['wake_sleep'])

    #plot



if __name__ == "__main__":
    main()
