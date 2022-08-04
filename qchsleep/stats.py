import os, sys

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score

date_groups = ['7-feb-2022','9-oct-2018','30-may-2018','2-may-2018','18-feb-2021']

def converter(vals):
    """
    Converts from U-Sleep to Remlogic, or vice versa
    i.e. SLEEP-S1 will become N1, or reverse
    """
    rem2U = {"SLEEP-S0":'WAKE',"SLEEP-S1":'N1',"SLEEP-S2":'N2',"SLEEP-S3":'N3',"SLEEP-REM":'REM'}
    to_rem = {v:k for k,v in rem2U.items()}

    new_vals = []
    for i in vals:
        if i in rem2U.keys():
            new_vals.append(rem2U[i])
        elif i in to_rem.keys():
            new_vals.append(to_rem[i])
        else:
            # If this flags below, its probably a Wake seen somewhere
            raise Exception("value {} not yet declared in converter()".format(i))
    return new_vals

def is_equal(l,r,rule):
    """
    Determines if 2 strings are equal,
    such as `n2` and `SLEEP-S2`
    U-Sleep: WAKE/N1/N2/N3/REM
    RemLogic: SLEEP-S0,SLEEP-S1,SLEEP-S2,SLEEP-S3,SLEEP-REM
    """
    wake = ['WAKE','SLEEP-S0']

    if rule == 'pure':
        if l[-1] == r[-1] or (l in wake and r in wake):
            return True
        else:
            return False
    elif rule == 'non_REM_merge':
        if (l[-1] == r[-1]) or (l[-1] in ['1','2','3'] and r[-1] in ['1','2','3']) or (l in wake and r in wake):
            return True
        else:
            return False

    elif rule == 'wake_sleep':
        if (l[-1] == r[-1]) or (l[-1] in ['1','2','3','M'] and r[-1] in ['1','2','3','M']) or (l in wake and r in wake):
            return True
        else:
            return False

    else:
        raise Exception("merge rule {} not defined, see is_equal()".format(rule))

def is_empty(cell):
    """
    Returns true if cell is nan or empty
    """
    if cell in ['','NaN'] or pd.isna(cell):
        return True
    else:
        return False

def ck_prep(lc, rc, rule):
    """
    converts two prediction series into forms that can be fed
    into sklearns cohens-kappa function
    NB: kappa is symmetric, so lc and rc can be swapped
    """
    fake_rc = converter(rc)
    ret_rc = []
    for i, j in zip(lc, fake_rc):
        if is_equal(i,j,rule):
            ret_rc.append(i)
        else:
            ret_rc.append(j)
    return ret_rc


def grade(df):
    """
    returns 2 3x3 pandas wheres rows = test, columns = leads,
    one for simularity, one for cohens-kappa
    """
    sim_tests = ['pure', 'non_REM_merge', 'wake_sleep']
    sim_df = pd.DataFrame(data = [], index = sim_tests)
    ck_df = pd.DataFrame(data = [], index = sim_tests)

    for lead in ['F4','C4','O2']:
        lc = df[lead]
        rc = df['gold_std']

        # mask both so they only have data where gold std labels are available
        lc = lc[[not is_empty(i) for i in rc]]
        rc = rc[[not is_empty(i) for i in rc]]

        try:
            assert np.all([i==j for i,j in zip(lc.index,rc.index)])
        except:
            raise Exception("indeces not matching for comparison")

        sim_results = []
        cohen_kappa_results = []
        for sim_test in ['pure', 'non_REM_merge', 'wake_sleep']:
            sim = np.sum([is_equal(i,j,sim_test) for i,j in zip(lc,rc)])/len(lc)

            fake_rc = ck_prep(lc, rc, sim_test)

            ck = cohen_kappa_score(lc, fake_rc)

            sim_results.append(sim)
            cohen_kappa_results.append(ck)
        sim_df[lead] = sim_results
        ck_df[lead] = cohen_kappa_results
    return sim_df, ck_df

def simularities():
    """
    Reads in the prediction tables, prints % simularity and cohen_kappa_score
    """
    data = "data/U-Sleep-v-Gold.xlsx"
    xlsx = pd.ExcelFile(data)
    for date_group in date_groups:
        df = pd.read_excel(xlsx, date_group)
        sim_df, ck_df = grade(df)

        print(date_group)
        print("Simularity")
        print(sim_df)
        print("\nCohen's Kappa")
        print(ck_df)
        print('-------------------------------------------')


def u_vs_q(out):
    """
    Runs all the relevant stats required for the
    usleep vs qsleep comparison
    """
    simularities()


if __name__ == "__main__":
    out = "results/usleep_vs_qsleep.csv"
    u_vs_q(out)
