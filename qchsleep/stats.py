import os, sys

import pandas as pd
import numpy as np

from sklearn.metrics import cohen_kappa_score

date_groups = ['30-may-2018','7-feb-2022','9-oct-2018','2-may-2018','18-feb-2021', '3-mar-2015']

expected_supports = {'30-may-2018':200,'7-feb-2022':200,'9-oct-2018':229,'2-may-2018':272,'18-feb-2021':241,'3-mar-2015':218, 'test':8}

def converter(vals):
    """
    Converts from U-Sleep to Remlogic, or vice versa
    i.e. SLEEP-S1 will become N1, or reverse
    """
    rem2U = {"SLEEP-S0":'Wake',"SLEEP-S1":'N1',"SLEEP-S2":'N2',"SLEEP-S3":'N3',"SLEEP-REM":'REM'}
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


def grade(df, date_group):
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
            assert len(lc)==len(rc)
            assert len(lc) == expected_supports[date_group]
        except:
            raise Exception("indeces not matching for comparison")
            raise

        sim_results = []
        cohen_kappa_results = []
        for sim_test in ['pure', 'non_REM_merge', 'wake_sleep']:
            sim = np.sum([is_equal(i,j,sim_test) for i,j in zip(lc,rc)])/len(lc)


            # error testing
            """
            if lead == 'F4' and sim_test == 'pure':
                np.save('data/test_lc.npy', lc)
                np.save('data/test_rc.npy', rc)
                sys.exit()
            """

            fake_rc = ck_prep(lc, rc, sim_test)

            ck = cohen_kappa_score(lc, fake_rc)

            # save all data for total epoch testing
            np.save("data/all_epochs/lc_sim_{}_{}_{}.npy".format(date_group,lead,sim_test),lc)
            np.save("data/all_epochs/rc_sim_{}_{}_{}.npy".format(date_group,lead,sim_test),rc)
            np.save("data/all_epochs/frc_sim_{}_{}_{}.npy".format(date_group,lead,sim_test),fake_rc)

            assert len(fake_rc) == expected_supports[date_group]


            sim_results.append(sim)
            cohen_kappa_results.append(ck)
        sim_df[lead] = sim_results
        ck_df[lead] = cohen_kappa_results
    return sim_df, ck_df

def similarities():
    """
    Reads in the prediction tables, prints % simularity and cohen_kappa_score
    """
    data = "data/U-Sleep-v-Gold.xlsx"
    xlsx = pd.ExcelFile(data)
    for date_group in date_groups:
        df = pd.read_excel(xlsx, date_group)
        sim_df, ck_df = grade(df, date_group)

        print(date_group)
        print("Similarity")
        print(sim_df)
        print("\nCohen's Kappa")
        print(ck_df)
        print('-------------------------------------------')

def all_epoch_scores():
    """
    Merges all date groups and reports u-sleep performance
    over all concordances
    """
    sim_tests = ['pure', 'non_REM_merge', 'wake_sleep']
    sim_df = pd.DataFrame(index = sim_tests)
    ck_df = pd.DataFrame(index = sim_tests)

    for lead in ['F4','C4','O2']:
        sim_res = []
        ck_res = []
        for sim_test in sim_tests:
            all_lc = []
            all_rc = []
            all_frc = []
            for date_group in date_groups:
                lc = np.load("data/all_epochs/lc_sim_{}_{}_{}.npy".format(date_group,lead,sim_test), allow_pickle = True)
                rc = np.load("data/all_epochs/rc_sim_{}_{}_{}.npy".format(date_group,lead,sim_test), allow_pickle = True)
                frc = np.load("data/all_epochs/frc_sim_{}_{}_{}.npy".format(date_group,lead,sim_test), allow_pickle = True)

                try:
                    assert len(lc) == len(rc) == len(frc)
                except:
                    print("unequal arrays for {}_{}_{}".format(date_group,lead,sim_test))
                    raise
                for i, j in zip([lc,rc,frc],[all_lc,all_rc,all_frc]):
                    for k in i:
                        j.append(k)

            sim = np.sum([is_equal(i,j,sim_test) for i,j in zip(all_lc,all_rc)])/len(all_lc)
            ck = cohen_kappa_score(all_lc, all_frc)

            #print("epochs",len(all_lc))
            sim_res.append(sim)
            ck_res.append(ck)
        sim_df[lead] = sim_res
        ck_df[lead] = ck_res

    print("All epoch similarity")
    print(sim_df)
    print("\nCohen's Kappa")
    print(ck_df)




def u_vs_q(out):
    """
    Runs all the relevant stats required for the
    usleep vs qsleep comparison
    """
    similarities()
    all_epoch_scores()


if __name__ == "__main__":
    out = "results/usleep_vs_qsleep.csv"
    u_vs_q(out)
