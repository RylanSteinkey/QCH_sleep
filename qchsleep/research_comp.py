import yaml
from qchsleep import events_to_csv

import numpy as np
import pandas as pd
import os, sys

import datetime

from qchsleep.stats import is_equal, is_empty, ck_prep

from sklearn.metrics import cohen_kappa_score



date_groups = ['30-may-2018','7-feb-2022','9-oct-2018','2-may-2018','18-feb-2021','3-mar-2015']

def read_job_title_yaml():
    """
    Reads yaml into dict
    """
    with open("data/staff.yaml", 'r') as jobs:
        titles = yaml.safe_load(jobs)

    return titles

def scores_to_df(files):
    """
    Takes directories of research PSG scoring
    turns into dataframes
    i in files look like:
    'data/researcher_reports/18-feb-2021/fname_last_name.txt'
    """

    pages = {}
    for date in date_groups:
        pages[date] = {}
    for file in files:
        file = file[0]
        try:
            pdseries, time = events_to_csv.dynamic_RemLogic_to_pandas(file)
        except:
            print("problem is with ",file)
            raise

        g,d, date, name = file.split('/')
        fname = name.split('.')[0].split('_')[0]
        pages[date][name]=[pdseries,time]

    return pages

def save_predictions(pages, out):
    """
    turns the pages dictionary into an excel sheet
    """
    fail_next = False
    writer = pd.ExcelWriter(out, engine='xlsxwriter')
    for date_group in date_groups:
        g, gold_time = pages[date_group]['Gold.txt']
        series_data = pages[date_group]
        max_len = max([len(i[0]) for i in series_data.values()])
        df = pd.DataFrame()
        for researcher in series_data.keys():
            preds, time = series_data[researcher]

            # if preds is shorter than the gold time, we need to pad on unscored.
            to_add = len(g)-len(preds)
            preds = list(preds.values)
            for i in range(to_add):
                preds.append('SLEEP-UNSCORED')

            # check that the series starts at the same time as the gold
            if time == gold_time:
                df[researcher] = preds[:len(g)]

            else:
                time_diff = gold_time - time
                epochs_to_shift = round(time_diff.total_seconds() / 30)
                if epochs_to_shift >= 0:
                    # researcher started before gold, need to remove first X gradings
                    df[researcher] = preds[epochs_to_shift:][:len(g)]

                else:
                    # research started late, need to add a blank, i hope we dont have this though
                    pads = ['SLEEP-UNSCORED' for i in range(abs(epochs_to_shift))]
                    new_pred = pads+preds
                    df[researcher] = new_pred[:len(g)]


        times = []
        for i in range(max_len):
            times.append(gold_time + datetime.timedelta(0,i*30))
        times = [i.ctime() for i in times]
        df['time_stamp'] = pd.Series(times)
        df = df.set_index('time_stamp')

        #df.index = times
        df.columns = [i.split('.')[0] for i in df.columns]

        df.to_excel(writer, sheet_name=date_group)
    writer.save()

def read_scorings_for_figs():
    """
    Reads the 'data/all_researcher_scorings.xlsx' file and returns
    the data in a way that seaborn can easily turn it into a figure
    i.e. no math is done in figures.py
    """
    #is_equal(l,r,rule)

    # merge in gold standards (all 3 leads)
        # keep in mind that the u-sleep v gold.xlsx will have 12h of data,
        # will need to line it up correctly
    dfs = []

    for date_group in date_groups:
        staff_data = pd.read_excel('data/all_researcher_scorings.xlsx',date_group)
        staff_data = staff_data.set_index('time_stamp')

        model_data = pd.read_excel('data/u_sleep_predictions.xlsx',date_group)
        model_data = model_data.set_index('U-Sleep_epoch_start')

        # indeces might not match, model needs to be moved to match staff_data
        staff_start = datetime.datetime.strptime(staff_data.index[0], '%c')
        model_start = datetime.datetime.strptime(model_data.index[0], '%c')
        time_diff = staff_start - model_start

        # if time diff < 15 seconds, just move model data by that much time
        # else, we move it whichever is closest
        if time_diff.seconds % 30 > 15:
            # reverse
            # means we are closer to the next epoch, reverse forward 30-d seconds
            # i.e. if the diff is 22 seconds, move back 8
            new_times = [datetime.datetime.strptime(i, '%c')-datetime.timedelta(seconds=30-(time_diff.seconds % 30)) for i in model_data.index]

        else:
            # advance
            # e.g. if 8 seconds, then we need to move forward 8 seconds to catch the next epoch
            new_times = [datetime.datetime.strptime(i, '%c')+datetime.timedelta(seconds=(time_diff.seconds % 30)) for i in model_data.index]

        model_data.index = [i.ctime() for i in new_times]

        all_data = pd.merge(staff_data, model_data, left_index = True, right_index = True)
        dfs.append(all_data)
        # could save the above df if you wanted all the raw data in one spot, i guess
    return dfs

def grade(lc, rc, is_USleep):
    """
    Takes 2 columns of data, compares them and returns
    [[pure, non_REM_merge, wake_sleep], #similarity
     [pure, non_REM_merge, wake_sleep]] #cohen_kappa_score
    """
    grades = []
    sim_tests = ['pure', 'non_REM_merge', 'wake_sleep']

    assert len(lc) == len(rc)

    sim_grades = []
    for sim_test in sim_tests:
        sim = np.sum([is_equal(i,j,sim_test) for i,j in zip(lc,rc)])/len(lc)
        sim_grades.append(sim)
    grades.append(sim_grades)

    ck_grades = []
    for sim_test in sim_tests:
        if is_USleep:
            temp_rc = ck_prep(lc, rc, sim_test)
        else:
            temp_rc = []
            for i, j in zip(lc, rc):
                if is_equal(i,j,sim_test):
                    temp_rc.append(i)
                else:
                    temp_rc.append(j)
        ck = cohen_kappa_score(lc, temp_rc)
        ck_grades.append(ck)
    grades.append(ck_grades)

    return grades

def grade_scorings_for_figs(stages):
    """
    takes an array of dataframes from read_scorings_for_figs()
    returns an array of dictionaries where each element of the array is:
    key: staff_first_name
    value: [[pure, non_REM_merge, wake_sleep], #similarity
            [pure, non_REM_merge, wake_sleep]] #cohen_kappa_score
    """
    grades = [] #array of 6 date groups
    for df in stages:
        staffs = {}
        for staff in [i for i in df.columns if i !='Gold']:
            comparable = [ not is_empty(i) and not is_empty(j) for i,j in zip(df['Gold'], df[staff])]
            staff_df = df
            staff_df['comp'] = comparable
            staff_df = staff_df[staff_df['comp']==True]

            if staff in ['F4','O2','C4']:
                usleep = True
            else:
                usleep = False

            staffs[staff] = grade(staff_df['Gold'], staff_df[staff], usleep)

        grades.append(staffs)

    return grades

def save_grades(grades):
    """
    Writes the grades into an xlsx sheet
    """
    writer = pd.ExcelWriter("all_gradings.xlsx", engine='xlsxwriter')
    for date_group, data in zip(date_groups,grades):
        df = pd.DataFrame()
        for staff in data:
            pass
    # TODO

def main():
    titles = read_job_title_yaml()
    files = events_to_csv.get_files_list("data/researcher_reports")
    pages = scores_to_df(files)
    outf = 'data/all_researcher_scorings.xlsx'
    save_predictions(pages, outf)

    #read researcher scorings
    stages = read_scorings_for_figs()

    #compare research scorings to gold standard
    grading = grade_scorings_for_figs(stages)

    #save_grads(grading)


if __name__ == "__main__":
    main()
