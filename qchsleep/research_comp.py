import yaml
from qchsleep import events_to_csv

import numpy as np
import pandas as pd

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
            pdseries, time = events_to_csv.RemLogic_to_pandas(file)
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
    writer = pd.ExcelWriter(out, engine='xlsxwriter')
    for date_group in date_groups:
        first_time = ''
        series_data = pages[date_group]
        df = pd.DataFrame()
        for researcher in series_data.keys():
            preds, time = series_data[researcher]
            if time != '': #check to make sure the series all start at the same time
                first_time = time
            else:
                assert time == first_time

            df[researcher] = preds

        df.to_excel(writer, sheet_name=date_group)
    writer.save()

def main():
    titles = read_job_title_yaml()
    files = events_to_csv.get_files_list("data/researcher_reports")
    pages = scores_to_df(files)
    outf = 'all_researcher_scorings.xlsx'
    save_predictions(pages, outf)

    #read researcher scorings
    #compare research scorings to gold standard
    #cluster is researcher title



if __name__ == "__main__":
    main()
