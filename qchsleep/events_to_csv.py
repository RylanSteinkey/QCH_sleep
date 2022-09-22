import os, sys

import pandas as pd
import numpy as np
from itertools import groupby
import datetime

"""
Turns a file or a directory of files that contain sleep events
into a time matched csv

Usage:
`python events_to_csv.py`
"""

def get_files_list(path):
    """
    Gets a list of files if a dir is passed,
    else return the path to the file
    """
    files_list = []
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for filename in files:
                files_list.append(os.path.join(root, filename))
    else:
        files_list.append(os.path.abspath(path))

    return[list(i) for j, i in groupby(sorted(files_list), lambda x: x.split(' ',1)[0])]

def read_time(t):
    """
    Converts weird text date into python readable
    U-Sleep: '2018/10/09-14:19:10'
    RemLogic:  ['7/02/2022', '11:30:09 PM']
    """
    if isinstance(t, list):
        date, time = t
        date = [date.split('/')[2], date.split('/')[1], date.split('/')[0]]
        if time == '12:00:00 AM':
            date[2] = str(int(date[2])+1)
        try:
            time, half = time.split(' ')
        except:
            print(time)
        time = time.split(':')
        if half == 'PM':
            pm = True
        else:
            pm = False
        vals = [int(i) for i in date + time]

        dt = datetime.datetime(*vals)
        if pm:
            return dt + datetime.timedelta(hours = 12)
        elif time[0] == '12':
            return dt - datetime.timedelta(hours = 12)
        else:
            return dt

    else:
        t = t.split('=')[1].split(' ')[0]
        date, time = t.split('-')
        vals = [int(i) for i in date.split('/') + time.split(':')]

        return datetime.datetime(*vals)

def U_Sleep(files, out):
    """
    Takes a directory of directories containing traces.
    returns pandas df
    """
    all_dfs = []
    writer = pd.ExcelWriter(out, engine='xlsxwriter')
    for date_group in files:
        df = pd.DataFrame()
        last_timestamp = 'empty'
        for EEG_lead in date_group:
            events = []
            with open(EEG_lead) as file:
                try:

                    for line_number, line in enumerate(file):
                        line = line.rstrip()
                        if line_number == 0:
                            try:
                                assert line == 'EPOCH=30.0s'
                            except:
                                raise Exception("Epoch not set to 30s")
                        elif line_number == 1:
                            time_stamp = read_time(line)
                            if last_timestamp == 'empty':
                                last_timestamp = time_stamp
                            else:
                                try:
                                    assert last_timestamp == time_stamp
                                except:
                                    raise Exception("starting timestamps dont match for 2 of the same EEG leads")
                        else:
                            events.append(line)

                except:
                    print("Error when reading file {}".format(file))
                    raise
            eeg = EEG_lead.split(' ')[-1].split('/')[0]
            df[eeg] = pd.Series(events)
        times = []
        for i in range(len(events)):
            times.append(last_timestamp + datetime.timedelta(0,i*30))
        times = [i.ctime() for i in times]
        df['U-Sleep_epoch_start'] = pd.Series(times)
        df = df.set_index('U-Sleep_epoch_start')

        df.to_excel(writer, sheet_name=date_group[0].split(' ')[0].split('/')[-1])
        all_dfs.append(df)
    writer.save()

    return all_dfs

def RemLogic_to_pandas(file_path):
    """
    Takes a path to a remlogic event export and
    turns it into a pandas series at the correct time
    !!!DEPRECATED, USED dynamic_RemLogic_to_pandas()!!!
    """
    events = []
    early_start = False # if starting 4 lines early,
    """
    note that for the above case, you must start 4 lines early, and the data might
    have an analysis start after 2 lines, but to match the times you have to do 4 lines,
    hence why this is hard coded instead of looking for the correct header
    """
    late_start = False # if starting 1 line late
    with open(file_path) as file:
        for i, line in enumerate(file):
            # skip the first 19 lines, but record date on line 3

            if i == 3:
                rdate = line.rstrip().split('\t')[-1]
            # double check the date starts here

            elif i == 15:
                if line[:15] == 'Time [hh:mm:ss]':
                    # This means the scoring start earlier
                    # skipping the scoring session info, also
                    # most likely doesnt have a sleep staging column
                    early_start = True

            elif i == 19 and early_start == False:
                try:
                    assert line[:11] == 'Sleep Stage'
                except:
                    raise Exception("data doesnt appear to start at the correct line")
            elif i == 20 and early_start == False:
                # record the starting epoch time

                #This is if there are 5 columns, including position
                if len(line.rstrip().split('\t')) == 5:
                    rtime = line.rstrip().split('\t')[2]

                #This is if there are 4 columns, NOT including position
                elif len(line.rstrip().split('\t')) == 4:
                    rtime = line.rstrip().split('\t')[1]
                else:
                    raise Exception("number of columns was unexpected")
                events.append(line.split('\t')[0])
                time = read_time([rdate, rtime])

            elif i == 16 and early_start == True:
                # we've already confirmed the first column has time
                # or we wouldnt be in this flag
                rtime = line.rstrip().split('\t')[0]
                events.append(line.split('\t')[1])
                time = read_time([rdate, rtime])

            elif i>20 and early_start == False:
                events.append(line.split('\t')[0])
            elif i>16 and early_start == True:
                # take the 2nd item instead of the first, because there is no
                # sleep stagin column, need to grab the event
                events.append(line.split('\t')[1])
            else:
                pass #skips the initial garbage
    return pd.Series(events), time

def add_gold_stds(files, usleep, out):
    """
    Takes a directory of directories containing traces
    that are determined gold standard.
    adds the gold standard to the u-sleep prediction excel
    """
    writer = pd.ExcelWriter(out, engine='xlsxwriter')
    xlsx = pd.ExcelFile(usleep)
    date_groups = [i[0].split(' ')[-1].split('.')[0] for i in files]
    for date_group, file in zip(date_groups, files):
        df = pd.read_excel(xlsx, date_group)
        u_time = df['U-Sleep_epoch_start'][0]
        u_time = datetime.datetime.strptime(u_time, '%c')
        events, stime = RemLogic_to_pandas(file[0])
        #print(len(events))

        time_diff = stime-u_time

        tms = [int(i) for i in str(time_diff).split(':')]
        secs = tms[0]*3600+tms[1]*60+tms[2]

        cells_to_skip = secs//30

        if secs % 30 >15:
            cells_to_skip +=1

        pad_after = len(df.index) - cells_to_skip - len(events)

        gold_events = pd.concat([pd.Series(['' for i in range(cells_to_skip)]),events,pd.Series(['' for i in range(pad_after)])])

        gold_times = [stime + datetime.timedelta(seconds=30)*i for i in range(len(gold_events))]
        times = ['' for i in range(cells_to_skip)]+[i.ctime() for i in gold_times]+['' for i in range(pad_after)]

        df['gold_std'] = gold_events.values
        df['gold_time_stamp'] = pd.Series(times)

        df.set_index('U-Sleep_epoch_start')

        df.to_excel(writer, sheet_name=date_group)
    writer.save()

def dynamic_RemLogic_to_pandas(file_path):
    """
    Too many files have different formats with different rules
    the above RemLogic function needs more general rules

    Anyways, converts a remlogic events output, in any format,
    and any amount of columns, into a pandas series. Unlike the other
    remlogic to pandas function, this finds the line the equivalent gold
    would start at, and searches for which columns give the Correct
    event info and time stamps.
    """
    events = []
    start_recording = False
    with open(file_path) as file:
        for i, line in enumerate(file):
            if start_recording == False:
                if i == 3:
                    rdate = line.rstrip().split('\t')[-1]
                elif 'Time [hh:mm:ss]' in line:
                    start_recording = i

                    if 'Event' in line:
                        method = 'Event'
                    elif 'Sleep Stage' in line:
                        method = 'Sleep Stage'
                    else:
                        print(line)
                        raise Exception("No column of sleep staging found in file {}".format(file_path))
                    stage_index = line.split('\t').index(method)
                    time_index = line.split('\t').index('Time [hh:mm:ss]')

                else:
                    pass
            else:
                if i == start_recording+1:
                    # only recording the first time
                    rtime = line.rstrip().split('\t')[time_index]
                    time = read_time([rdate,rtime])

                # recording each event, for every line
                event  = line.rstrip().split('\t')[stage_index]
                events.append(event)
    return pd.Series(events), time

def main():
    # Primary data aggregation
    outu = 'data/u_sleep_predictions.xlsx'
    dfs = U_Sleep(get_files_list("data/U-Sleep_predictions"),outu)
    outf = 'data/U-Sleep-v-Gold.xlsx'
    add_gold_stds(get_files_list("data/gold_labels"), outu, outf)


if __name__ == "__main__":
    main()
