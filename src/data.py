from pathlib import Path
import zipfile
from loguru import logger
from tqdm import tqdm
from src.voting_experts import VotingExperts
import numpy as np
import pandas as pd

def extract(example_name):

    data_folder = Path('./data')

    extract_to = data_folder / example_name
    file_name_to_extract = data_folder / f"{example_name}.zip"
    extract_to.mkdir(exist_ok=True)

    logger.info(f"extracting dataset {file_name_to_extract} to {extract_to}")
    try:
        with zipfile.ZipFile(file_name_to_extract, "r") as zip_ref:
            zip_ref.extractall(extract_to)

    except:
        logger.info(f"Cannot extract {file_name_to_extract} to {extract_to}")

    return extract_to

def preprocessing(dataframe=None, source_dir=".", source_file=None):
    
        
    if dataframe is None:
        df = pd.read_csv(source_dir/source_file, engine='c', na_filter=False, sep=';', memory_map=True, header=None)
        # TODO: check if file contains timestamps and change this ifs
        df_len = len(df.columns)
        # logger.info(f"there are {df_len} columns in file")

        if df_len == 5: 
            df.columns = ['LineNumber','Time', 'ThreadID', 'Level', 'EventId']
        elif df_len == 4:
            df.columns = ['LineNumber','ThreadID', 'Level', 'EventId']
        # df['Label'] = df['Label'].ne('-').astype(int)
        
    else:
        df = dataframe
    # print('There are %d instances in this dataset\n' % len(df))
        

    df = df[df['EventId'] != '']
    df['EventId'] = df['EventId'].astype(str)
    valid_rows = df[df['Time'].str.strip() != '']
    # Apply the fix function and only keep the corrected timestamps
    # df = fix_timestamp(valid_rows)
    df.loc[:, 'FixedTime'] = valid_rows['Time'].apply(fix_timestamp)

    np_datetime = df['FixedTime'].to_numpy(dtype='datetime64')
    df.loc[:, 'timestamp'] = np.array(np_datetime).astype(np.int64)
    df = df.sort_values('timestamp')

    df.set_index('timestamp', drop=False, inplace=True)

    df['file_name'] = source_file
 
    return df

def ve_segmentation(grouped_by_node, segmentation):
    
    new_data = []
    for group in grouped_by_node:
        start_idx = 0
        for segment in segmentation[group[0]]:
            
            df_window = group[1].iloc[start_idx:start_idx+len(segment)]
            if len(df_window) > 1:
                if len(df_window) > 512:

                    start = 0
                    end = len(segment)
                    

                    while (end - start) > 512:

                        df_window_inner = df_window.iloc[start:start+512]

                        new_data.append([
                            df_window_inner['LineNumber'].values.tolist(),
                            df_window_inner['EventId'].values.tolist(),
                            group[0],
                            df_window_inner['timestamp'].values.tolist(),
                        ])
                        start += 512 // 2
                    if start < end:
                        df_window = df_window.iloc[start:end]
                        new_data.append([
                            df_window['LineNumber'].values.tolist(),
                            df_window['EventId'].values.tolist(),
                            group[0],
                            df_window['timestamp'].values.tolist(),
                        ])

                    start_idx += len(segment)
                else:
                    new_data.append([
                        df_window['LineNumber'].values.tolist(),
                        df_window['EventId'].values.tolist(),
                        group[0],
                        df_window['timestamp'].values.tolist(),
                    ])  
                    start_idx += len(segment) 
            else:
                start_idx += 1

    #print('there are %d instances (voting experts segments) in this dataset\n' % len(new_data))
    return pd.DataFrame(new_data, columns=['LineNumber','EventSequence', 'ThreadId', 'timestamp'])

def prepare_train_test(dest_dir):
    log_structured_files = Path(dest_dir).glob('*.log_structured')

    window = 7
    threshold = 4
    ve = VotingExperts(window, threshold)
    all_grouped_by = {}
    all_event_dict = {}
    for log_file in tqdm(log_structured_files, desc="preprocessing log files"):
        file_name = log_file.name
        df = preprocessing(source_dir=dest_dir, source_file=file_name)
        grouped_by = df.groupby('ThreadId')
        all_grouped_by[file_name]=grouped_by
        event_sequence = grouped_by['EventId'].apply(list).reset_index()
        all_event_dict.update(event_sequence.set_index('ThreadId').to_dict()['EventId'])
        
    ve.fit(all_event_dict)

    dataframes = []
    test_dataframes = []
    segmentation = ve.transform(all_event_dict, window ,threshold)
    for file_name, grouped_by in all_grouped_by.items():

        new_df = ve_segmentation(grouped_by, segmentation=segmentation)
        if 'trunk' in file_name:
            dataframes.append(new_df)
            
        else:
            test_dataframes.append(new_df)

    new_df = pd.concat(dataframes, ignore_index=True)
    inference_df = pd.concat(test_dataframes, ignore_index=True)
    train_file = dest_dir/"train.csv"
    test_file = dest_dir/"inference.csv"
    new_df.to_csv(train_file)
    inference_df.to_csv(test_file)
    
    return train_file, test_file