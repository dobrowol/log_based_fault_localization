import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from loguru import logger

def fault_localization(train_file, test_file):
    train = pd.read_csv(train_file, index_col=0)
    # train = pd.read_csv('../loggpt/train_PR777496.csv', index_col=0)
    train["LineNumber"] = train["LineNumber"].apply(lambda x: eval(x))
    train["EventSequence"] = train["EventSequence"].apply(lambda x: eval(x))
    train["timestamp"] = train["timestamp"].apply(lambda x: eval(x))

    bigdoc = []
    class_cnt = 0
    for idx, curr_label in enumerate(train.iterrows()):
        bigdoc.extend(list(set(train.iloc[idx]["EventSequence"])))
        class_cnt += 1
    vocab = set(bigdoc)
    cnt = Counter(bigdoc)

    inference = pd.read_csv(test_file, index_col=0)
    inference["LineNumber"] = inference["LineNumber"].apply(lambda x: eval(x))
    inference["EventSequence"] = inference["EventSequence"].apply(lambda x: eval(x))
    inference["timestamp"] = inference["timestamp"].apply(lambda x: eval(x))
    inference['timestamp'] = inference['timestamp'].apply(lambda x: [np.int64(i) for i in x])

    fail_bigdoc = []
    class_cnt = 0
    for idx, curr_label in enumerate(inference.iterrows()):
        fail_bigdoc.extend(list(set(inference.iloc[idx]["EventSequence"])))
        class_cnt += 1
    fail_vocab = set(fail_bigdoc)
    fail_cnt = Counter(fail_bigdoc)

    suspicious_score = {}
    for word in fail_vocab:
        #suspicious_score[word] = np.log10((fail_cnt[word]+1)/(len(fail_bigdoc))) - np.log10((cnt[word]+1)/(len(bigdoc)))
        suspicious_score[word] = fail_cnt[word]/(fail_cnt[word] + cnt[word])

    suspicious_words = []
    for word in suspicious_score:
        if suspicious_score[word] == 1:
            suspicious_words.append(word)
    logger.info(f"percentage of rank 1 susp lines {len(suspicious_words)/len(suspicious_score)*100}")
    logger.info("number of rank 1 suspicious lines", len(suspicious_words))

    sbfl_anomalies_segments = []
    sbfl_anomalies_lines = []
    sbfl_anomalous_timestamp = []
    sbfl_anomalous_confidence = []
    confidences=[]
    all_components=set()

    for row in range(len(inference)):
        # print(inference.iloc[0])
        # print(row)
        one_word_not_in_vocab = False
        all_components.add(inference.iloc[row]['ThreadId'])
        scores = []
        for idx, word in enumerate(inference.iloc[row]['EventSequence']):
            if word not in vocab:
                one_word_not_in_vocab = True
                sbfl_anomalies_lines.append(inference.iloc[row]['LineNumber'][idx])
            if word in suspicious_score:
                # print(f"drain {word} at index {idx} in row {row} not in vocab")
                scores.append(suspicious_score[word])
            else:
                scores.append(0)

                # no_word_in_vocab = False

        confidences.append(np.mean(scores))

        if one_word_not_in_vocab:
            # conf_sc = (max(scores) - min(scores)) 
            # sbfl_anomalous_confidence.append(conf_sc)
            conf_sc = np.mean((scores))
            sbfl_anomalous_confidence.append(1-conf_sc)
            sbfl_anomalies_segments.append(row)
            sbfl_anomalous_timestamp.append(np.int64(inference.iloc[row]['timestamp'][0]))
    sbfl_anomalies = inference.iloc[sbfl_anomalies_segments]
    sbfl_anomalies['timestamp'] = sbfl_anomalous_timestamp
    sbfl_anomalies['confidence'] = sbfl_anomalous_confidence
    # # Sorting the DataFrame correctly by the first element in the 'LineNumber' lists
    sbfl_anomalies_sorted_by_timestamp = sbfl_anomalies.sort_values(by='timestamp')
    sbfl_anomalies_sorted_by_confidence = sbfl_anomalies.sort_values(by='confidence', ascending=False)
    logger.info(f"number of sbfl anomalous segments {len(sbfl_anomalies_segments)}")
    sbfl_anomalies_sorted_by_confidence.to_csv('sbfl_anomalies_sorted_by_confidence.csv')
    sbfl_anomalies_sorted_by_timestamp.to_csv('sbfl_anomalies_sorted_by_timestamp.csv')

    logger.info("results written to sbfl_anomalies_sorted_by_confidence.csv and sbfl_anomalies_sorted_by_timestamp.csv")