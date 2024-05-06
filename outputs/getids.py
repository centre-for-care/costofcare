import os
import pandas as pd

def main(folder_path):
    files = os.listdir(folder_path)
    files = [file for file in files if file.endswith('.pkl')]
    list_ids = []
    for file in files:
        df = pd.read_pickle(file_path)
        for ele in df:
            ids = ele['data'].columns.tolist()
            list_ids.extend(ids)
    unique_ids = list(set(list_ids))
    pd.DataFrame({'ids': unique_ids}).to_csv('./unique_ids.csv')


if __name__ == "__main__":
    folder_path = os.getcwd()
    main(folder_path)
