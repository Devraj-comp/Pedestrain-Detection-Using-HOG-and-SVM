import os
import pathlib
import glob

list_txt = os.listdir('train_data')

for file_path in os.listdir('train_data'):

    # print(file_path)
    pass

txt_file_paths = glob.glob(r"train_data/*.txt")

for i, file_path in enumerate(txt_file_paths):

    print(file_path)