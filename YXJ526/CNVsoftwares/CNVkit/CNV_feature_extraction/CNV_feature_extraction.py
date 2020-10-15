import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# This program was created to extract the read depths from the CNV profile from patient targeted next generation sequencing data and export the 
# data into a csv file which can be used to compare the differences between tunour and non-tumour in the machine learning pipeline.

# Calculate the proportional bin size in relation to the whole chromosome 
def calculate_bin_proportion(df):
    bin_size = dict() 
    for index, row in df.iterrows():
        chr = row['chromosome'] 
        bin_size[chr] = bin_size.get(chr, 0) + (row['end'] - row['start'])

    for index, row in df.iterrows():
        df.loc[index,'bin_proportion'] = (row['end'] - row['start']) / bin_size.get(row['chromosome']) 
    return df 

def create_df(dictionary, filename):
    temp_dict = pd.DataFrame(dictionary, index = [filename.split('.')[0]])
    temp_dict['Tumour'] = 1 if filename.startswith('s') or filename.startswith('N154') or filename.startswith('N177') or filename.startswith('N379') or filename.startswith('N586') or filename.startswith('N621') or filename.startswith('N679') or filename.startswith('N750') or filename.startswith('N794') or filename.startswith('N874') else 0 
    return temp_dict

# The name of the sample run 
sample_series = 'Test'

directoryFilePath = os.getcwd()

os.system("dot_clean " + directoryFilePath)
filenames = os.listdir(directoryFilePath)

callFlag = False

log_sum_abs_csv =  pd.DataFrame()
log_sum_rel_csv =  pd.DataFrame()
log_avg_abs_csv =  pd.DataFrame()
log_avg_rel_csv =  pd.DataFrame()

raw_sum_abs_csv =  pd.DataFrame()
raw_sum_rel_csv =  pd.DataFrame()
raw_avg_abs_csv =  pd.DataFrame()
raw_avg_rel_csv =  pd.DataFrame()  

for filename in filenames:

    if filename.endswith(".cns") and not filename.endswith("call.cns"):
        callFlag = True

        df = pd.read_csv(os.path.join(directoryFilePath,filename), delimiter = '\t')
        df = calculate_bin_proportion(df) # Calculate the percentage of each bin 

        log_sum_abs_dict = dict() 
        log_sum_rel_dict = dict()
        log_avg_abs_dict = dict()
        log_avg_rel_dict = dict()

        raw_sum_abs_dict = dict() 
        raw_sum_rel_dict = dict()
        raw_avg_abs_dict = dict()
        raw_avg_rel_dict = dict()

        for index, row in df.iterrows():
            chr = row['chromosome'] 

            log_sum_abs_dict[chr] = log_sum_abs_dict.get(chr, 0) + abs(row['log2']) 
            log_sum_rel_dict[chr] = log_sum_rel_dict.get(chr, 0) + (row['log2']) 
            log_avg_abs_dict[chr] = log_avg_abs_dict.get(chr, 0) + abs(row['log2']) * row['bin_proportion']
            log_avg_rel_dict[chr] = log_avg_rel_dict.get(chr, 0) + (row['log2']) * row['bin_proportion']

            raw_sum_abs_dict[chr] = raw_sum_abs_dict.get(chr, 0) + (2 ** row['log2']) 
            raw_sum_rel_dict[chr] = raw_sum_rel_dict.get(chr, 0) + (2 ** abs(row['log2'])) 
            raw_avg_abs_dict[chr] = raw_avg_abs_dict.get(chr, 0) + (2 ** abs(row['log2'])) * row['bin_proportion']
            raw_avg_rel_dict[chr] = raw_avg_rel_dict.get(chr, 0) + (2 ** row['log2']) * row['bin_proportion']

        log_sum_abs_res = create_df(log_sum_abs_dict, filename)
        log_sum_rel_res = create_df(log_sum_rel_dict, filename)
        log_avg_abs_res = create_df(log_avg_abs_dict, filename)
        log_avg_rel_res = create_df(log_avg_rel_dict, filename)

        raw_sum_abs_res = create_df(raw_sum_abs_dict, filename)
        raw_sum_rel_res = create_df(raw_sum_rel_dict, filename)
        raw_avg_abs_res = create_df(raw_avg_abs_dict, filename)
        raw_avg_rel_res = create_df(raw_avg_rel_dict, filename)
     
        # Add each sample onto the final dataframe

        log_sum_abs_csv =  pd.concat([log_sum_abs_csv, log_sum_abs_res])
        log_sum_rel_csv =  pd.concat([log_sum_rel_csv, log_sum_rel_res])
        log_avg_abs_csv =  pd.concat([log_avg_abs_csv, log_avg_abs_res])
        log_avg_rel_csv =  pd.concat([log_avg_rel_csv, log_avg_rel_res])

        raw_sum_abs_csv =  pd.concat([raw_sum_abs_csv, raw_sum_abs_res])
        raw_sum_rel_csv =  pd.concat([raw_sum_rel_csv, raw_sum_rel_res])
        raw_avg_abs_csv =  pd.concat([raw_avg_abs_csv, raw_avg_abs_res])
        raw_avg_rel_csv =  pd.concat([raw_sum_rel_csv, raw_sum_rel_res])

if callFlag is True :

    print("Printing log and raw csv files...")  

    log_sum_abs_csv.to_csv(directoryFilePath + sample_series + '_log_sum_abs.csv', sep = ',')
    log_sum_rel_csv.to_csv(directoryFilePath + sample_series + '_log_sum_rel.csv', sep = ',')
    log_avg_abs_csv.to_csv(directoryFilePath + sample_series + '_log_avg_abs.csv', sep = ',')
    log_avg_rel_csv.to_csv(directoryFilePath + sample_series + '_log_avg_rel.csv', sep = ',')

    raw_sum_abs_csv.to_csv(directoryFilePath + sample_series + '_raw_sum_abs.csv', sep = ',')
    raw_sum_rel_csv.to_csv(directoryFilePath + sample_series + '_raw_sum_rel.csv', sep = ',')
    raw_avg_abs_csv.to_csv(directoryFilePath + sample_series + '_raw_avg_abs.csv', sep = ',')
    raw_avg_rel_csv.to_csv(directoryFilePath + sample_series + '_raw_avg_rel.csv', sep = ',')

else :
    print("No log files are found in this directory")

# Reset call flag to be used on call files 
callFlag = False

cll_sum_abs_csv =  pd.DataFrame()
cll_sum_rel_csv =  pd.DataFrame()
cll_avg_abs_csv =  pd.DataFrame()
cll_avg_rel_csv =  pd.DataFrame()   

for filename in filenames:
    if filename.endswith("call.cns"):
        # call file exists 
        callFlag = True 

        df = pd.read_csv(os.path.join(directoryFilePath,filename), delimiter = '\t')
        df = calculate_bin_proportion(df) # Calculate the percentage of each bin 
        df = pd.read_csv(os.path.join(directoryFilePath,filename), delimiter = '\t')
        df = calculate_bin_proportion(df) # Calculate the percentage of each bin 

        # Creates a dictionary to combine all the log values for depth/ call values for each choromsome
        
        cll_sum_abs_dict = dict() 
        cll_sum_rel_dict = dict()
        cll_avg_abs_dict = dict()
        cll_avg_rel_dict = dict()
        
        for index, row in df.iterrows():
            chr = row['chromosome'] 

            # For each row, take the absolute cn value minus the expected diploid from normal and multiply the value by the bin proportion
            # and then add it to the dictionary

            # NB change 'log2' back to cn after testing

            cll_sum_abs_dict[chr] = cll_sum_abs_dict.get(chr, 0) + (abs(row['cn'] - 2))  # summation of the absolute copy numbers values of each chromosome
            cll_sum_rel_dict[chr] = cll_sum_rel_dict.get(chr, 0) + (row['cn'] - 2)  # summation of the copy number values of each chromosome
            cll_avg_abs_dict[chr] = cll_avg_abs_dict.get(chr, 0) + (abs(row['cn']) - 2) * row['bin_proportion']  # summation of 
            cll_avg_rel_dict[chr] = cll_avg_rel_dict.get(chr, 0) + (row['cn'] - 2) * row['bin_proportion']  #

        cll_sum_abs_res = create_df(cll_sum_abs_dict, filename)
        cll_sum_rel_res = create_df(cll_sum_rel_dict, filename)
        cll_avg_abs_res = create_df(cll_avg_abs_dict, filename)
        cll_avg_rel_res = create_df(cll_avg_rel_dict, filename) 

        cll_sum_abs_csv =  pd.concat([cll_sum_abs_csv, cll_sum_abs_res])
        cll_sum_rel_csv =  pd.concat([cll_sum_rel_csv, cll_sum_rel_res])
        cll_avg_abs_csv =  pd.concat([cll_avg_abs_csv, cll_avg_abs_res])
        cll_avg_rel_csv =  pd.concat([cll_avg_rel_csv, cll_avg_rel_res])

if callFlag is True:
    print("Printing call csv files...")    
    cll_sum_abs_csv.to_csv( directoryFilePath + sample_series + '_cll_sum_abs.csv', sep = ',')
    cll_sum_rel_csv.to_csv( directoryFilePath + sample_series + '_cll_sum_rel.csv', sep = ',')
    cll_avg_abs_csv.to_csv( directoryFilePath + sample_series + '_cll_avg_abs.csv', sep = ',')
    cll_avg_rel_csv.to_csv( directoryFilePath + sample_series + '_cll_avg_rel.csv', sep = ',')

    print('Finished')

else : 
    print("Call files not found")
