## This program will flag practices with suspicious behavior.
## It is implemented as a command line script argsparse.
## Steps:
## 1. Load and clean the data.
## 2. Flag opioids.
## 3. Perform data analysis.
## 4. Find suspicious practices.

# Import needed libraries
import pandas as pd
import numpy as np

# Global variables
DIR_DRUGFILES = r'../data/'
OPIOIDS = ['morphine', 'oxycodone', 'methadone', 'fentanyl', 'pethidine', 'buprenorphine', 'propoxyphene', 'codeine']
Z_SCORE_CUTTOFF = 1.96 #z=1.64 (90%); z=1.96 (95%); z=2.58 (99%)
RAW_COUNT_CUTTOFF = 0
OUTPUTFILE = 'flagged_practices.csv'

# Global functions
def load_and_clean_data(dir_drugfiles = DIR_DRUGFILES):
    """Return the cleaned scripts, chemical and practices data sets."""
    
    scripts = pd.read_csv(dir_drugfiles + '201701scripts_sample.csv.gz', compression='gzip')
    
    # Need to drop duplicate CHEM SUB rows.
    chem = pd.read_csv(dir_drugfiles + 'chem.csv.gz', compression='gzip')
    chem = chem.sort_values('CHEM SUB').drop_duplicates(subset='CHEM SUB', keep='first') 

    col_names = [ 'code', 'name', 'addr_1', 'addr_2', 'borough', 'village', 'post_code']
    practices = pd.read_csv(dir_drugfiles + 'practices.csv.gz', compression='gzip', header=None, names=col_names)
    
    return scripts, chem, practices
    
    
def flag_opioids(scripts, chem, opioids = OPIOIDS):
    """Return a merged scripts with chem dataframe, flagging scripts with opioids."""
    # Merge scripts with chem data to get the name of the chemical involve
    scripts_and_chem = scripts.merge(chem, how='left', left_on='bnf_code', right_on='CHEM SUB')

    # Rename columns to clarify and delete the duplicate data in the column CHEM SUB (is the same bnf_code)
    scripts_and_chem = scripts_and_chem.rename(index=str, columns={"NAME": "chemical_name"}).drop(['CHEM SUB'], axis=1)
    scripts_and_chem["chemical_name"].fillna('', inplace=True)
    
    # Flag which chemical belongs to opiod list
    scripts_and_chem['is_opioid'] = scripts_and_chem['chemical_name'].str.lower().str.contains(r'|'.join(opioids))
    
    return scripts_and_chem 
    
    
def calculate_z_score(scripts_and_chem):
    """Return the zscore for each scripts as a pandas series."""
    
    # grouping by practice to get the proportion of opioids prescriptions
    opioids_per_practice = scripts_and_chem.groupby('practice').agg(
                                opiods_proportion = pd.NamedAgg(column='is_opioid', aggfunc='mean'),
                                total_samples     = pd.NamedAgg(column='bnf_name', aggfunc='count')
                           )
    
    # Getting the overall opioid prescription rate
    overall_opioid_prescription = scripts_and_chem['is_opioid'].mean()
    
    # Subtract off the proportion of all prescriptions that are opioids from each practice's proportion.
    relative_opioids_per_practice = opioids_per_practice['opiods_proportion'] - overall_opioid_prescription
    
    # Get the standard error
    standard_error_per_practice = scripts_and_chem['is_opioid'].std() / np.sqrt(opioids_per_practice['total_samples'])

    # Calculate the z-score
    z_score = relative_opioids_per_practice / standard_error_per_practice
    
    return z_score
    
    
def flag_suspicious_practices(practices, z_score, z_score_cuttoff=Z_SCORE_CUTTOFF, raw_count_cuttoff=RAW_COUNT_CUTTOFF):
    """Return practices flagging those one that are suspicious (with a z-score greater than z_score_cuttoff and raw count greater than raw_count_cuttoff)."""
    
    # Merge with practice data to prepare the result 
    anomalous_practices = practices.sort_values('name').drop_duplicates(subset='code', keep='first')
    anomalous_practices.set_index('code', inplace=True)
    anomalous_practices['z_score'] = z_score
    anomalous_practices['count'] = scripts_and_chem['practice'].value_counts()
    #anomalous_practices['suspicious'] = unique_practices['z_score'] > z_score_cuttoff
    
    # Return the n rows flagged as suspicious ordered by columns in descending order.
    return anomalous_practices.query('z_score > @z_score_cuttoff and count > @raw_count_cuttoff')[['name', 'post_code', 'z_score', 'count']].reset_index()


def dump_result(anomalous_practices, output_file=OUTPUTFILE):
    """Dump pandas data frame of the results to disk"""
    anomalous_practices.sort_values(by='z_score', ascending=False).to_csv(output_file, index=False)
    
    
# Main part
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='This program will flag practices with suspicious behavior.')
    parser.add_argument('-z', metavar='float', type=float, default = Z_SCORE_CUTTOFF, 
                        help=f'Z-score boorder to flag extreme practices as suspicious. {Z_SCORE_CUTTOFF} is default if none.')
    parser.add_argument('-c', metavar='integer', type=int, default = RAW_COUNT_CUTTOFF, 
                        help=f'Raw count to flag as suspicious, in adition to z-score filter. {RAW_COUNT_CUTTOFF} is default if none.')
    parser.add_argument('-f', metavar='<filename>', type=str, default = OUTPUTFILE, 
                        help=f'Filename of the result output. {OUTPUTFILE} is default if none.')
    parser.add_argument("--verbosity", action="store_true",
                        help="Enable output verbosity.")
    
    args = parser.parse_args()
    z_score_cuttoff = args.z
    raw_count_cuttoff = args.c
    output_file = args.f
    verbosity = args.verbosity
    
    if verbosity: print(f"Running... z (z-score cuttoff): {z_score_cuttoff:0.2f}, c (raw count cuttoff): {raw_count_cuttoff}...")
    scripts, chem, practices = load_and_clean_data()
    scripts_and_chem = flag_opioids(scripts, chem)
    z_score = calculate_z_score(scripts_and_chem)
    anomalous_practices = flag_suspicious_practices(practices, z_score, z_score_cuttoff, raw_count_cuttoff)
    
    if verbosity: print(f"Saving the result to '{output_file}' file...")
    dump_result(anomalous_practices, output_file)
    
    if verbosity: print('End with success!')
    