# -*- coding: utf-8 -*-
"""

@author: Efrain Boza N. 

Script for RA test at World Bank.
Using AI for cleaning names. 

    
"""


# In[93]:
#Relevant packages:
    
import os   
import pandas as pd 
import re
import numpy as np
##################################################
#This algorithm was to slow so I am not using it anymore
#!pip install fuzzywuzzy[speedup] pandas    
#from fuzzywuzzy import fuzz, process
#!pip install catboost[cuda]

#############################################
#!pip install rapidfuzz
#!pip install lightgbm xgboost
#!pip install scikit-learn pandas rapidfuzz
#!pip install catboost
from rapidfuzz import fuzz, process
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.utils import resample
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from sklearn.metrics import pairwise_distances
# In[93]:
    
# Set the working directory to the directory where the .py file is located
# Define a default path to Python_Efrain_Boza
DEFAULT_PROJECT_DIRECTORY = r"C:\Users\Efrain\Desktop\Python_Efrain_Boza"

# Check if Python_Efrain_Boza is present; use it if accessible, if it is not then need to change the default working directory.
if os.path.exists(DEFAULT_PROJECT_DIRECTORY):
    working_directory = DEFAULT_PROJECT_DIRECTORY
else:
    raise EnvironmentError("Could not locate the Python_Efrain_Boza directory.")

# Change to the working directory
os.chdir(working_directory)

# Define input and output folders within this directory
input_folder = os.path.join(working_directory, "data")
output_folder = os.path.join(working_directory, "outputs")

# Create folders if they don't exist
os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
    

# In[93]:
#Loading inputs. 
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Load each CSV file into a dictionary of DataFrames
for file in csv_files:
    file_name = os.path.splitext(file)[0]  # Get the file name without the extension
    file_path = os.path.join(input_folder, file)
    globals()[file_name] = pd.read_csv(file_path)  # Assign the DataFrame to the file name

    # Print confirmation
    print(f"Loaded '{file}' into DataFrame '{file_name}'")
    

# In[93]:
#EXERCISE 1

df = ForeignNames_2019_2020.merge(
    Country_Name_ISO3,
    how='outer',
    left_on='foreigncountry_cleaned',
    right_on='country_name',
    indicator=True  # Adds a '_merge' column for tracking match results
)


# Tabulate the merge indicator
merge_summary = df['_merge'].value_counts()

print(merge_summary)
right_only_df = df[df['_merge'] == 'right_only']

# Print the `right_only` rows
print("Rows in `Country_Name_ISO3` but not in `ForeignNames_2019_2020`:")
print(right_only_df)
#I would have liked to check the unmatched problems but I did not count with the enough time to be more precise 
#It is important to check them out carefully to avoid loosing observations. 
left_only_df = df[df['_merge'] == 'left_only']
print(left_only_df)
# Export to Excel
output_path = os.path.join(output_folder, "left_only_df.xlsx")
left_only_df.to_excel(output_path, index=False)
print(f"left_only_df has been saved to {output_folder} to check for the matches manually")    
#There are 5 problematic countries, I am going to fix this manualy. They are: South Korea, Antigua, Congo, Iran, Tanzania
country_updates = {
    "Antigua And Barbuda": "ATG",
    "Congo, the Democratic Republic of the": "BHS",
    "South Korea": "KOR",
    "Iran": "IRN",
    "Tanzania": "TZA",
    # Add more mappings as needed
}

# Apply the updates based on the dictionary
for country, iso3_code in country_updates.items():
    df.loc[df['foreigncountry_cleaned'] == country, 'country_iso3'] = iso3_code

# Verify some updates
print(df[df['foreigncountry_cleaned'].isin(country_updates.keys())])
# Drop rows where the 'foreign' column is an empty string
# Ensure all values in 'foreign' are strings initially
df['foreign'] = df['foreign'].fillna("").astype(str)

initial_count = len(df)
df = df[df['foreign'] != ""]
dropped_count = initial_count - len(df)
print(f"Number of observations dropped: {dropped_count}")


# Drop the '_merge' column 
df = df.drop(columns=['_merge'], errors='ignore')
# Create lists to store the final cleaned names and IDs
all_cleaned_names = []
all_cleaned_ids = []



# fuzzy matching within each group

#  remove common suffixes for standardization
def clean_company_name(name):
    name = re.sub(r'\b(ltd|limited|inc|corporation|corp|llc)\b', '', name, flags=re.IGNORECASE)
    return name.strip()

#To test data
#df = df[df['country_iso3'].isin(['AUS' ])]

# Get unique countries from 'country_iso3' column
unique_countries = df['country_iso3'].unique()

# Process each country independently
for country in unique_countries:
    print(f"Processing country: {country}")

    # Filter the DataFrame to only include records for the current country
    country_df = df[df['country_iso3'] == country].copy()

    # Clean company names
    country_df['cleaned_foreign'] = country_df['foreign'].apply(clean_company_name)

    # Get unique cleaned names
    unique_names = country_df['cleaned_foreign'].unique()

    # Create an empty list to store clustering labels
    labels = [-1] * len(unique_names)

    # Perform clustering using a more efficient approach
    threshold = 90  # Threshold for similarity (max is 100)

    for i, name in enumerate(unique_names):
        if labels[i] != -1:
            continue

        # Find similar names to the current one
        matches = process.extract(name, unique_names, scorer=fuzz.token_sort_ratio, limit=len(unique_names))

        # Cluster names with a similarity above the threshold
        cluster_label = max(labels) + 1
        for match_name, score in matches:
            if score >= threshold:
                idx = np.where(unique_names == match_name)[0][0]
                if labels[idx] == -1:
                    labels[idx] = cluster_label

    # Assign cleaned names and unique IDs
    cluster_to_name = {}
    cleaned_names = []
    cleaned_ids = []
    country_id_counter = 1

    for label, name in zip(labels, unique_names):
        if label not in cluster_to_name:
            cluster_to_name[label] = f"{country}_{country_id_counter}"
            country_id_counter += 1

        cleaned_name = cluster_to_name[label]
        cleaned_names.append(cleaned_name)

    # Map back to the original DataFrame
    name_to_cluster = dict(zip(unique_names, cleaned_names))
    country_df['cleaned_ID'] = country_df['cleaned_foreign'].map(name_to_cluster)

    # Append results for this country to the master lists
    all_cleaned_names.extend(country_df['cleaned_foreign'])
    all_cleaned_ids.extend(country_df['cleaned_ID'])

# Add the cleaned names and IDs to the original DataFrame
df['cleaned_name'] = all_cleaned_names
df['cleaned_ID'] = all_cleaned_ids

def clean_company_name(name):
    # Remove common suffixes
    name = re.sub(r'\b(ltd|limited|inc|corporation|corp|llc)\b', '', name, flags=re.IGNORECASE)
    # Remove & and other non-alphanumeric characters (keep spaces and alphanumeric)
    name = re.sub(r'[&.,/\\\'\"!@#$%^*()_+=-]', '', name)
    # Replace multiple spaces with a single space
    name = re.sub(r'\s+', ' ', name)
    return name.strip()

df['cleaned_foreign'] = df['cleaned_foreign'].apply(clean_company_name)
first_names = df.groupby('cleaned_ID')['cleaned_foreign'].transform('first')

# Replace all cleaned_name values with the first value within each cleaned_ID group
df['cleaned_name'] = first_names
# Review a sample of the results
print(df[['foreign', 'country_iso3', 'cleaned_name', 'cleaned_ID']].head(10))
df.drop(columns=['cleaned_foreign'], inplace=True)
output_path = os.path.join(output_folder, "outputfile_Efrain_1.csv")
# Save the cleaned DataFrame to a CSV file
df.to_csv(output_path, index=False)


#AI TRAINING

#Test data

# Identify and merge rare classes into "Other" category
df['cleaned_foreign'] = df['foreign'].apply(clean_company_name)
