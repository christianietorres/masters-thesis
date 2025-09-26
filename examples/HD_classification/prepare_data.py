import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import os

from typing import List

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_for_clients(input_file: str, binary: bool = True, target_size: int = 200, non_iid: bool = False) ->  None:
    """" 
    Process data and split the processed data further for each client.
    
    Args:
        input_file (str): Path to the CSV data file.
        binary (bool): Whether to simplify the task to binary classification.
        target_size (int): Minimum number of samples to upsample each class to.
        non_iid (bool): Whether to use non-IID distribution when splitting data for clients. 

    Returns:
        None
    """

    RANDOM_STATE_1 = 42
    RANDOM_STATE_2 = 43
    FRAC_IID = 1
    FRAC_NONIID_B1 = 0.9
    FRAC_NONIID_B2 = 0.1
    FRAC_NONIID_M = 1

    LABEL_COL = 13
    TEST_SIZE = 0.5

    class_0_upsampled, class_1_upsampled, class_2_upsampled, class_3_upsampled, class_4_upsampled = preprocess_data(input_file, binary, target_size)

    # Split data for clients using either IID or non-IID distribution
    if non_iid:
        logging.info("Creating non-IID split...")

        if binary:
            client1_df = pd.concat([
                class_0_upsampled.sample(frac = FRAC_NONIID_B1, random_state = RANDOM_STATE_1),
                class_1_upsampled.sample(frac = FRAC_NONIID_B2, random_state = RANDOM_STATE_1)
            ])
            client2_df = pd.concat([
                class_0_upsampled.sample(frac = FRAC_NONIID_B2, random_state = RANDOM_STATE_2),
                class_1_upsampled.sample(frac = FRAC_NONIID_B1, random_state= RANDOM_STATE_2)
            ])
        else:
            client1_df = pd.concat([class_0_upsampled, class_1_upsampled, class_2_upsampled])
            client2_df = pd.concat([class_3_upsampled, class_4_upsampled])

            client1_df = client1_df.sample(frac = FRAC_NONIID_M , random_state = RANDOM_STATE_1)
            client2_df = client2_df.sample(frac = FRAC_NONIID_M , random_state = RANDOM_STATE_2)

    else:
        logging.info("Creating IID split...")

        # Combine classes and shuffle
        if binary:
            balanced_df = pd.concat([class_0_upsampled, class_1_upsampled]
            ).sample(frac = FRAC_IID, random_state = RANDOM_STATE_1)

        else:
            balanced_df = pd.concat([class_0_upsampled, class_1_upsampled, class_2_upsampled, 
            class_3_upsampled, class_4_upsampled]).sample(frac = FRAC_IID, random_state = RANDOM_STATE_1)

        client1_df, client2_df = train_test_split(balanced_df, test_size = TEST_SIZE, 
        stratify = balanced_df[LABEL_COL], random_state = RANDOM_STATE_1)

    logging.info("Client 1 label distribution:\n%s", client1_df[LABEL_COL].value_counts(normalize = True))
    logging.info("Client 2 label distribution:\n%s", client2_df[LABEL_COL].value_counts(normalize = True))

    # save client dataframets to csv-files
    CLIENT_DIR = "examples/HD_classification/data/clients/"
    CLIENT1_PATH = os.path.join(CLIENT_DIR, "client1.csv")
    CLIENT2_PATH = os.path.join(CLIENT_DIR, "client2.csv")

    client1_df.to_csv(CLIENT1_PATH, index = False, header = False)
    client2_df.to_csv(CLIENT2_PATH, index = False, header = False)

    if non_iid:
        if binary == True:
            logging.info("Oversampled 90/10 unbalanced data split into two clients.")
        else:
            logging.info("Oversampled split: client1 has classes 0–2, client2 has classes 3–4.")
    else:
        if binary == True:
            logging.info("Oversampled 50/50 balanced data split into two clients.")
        else:
            logging.info("Oversampled 20/20/20/20/20 balanced data split into two clients.")
    logging.info(f" - Client 1 file: {CLIENT1_PATH}")
    logging.info(f" - Client 2 file: {CLIENT2_PATH}")


def preprocess_data(input_file: str, binary: bool = True, target_size: int = 200) -> List[pd.DataFrame]:
    RANDOM_STATE_1 = 42
    LABEL_COL = 13
    MISSING_THRESHOLD = 5

    """
    Preprocesses data by handling missing values, applying normalization, 
    and returning upsampled class DataFrames.

    Args:
        input_file (str): Path to the CSV data file.
        binary (bool): Whether to simplify the task to binary classification.
        target_size (int): Minimum number of samples to upsample each class to.

    Returns:
        [pd.DataFrame]: Upsampled DataFrames for each class.
    """

    try: 
        # Load the health dataset
        df = pd.read_csv(input_file, header = None, na_values = "?")
        logging.info("Loading dataset")
    except FileNotFoundError as e:
        logging.INFO(f"File not found: {input_file}")
        raise e
    except pd.errors.EmptyDataError:
        logging.error(f"File is empty or unreadable: {input_file}")
        raise
        return



    # Evaluate if one should drop missing values
    total_rows = len(df)
    num_missing_rows = df.isnull().any(axis = 1).sum()
    percent_missing_rows = (num_missing_rows / total_rows) * 100
    
    # Drop missing values if missing values are less than 5%
    if percent_missing_rows < MISSING_THRESHOLD:
        df = df.dropna()
        logging.info("Dropping rows with missing values")

    # Normalize labels to binary (0: no disease, 1: disease)
    if binary:

        df[LABEL_COL] = df[LABEL_COL].apply(lambda x: 0 if x == 0 else 1)

    # Split features and labels
    x = df.drop(columns =[LABEL_COL])
    y = df[LABEL_COL]

    # Normalize features
    scaler = StandardScaler()
    x_normalized = scaler.fit_transform(x)

    # Recombine features and labels into a DataFrame
    df = pd.DataFrame(x_normalized, columns=x.columns)
    df[LABEL_COL] = y.values
    
    logging.info("Normalizing features")

    # Split medical data into multiple classes: 
    # absence of heart disease (0) and presence — either as binary (1) or multiclass (1–4)

    if binary:    
        class_0 = df[df[LABEL_COL] == 0]
        class_1 = df[df[LABEL_COL] == 1]

    else:
        class_0 = df[df[LABEL_COL] == 0]
        class_1 = df[df[LABEL_COL] == 1]    	   
        class_2 = df[df[LABEL_COL] == 2] 
        class_3 = df[df[LABEL_COL] == 3] 
        class_4 = df[df[LABEL_COL] == 4] 
        
    # Get the size of the largest class    
    if binary: 
        max_size = max(len(class_0), len(class_1))
    else:
        max_size = max(len(class_0), len(class_1), len(class_2), len(class_3), len(class_4))
    
    if max_size < target_size:
        max_size = target_size

    # Oversample all the classes to match the max size
    if binary:
        class_0_upsampled = resample(class_0, replace = True, n_samples = max_size, random_state = RANDOM_STATE_1)
        class_1_upsampled = resample(class_1, replace = True, n_samples = max_size, random_state = RANDOM_STATE_1)
        class_2_upsampled = None
        class_3_upsampled = None
        class_4_upsampled = None

    else:
        class_0_upsampled = resample(class_0, replace = True, n_samples = max_size, random_state = RANDOM_STATE_1)
        class_1_upsampled = resample(class_1, replace = True, n_samples = max_size, random_state = RANDOM_STATE_1)     
        class_2_upsampled = resample(class_2, replace = True, n_samples = max_size, random_state = RANDOM_STATE_1)
        class_3_upsampled = resample(class_3, replace = True, n_samples = max_size, random_state = RANDOM_STATE_1)
        class_4_upsampled = resample(class_4, replace = True, n_samples = max_size, random_state = RANDOM_STATE_1)


    return class_0_upsampled, class_1_upsampled, class_2_upsampled, class_3_upsampled, class_4_upsampled   

DATA_PATH = "examples/HD_classification/data/heart+disease/"
CLEVELAND_DATA = os.path.join(DATA_PATH, "processed.cleveland.data")

BINARY = True

split_for_clients(CLEVELAND_DATA, BINARY, non_iid = True)
