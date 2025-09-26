import logging
logging.basicConfig(level=logging.INFO)

import pandas as pd
import numpy as np
import os

from .prepare_data import preprocess_data


def save_dataframe(class_0_upsampled: pd.DataFrame, 
    class_1_upsampled: pd.DataFrame, 
    class_2_upsampled: pd.DataFrame, 
    class_3_upsampled: pd.DataFrame, 
    class_4_upsampled: pd.DataFrame, binary: bool) -> None:
    """
    Combining the classes to a full dataset 
    and saving the dataframe to the appropriate path. 
    
    Args:
        class_0_upsampled (pd.DataFrame): Upsampled DataFrames for class 0.
        class_1_upsampled (pd.DataFrame): Upsampled DataFrames for class 1.
        class_2_upsampled (pd.DataFrame): Upsampled DataFrames for class 2.
        class_3_upsampled (pd.DataFrame): Upsampled DataFrames for class 3.
        class_4_upsampled (pd.DataFrame): Upsampled DataFrames for class 4.

    Returns:
        None
    """
    
    WHOLE_DATA_PATH = "examples/HD_classification/data/CL/whole_data.csv"

    RANDOM_STATE_1 = 42
    FRAC_IID = 1

    # Combine classes and shuffle
    if binary:
        balanced_df = pd.concat([class_0_upsampled, class_1_upsampled]
        ).sample(frac=FRAC_IID, random_state=RANDOM_STATE_1)

    else:
        balanced_df = pd.concat([
            class_0_upsampled, 
            class_1_upsampled, 
            class_2_upsampled, 
            class_3_upsampled, 
            class_4_upsampled]
        ).sample(frac=FRAC_IID, random_state=RANDOM_STATE_1)

    # save dataframe to csv-file
    balanced_df.to_csv(WHOLE_DATA_PATH, index=False, header=False)


    if binary:
        logging.info("Oversampled 50/50 balanced data has been prepared")
    else:
        logging.info("Oversampled 20/20/20/20/20 balanced data has been prepared")
    logging.info(f" - whole data file: {WHOLE_DATA_PATH}")


DATA_PATH = "examples/HD_classification/data/heart+disease/"
CLEVELAND_DATA = os.path.join(DATA_PATH, "processed.cleveland.data")

BINARY = False 

class_0_upsampled, class_1_upsampled, class_2_upsampled, class_3_upsampled, class_4_upsampled = preprocess_data(
    CLEVELAND_DATA, binary = BINARY)

save_dataframe(class_0_upsampled, class_1_upsampled, class_2_upsampled, 
class_3_upsampled, class_4_upsampled, BINARY)



