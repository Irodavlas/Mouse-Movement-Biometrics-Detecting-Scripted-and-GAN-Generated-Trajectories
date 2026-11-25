import numpy as np
import pandas as pd
from pathlib import Path 

root_path = Path("data/raw/balabit/training_files")

# for now we focus on training files but also test files are good for short trajectories, we would need to add them later on 
def parse_training_directory():
    session_files = sorted([p for p in root_path.rglob("session*")])
    for file_path in session_files:
        user_id = file_path.parent.name
        session_id = file_path.stem.split("_")[1]
        print(user_id, session_id)
        

        df = pd.read_csv(file_path)
        print(df.head(15))
        break 


def parse_csv():
    '''
        We need to construct the new dataset by dropping:
         - non movement or noisy rows , we keep the state MOVE and segment strokes between pressed and released.
         - drop identical rows 
         - if time delta is 0 instead we clamp to 0.0001
         - we can keep the dx and dy rows if we use dt to discrinate, as pauses are human'like 
         
        At each Move we create another sequence, if its too short we drop it (few points or tiny start and end distances) - post processing maybe 

    '''
    pass

parse_training_directory()