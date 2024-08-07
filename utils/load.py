import os
import pandas as pd

def load_table_file(
    folder_path: str,
    file_name: str
) -> pd.DataFrame:
    
    if file_name[-5:] == ".xlsx":
        return pd.read_excel(
            os.path.join(
                folder_path, file_name
            )
        )
