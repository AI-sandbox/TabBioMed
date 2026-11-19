# code from https://github.com/CIMCB/cimcb
from os import path
import numpy as np
import pandas as pd


def table_check(DataTable, PeakTable, print_statement=True):
    data_columns = DataTable.columns.values
    if "Idx" not in data_columns:
        raise ValueError("Data Table does not contain the required 'Idx' column")
    if DataTable.Idx.isnull().values.any() == True:
        raise ValueError("Data Table Idx column cannot contain missing values")
    if len(np.unique(DataTable.Idx)) != len(DataTable.Idx):
        raise ValueError("Data Table Idx numbers are not unique. Please change")
    if "SampleID" not in data_columns:
        raise ValueError("Data Table does not contain the required 'SampleID' column")
    peak_columns = PeakTable.columns.values
    if "Idx" not in peak_columns:
        raise ValueError("Peak Table does not contain the required 'Idx' column")
    if PeakTable.Idx.isnull().values.any() == True:
        raise ValueError("Peak Table Idx column cannot contain missing values")
    if len(np.unique(PeakTable.Idx)) != len(PeakTable.Idx):
        raise ValueError("Peak Table Idx numbers are not unique. Please change")
    if "Name" not in peak_columns:
        raise ValueError("Peak Table does not contain the required 'Name' column")
    if PeakTable.Idx.isnull().values.any() == True:
        raise ValueError("Peak Table Name column cannot contain missing values")
    if len(np.unique(PeakTable.Idx)) != len(PeakTable.Idx):
        raise ValueError("Peak Table Name numbers are not unique. Please change")
    if "Label" not in peak_columns:
        raise ValueError("Data Table does not contain the required 'Label' column")
    peak_list = PeakTable.Name
    data_columns = DataTable.columns.values
    temp = np.intersect1d(data_columns, peak_list)
    if len(temp) != len(peak_list):
        raise ValueError(
            "The Peak Names in Data Table should exactly match the Peak Names in Peak Table. Remember that all Peak Names should be unique."
        )
    if print_statement is True:
        print("Data Table & Peak Table is suitable.")


def load_dataXL(filename: str, DataSheet: str, PeakSheet: str):
    if path.isfile(filename) is False:
        raise ValueError("{} does not exist.".format(filename))
    if not filename.endswith(".xlsx"):
        raise ValueError("{} should be a .xlsx file.".format(filename))
    print("Loadings PeakFile: {}".format(PeakSheet))
    PeakTable = pd.read_excel(filename, sheet_name=PeakSheet)
    print("Loadings DataFile: {}".format(DataSheet))
    DataTable = pd.read_excel(filename, sheet_name=DataSheet)
    DataTable = DataTable.replace(-99, np.nan)
    DataTable = DataTable.replace(".", np.nan)
    DataTable = DataTable.replace(" ", np.nan)
    table_check(DataTable, PeakTable, print_statement=True)
    DataTable.index = np.arange(1, len(DataTable) + 1)
    PeakTable.index = np.arange(1, len(PeakTable) + 1)
    print("TOTAL SAMPLES: {} TOTAL PEAKS: {}".format(len(DataTable), len(PeakTable)))
    print("Done!")
    return (DataTable, PeakTable)
