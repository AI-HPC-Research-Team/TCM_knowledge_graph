import pandas as pd
import json
import os
import glob
import re
from tqdm import tqdm
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
import ast
from merge_utility import merge_database_by_id

if __name__ == "__main__":
    data_root_dir = "../data"
    merge_result_dir = "../merge_result"
    cpmcp_dir = "CPMCP"
    tcmid_dir = "TCMID"

    cpmcp_cpm_prescription = pd.read_csv(
        os.path.join(data_root_dir, cpmcp_dir, "cpm.csv")
    )
    cpmcp_cmp_prescription = pd.read_csv(
        os.path.join(data_root_dir, cpmcp_dir, "cmp.csv")
    )

    # 'ID'('CPMCP_ID'), 'Name'('Chinese_Name'), 'Pinyin Name'('Pinyin_Name'), 'Source', 'Source_en', 'Prescription'(带剂量), 'Prescription_en','Components', 'Components_en', 'Functions'(适应症以及功效), 'Functions_en', 'Direction_for_use'（使用方法）, 'Direction_for_use_en', 'Contraindicatons'（'禁忌症'）, 'Contraindicatons_en','Processing'(制备过程), 'Processing_en', 'Character','Character_en', 'Verification'(验证方法), 'Verification_en', 'Check'(标准), 'Check_en'(标准), 'Identification'(检验方法), 'Identification_en', 'Specification', 'Specification_en', 'Storage', 'Storage_en','Expiration Date'
    cpmcp_cpm_prescription["ID"] = cpmcp_cpm_prescription["ID"].apply(
        lambda x: "CPM{:05d}".format(x)
    )
    function = cpmcp_cpm_prescription["Functions"].apply(lambda x: x.split("。", 1)[0])
    indication = cpmcp_cpm_prescription["Functions"].apply(lambda x: x.split("。", 1)[1])
    function_en = cpmcp_cpm_prescription["Functions_en"].apply(
        lambda x: x.split(".", 1)[0]
    )
    indication_en = cpmcp_cpm_prescription["Functions_en"].apply(
        lambda x: x.split(".", 1)[1]
    )
    cpmcp_cpm_prescription["Functions"] = function
    cpmcp_cpm_prescription["Functions_en"] = function_en
    cpmcp_cpm_prescription["Indications"] = indication
    cpmcp_cpm_prescription["Indications_en"] = indication_en

    cpmcp_cpm_prescription.rename(
        columns={
            "ID": "CPMCP_id",
            "Name": "Chinese_Name",
            "Pinyin Name": "Pinyin_Name",
        },
        inplace=True,
    )

    # rename columns names
    cpmcp_cmp_prescription.rename(
        columns={
            "Indication": "Functions",
            "Indication_en": "Functions_en",
            "Function": "Decocting_Procedure",
            "Function_en": "Decocting_Procedure_en",
            "Treatment": "Indications",
            "Treatment_en": "Indications_en",
        },
        inplace=True,
    )

    cpmcp_cmp_prescription["ID"] = cpmcp_cmp_prescription["ID"].apply(
        lambda x: "CMP{:05d}".format(x)
    )
    cpmcp_cmp_prescription.replace(to_replace=["-"], value=None, inplace=True)
    cpmcp_cmp_prescription.rename(
        columns={
            "ID": "CPMCP_id",
            "Name": "Chinese_Name",
            "Pinyin Name": "Pinyin_Name",
        },
        inplace=True,
    )
    # discard redundant information
    cpmcp_cmp_prescription.drop(["Note", "Note_en"], axis=1, inplace=True)

    cpmcp_prescription = pd.concat(
        [cpmcp_cmp_prescription, cpmcp_cpm_prescription], ignore_index=True
    )
    tcmid_prescription = pd.read_csv(
        os.path.join(data_root_dir, tcmid_dir, "prescription-TCMID-v2.0.1.txt"),
        sep="\t",
    )

    # tcmid pinyin_name 'chinese_name', 'composition', 'pinyin_composition',
    # 'indication', 'use_method', 'references',
    tcmid_prescription.rename(
        columns={
            "pinyin_name": "Pinyin_Name",
            "chinese_name": "Chinese_Name",
            "composition": "Prescription",
            "indication": "Indications",
            "use_method": "Decocting_Procedure",
            "references": "Source",
        },
        inplace=True,
    )
    tcmid_prescription["TCMID_id"] = [
        "TCMID{:05d}".format(tcmid + 1) for tcmid in range(tcmid_prescription.shape[0])
    ]
    tcmid_prescription.drop(["Unnamed: 7", "Unnamed: 8"], axis=1, inplace=True)
    all_prescription = pd.concat(
        [tcmid_prescription, cpmcp_prescription], ignore_index=True
    )
    all_prescription.replace(np.nan, None, inplace=True)
    # Merge two prescriptions with the same prescription description
    all_prescription = merge_database_by_id(
        all_prescription, "Prescription", debug=True
    )
    all_prescription["TMDB_id"] = [
        "TMPRE{:05d}".format(i + 1) for i in range(all_prescription.shape[0])
    ]
    all_prescription.to_csv(
        os.path.join(merge_result_dir, "entity/prescription.csv"), index=False
    )
