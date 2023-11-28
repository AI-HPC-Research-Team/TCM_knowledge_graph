import pandas as pd
import numpy as np
from merge_utility import merge_two_row, merge_database_by_id
import glob
import json
import os


if __name__ == "__main__":
    symmap_data_dir = "../data/symmap"
    cpmcp_data_dir = "../data/CPMCP"
    merge_result_dir = "../merge_result"
    data_root_dir = "../data"
    
    # Merge two symptoms whose description similarity is greater than 0.98
    merged_group = pd.read_csv(
        os.path.join(data_root_dir, "resource/threshold_98.csv"), header=None
    )

    symmap_tcm_symptom = pd.read_excel(
        os.path.join(symmap_data_dir, "symmap_tcm_symptom.xlsx")
    )
    symmap_tcm_symptom.drop(["Version", "Suppress"], inplace=True, axis=1)
    symmap_tcm_symptom.drop(203, inplace=True)
    symmap_tcm_symptom.rename(
        columns={
            "TCM_symptom_id": "SymMap_id",
            "TCM_symptom_name": "Chinese_Name",
            "Symptom_pinYin": "Pinyin_Name",
            "Symptom_definition": "Definition",
            "Symptom_locus": "Locus",
            "Symptom_property": "Property",
        },
        inplace=True,
    )
    symmap_tcm_symptom.SymMap_id = symmap_tcm_symptom.SymMap_id.apply(
        lambda x: "SMTS{:05d}".format(int(x))
    )
    cpmcp_tcm_symptom = pd.read_csv(os.path.join(cpmcp_data_dir, "tcm_symptom.csv"))
    cpmcp_tcm_symptom.rename(
        columns={
            "ID": "CPMCP_id",
            "Chinese name": "Chinese_Name",
            "English name": "English_Name",
            "Pinyin name": "Pinyin_Name",
            "Chinese locus": "Locus",
            "English locus": "Locus_English",
            "Chinese property": "Property",
            "English property": "Property_English",
        },
        inplace=True,
    )
    cpmcp_tcm_symptom.CPMCP_id = cpmcp_tcm_symptom.CPMCP_id.apply(
        lambda x: "TCM{:05d}".format(int(x))
    )
    cpmcp_tcm_symptom.Locus.replace(to_replace=["-"], value=np.nan, inplace=True)
    cpmcp_tcm_symptom.Property.replace(to_replace=["-"], value=np.nan, inplace=True)

    merge_symptom = cpmcp_tcm_symptom.merge(
        symmap_tcm_symptom,
        on=["Chinese_Name", "Locus", "Property", "Pinyin_Name"],
        how="outer",
    )
    merge_symptom.replace(np.nan, None, inplace=True)
    Chinese_Name_map_id = dict(
        zip(merge_symptom.Chinese_Name.tolist(), range(merge_symptom.shape[0]))
    )

    merged_id = []
    for symptom_group in merged_group[0].tolist():
        symptom_list = symptom_group.split(";")
        if len(symptom_list) > 1:
            first_id = Chinese_Name_map_id[symptom_list[0]]
            first_row = merge_symptom.iloc[first_id]
            for symptom in symptom_list[1:]:
                other_id = Chinese_Name_map_id[symptom]
                first_row = merge_two_row(first_row, merge_symptom.iloc[other_id])
                merged_id.append(other_id)
            merge_symptom.iloc[first_id] = first_row
    assert len(merged_id) == len(set(merged_id))
    merge_symptom = merge_symptom[~merge_symptom.index.isin(merged_id)]
    merge_symptom.reset_index(drop=True, inplace=True)
    merge_symptom["TMDB_id"] = [
        "TMTS{:05d}".format(index + 1) for index in range(merge_symptom.shape[0])
    ]
    merge_symptom.to_csv(
        os.path.join(merge_result_dir, "entity/tcm_symptom.csv"), index=False
    )

    # Define locus

    locus_df = pd.read_csv(os.path.join(data_root_dir, 'resource/locus.csv'))
    locus_df["TMDB_id"] = [
        "TMLC{:05d}".format(index + 1) for index in range(locus_df.shape[0])
    ]
    locus_df[["TMDB_id", "Locus", "Locus_English"]].to_csv(
        os.path.join(merge_result_dir, "entity/locus.csv"), index=False
    )

    locus_map = dict(zip(locus, locus))
    locus_map.update(
        {
            "肩臂": "上肢",
            "项": "颈部",
            "体表": "皮肤",
            "通身": "全身",
            "任何部位": "全身",
            "躯体": "全身",
            "目": "眼",
            "咽喉": "咽部",
            "肢体": "四肢",
        }
    )
    locus_map_id = dict(zip(locus_df.Locus.tolist(), locus_df.TMDB_id.tolist()))
    # tcm_symptom locus
    tcm_symptom2locus = []
    for index, row in merge_symptom.iterrows():
        if not isinstance(row["Locus"], str):
            continue
        for string, locus in locus_map.items():
            if string in row["Locus"]:
                tcm_symptom2locus.append((row["TMDB_id"], locus_map_id[locus]))

    # extract relations
    tcm_symptom2locus = pd.DataFrame(
        tcm_symptom2locus, columns=["source_id", "target_id"]
    )
    tcm_symptom2locus["Relation_type"] = ["symptom_locus"] * tcm_symptom2locus.shape[0]
    tcm_symptom2locus.to_csv(
        os.path.join(merge_result_dir, "relation/tcm_symptom2locus.csv"), index=False
    )

    cpmcp_tcm_symptom_map = {}
    symmap_tcm_symptom_map = {}
    for index, row in merge_symptom.iterrows():
        if isinstance(row["CPMCP_id"], str):
            for cpmcp_id in row["CPMCP_id"].split(";"):
                cpmcp_tcm_symptom_map[cpmcp_id] = row["TMDB_id"]
        if isinstance(row["SymMap_id"], str):
            for SymMap_id in row["SymMap_id"].split(";"):
                symmap_tcm_symptom_map[SymMap_id] = row["TMDB_id"]

    mm_symptom = pd.read_csv(os.path.join(merge_result_dir, "entity/mm_symptom.csv"))
    cpmcp_mm_symptom_map = {}
    symmap_mm_symptom_map = {}
    for index, row in mm_symptom.iterrows():
        if isinstance(row["CPMCP_id"], str):
            for cpmcp_id in row["CPMCP_id"].split(";"):
                cpmcp_mm_symptom_map[cpmcp_id] = row["TMDB_id"]
        if isinstance(row["SymMap_id"], str):
            for SymMap_id in row["SymMap_id"].split(";"):
                symmap_mm_symptom_map[SymMap_id] = row["TMDB_id"]

    herb = pd.read_csv(os.path.join(merge_result_dir, "entity/medicinal_material.csv"))
    cpmcp_herb_map = {}
    symmap_herb_map = {}
    for index, row in herb.iterrows():
        if isinstance(row["CPMCP_id"], str):
            for cpmcp_id in row["CPMCP_id"].split(";"):
                cpmcp_herb_map[cpmcp_id] = row["TMDB_id"]
        if isinstance(row["SymMap_id"], str):
            for SymMap_id in row["SymMap_id"].split(";"):
                symmap_herb_map["SMHB{:05d}".format(int(SymMap_id))] = row["TMDB_id"]

    syndrome = pd.read_csv(os.path.join(merge_result_dir, "entity/syndrome.csv"))
    symmap_syndrome_map = {}
    for index, row in syndrome.iterrows():
        if isinstance(row["SymMap_id"], str):
            for SymMap_id in row["SymMap_id"].split(";"):
                symmap_syndrome_map[SymMap_id] = row["TMDB_id"]

    prescription = pd.read_csv(
        os.path.join(merge_result_dir, "entity/prescription.csv")
    )
    cpmcp_prescription_map = {}
    for index, row in prescription.iterrows():
        if isinstance(row["CPMCP_ID"], str):
            for cpmcp_id in row["CPMCP_ID"].split(";"):
                cpmcp_prescription_map[cpmcp_id] = row["TMDB_id"]

    # herb2symptom
    herb2symptom = []
    # symmap_relation
    symmap_tcm_data = os.path.join(symmap_data_dir, "symptom")
    for tcm_herb_path in glob.glob(symmap_tcm_data + "/*/herb.json"):
        symmap_tcm_id = tcm_herb_path.split("/")[-2]
        TMDB_tcm_id = symmap_tcm_symptom_map[symmap_tcm_id]
        herb_json = json.load(open(tcm_herb_path))
        for herb in herb_json["data"]:
            TMDB_herb_id = symmap_herb_map[herb["Herb_id"]]
            herb2symptom.append((TMDB_herb_id, TMDB_tcm_id))

    # cpmcp_relation
    cpmcp_tcm_data = os.path.join(cpmcp_data_dir, "TCM symptom")
    for tcm_herb_path in glob.glob(cpmcp_tcm_data + "/*/herb.json"):
        cpmcp_tcm_id = "TCM{:05d}".format(int(tcm_herb_path.split("/")[-2]))
        TMDB_tcm_id = cpmcp_tcm_symptom_map[cpmcp_tcm_id]
        herb_json = json.load(open(tcm_herb_path))
        for herb in herb_json["items"]:
            TMDB_herb_id = cpmcp_herb_map[str(herb["id"])]
            herb2symptom.append((TMDB_herb_id, TMDB_tcm_id))

    herb2symptom_df = pd.DataFrame(
        set(herb2symptom), columns=["source_id", "target_id"]
    )
    herb2symptom_df["Relation_type"] = ["herb_treat_symptom"] * herb2symptom_df.shape[0]
    herb2symptom_df.to_csv(
        os.path.join(merge_result_dir, "relation/herb2symptom.csv"), index=False
    )

    # mm_symptom 2 tcm_symptom
    tcm_symptom2mm_symptom = []
    # symmap_relation
    symmap_tcm_data = os.path.join(symmap_data_dir, "symptom")
    for mm_tcm_path in glob.glob(symmap_tcm_data + "/*/mm_symptom.json"):
        symmap_tcm_id = mm_tcm_path.split("/")[-2]
        TMDB_tcm_id = symmap_tcm_symptom_map[symmap_tcm_id]
        mm_json = json.load(open(mm_tcm_path))
        for mm in mm_json["data"]:
            TMDB_mm_id = symmap_mm_symptom_map[mm["MM_symptom_id"]]
            tcm_symptom2mm_symptom.append((TMDB_tcm_id, TMDB_mm_id))

    # cpmcp_relation
    cpmcp_mm_data = os.path.join(cpmcp_data_dir, "MM symptom")
    for mm_tcm_path in glob.glob(cpmcp_mm_data + "/*/tcm_symptom.json"):
        cpmcp_mm_id = "SYM{:05d}".format(int(mm_tcm_path.split("/")[-2]))
        TMDB_mm_id = cpmcp_mm_symptom_map[cpmcp_mm_id]
        tcm_json = json.load(open(mm_tcm_path))
        for tcm in tcm_json["items"]:
            TMDB_tcm_id = cpmcp_tcm_symptom_map["TCM{:05d}".format(tcm["id"])]
            tcm_symptom2mm_symptom.append((TMDB_tcm_id, TMDB_mm_id))
    tcm_symptom2mm_symptom_df = pd.DataFrame(
        set(tcm_symptom2mm_symptom), columns=["source_id", "target_id"]
    )
    tcm_symptom2mm_symptom_df["Relation_type"] = [
        "tcm_symptom_map_mm_symptom"
    ] * tcm_symptom2mm_symptom_df.shape[0]
    tcm_symptom2mm_symptom_df.to_csv(
        os.path.join(merge_result_dir, "relation/tcm_symptom2mm_symptom.csv"),
        index=False,
    )
    # syndrome 2 tcm_symptom
    syndrome2tcm_symptom = []
    symmap_syndrome_data = os.path.join(symmap_data_dir, "syndrome")
    for syndrome_tcm_path in glob.glob(symmap_syndrome_data + "/*/tcm_symptom.json"):
        symmap_syndrome_id = syndrome_tcm_path.split("/")[-2]
        TMDB_syndrome_id = symmap_syndrome_map[symmap_syndrome_id]
        tcm_json = json.load(open(syndrome_tcm_path))
        for tcm in tcm_json["data"]:
            TMDB_tcm_id = symmap_tcm_symptom_map[tcm["TCM_symptom_id"]]
            syndrome2tcm_symptom.append((TMDB_syndrome_id, TMDB_tcm_id))
    syndrome2tcm_symptom_df = pd.DataFrame(
        set(syndrome2tcm_symptom), columns=["source_id", "target_id"]
    )
    syndrome2tcm_symptom_df["Relation_type"] = [
        "syndrome_present_symptom"
    ] * syndrome2tcm_symptom_df.shape[0]
    syndrome2tcm_symptom_df.to_csv(
        os.path.join(merge_result_dir, "relation/syndrome2tcm_symptom.csv"), index=False
    )
    
    # prescription2symptom
    prescription2symptom = []
    for tcm_prescription_path in glob.glob(cpmcp_tcm_data + "/*/cmp.json"):
        cpmcp_tcm_id = "TCM{:05d}".format(int(tcm_prescription_path.split("/")[-2]))
        TMDB_tcm_id = cpmcp_tcm_symptom_map[cpmcp_tcm_id]
        prescription_json = json.load(open(tcm_prescription_path))
        for prescription in prescription_json["items"]:
            TMDB_prescription_id = cpmcp_prescription_map[
                "CMP{:05d}".format(prescription["id"])
            ]
            prescription2symptom.append((TMDB_prescription_id, TMDB_tcm_id))

    for tcm_prescription_path in glob.glob(cpmcp_tcm_data + "/*/cpm.json"):
        cpmcp_tcm_id = "TCM{:05d}".format(int(tcm_prescription_path.split("/")[-2]))
        TMDB_tcm_id = cpmcp_tcm_symptom_map[cpmcp_tcm_id]
        prescription_json = json.load(open(tcm_prescription_path))
        for prescription in prescription_json["items"]:
            TMDB_prescription_id = cpmcp_prescription_map[
                "CPM{:05d}".format(prescription["id"])
            ]
            prescription2symptom.append((TMDB_prescription_id, TMDB_tcm_id))

    prescription2symptom_df = pd.DataFrame(
        set(prescription2symptom), columns=["source_id", "target_id"]
    )

    # delete some relations
    count_info = prescription2symptom_df.groupby("source_id").count()
    prescription2symptom_df = prescription2symptom_df[
        prescription2symptom_df.source_id.isin(
            count_info[count_info.target_id < 50].index
        )
    ]
    prescription2symptom_df["Relation_type"] = [
        "prescription_treat_symptom"
    ] * prescription2symptom_df.shape[0]
    prescription2symptom_df.to_csv(
        os.path.join(merge_result_dir, "relation/prescription2symptom.csv"), index=False
    )
