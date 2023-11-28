import pandas as pd
import os
from tqdm import tqdm
import numpy as np


merge_result_dir = "merge_result"
# Read the symptoms, symptoms, and prescription data sets respectively, and use a complete matching method to find whether the symptoms appear in the prescription indications. If so, add edges.
symptom_data = pd.read_csv(os.path.join(merge_result_dir, "entity/tcm_symptom.csv"))
prescription_data = pd.read_csv(
    os.path.join(merge_result_dir, "entity/prescription.csv")
)
prescription_data.replace(np.nan, None, inplace=True)
syndrome_data = pd.read_csv(os.path.join(merge_result_dir, "entity/syndrome.csv"))

symptom_key_map = dict()

for index, row in symptom_data.iterrows():
    for item in row["Chinese_Name"].split(";"):
        symptom_key_map[item] = row["TMDB_id"]

prescription_symptom_relation = []
for index, row in tqdm(prescription_data.iterrows()):
    if row["Indications"] is None:
        continue
    for symptom_word, symptom_id in symptom_key_map.items():
        if symptom_word in row["Indications"]:
            prescription_symptom_relation.append((row["TMDB_id"], symptom_id))

prescription_symptom_relation_new = pd.DataFrame(
    set(prescription_symptom_relation), columns=["source_id", "target_id"]
)

prescription_symptom_relation_extracted = pd.read_csv(
    os.path.join(merge_result_dir, "relation/prescription2symptom.csv")
)
prescription_data.index = prescription_data.TMDB_id
symptom_data.index = symptom_data.TMDB_id

# There are 5701 places where two relational tables intersect.
prescription_symptom_relation_new[
    "Indications"
] = prescription_symptom_relation_new.source_id.apply(
    lambda x: prescription_data.loc[x, "Indications"]
)
prescription_symptom_relation_new[
    "symptom_words"
] = prescription_symptom_relation_new.target_id.apply(
    lambda x: symptom_data.loc[x, "Chinese_Name"]
)

# Drop Some Symptoms TMTS01877、TMTS01878
prescription_symptom_relation_new.drop(
    prescription_symptom_relation_new[
        prescription_symptom_relation_new.target_id == "TMTS01877"
    ].index,
    inplace=True,
)

prescription_symptom_relation_new.drop(
    prescription_symptom_relation_new[
        prescription_symptom_relation_new.target_id == "TMTS01878"
    ].index,
    inplace=True,
)

prescription_symptom_relation_new.drop(
    prescription_symptom_relation_new[
        prescription_symptom_relation_new.Indications.str.contains("不渴")
    ].index,
    inplace=True,
)

prescription_symptom_relation_new.drop(
    prescription_symptom_relation_new[
        (prescription_symptom_relation_new.target_id == "TMTS00409")
        & prescription_symptom_relation_new.Indications.apply(lambda x: "不渴" in x)
    ].index,
    inplace=True,
)

prescription_symptom_relation_new.drop(
    ["symptom_words", "Indications"], axis=1, inplace=True
)

prescription_symptom_relation_new["Relation_type"] = "prescription_treat_symptom"

prescription_symptom_relation = pd.concat(
    [prescription_symptom_relation_extracted, prescription_symptom_relation_new]
)

prescription_symptom_relation.drop_duplicates(inplace=True)

prescription_symptom_relation.to_csv(
    os.path.join(merge_result_dir, "relation/prescription2symptom.csv"), index=False
)

syndrome_key_map = dict()

for index, row in syndrome_data.iterrows():
    for item in row["Syndrome_name"].split(";"):
        syndrome_key_map[item] = row["TMDB_id"]

prescription_syndrome_relation = []
for index, row in tqdm(prescription_data.iterrows()):
    if row["Indications"] is None:
        continue
    for syndrome_word, syndrome_id in syndrome_key_map.items():
        if syndrome_word in row["Indications"]:
            prescription_syndrome_relation.append((row["TMDB_id"], syndrome_id))

syndrome_data.index = syndrome_data.TMDB_id

prescription_syndrome_relation = pd.DataFrame(
    set(prescription_syndrome_relation), columns=["source_id", "target_id"]
)

prescription_syndrome_relation[
    "Indications"
] = prescription_syndrome_relation.source_id.apply(
    lambda x: prescription_data.loc[x, "Indications"]
)

prescription_syndrome_relation[
    "Syndrome"
] = prescription_syndrome_relation.target_id.apply(
    lambda x: syndrome_data.loc[x, "Syndrome_name"]
)

prescription_syndrome_relation.drop(["Syndrome", "Indications"], axis=1, inplace=True)
prescription_syndrome_relation["Relation_type"] = "prescription_treat_syndrome"
prescription_syndrome_relation.to_csv(
    os.path.join(merge_result_dir, "relation/prescription2syndrome.csv"), index=False
)
