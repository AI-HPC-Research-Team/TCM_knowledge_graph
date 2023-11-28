import pandas as pd
import json
import os
import ast
import sys
import re
from merge_utility import *
import glob
import math
import numpy as np
from collections import Counter

cpmcp_root_dir = "../data/CPMCP"
symmap_root_dir = "../data/symmap"
tcmbank_root_dir = "../data/TCMBANK"
primekg_root_dir = "../data/primeKG"
pharmebinet_root_dir = "../data/PharMeBINet"
merge_result_dir = "../merge_result"

# cpmcp disease
cpmcp_disease = []
for disease_path in glob.glob(os.path.join(cpmcp_root_dir, "disease/*/disease.json")):
    res_json = json.load(open(disease_path))
    cpmcp_disease.append(res_json)

cpmcp_disease = pd.DataFrame(cpmcp_disease)

cpmcp_disease["ICD-10_id"] = cpmcp_disease.externals.apply(
    lambda x: ";".join(x["ICD-10 id"]) if x["ICD-10 id"] != [] else None
)
cpmcp_disease["MeSH_id"] = cpmcp_disease.externals.apply(
    lambda x: ";".join(x["MeSH id"]) if x["MeSH id"] != [] else None
)
cpmcp_disease["MedDRA_id"] = cpmcp_disease.externals.apply(
    lambda x: ";".join(x["MedDRA id"]) if x["MedDRA id"] != [] else None
)
cpmcp_disease["OMIM_id"] = cpmcp_disease.externals.apply(
    lambda x: ";".join(x["OMIM id"]) if x["OMIM id"] != [] else None
)
cpmcp_disease["Orphanet_id"] = cpmcp_disease.externals.apply(
    lambda x: ";".join([str(item) for item in x["Orphanet id"]])
    if x["Orphanet id"] != []
    else None
)
cpmcp_disease["SymMap_id"] = cpmcp_disease.externals.apply(
    lambda x: ";".join(x["SymMap id"]) if x["SymMap id"] != [] else None
)
cpmcp_disease["UMLS_id"] = cpmcp_disease.externals.apply(
    lambda x: ";".join(x["UMLS id"]) if x["UMLS id"] != [] else None
)

cpmcp_disease.drop(
    ["suppress", "symmap_id", "_show", "externals", "link_disease_id"],
    axis=1,
    inplace=True,
)
cpmcp_disease.rename(
    columns={"name": "Disease_name", "id": "CPMCP_id", "definition": "Definition"},
    inplace=True,
)
cpmcp_disease["Disease_name"] = cpmcp_disease["Disease_name"].str.lower()
# tcmbank disease
tcmbank_disease = pd.read_csv(os.path.join(tcmbank_root_dir, "disease_all.csv"))
tcmbank_disease["Disease_name"] = tcmbank_disease["Disease_name"].str.lower()
tcmbank_disease["Disease_id"] = tcmbank_disease["Disease_id"].apply(
    lambda x: "TCMBANKDI{:06d}".format(x)
)

tcmbank_disease.rename(columns={"Disease_id": "TCMBank_id"}, inplace=True)  # 32529
concate_disease = pd.concat(
    [tcmbank_disease, cpmcp_disease], ignore_index=True
)  # 14434
concate_disease.replace(np.nan, None, inplace=True)
concate_disease = merge_database_by_id_group(concate_disease, "Disease_name")
# 46963 records before merged, 37281 records after merged
# primekg disease 15163/15813 overlaps with pharMeBINet database
primekg_disease = pd.read_csv(os.path.join(primekg_root_dir, "disease.csv"))
primekg_disease = primekg_disease[primekg_disease.node_source == "MONDO"]
primekg_disease.rename(
    columns={
        "node_index": "PrimeKG_id",
        "node_id": "MONDO_id",
        "node_name": "Disease_name",
    },
    inplace=True,
)
primekg_disease.Disease_name = primekg_disease.Disease_name.str.lower()
primekg_disease.drop(["node_type", "node_source"], axis=1, inplace=True)
concate_disease = pd.concat([concate_disease, primekg_disease], ignore_index=True)
concate_disease.replace(np.nan, None, inplace=True)
concate_disease = merge_database_by_id_group(concate_disease, "Disease_name")
# 53094 records before merged, 48913 records after merged

# PharMeBINet
# ['node_id', 'labels', 'properties', 'name', 'identifier', 'resource', 'license', 'source', 'url']
pharmebinet_disease = pd.read_csv(os.path.join(pharmebinet_root_dir, "disease.csv"))
pharmebinet_disease.resource = pharmebinet_disease.resource.apply(
    lambda x: ast.literal_eval(x)
)
pharmebinet_disease.properties = pharmebinet_disease.properties.apply(
    lambda x: ast.literal_eval(x)
)
pharmebinet_disease.rename(
    columns={
        "node_id": "pharmebinet_id",
        "name": "Disease_name",
        "identifier": "MONDO_id",
    },
    inplace=True,
)

pharmebinet_disease["MONDO_id"] = pharmebinet_disease["MONDO_id"].apply(
    lambda x: str(int(x[6:]))
)

OMIM_ids = []
MESH_ids = []
Orphanet_ids = []
UMLS_ids = []
for property_item in pharmebinet_disease["properties"]:
    # extract omim id
    OMIM_id = []
    MESH_id = []
    Orphanet_id = []
    UMLS_id = []
    if "xrefs" in property_item:
        for id_ref in property_item["xrefs"]:
            if "OMIM" in id_ref:
                OMIM_id.append(id_ref[5:])
            if "MESH" in id_ref:
                MESH_id.append(id_ref[5:])
            if "Orphanet" in id_ref:
                Orphanet_id.append(id_ref[9:])
            if "UMLS" in id_ref:
                UMLS_id.append(id_ref[5:])
    if OMIM_id != []:
        OMIM_ids.append(";".join(OMIM_id))
    else:
        OMIM_ids.append(None)
    if MESH_id != []:
        MESH_ids.append(";".join(MESH_id))
    else:
        MESH_ids.append(None)
    if Orphanet_id != []:
        Orphanet_ids.append(";".join(Orphanet_id))
    else:
        Orphanet_ids.append(None)
    if UMLS_id != []:
        UMLS_ids.append(";".join(UMLS_id))
    else:
        UMLS_ids.append(None)

pharmebinet_disease["OMIM_id"] = OMIM_ids
pharmebinet_disease["MeSH_id"] = MESH_ids
pharmebinet_disease["Orphanet_id"] = Orphanet_ids
pharmebinet_disease["UMLS_id"] = UMLS_ids
pharmebinet_disease.drop(
    ["labels", "resource", "source", "properties"], inplace=True, axis=1
)

concate_disease = pd.concat([concate_disease, pharmebinet_disease], ignore_index=True)
concate_disease.replace(np.nan, None, inplace=True)
concate_disease["CPMCP_id"] = concate_disease["CPMCP_id"].apply(
    lambda x: str(int(x)) if x else None
)
concate_disease["PrimeKG_id"] = concate_disease["PrimeKG_id"].apply(
    lambda x: str(int(x)) if x else None
)
concate_disease["pharmebinet_id"] = concate_disease["pharmebinet_id"].apply(
    lambda x: str(int(x)) if x else None
)
# 71132 --> 48231
concate_disease = merge_database_by_id_group(concate_disease, "MONDO_id")
concate_disease = merge_database_by_id_group(concate_disease, "OMIM_id")
concate_disease = merge_database_by_id_group(concate_disease, "MeSH_id")
concate_disease = merge_database_by_id_group(concate_disease, "Orphanet_id")
concate_disease = merge_database_by_id_group(concate_disease, "UMLS_id")
concate_disease = concate_disease[
    concate_disease.PrimeKG_id.notnull() | concate_disease.pharmebinet_id.notnull()
]  # 删除冗余的disease

concate_disease["TMDB_id"] = [
    "TMDIS{:05d}".format(index + 1) for index in range(concate_disease.shape[0])
]
concate_disease.to_csv(
    os.path.join(merge_result_dir, "entity/disease.csv"), index=False
)

concate_disease = pd.read_csv(os.path.join(merge_result_dir, "entity/disease.csv"))
concate_disease.replace(np.nan, None, inplace=True)
# extract disease2mm symptom
# disease map
cpmcp_disease_map = {}
symmap_disease_map = {}
PharMeBINet_disease_map = {}
for index, row in concate_disease.iterrows():
    if row["CPMCP_id"]:
        for CPMCP_id in row["CPMCP_id"].split(";"):
            cpmcp_disease_map[CPMCP_id] = row["TMDB_id"]
    if row["SymMap_id"]:
        for Symmap_id in row["SymMap_id"].split(";"):
            symmap_disease_map[Symmap_id] = row["TMDB_id"]
    if row["pharmebinet_id"]:
        for PharMeBINet_id in row["pharmebinet_id"].split(";"):
            PharMeBINet_disease_map[PharMeBINet_id] = row["TMDB_id"]

mm_symptom = pd.read_csv(os.path.join(merge_result_dir, "entity/mm_symptom.csv"))
mm_symptom.replace(np.nan, None, inplace=True)
cpmcp_mm_symptom_map = {}
symmap_mm_symptom_map = {}
PharMeBINet_mm_symptom_map = {}
for index, row in mm_symptom.iterrows():
    if row["CPMCP_id"]:
        for CPMCP_id in row["CPMCP_id"].split(";"):
            cpmcp_mm_symptom_map[CPMCP_id] = row["TMDB_id"]
    if row["SymMap_id"]:
        for Symmap_id in row["SymMap_id"].split(";"):
            symmap_mm_symptom_map[Symmap_id] = row["TMDB_id"]
    if row["pharmebinet_id"]:
        for PharMeBINet_id in row["pharmebinet_id"].split(";"):
            PharMeBINet_mm_symptom_map[PharMeBINet_id] = row["TMDB_id"]

# read cpmcp disease mm symptom
disease2mm = []
cpmcp_disease_data = os.path.join(cpmcp_root_dir, "disease")
for disease_mm_symptom_path in glob.glob(cpmcp_disease_data + "/*/mm_symptom.json"):
    cpmcp_disease_id = disease_mm_symptom_path.split("/")[-2]
    if cpmcp_disease_id not in cpmcp_disease_map:
        continue
    TMDB_disease_id = cpmcp_disease_map[cpmcp_disease_id]
    mm_symptom_json = json.load(open(disease_mm_symptom_path))
    for mm_symptom in mm_symptom_json["items"]:
        TMDB_mm_symptom_id = cpmcp_mm_symptom_map["SYM{:05d}".format(mm_symptom["id"])]
        disease2mm.append((TMDB_disease_id, TMDB_mm_symptom_id))
# read symmap disease mm symptom
symmap_ms_data = os.path.join(symmap_root_dir, "mm_symptom")
for disease_mm_symptom_path in glob.glob(symmap_ms_data + "/*/disease.json"):
    symmap_ms_id = disease_mm_symptom_path.split("/")[-2]
    TMDB_ms_id = symmap_mm_symptom_map[symmap_ms_id]
    disease_json = json.load(open(disease_mm_symptom_path))
    for disease in disease_json["data"]:
        if disease["Disease_id"] not in symmap_disease_map:
            continue
        TMDB_disease_id = symmap_disease_map[disease["Disease_id"]]
        disease2mm.append((TMDB_disease_id, TMDB_ms_id))

# read PharMeBINet disease mm symptom
temp_relation = []
PharMeBINet_disease2mm = pd.read_csv(
    os.path.join(pharmebinet_root_dir, "disease2symptom.csv")
)
for index, relation in PharMeBINet_disease2mm.iterrows():
    PharMeBINet_start_id = str(int(relation["start_id"]))
    PharMeBINet_end_id = str(int(relation["end_id"]))
    if (PharMeBINet_start_id in PharMeBINet_disease_map) and (
        PharMeBINet_end_id in PharMeBINet_mm_symptom_map
    ):
        TMDB_disease_id = PharMeBINet_disease_map[PharMeBINet_start_id]
        TMDB_ms_id = PharMeBINet_mm_symptom_map[PharMeBINet_end_id]
        temp_relation.append((TMDB_disease_id, TMDB_ms_id))

disease2mm.extend(temp_relation)
disease2mm_df = pd.DataFrame(set(disease2mm), columns=["source_id", "target_id"])
disease2mm_df["Relation_type"] = ["disease_present_symptom"] * disease2mm_df.shape[0]
disease2mm_df.to_csv(
    os.path.join(merge_result_dir, "relation/disease2mm_symptom.csv"), index=False
)

pharmebinet_relations = pd.read_table(os.path.join(pharmebinet_root_dir, "edges.tsv"))
primekg_relations = pd.read_csv(os.path.join(primekg_root_dir, "edges.csv"))

disease_id_maps = id_map(
    concate_disease,
    ["pharmebinet_id", "PrimeKG_id", "CPMCP_id", "SymMap_id", "TCMBank_id"],
)

# disease is a disease child-parent
disease_is_a_disease = extract_specific_relation(
    pharmebinet_relations,
    disease_id_maps["pharmebinet_id"],
    disease_id_maps["pharmebinet_id"],
    "IS_A_DiaD",
)
disease_is_a_disease_df = pd.DataFrame(
    set(disease_is_a_disease), columns=["source_id", "target_id"]
)

# need to drop source_id == target_id, source 是child， target 是 parent，target出现的次数要比source多
select_condition = calulate_reverse_relation(disease_is_a_disease_df)

disease_is_a_disease_df["Relation_type"] = [
    "disease_is_a_disease"
] * disease_is_a_disease_df.shape[0]
disease_is_a_disease_df = disease_is_a_disease_df[~select_condition]
disease_is_a_disease_df.to_csv(
    os.path.join(merge_result_dir, "relation/disease_is_a_disease.csv"), index=False
)

disease_is_a_disease = disease_is_a_disease_df.apply(
    lambda row: (row[0], row[1]), axis=1
).tolist()

# disease resemble disease
disease_resemble_disease = extract_specific_relation(
    pharmebinet_relations,
    disease_id_maps["pharmebinet_id"],
    disease_id_maps["pharmebinet_id"],
    "RESEMBLES_DrD",
)
disease_resemble_disease = set(disease_resemble_disease)
# If an entity pair appears in both relationship types "disease_resemble_disease" and "disease is a disease", remove the entity pair in relationship "disease_resemble_disease".
remove_rel = set()
for rel in disease_resemble_disease:
    if rel in disease_is_a_disease or (rel[1], rel[0]) in disease_is_a_disease:
        remove_rel.add(rel)
disease_resemble_disease -= remove_rel
disease_resemble_disease_df = pd.DataFrame(
    disease_resemble_disease, columns=["source_id", "target_id"]
)
disease_resemble_disease_df["Relation_type"] = [
    "disease_resemble_disease"
] * disease_resemble_disease_df.shape[0]
disease_resemble_disease_df.to_csv(
    os.path.join(merge_result_dir, "relation/disease_resemble_disease.csv"), index=False
)


# pathway lead to disease
pathway = pd.read_csv(os.path.join(merge_result_dir, "entity/pathway.csv"))
pathway.replace(np.nan, None, inplace=True)
pharmebinet_pathway_map = {}
for index, row in pathway.iterrows():
    if row["pharmebinet_id"]:
        pharmebinet_pathway_map[str(int(row["pharmebinet_id"]))] = row["TMDB_id"]
pathway_lead_to_disease = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_pathway_map,
    disease_id_maps["pharmebinet_id"],
    "LEADS_TO_PWltD",
)
pathway_lead_to_disease_df = pd.DataFrame(
    set(pathway_lead_to_disease), columns=["source_id", "target_id"]
)
pathway_lead_to_disease_df["Relation_type"] = [
    "pathway_lead_to_disease"
] * pathway_lead_to_disease_df.shape[0]
pathway_lead_to_disease_df.to_csv(
    os.path.join(merge_result_dir, "relation/pathway2disease.csv"), index=False
)
