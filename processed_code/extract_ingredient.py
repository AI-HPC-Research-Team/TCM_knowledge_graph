import pandas as pd
import json
import os
import ast
import sys
import re
from merge_utility import (
    merge_database_by_id,
    merge_database_by_id_group,
    id_map,
    extract_specific_relation,
)
import glob
import math
import numpy as np
from merge_utility import extract_external_ids, extract_properties


def Q2B(uchar):
    if len(uchar) != 1:
        raise TypeError("expected a character, but a string found!")
    inner_code = ord(uchar)
    if inner_code == 0x3000:
        inner_code = 0x0020
    else:
        inner_code -= 0xFEE0
    if inner_code < 0x0020 or inner_code > 0x7E:
        return uchar
    return chr(inner_code)


def stringQ2B(ustring):
    return "".join([Q2B(uchar) for uchar in ustring])


def str_clean(string):
    string = string.lower()
    string = stringQ2B(string)
    string = string.replace("〔", "(")
    string = string.replace("〕", "(")
    string = re.sub("(?<![\u4e00-\u9fa5])[一‐−–—―→‑]", "-", string)
    string = re.sub("\s*-\s*", "-", string)
    return string


# database dir
cpmcp_root_dir = "../data/CPMCP"
symmap_root_dir = "../data/symmap"
tcmbank_root_dir = "../data/TCMBANK"
primekg_root_dir = "../data/primeKG"
pharmebinet_root_dir = "../data/PharMeBINet"
merge_result_dir = "../merge_result"

# # read cpmcp ingredient
print("reading cpmcp database")
cpmcp_ingredients = []
for file_path in glob.glob(os.path.join(cpmcp_root_dir, "ingredient/*/*.json")):
    res_json = json.load(open(file_path))
    cpmcp_ingredients.append(res_json)
cpmcp_ingredients = pd.DataFrame(cpmcp_ingredients)
cpmcp_ingredients["CAS_number"] = cpmcp_ingredients.externals.apply(
    lambda x: ";".join(x["CAS number"]) if x["CAS number"] != [] else None
)
cpmcp_ingredients["PubChem_id"] = cpmcp_ingredients.externals.apply(
    lambda x: ";".join(x["PubChem id"]) if x["PubChem id"] != [] else None
)
cpmcp_ingredients["SymMap_id"] = cpmcp_ingredients.externals.apply(
    lambda x: ";".join(x["SymMap id"]) if x["SymMap id"] != [] else None
)
cpmcp_ingredients["id"] = cpmcp_ingredients["id"].apply(lambda x: str(int(x)))
cpmcp_ingredients.rename(
    columns={
        "id": "CPMCP_id",
        "link_ingredient_id": "cpmcp_link_id",
        "molecule_formula": "Molecule_formula",
        "molecule_name": "Molecule_name",
        "molecule_structure": "Molecule_structure",
        "molecule_weight": "Molecule_weight",
        "ob_score": "OB_score",
        "suppress": "Suppress",
        "type": "Type",
    },
    inplace=True,
)

# cas_number and ingredient do not correspond.
cpmcp_ingredients.drop(
    ["_show", "externals", "Suppress", "symmap_id", "cpmcp_link_id", "Type"],
    axis=1,
    inplace=True,
)
# read tcmbank ingredient
cpmcp_ingredients["Molecule_name"] = cpmcp_ingredients["Molecule_name"].apply(
    lambda x: str_clean(x)
)
cpmcp_ingredients["OB_score"] = cpmcp_ingredients["OB_score"].apply(
    lambda x: str(x) + "::CPMCP" if x else None
)
cpmcp_ingredients["Molecule_weight"] = cpmcp_ingredients["Molecule_weight"].apply(
    lambda x: str(x) + "::CPMCP" if x else None
)
print("reading tcmbank database")
tcmbank_ingredient = pd.read_csv(
    os.path.join(tcmbank_root_dir, "refine_ingredient.csv")
)
tcmbank_ingredient.rename(
    columns={
        "Ingredient_id": "HERB_id",
        "TCMBank_ID": "TCMBank_id",
        "Molecular_Formula": "Molecule_formula",
        "Molecular_Weight": "Molecule_weight",
        "name": "Molecule_name",
        "CAS_id": "CAS_number",
    },
    inplace=True,
)
# tcmbank_ingredient = tcmbank_ingredient[['TCMBank_ID', 'level1_name', 'level1_name_en', 'level2_name', 'level2_name_en', 'TCM_name', 'TCM_name_en', 'TCM_name2', 'Molecule_name', 'Alias', 'Molecule_formula', 'Smiles', 'Molecule_weight', 'Molecular_Volume', 'HERB_id', 'OB_score', 'CAS_number', 'SymMap_id', 'TCMID_id', 'TCMSP_id', 'TCM-ID_id', 'PubChem_id', 'DrugBank_id', 'ALogP', 'Molecular_PolarSurfaceArea', 'Num_H_Acceptors', 'Num_H_Donors', 'Num_RotatableBonds', 'mol2_path', 'reference', '37_flag', 'ALogP_MR']]
tcmbank_ingredient = tcmbank_ingredient[
    [
        "TCMBank_id",
        "level1_name",
        "level1_name_en",
        "level2_name",
        "level2_name_en",
        "TCM_name",
        "TCM_name_en",
        "TCM_name2",
        "Molecule_name",
        "Alias",
        "Molecule_formula",
        "Smiles",
        "Molecule_weight",
        "Molecular_Volume",
        "HERB_id",
        "OB_score",
        "CAS_number",
        "SymMap_id",
        "TCMID_id",
        "TCMSP_id",
        "TCM-ID_id",
        "PubChem_id",
        "DrugBank_id",
        "Molecular_PolarSurfaceArea",
        "Num_H_Acceptors",
        "Num_H_Donors",
        "Num_RotatableBonds",
        "mol2_path",
        "reference",
        "37_flag",
        "ALogP",
        "ALogP_MR",
    ]
]
tcmbank_ingredient["OB_score"] = tcmbank_ingredient["OB_score"].apply(
    lambda x: str(x) + "::TCMBANK" if x else None
)
tcmbank_ingredient["Molecule_weight"] = tcmbank_ingredient["Molecule_weight"].apply(
    lambda x: str(x) + "::TCMBANK" if x else None
)

# concate_database = tcmbank_ingredient.merge(cpmcp_ingredients, on=['SymMap_id', 'Molecule_formula', 'Molecule_name', 'Molecule_weight', 'OB_score', 'PubChem_id'], how='outer')
concate_database = pd.concat([cpmcp_ingredients, tcmbank_ingredient], ignore_index=True)
concate_database.replace([np.nan, "None"], None, inplace=True)
# Because entities are not distinguished based on molecular structure in some databases,
# one record may correspond to multiple Ingredients, and its attributes vary in databases.
concate_database["Molecule_name"] = concate_database["Molecule_name"].apply(
    lambda x: str_clean(x) if x else None
)
concate_database = merge_database_by_id_group(concate_database, "SymMap_id")
ingredient_names = []
# read PharMeBINet
print("reading PharMeBINet database")
pharmebinet_node = pd.read_csv(os.path.join(pharmebinet_root_dir, "ingredient.csv"))
pharmebinet_node["properties"] = pharmebinet_node["properties"].apply(
    lambda x: ast.literal_eval(x)
)


pharmebinet_node.rename(
    columns={
        "node_id": "pharmebinet_id",
        "labels": "pharmebint_labels",
        "name": "Molecule_name",
    },
    inplace=True,
)
pharmebinet_node["Molecule_name"] = pharmebinet_node["Molecule_name"].apply(
    lambda x: str_clean(x)
)
# pharmebinet property
external_properties = extract_properties(
    pharmebinet_node,
    [
        "ctd_url",
        "treeNumbers",
        "parentTreeNumbers",
        "parentIDs",
        "synonyms",
        "cas_number",
        "xrefs",
        "type",
        "groups",
        "inchikey",
        "unii",
        "calculated_properties_kind_value_source",
        "inchi",
    ],
)

external_properties_df = pd.DataFrame(external_properties)
pharmebinet_node = pd.concat([pharmebinet_node, external_properties_df], axis=1)
pharmebinet_node.drop(["properties"], axis=1, inplace=True)
concate_database = concate_database.merge(
    pharmebinet_node, on="Molecule_name", how="left"
)

concate_database["TMDB_id"] = [
    "TMIN{:05d}".format(index + 1) for index in range(concate_database.shape[0])
]

concate_database.replace("nan::TCMBANK", None, inplace=True)
concate_database["OB_score"] = concate_database["OB_score"].apply(
    lambda x: x.replace("nan::TCMBANK;", "").replace(";nan::TCMBANK", "") if x else None
)
concate_database.to_csv(
    os.path.join(merge_result_dir, "entity/ingredient.csv"), index=False
)

concate_database = pd.read_csv(os.path.join(merge_result_dir, "entity/ingredient.csv"))

# origin_ingredient.replace(np.nan, None, inplace=True)
# concate_database.replace(np.nan, None, inplace=True)
# origin_ingredient['CPMCP_id'] = origin_ingredient['CPMCP_id'].apply(lambda x: str(int(x)) if x else None)
# assert(origin_ingredient['CPMCP_id'].equals(concate_database['CPMCP_id']))
# assert(origin_ingredient['pharmebinet_id'].equals(concate_database['pharmebinet_id']))
concate_database.replace(np.nan, None, inplace=True)
concate_database["CPMCP_id"] = concate_database["CPMCP_id"].apply(
    lambda x: str(int(x)) if x else None
)
# ingredient map
cpmcp_ingredient_map = {}
symmap_ingredient_map = {}
TCMBank_ingredient_map = {}
pharmebinet_ingredient_map = {}
for index, row in concate_database.iterrows():
    if row["CPMCP_id"]:
        cpmcp_ingredient_map[row["CPMCP_id"]] = row["TMDB_id"]
    if row["SymMap_id"] and len(row["SymMap_id"].split(";")) == 1:
        symmap_ingredient_map[row["SymMap_id"]] = row["TMDB_id"]
    if row["TCMBank_id"]:
        for tcmbank_id in row["TCMBank_id"].split(";"):
            TCMBank_ingredient_map[tcmbank_id] = row["TMDB_id"]
    if row["pharmebinet_id"]:
        pharmebinet_ingredient_map[str(int(row["pharmebinet_id"]))] = row["TMDB_id"]

herb = pd.read_csv(os.path.join(merge_result_dir, "entity/medicinal_material.csv"))
herb.replace(np.nan, None, inplace=True)
cpmcp_herb_map = {}
symmap_herb_map = {}
TCMBank_herb_map = {}
for index, row in herb.iterrows():
    if isinstance(row["CPMCP_id"], str):
        for cpmcp_id in row["CPMCP_id"].split(";"):
            cpmcp_herb_map[cpmcp_id] = row["TMDB_id"]
    if isinstance(row["SymMap_id"], str):
        for symmap_id in row["SymMap_id"].split(";"):
            symmap_herb_map["SMHB{:05d}".format(int(symmap_id))] = row["TMDB_id"]
    if row["TCMBank_id"]:
        for tcmbank_id in row["TCMBank_id"].split(";"):
            TCMBank_herb_map[tcmbank_id] = row["TMDB_id"]

# read cpmcp herb2ingredient relation
herb2ingredient = []

# cpmcp_relation
cpmcp_herb_data = os.path.join(cpmcp_root_dir, "herb")
for herb_ingredient_path in glob.glob(cpmcp_herb_data + "/*/ingredient.json"):
    cpmcp_herb_id = herb_ingredient_path.split("/")[-2]
    TMDB_herb_id = cpmcp_herb_map[cpmcp_herb_id]
    ingredient_json = json.load(open(herb_ingredient_path))
    for ingredient in ingredient_json["items"]:
        TMDB_ingredient_id = cpmcp_ingredient_map[str(ingredient["id"])]
        herb2ingredient.append((TMDB_herb_id, TMDB_ingredient_id))

# read symmap herb2ingredient relation
symmap_herb_data = os.path.join(symmap_root_dir, "herb")
for herb_ingredient_path in glob.glob(symmap_herb_data + "/*/ingredient.json"):
    symmap_herb_id = herb_ingredient_path.split("/")[-2]
    TMDB_herb_id = symmap_herb_map[symmap_herb_id]
    ingredient_json = json.load(open(herb_ingredient_path))
    for ingredient in ingredient_json["data"]:
        if ingredient["MOL_id"] not in symmap_ingredient_map:
            continue
        TMDB_ingredient_id = symmap_ingredient_map[ingredient["MOL_id"]]
        herb2ingredient.append((TMDB_herb_id, TMDB_ingredient_id))
# read tcmbank herb2ingredient relation
tcmbank_herb_data = os.path.join(tcmbank_root_dir, "herb")
for herb_ingredient_path in glob.glob(tcmbank_herb_data + "/*.json"):
    tcmbank_herb_id = herb_ingredient_path.split("/")[-1][:-5]
    TMDB_herb_id = TCMBank_herb_map[tcmbank_herb_id]
    ingredient_json = json.load(open(herb_ingredient_path))
    for ingredient in ingredient_json["data"]["chart_data"]:
        if (
            ingredient["record_type"] == "Ingredients"
            and ingredient["TCMBank_ID"] in TCMBank_ingredient_map
        ):
            TMDB_ingredient_id = TCMBank_ingredient_map[ingredient["TCMBank_ID"]]
            herb2ingredient.append((TMDB_herb_id, TMDB_ingredient_id))

herb2ingredient_df = pd.DataFrame(
    set(herb2ingredient), columns=["source_id", "target_id"]
)
herb2ingredient_df["Relation_type"] = [
    "herb_consistof_ingredient"
] * herb2ingredient_df.shape[0]
herb2ingredient_df.to_csv(
    os.path.join(merge_result, "relation/herb2ingredient.csv"), index=False
)

ingredient_resemble_ingredient = []
ingredient_interact_ingredient = []

pharmebinet_relations = pd.read_table(os.path.join(pharmebinet_root_dir, "edges.tsv"))
for index, row in pharmebinet_relations[
    pharmebinet_relations.type == "INTERACTS_CiC"
].iterrows():
    origin_ingredient1_id = str(int(row["start_id"]))
    origin_ingredient2_id = str(int(row["end_id"]))
    if (
        origin_ingredient1_id in pharmebinet_ingredient_map
        and origin_ingredient2_id in pharmebinet_ingredient_map
    ):  # 只取了一部分和中医方面重合的ingredients
        pharmebinet_ingredient1_id = pharmebinet_ingredient_map[origin_ingredient1_id]
        pharmebinet_ingredient2_id = pharmebinet_ingredient_map[origin_ingredient2_id]
        ingredient_interact_ingredient.append(
            (pharmebinet_ingredient1_id, pharmebinet_ingredient2_id)
        )

for index, row in pharmebinet_relations[
    pharmebinet_relations.type == "RESEMBLES_CrC"
].iterrows():
    origin_ingredient1_id = str(int(row["start_id"]))
    origin_ingredient2_id = str(int(row["end_id"]))
    if (
        origin_ingredient1_id in pharmebinet_ingredient_map
        and origin_ingredient2_id in pharmebinet_ingredient_map
    ):  # 只取了一部分和中医方面重合的ingredients
        pharmebinet_ingredient1_id = pharmebinet_ingredient_map[origin_ingredient1_id]
        pharmebinet_ingredient2_id = pharmebinet_ingredient_map[origin_ingredient2_id]
        ingredient_resemble_ingredient.append(
            (pharmebinet_ingredient1_id, pharmebinet_ingredient2_id)
        )

ingredient_resemble_ingredient = set(ingredient_resemble_ingredient)
remove_rel = set()
for rel in ingredient_resemble_ingredient:
    if ((rel[1], rel[0]) in ingredient_resemble_ingredient) and (
        (rel[1], rel[0]) not in remove_rel
    ):
        remove_rel.add(rel)
ingredient_resemble_ingredient -= remove_rel

ingredient_resemble_ingredient_df = pd.DataFrame(
    ingredient_resemble_ingredient, columns=["source_id", "target_id"]
)
ingredient_resemble_ingredient_df["Relation_type"] = [
    "ingredient_resemble_ingredient"
] * ingredient_resemble_ingredient_df.shape[0]
ingredient_resemble_ingredient_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_resemble_ingredient.csv"),
    index=False,
)

ingredient_interact_ingredient = set(ingredient_interact_ingredient)
remove_rel = set()
for rel in ingredient_interact_ingredient:
    if ((rel[1], rel[0]) in ingredient_interact_ingredient) and (
        (rel[1], rel[0]) not in remove_rel
    ):
        remove_rel.add(rel)
ingredient_interact_ingredient -= remove_rel

ingredient_interact_ingredient_df = pd.DataFrame(
    set(ingredient_interact_ingredient), columns=["source_id", "target_id"]
)
ingredient_interact_ingredient_df["Relation_type"] = [
    "ingredient_associate_ingredient"
] * ingredient_interact_ingredient_df.shape[0]
ingredient_interact_ingredient_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_associate_ingredient.csv"),
    index=False,
)

pharmacologi_class = pd.read_csv(
    os.path.join(pharmebinet_root_dir, "PharmacologicClass.csv")
)
pharmacologi_class.properties = pharmacologi_class.properties.apply(
    lambda x: ast.literal_eval(x.replace("true", "True"))
)
# {'xrefs', 'class_type', 'synonyms', 'atc_codes' }
pharmacologi_class["xrefs"] = pharmacologi_class.properties.apply(
    lambda x: x["xrefs"] if "xrefs" in x else None
)
pharmacologi_class["class_type"] = pharmacologi_class.properties.apply(
    lambda x: x["class_type"] if "class_type" in x else None
)
pharmacologi_class["synonyms"] = pharmacologi_class.properties.apply(
    lambda x: x["synonyms"] if "synonyms" in x else None
)
pharmacologi_class["atc_codes"] = pharmacologi_class.properties.apply(
    lambda x: x["atc_codes"] if "atc_codes" in x else None
)
pharmacologi_class.rename(
    columns={"node_id": "pharmebinet_id", "name": "Pharmacologi_class_name"},
    inplace=True,
)
pharmacologi_class.drop(
    ["labels", "properties", "resource", "license"], axis=1, inplace=True
)
pharmacologi_class["pharmebinet_id"] = pharmacologi_class["pharmebinet_id"].apply(
    lambda x: str(int(x))
)
pharmacologi_class["TMDB_id"] = [
    "TMPC{:05d}".format(index + 1) for index in range(pharmacologi_class.shape[0])
]
pharmacologi_class.to_csv(
    os.path.join(merge_result_dir, "entity/pharmacologic_class.csv"), index=False
)

pharmacologi_class = pd.read_csv(
    os.path.join(merge_result_dir, "entity/pharmacologic_class.csv")
)
pharmacologi_class.replace(np.nan, None, inplace=True)
pharmacologi_class_id_maps = id_map(pharmacologi_class, ["pharmebinet_id"])
ingredient2pharmacologi_class = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    pharmacologi_class_id_maps["pharmebinet_id"],
    "BELONGS_TO_CHbtPC",
)
pharmacologi_class2ingredient = extract_specific_relation(
    pharmebinet_relations,
    pharmacologi_class_id_maps["pharmebinet_id"],
    pharmebinet_ingredient_map,
    "INCLUDES_PCiCH",
)
ingredient2pharmacologi_class.extend(
    [(rel[1], rel[0]) for rel in pharmacologi_class2ingredient]
)

ingredient2pharmacologi_class_df = pd.DataFrame(
    set(ingredient2pharmacologi_class), columns=["source_id", "target_id"]
)
ingredient2pharmacologi_class_df["Relation_type"] = [
    "ingredient_belong_to_pharmacologic_class"
] * ingredient2pharmacologi_class_df.shape[0]
ingredient2pharmacologi_class_df.to_csv(
    os.path.join(
        merge_result_dir, "relation/ingredient_belong_to_pharmacologic_class.csv"
    ),
    index=False,
)

cell_component = pd.read_csv(
    os.path.join(merge_result_dir, "entity/cellular_component.csv")
)
cell_component.replace(np.nan, None, inplace=True)
cell_component_id_maps = id_map(cell_component, ["pharmebinet_id"])

biological_process = pd.read_csv(
    os.path.join(merge_result_dir, "entity/biological_process.csv")
)
biological_process.replace(np.nan, None, inplace=True)
biological_process_id_maps = id_map(biological_process, ["pharmebinet_id"])

molecular_function = pd.read_csv(
    os.path.join(merge_result_dir, "entity/molecular_function.csv")
)
molecular_function.replace(np.nan, None, inplace=True)
molecular_function_id_maps = id_map(molecular_function, ["pharmebinet_id"])

disease = pd.read_csv(os.path.join(merge_result_dir, "entity/disease.csv"))
disease.replace(np.nan, None, inplace=True)
disease_id_maps = id_map(disease, ["pharmebinet_id"])

pathway = pd.read_csv(os.path.join(merge_result_dir, "entity/pathway.csv"))
pathway.replace(np.nan, None, inplace=True)
pathway_id_maps = id_map(pathway, ["pharmebinet_id"])


# extract relation

INDUCES_CHiD = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    disease_id_maps["pharmebinet_id"],
    "INDUCES_CHiD",
)

TREATS_CHtD = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    disease_id_maps["pharmebinet_id"],
    "TREATS_CHtD",
)
CONTRAINDICATES_CHcD = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    disease_id_maps["pharmebinet_id"],
    "CONTRAINDICATES_CHcD",
)


conflict_ingredient_disease = set(INDUCES_CHiD) & set(TREATS_CHtD)
conflict_ingredient_disease.update(set(CONTRAINDICATES_CHcD) & set(TREATS_CHtD))

ingredient_induce_disease_df = pd.DataFrame(
    set(INDUCES_CHiD) - conflict_ingredient_disease, columns=["source_id", "target_id"]
)
ingredient_induce_disease_df["Relation_type"] = [
    "ingredient_induce_disease"
] * ingredient_induce_disease_df.shape[0]
ingredient_induce_disease_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_induce_disease.csv"),
    index=False,
)

ingredient_treat_disease_df = pd.DataFrame(
    set(TREATS_CHtD) - conflict_ingredient_disease, columns=["source_id", "target_id"]
)
ingredient_treat_disease_df["Relation_type"] = [
    "ingredient_treat_disease"
] * ingredient_treat_disease_df.shape[0]
ingredient_treat_disease_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_treat_disease.csv"), index=False
)

ingredient_contraindicate_disease_df = pd.DataFrame(
    set(CONTRAINDICATES_CHcD) - conflict_ingredient_disease,
    columns=["source_id", "target_id"],
)
ingredient_contraindicate_disease_df["Relation_type"] = [
    "ingredient_contraindicate_disease"
] * ingredient_contraindicate_disease_df.shape[0]
ingredient_contraindicate_disease_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_contraindicate_disease.csv"),
    index=False,
)

ingredient_associate_biological_process = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    biological_process_id_maps["pharmebinet_id"],
    "ASSOCIATES_CHaBP",
)

ingredient_associate_biological_process.extend(
    extract_specific_relation(
        pharmebinet_relations,
        pharmebinet_ingredient_map,
        biological_process_id_maps["pharmebinet_id"],
        "INCREASES_CHiBP",
    )
)
ingredient_associate_biological_process.extend(
    extract_specific_relation(
        pharmebinet_relations,
        pharmebinet_ingredient_map,
        biological_process_id_maps["pharmebinet_id"],
        "DECREASES_CHdBP",
    )
)

ingredient_associate_biological_process_df = pd.DataFrame(
    set(ingredient_associate_biological_process), columns=["source_id", "target_id"]
)
ingredient_associate_biological_process_df["Relation_type"] = [
    "ingredient_associate_biological_process"
] * ingredient_associate_biological_process_df.shape[0]
ingredient_associate_biological_process_df.to_csv(
    os.path.join(
        merge_result_dir, "relation/ingredient_associate_biological_process.csv"
    ),
    index=False,
)

ingredient_associate_cellular_component = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    cell_component_id_maps["pharmebinet_id"],
    "ASSOCIATES_CHaCC",
)
ingredient_associate_cellular_component.extend(
    extract_specific_relation(
        pharmebinet_relations,
        pharmebinet_ingredient_map,
        cell_component_id_maps["pharmebinet_id"],
        "INCREASES_CHiCC",
    )
)
ingredient_associate_cellular_component.extend(
    extract_specific_relation(
        pharmebinet_relations,
        pharmebinet_ingredient_map,
        cell_component_id_maps["pharmebinet_id"],
        "DECREASES_CHdCC",
    )
)
ingredient_associate_cellular_component_df = pd.DataFrame(
    set(ingredient_associate_cellular_component), columns=["source_id", "target_id"]
)
ingredient_associate_cellular_component_df["Relation_type"] = [
    "ingredient_associate_cellular_component"
] * ingredient_associate_cellular_component_df.shape[0]
ingredient_associate_cellular_component_df.to_csv(
    os.path.join(
        merge_result_dir, "relation/ingredient_associate_cellular_component.csv"
    ),
    index=False,
)


ingredient_associate_molecular_function = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    molecular_function_id_maps["pharmebinet_id"],
    "ASSOCIATES_CHaMF",
)

ingredient_associate_molecular_function.extend(
    extract_specific_relation(
        pharmebinet_relations,
        pharmebinet_ingredient_map,
        molecular_function_id_maps["pharmebinet_id"],
        "DECREASES_CHdMF",
    )
)

ingredient_associate_molecular_function.extend(
    extract_specific_relation(
        pharmebinet_relations,
        pharmebinet_ingredient_map,
        molecular_function_id_maps["pharmebinet_id"],
        "INCREASES_CHiMF",
    )
)

ingredient_associate_molecular_function_df = pd.DataFrame(
    set(ingredient_associate_molecular_function), columns=["source_id", "target_id"]
)
ingredient_associate_molecular_function_df["Relation_type"] = [
    "ingredient_associate_molecular_function"
] * ingredient_associate_molecular_function_df.shape[0]
ingredient_associate_molecular_function_df.to_csv(
    os.path.join(
        merge_result_dir, "relation/ingredient_associate_molecular_function.csv"
    ),
    index=False,
)


ingredient_associate_pathway = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    pathway_id_maps["pharmebinet_id"],
    "ASSOCIATES_CaPW",
)
ingredient_associate_pathway_df = pd.DataFrame(
    set(ingredient_associate_pathway), columns=["source_id", "target_id"]
)
ingredient_associate_pathway_df["Relation_type"] = [
    "ingredient_associate_pathway"
] * ingredient_associate_pathway_df.shape[0]
ingredient_associate_pathway_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_associate_pathway.csv"),
    index=False,
)


sideeffect = pd.read_csv(os.path.join(merge_result_dir, "entity/sideeffect.csv"))
sideeffect.replace(np.nan, None, inplace=True)
sideeffect_id_maps = id_map(sideeffect, ["pharmebinet_id"])
pharmebinet_side_effect_map = sideeffect_id_maps["pharmebinet_id"]


ingredient_cause_sideeffect = []
ingredient_might_cause_sideeffect = []

for index, row in pharmebinet_relations[
    pharmebinet_relations.type == "CAUSES_CHcSE"
].iterrows():
    if (
        str(int(row["start_id"])) in pharmebinet_ingredient_map
    ):  # 只取了一部分和中医方面重合的ingredients
        pharmebinet_ingredient_id = pharmebinet_ingredient_map[
            str(int(row["start_id"]))
        ]
        pharmebinet_sideeffect_id = pharmebinet_side_effect_map[str(int(row["end_id"]))]
        ingredient_cause_sideeffect.append(
            (pharmebinet_ingredient_id, pharmebinet_sideeffect_id)
        )

for index, row in pharmebinet_relations[
    pharmebinet_relations.type == "MIGHT_CAUSES_CHmcSE"
].iterrows():
    if (
        str(int(row["start_id"])) in pharmebinet_ingredient_map
    ):  # 只取了一部分和中医方面重合的ingredients
        pharmebinet_ingredient_id = pharmebinet_ingredient_map[
            str(int(row["start_id"]))
        ]
        pharmebinet_sideeffect_id = pharmebinet_side_effect_map[str(int(row["end_id"]))]
        ingredient_might_cause_sideeffect.append(
            (pharmebinet_ingredient_id, pharmebinet_sideeffect_id)
        )

ingredient_cause_sideeffect_df = pd.DataFrame(
    set(ingredient_cause_sideeffect), columns=["source_id", "target_id"]
)
ingredient_might_cause_sideeffect_df = pd.DataFrame(
    set(ingredient_might_cause_sideeffect), columns=["source_id", "target_id"]
)
ingredient_cause_sideeffect_df["Relation_type"] = [
    "ingredient_cause_sideeffect"
] * ingredient_cause_sideeffect_df.shape[0]
ingredient_might_cause_sideeffect_df["Relation_type"] = [
    "ingredient_might_cause_sideeffect"
] * ingredient_might_cause_sideeffect_df.shape[0]

ingredient_cause_sideeffect_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_cause_sideeffect.csv"),
    index=False,
)
ingredient_might_cause_sideeffect_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_might_cause_sideeffect.csv"),
    index=False,
)
