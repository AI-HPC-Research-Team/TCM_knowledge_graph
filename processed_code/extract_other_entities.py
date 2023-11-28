import pandas as pd
import json
import os
import ast
import sys
import re
from merge_utility import merge_database_by_id, merge_database_by_id_group
from merge_utility import extract_external_ids, extract_properties
import glob
import math
import numpy as np


primekg_root_dir = "../data/primeKG"
pharmebinet_root_dir = "../data/PharMeBINet"
merge_result_dir = "../merge_result"
# ---------------Cellular component----------------------

primekg_cell_comp = pd.read_csv(
    os.path.join(primekg_root_dir, "cellular_component.csv")
)
primekg_cell_comp.rename(
    columns={
        "node_index": "PrimeKG_id",
        "node_id": "GO_id",
        "node_name": "Cellular_component_name",
    },
    inplace=True,
)
primekg_cell_comp["GO_id"] = primekg_cell_comp["GO_id"].apply(
    lambda x: "GO:{:07d}".format(x)
)
primekg_cell_comp.drop(["node_type", "node_source"], inplace=True, axis=1)
primekg_cell_comp["PrimeKG_id"] = primekg_cell_comp["PrimeKG_id"].apply(
    lambda x: str(int(x))
)

pharmebinet_cell_comp = pd.read_csv(
    os.path.join(pharmebinet_root_dir, "CellularComponent.csv")
)
pharmebinet_cell_comp["properties"] = pharmebinet_cell_comp["properties"].apply(
    lambda x: ast.literal_eval(x.replace("true", "True"))
)

database_name = ["EC", "NIF_Subcellular"]
external_ids = extract_external_ids(pharmebinet_cell_comp, database_name)
external_ids_df = pd.DataFrame(external_ids)
columns_name = dict(zip(database_name, [item + "_id" for item in database_name]))
external_ids_df.rename(columns=columns_name, inplace=True)
pharmebinet_cell_comp = pd.concat([pharmebinet_cell_comp, external_ids_df], axis=1)

external_properties = extract_properties(
    pharmebinet_cell_comp,
    [
        "definition",
        "related_synonyms",
        "url_ctd",
        "subsets",
        "comment",
        "highestGOLevel",
        "broad_synonyms",
        "narrow_synonyms",
        "alternative_ids",
        "synonyms",
    ],
)
external_properties_df = pd.DataFrame(external_properties)

pharmebinet_cell_comp = pd.concat(
    [pharmebinet_cell_comp, external_properties_df], axis=1
)
pharmebinet_cell_comp.rename(
    columns={
        "node_id": "pharmebinet_id",
        "name": "Cellular_component_name",
        "identifier": "GO_id",
    },
    inplace=True,
)

pharmebinet_cell_comp["pharmebinet_id"] = pharmebinet_cell_comp["pharmebinet_id"].apply(
    lambda x: str(int(x))
)
pharmebinet_cell_comp.drop(["labels", "properties", "resource"], axis=1, inplace=True)

concate_cell_comp = pd.concat(
    [primekg_cell_comp, pharmebinet_cell_comp], ignore_index=True
)
concate_cell_comp.replace(np.nan, None, inplace=True)
concate_cell_comp = merge_database_by_id_group(concate_cell_comp, "GO_id")
concate_cell_comp["TMDB_id"] = [
    "TMCC{:05d}".format(index + 1) for index in range(concate_cell_comp.shape[0])
]
concate_cell_comp.to_csv(
    os.path.join(merge_result, "entity/cellular_component.csv"), index=False
)

# --------------------Biological Process-------------------

primekg_bio_process = pd.read_csv(
    os.path.join(primekg_root_dir, "biological_process.csv")
)
primekg_bio_process.rename(
    columns={
        "node_index": "PrimeKG_id",
        "node_id": "GO_id",
        "node_name": "Biological_process_name",
    },
    inplace=True,
)
primekg_bio_process["GO_id"] = primekg_bio_process["GO_id"].apply(
    lambda x: "GO:{:07d}".format(x)
)
primekg_bio_process.drop(["node_type", "node_source"], inplace=True, axis=1)
primekg_bio_process["PrimeKG_id"] = primekg_bio_process["PrimeKG_id"].apply(
    lambda x: str(int(x))
)

pharmebinet_bio_process = pd.read_csv(
    os.path.join(pharmebinet_root_dir, "BiologicalProcess.csv")
)
pharmebinet_bio_process["properties"] = pharmebinet_bio_process["properties"].apply(
    lambda x: ast.literal_eval(x.replace("true", "True"))
)


database_name = ["UM-BBD_pathwayID", "VZ", "Wikipedia", "MetaCyc", "RESID"]
external_ids = extract_external_ids(pharmebinet_bio_process, database_name)
external_ids_df = pd.DataFrame(external_ids)
columns_name = dict(zip(database_name, [item + "_id" for item in database_name]))
external_ids_df.rename(columns=columns_name, inplace=True)
pharmebinet_bio_process = pd.concat([pharmebinet_bio_process, external_ids_df], axis=1)

external_properties = extract_properties(
    pharmebinet_bio_process,
    [
        "definition",
        "subsets",
        "comment",
        "related_synonyms",
        "url_ctd",
        "broad_synonyms",
        "alternative_ids",
        "narrow_synonyms",
        "synonyms",
        "highestGOLevel",
    ],
)
external_properties_df = pd.DataFrame(external_properties)

pharmebinet_bio_process = pd.concat(
    [pharmebinet_bio_process, external_properties_df], axis=1
)
pharmebinet_bio_process.rename(
    columns={
        "node_id": "pharmebinet_id",
        "name": "Biological_process_name",
        "identifier": "GO_id",
    },
    inplace=True,
)

pharmebinet_bio_process["pharmebinet_id"] = pharmebinet_bio_process[
    "pharmebinet_id"
].apply(lambda x: str(int(x)))
pharmebinet_bio_process.drop(["labels", "properties", "resource"], axis=1, inplace=True)

concate_bio_process = pd.concat(
    [primekg_bio_process, pharmebinet_bio_process], ignore_index=True
)
concate_bio_process.replace(np.nan, None, inplace=True)
concate_bio_process = merge_database_by_id_group(concate_bio_process, "GO_id")
concate_bio_process["TMDB_id"] = [
    "TMBP{:05d}".format(index + 1) for index in range(concate_bio_process.shape[0])
]
concate_bio_process.to_csv(
    os.path.join(merge_result, "entity/biological_process.csv"), index=False
)

# --------------------Molecular Function--------------------

primekg_mol_function = pd.read_csv(
    os.path.join(primekg_root_dir, "molecular_function.csv")
)
primekg_mol_function.rename(
    columns={
        "node_index": "PrimeKG_id",
        "node_id": "GO_id",
        "node_name": "Molecular_function_name",
    },
    inplace=True,
)
primekg_mol_function["GO_id"] = primekg_mol_function["GO_id"].apply(
    lambda x: "GO:{:07d}".format(x)
)
primekg_mol_function.drop(["node_type", "node_source"], inplace=True, axis=1)
primekg_mol_function["PrimeKG_id"] = primekg_mol_function["PrimeKG_id"].apply(
    lambda x: str(int(x))
)

pharmebinet_mol_function = pd.read_csv(
    os.path.join(pharmebinet_root_dir, "MolecularFunction.csv")
)
pharmebinet_mol_function["properties"] = pharmebinet_mol_function["properties"].apply(
    lambda x: ast.literal_eval(x.replace("true", "True"))
)

database_name = [
    "KEGG_REACTION",
    "TC",
    "Reactome",
    "Wikipedia",
    "RHEA",
    "EC",
    "MetaCyc",
    "UM-BBD_reactionID",
    "UM-BBD_enzymeID",
]
external_ids = extract_external_ids(pharmebinet_mol_function, database_name)
external_ids_df = pd.DataFrame(external_ids)
columns_name = dict(zip(database_name, [item + "_id" for item in database_name]))
external_ids_df.rename(columns=columns_name, inplace=True)
pharmebinet_mol_function = pd.concat(
    [pharmebinet_mol_function, external_ids_df], axis=1
)

external_properties = extract_properties(
    pharmebinet_mol_function,
    [
        "definition",
        "subsets",
        "comment",
        "url_ctd",
        "broad_synonyms",
        "synonyms",
        "alternative_ids",
        "narrow_synonyms",
        "related_synonyms",
        "highestGOLevel",
    ],
)
external_properties_df = pd.DataFrame(external_properties)

pharmebinet_mol_function = pd.concat(
    [pharmebinet_mol_function, external_properties_df], axis=1
)
pharmebinet_mol_function.rename(
    columns={
        "node_id": "pharmebinet_id",
        "name": "Molecular_function_name",
        "identifier": "GO_id",
    },
    inplace=True,
)

pharmebinet_mol_function["pharmebinet_id"] = pharmebinet_mol_function[
    "pharmebinet_id"
].apply(lambda x: str(int(x)))
pharmebinet_mol_function.drop(
    ["labels", "properties", "resource"], axis=1, inplace=True
)

concate_mol_function = pd.concat(
    [primekg_mol_function, pharmebinet_mol_function], ignore_index=True
)
concate_mol_function.replace(np.nan, None, inplace=True)
concate_mol_function = merge_database_by_id_group(concate_mol_function, "GO_id")
concate_mol_function["TMDB_id"] = [
    "TMMF{:05d}".format(index + 1) for index in range(concate_mol_function.shape[0])
]
concate_mol_function.to_csv(
    os.path.join(merge_result, "entity/molecular_function.csv"), index=False
)


# ---------------Anatomy-----------------
primekg_anatomy = pd.read_csv(os.path.join(primekg_root_dir, "anatomy.csv"))
primekg_anatomy.rename(
    columns={
        "node_index": "PrimeKG_id",
        "node_id": "UBERON_id",
        "node_name": "Anatomy_name",
    },
    inplace=True,
)
primekg_anatomy["UBERON_id"] = primekg_anatomy["UBERON_id"].apply(
    lambda x: "UBERON:{:07d}".format(x)
)
primekg_anatomy.drop(["node_type", "node_source"], inplace=True, axis=1)
primekg_anatomy["PrimeKG_id"] = primekg_anatomy["PrimeKG_id"].apply(
    lambda x: str(int(x))
)

pharmebinet_anatomy = pd.read_csv(os.path.join(pharmebinet_root_dir, "anatomy.csv"))
pharmebinet_anatomy["properties"] = pharmebinet_anatomy["properties"].apply(
    lambda x: ast.literal_eval(x.replace("true", "True"))
)
database_name = ["MeSH", "BTO"]
external_ids = extract_external_ids(pharmebinet_anatomy, database_name)
external_ids_df = pd.DataFrame(external_ids)
columns_name = dict(zip(database_name, [item + "_id" for item in database_name]))
external_ids_df.rename(columns=columns_name, inplace=True)
pharmebinet_anatomy = pd.concat([pharmebinet_anatomy, external_ids_df], axis=1)
pharmebinet_anatomy.rename(
    columns={
        "node_id": "pharmebinet_id",
        "name": "Anatomy_name",
        "identifier": "UBERON_id",
    },
    inplace=True,
)

pharmebinet_anatomy["pharmebinet_id"] = pharmebinet_anatomy["pharmebinet_id"].apply(
    lambda x: str(int(x))
)
pharmebinet_anatomy.drop(["labels", "properties", "resource"], axis=1, inplace=True)

concate_anatomy = pd.concat([primekg_anatomy, pharmebinet_anatomy], ignore_index=True)
concate_anatomy.replace(np.nan, None, inplace=True)
concate_anatomy = merge_database_by_id_group(concate_anatomy, "UBERON_id")
concate_anatomy["TMDB_id"] = [
    "TMAT{:05d}".format(index + 1) for index in range(concate_anatomy.shape[0])
]
concate_anatomy.to_csv(os.path.join(merge_result, "entity/anatomy.csv"), index=False)


# ---------------Pathway-----------------

primekg_pathway = pd.read_csv(os.path.join(primekg_root_dir, "pathway.csv"))
primekg_pathway.rename(
    columns={
        "node_index": "PrimeKG_id",
        "node_id": "reactome_id",
        "node_name": "Pathway_name",
    },
    inplace=True,
)
primekg_pathway.drop(["node_type", "node_source"], inplace=True, axis=1)
primekg_pathway["PrimeKG_id"] = primekg_pathway["PrimeKG_id"].apply(
    lambda x: str(int(x))
)

pharmebinet_pathway = pd.read_csv(os.path.join(pharmebinet_root_dir, "pathway.csv"))
pharmebinet_pathway["properties"] = pharmebinet_pathway["properties"].apply(
    lambda x: ast.literal_eval(x.replace("true", "True"))
)

# 'pathbank', 'reactome', 'panther', 'netpath', 'wikipathways'
external_ids = extract_external_ids(
    pharmebinet_pathway, ["pathbank", "reactome", "panther", "netpath", "wikipathways"]
)
external_ids_df = pd.DataFrame(external_ids)
external_ids_df.rename(
    columns={
        "pathbank": "pathbank_id",
        "reactome": "reactome_id",
        "panther": "panther_id",
        "netpath": "netpath_id",
        "wikipathways": "wikipathways_id",
    },
    inplace=True,
)
pharmebinet_pathway = pd.concat([pharmebinet_pathway, external_ids_df], axis=1)
external_properties = extract_properties(
    pharmebinet_pathway,
    [
        "doi",
        "figure_urls",
        "all_mapped_ids",
        "ctd_url",
        "alternative_id",
        "definition",
        "synonyms",
        "pubMed_ids",
        "publication_urls",
        "books",
    ],
)
external_properties_df = pd.DataFrame(external_properties)
pharmebinet_pathway = pd.concat([pharmebinet_pathway, external_properties_df], axis=1)
pharmebinet_pathway.rename(
    columns={"node_id": "pharmebinet_id", "name": "Pathway_name"}, inplace=True
)
pharmebinet_pathway.drop(["labels", "properties", "resource"], axis=1, inplace=True)
pharmebinet_pathway["pharmebinet_id"] = pharmebinet_pathway["pharmebinet_id"].apply(
    lambda x: str(int(x))
)
concate_pathway = pd.concat([primekg_pathway, pharmebinet_pathway], ignore_index=True)
concate_pathway.replace(np.nan, None, inplace=True)
concate_pathway = merge_database_by_id_group(concate_pathway, "reactome_id")
concate_pathway["TMDB_id"] = [
    "TMPW{:05d}".format(index + 1) for index in range(concate_pathway.shape[0])
]
concate_pathway.to_csv(os.path.join(merge_result, "entity/pathway.csv"), index=False)


# ---------------Pharmacologic Class-----------------
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


# ---------------Side Effect-----------------
primekg_side_effect = pd.read_csv(
    os.path.join(primekg_root_dir, "effect_phenotype.csv")
)
primekg_side_effect.rename(
    columns={
        "node_index": "PrimeKG_id",
        "node_id": "HPO_id",
        "node_name": "side_effect_name",
    },
    inplace=True,
)
primekg_side_effect["HPO_id"] = primekg_side_effect["HPO_id"].apply(
    lambda x: "HP:{:07d}".format(x)
)
primekg_side_effect.drop(["node_type", "node_source"], axis=1, inplace=True)
primekg_side_effect["PrimeKG_id"] = primekg_side_effect["PrimeKG_id"].apply(
    lambda x: str(int(x))
)
pharmebinet_side_effect = pd.read_csv(
    os.path.join(pharmebinet_root_dir, "sideeffect.csv")
)

pharmebinet_side_effect["properties"] = pharmebinet_side_effect["properties"].apply(
    lambda x: ast.literal_eval(x.replace("true", "True"))
)
external_ids = extract_external_ids(
    pharmebinet_side_effect,
    ["OMIM", "MESH", "Orphanet", "UMLS", "HPO", "DOID", "MonDO"],
)
pharmebinet_side_effect["OMIM_id"] = external_ids["OMIM"]
pharmebinet_side_effect["MESH_id"] = external_ids["MESH"]
pharmebinet_side_effect["Orphanet_id"] = external_ids["Orphanet"]
pharmebinet_side_effect["HPO_id"] = external_ids["HPO"]
pharmebinet_side_effect["DOID_id"] = external_ids["DOID"]
pharmebinet_side_effect["MonDO_id"] = external_ids["MonDO"]
pharmebinet_side_effect["UMLS_id"] = external_ids["UMLS"]
pharmebinet_side_effect.rename(
    columns={"node_id": "pharmebinet_id", "name": "side_effect_name"}, inplace=True
)
pharmebinet_side_effect["pharmebinet_id"] = pharmebinet_side_effect[
    "pharmebinet_id"
].apply(lambda x: str(int(x)))
pharmebinet_side_effect.drop(
    ["labels", "identifier", "resource", "source", "properties"], axis=1, inplace=True
)
concate_sideeffect = pd.concat(
    [primekg_side_effect, pharmebinet_side_effect], ignore_index=True
)
concate_sideeffect.replace(np.nan, None, inplace=True)
concate_sideeffect = merge_database_by_id(concate_sideeffect, "HPO_id")
concate_sideeffect = merge_database_by_id(concate_sideeffect, "OMIM_id")
concate_sideeffect = merge_database_by_id(concate_sideeffect, "Orphanet_id")
concate_sideeffect = merge_database_by_id(concate_sideeffect, "DOID_id")
concate_sideeffect = merge_database_by_id(concate_sideeffect, "MonDO_id")
concate_sideeffect = merge_database_by_id(concate_sideeffect, "UMLS_id")
concate_sideeffect["TMDB_id"] = [
    "TMSE{:5d}".format(index + 1) for index in range(concate_sideeffect.shape[0])
]
concate_sideeffect.to_csv(
    os.path.join(merge_result_dir, "entity/sideeffect.csv"), index=False
)
