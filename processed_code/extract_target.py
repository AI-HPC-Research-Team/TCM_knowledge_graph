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


cpmcp_root_dir = "../data/CPMCP"
symmap_root_dir = "../data/symmap"
tcmbank_root_dir = "../data/TCMBANK"
primekg_root_dir = "../data/primeKG"
pharmebinet_root_dir = "../data/PharMeBINet"
merge_result_dir = "../merge_result"

# merge two target with the same (gene symbol/symmap if/NCBI id)

# collect target info from cpmcp database
cpmcp_targets = []
for file_path in glob.glob(os.path.join(cpmcp_root_dir, "target/*/[0-9]*.json")):
    res_json = json.load(open(file_path))
    cpmcp_targets.append(res_json)
cpmcp_targets = pd.DataFrame(cpmcp_targets)

external_ids = pd.DataFrame(cpmcp_targets["externals"].tolist())
external_ids["Ensembl id"] = external_ids["Ensembl id"].apply(
    lambda x: x if x != [] else None
)
external_ids["GenBank gene id"] = external_ids["GenBank gene id"].apply(
    lambda x: x if x != [] else None
)
external_ids["GenBank protein id"] = external_ids["GenBank protein id"].apply(
    lambda x: x if x != [] else None
)
external_ids["HGNC id"] = external_ids["HGNC id"].apply(
    lambda x: x if x != [] else None
)
external_ids["NCBI id"] = external_ids["NCBI id"].apply(
    lambda x: x[0] if x != [] else None
)
external_ids["OMIM id"] = external_ids["OMIM id"].apply(
    lambda x: x if x != [] else None
)
external_ids["SymMap id"] = external_ids["SymMap id"].apply(
    lambda x: x[0] if x != [] else None
)
external_ids["UniProt id"] = external_ids["UniProt id"].apply(
    lambda x: x if x != [] else None
)
external_ids["Vega id"] = external_ids["Vega id"].apply(
    lambda x: x if x != [] else None
)
database_name = [
    "Ensembl id",
    "GenBank gene id",
    "GenBank protein id",
    "HGNC id",
    "NCBI id",
    "OMIM id",
    "SymMap id",
    "UniProt id",
    "Vega id",
]
columns_name = dict(
    zip(database_name, [item.replace(" ", "_") for item in database_name])
)
external_ids.rename(columns=columns_name, inplace=True)
# 以基因名为主
cpmcp_targets = pd.concat([cpmcp_targets, external_ids], axis=1)
cpmcp_targets.drop(
    ["_show", "suppress", "externals", "symmap_id"], axis=1, inplace=True
)
cpmcp_targets.rename(
    columns={
        "chromosome": "Chromosome",
        "gene_name": "Gene_name",
        "protein_name": "Protein_name",
        "symbol": "Gene_symbol",
        "GenBank_gene_id": "GenBank_Gene_id",
        "GenBank_protein_id": "GenBank_Protein_id",
        "id": "CPMCP_id",
    },
    inplace=True,
)
cpmcp_targets["CPMCP_id"] = cpmcp_targets["CPMCP_id"].apply(
    lambda x: str(int(x)) if x else None
)
cpmcp_targets.NCBI_id = cpmcp_targets.NCBI_id.apply(
    lambda x: x.split(".")[0] if x else None
)

symmap_targets = pd.read_excel(os.path.join(symmap_root_dir, "Symmap target.xlsx"))
symmap_targets.replace(np.nan, None, inplace=True)
symmap_targets.rename(
    columns={"MIM_id": "OMIM_id", "Gene_id": "SymMap_id"}, inplace=True
)
symmap_targets["SymMap_id"] = symmap_targets["SymMap_id"].apply(
    lambda x: "SMTT{:05d}".format(x)
)
symmap_targets.NCBI_id = symmap_targets.NCBI_id.apply(
    lambda x: str(int(x)) if x else None
)
symmap_targets.drop(["Suppress", "Version"], axis=1, inplace=True)

concate_database = pd.concat([cpmcp_targets, symmap_targets], ignore_index=True)
concate_database.replace(np.nan, None, inplace=True)
concate_database = merge_database_by_id_group(concate_database, "SymMap_id")

# tcmbank
tcmbank_target = pd.read_csv(os.path.join(tcmbank_root_dir, "target_all.csv"))
# 'TCMBank_ID', 'Gene_name', 'Gene_alias', 'Description', 'Map_location',
# 'Type_of_gene', 'TTD_gene_type'
tcmbank_target.replace(["/", np.nan], None, inplace=True)
tcmbank_target.rename(
    columns={"Gene_name": "Gene_symbol", "TCMBank_ID": "TCMBank_id"}, inplace=True
)

concate_database = pd.concat([concate_database, tcmbank_target], ignore_index=True)
concate_database.replace(np.nan, None, inplace=True)
concate_database = merge_database_by_id_group(concate_database, "Gene_symbol")

primekg_target = pd.read_csv(os.path.join(primekg_root_dir, "gene_protein.csv"))
primekg_target.rename(
    columns={
        "node_index": "PrimeKG_id",
        "node_id": "NCBI_id",
        "node_name": "Gene_symbol",
    },
    inplace=True,
)

primekg_target["PrimeKG_id"] = primekg_target["PrimeKG_id"].apply(lambda x: str(int(x)))
primekg_target["NCBI_id"] = primekg_target["NCBI_id"].apply(lambda x: str(int(x)))
primekg_target.drop(["node_type", "node_source"], inplace=True, axis=1)
concate_database = pd.concat([concate_database, primekg_target], ignore_index=True)
concate_database.replace(np.nan, None, inplace=True)
concate_database = merge_database_by_id_group(concate_database, "NCBI_id")
concate_database = merge_database_by_id_group(concate_database, "Gene_symbol")


pharmebinet_gene = pd.read_csv(os.path.join(pharmebinet_root_dir, "gene.csv"))
pharmebinet_gene["properties"] = pharmebinet_gene["properties"].apply(
    lambda x: ast.literal_eval(x.replace("true", "True"))
)

external_ids = extract_external_ids(pharmebinet_gene, ["OMIM", "PharmGKB", "bioGRID"])
external_ids_df = pd.DataFrame(external_ids)
external_ids_df.rename(
    columns={"OMIM": "OMIM_id", "PharmGKB": "PharmGKB_id", "bioGRID": "bioGRID_id"},
    inplace=True,
)
property_set = [
    "synonyms",
    "type_of_gene",
    "xrefs",
    "feature_type",
    "description",
    "map_location",
    "gene_symbols",
    "chromosome",
]
external_properties = extract_properties(pharmebinet_gene, property_set)
external_properties_df = pd.DataFrame(external_properties)

pharmebinet_gene = pd.concat(
    [pharmebinet_gene, external_properties_df, external_ids_df], axis=1
)
pharmebinet_gene.rename(
    columns={
        "node_id": "pharmebinet_id",
        "name": "Gene_name",
        "identifier": "NCBI_id",
        "type_of_gene": "Type_of_gene",
        "feature_type": "Feature_type",
        "map_location": "Map_location",
        "gene_symbols": "Gene_symbol",
        "chromosome": "Chromosome",
    },
    inplace=True,
)
pharmebinet_gene.drop(["labels", "properties", "resource"], axis=1, inplace=True)
pharmebinet_gene.Gene_symbol = pharmebinet_gene.Gene_symbol.apply(
    lambda x: x[0] if x else None
)
pharmebinet_gene.pharmebinet_id = pharmebinet_gene.pharmebinet_id.apply(
    lambda x: str(int(x))
)
pharmebinet_gene.NCBI_id = pharmebinet_gene.NCBI_id.apply(
    lambda x: str(int(x)) if x else None
)
pharmebinet_gene.OMIM_id = pharmebinet_gene.OMIM_id.apply(
    lambda x: str(int(x)) if x else None
)
concate_database = pd.concat([pharmebinet_gene, concate_database], ignore_index=True)

concate_database.replace(np.nan, None, inplace=True)
concate_database = merge_database_by_id(concate_database, "Gene_symbol")
concate_database = merge_database_by_id(concate_database, "NCBI_id")
concate_database["TMDB_id"] = [
    "TMGE{:05d}".format(index + 1) for index in range(concate_database.shape[0])
]
concate_database.to_csv(os.path.join(merge_result_dir, "entity/gene.csv"), index=False)

concate_database = pd.read_csv(os.path.join(merge_result_dir, "entity/gene.csv"))
concate_database.replace(np.nan, None, inplace=True)
gene_id_maps = id_map(
    concate_database,
    ["pharmebinet_id", "PrimeKG_id", "CPMCP_id", "SymMap_id", "TCMBank_id"],
)

cpmcp_gene_map = gene_id_maps["CPMCP_id"]
symmap_gene_map = gene_id_maps["SymMap_id"]
TCMBank_gene_map = gene_id_maps["TCMBank_id"]
pharmebinet_gene_map = gene_id_maps["pharmebinet_id"]
primekg_gene_map = gene_id_maps["PrimeKG_id"]


ingredient = pd.read_csv(os.path.join(merge_result_dir, "entity/ingredient.csv"))
ingredient.replace(np.nan, None, inplace=True)
cpmcp_ingredient_map = {}
symmap_ingredient_map = {}
TCMBank_ingredient_map = {}
pharmebinet_ingredient_map = {}


for index, row in ingredient.iterrows():
    if row["CPMCP_id"]:
        cpmcp_ingredient_map[row["CPMCP_id"]] = row["TMDB_id"]
    if row["SymMap_id"] and len(row["SymMap_id"].split(";")) == 1:
        symmap_ingredient_map[row["SymMap_id"]] = row["TMDB_id"]
    if row["TCMBank_id"]:
        for tcmbank_id in row["TCMBank_id"].split(";"):
            TCMBank_ingredient_map[tcmbank_id] = row["TMDB_id"]
    if row["pharmebinet_id"]:
        pharmebinet_ingredient_map[str(int(row["pharmebinet_id"]))] = row["TMDB_id"]


disease = pd.read_csv(os.path.join(merge_result_dir, "entity/disease.csv"))
disease.replace(np.nan, None, inplace=True)
disease_id_maps = id_map(
    disease, ["pharmebinet_id", "PrimeKG_id", "CPMCP_id", "SymMap_id", "TCMBank_id"]
)

pathway = pd.read_csv(os.path.join(merge_result_dir, "entity/pathway.csv"))
pathway.replace(np.nan, None, inplace=True)
pharmebinet_pathway_map = {}
primekg_pathway_map = {}
for index, row in pathway.iterrows():
    if row["pharmebinet_id"]:
        pharmebinet_pathway_map[str(int(row["pharmebinet_id"]))] = row["TMDB_id"]
    if row["PrimeKG_id"]:
        primekg_pathway_map[str(int(row["PrimeKG_id"]))] = row["TMDB_id"]

anatomy = pd.read_csv(os.path.join(merge_result_dir, "entity/anatomy.csv"))
anatomy.replace(np.nan, None, inplace=True)
anatomy_id_maps = id_map(anatomy, ["pharmebinet_id", "PrimeKG_id"])

cell_component = pd.read_csv(
    os.path.join(merge_result_dir, "entity/cellular_component.csv")
)
cell_component.replace(np.nan, None, inplace=True)
cell_component_id_maps = id_map(cell_component, ["pharmebinet_id", "PrimeKG_id"])

biological_process = pd.read_csv(
    os.path.join(merge_result_dir, "entity/biological_process.csv")
)
biological_process.replace(np.nan, None, inplace=True)
biological_process_id_maps = id_map(
    biological_process, ["pharmebinet_id", "PrimeKG_id"]
)

molecular_function = pd.read_csv(
    os.path.join(merge_result_dir, "entity/molecular_function.csv")
)
molecular_function.replace(np.nan, None, inplace=True)
molecular_function_id_maps = id_map(
    molecular_function, ["pharmebinet_id", "PrimeKG_id"]
)

pharmebinet_relations = pd.read_table(os.path.join(pharmebinet_root_dir, "edges.tsv"))
primekg_relations = pd.read_csv(os.path.join(primekg_root_dir, "edges.csv"))

# ingredient-bind-gene BINDS_CHbG
# ingredient-up-gene UPREGULATES_CHuG
# ingredient-down-gene DOWNREGULATES_CHdG
# ingredient-?-gene(overall mean associate?) ASSOCIATES_CHaG
ingredient_down_gene = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    pharmebinet_gene_map,
    "DOWNREGULATES_CHdG",
)
ingredient_up_gene = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    pharmebinet_gene_map,
    "UPREGULATES_CHuG",
)
ingredient_bind_gene = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    pharmebinet_gene_map,
    "BINDS_CHbG",
)
ingredient_a_gene = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    pharmebinet_gene_map,
    "ASSOCIATES_CHaG",
)
gene_a_ingredient = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_ingredient_map,
    pharmebinet_gene_map,
    "ASSOCIATES_GaCH",
)
ingredient_a_gene.extend([(rel[1], rel[0]) for rel in gene_a_ingredient])

from collections import defaultdict

iag_record = defaultdict(list)
for record in ingredient_a_gene:
    iag_record[record].append("pharmebinet")
# read cpmcp symmap tcmbank ingredient related target
# cpmcp_relation

cpmcp_gene_data = os.path.join(cpmcp_root_dir, "target")
for gene_ingredient_path in glob.glob(cpmcp_gene_data + "/*/ingredient.json"):
    cpmcp_gene_id = gene_ingredient_path.split("/")[-2]
    TMDB_gene_id = cpmcp_gene_map[cpmcp_gene_id]
    ingredient_json = json.load(open(gene_ingredient_path))
    for ingredient in ingredient_json["items"]:
        TMDB_ingredient_id = cpmcp_ingredient_map[ingredient["id"]]
        ingredient_a_gene.append((TMDB_ingredient_id, TMDB_gene_id))
        iag_record[(TMDB_ingredient_id, TMDB_gene_id)].append("cpmcp")

# read symmap relation
symmap_gene_data = os.path.join(symmap_root_dir, "target")
for gene_ingredient_path in glob.glob(symmap_gene_data + "/*/ingredient.json"):
    symmap_gene_id = gene_ingredient_path.split("/")[-2]
    TMDB_gene_id = symmap_gene_map[symmap_gene_id]
    ingredient_json = json.load(open(gene_ingredient_path))
    for ingredient in ingredient_json["data"]:
        TMDB_ingredient_id = symmap_ingredient_map[ingredient["MOL_id"]]
        ingredient_a_gene.append((TMDB_ingredient_id, TMDB_gene_id))
        iag_record[(TMDB_ingredient_id, TMDB_gene_id)].append("symmap")

# read tcmbank relation
tcmbank_ingredient_data = os.path.join(tcmbank_root_dir, "ingredient")
for gene_ingredient_path in glob.glob(tcmbank_ingredient_data + "/*.json"):
    tcmbank_ingredient_id = gene_ingredient_path.split("/")[-1][:-5]
    if tcmbank_ingredient_id not in TCMBank_ingredient_map:
        continue
    TMDB_ingredient_id = TCMBank_ingredient_map[tcmbank_ingredient_id]
    ingredient_json = json.load(open(gene_ingredient_path))
    for ingredient in ingredient_json["data"]["chart_data"]:
        if ingredient["record_type"] == "Targets":
            TMDB_gene_id = TCMBank_gene_map[ingredient["TCMBank_ID"]]
            ingredient_a_gene.append((TMDB_ingredient_id, TMDB_gene_id))
            iag_record[(TMDB_ingredient_id, TMDB_gene_id)].append("tcmbank")

ingredient_down_gene = set(ingredient_down_gene)
ingredient_up_gene = set(ingredient_up_gene)
conflict_ingredient_rel_gene = ingredient_down_gene & ingredient_up_gene
ingredient_down_gene -= conflict_ingredient_rel_gene
ingredient_up_gene -= conflict_ingredient_rel_gene
ingredient_a_gene = set(ingredient_a_gene)
ingredient_a_gene.update(conflict_ingredient_rel_gene)


ingredient_down_gene_df = pd.DataFrame(
    ingredient_down_gene, columns=["source_id", "target_id"]
)
ingredient_down_gene_df["Relation_type"] = [
    "ingredient_downregulate_gene"
] * ingredient_down_gene_df.shape[0]
ingredient_down_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_downregulate_gene.csv"),
    index=False,
)

ingredient_up_gene_df = pd.DataFrame(
    ingredient_up_gene, columns=["source_id", "target_id"]
)
ingredient_up_gene_df["Relation_type"] = [
    "ingredient_upregulate_gene"
] * ingredient_up_gene_df.shape[0]
ingredient_up_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_upregulate_gene.csv"),
    index=False,
)

ingredient_bind_gene_df = pd.DataFrame(
    set(ingredient_bind_gene), columns=["source_id", "target_id"]
)
ingredient_bind_gene_df["Relation_type"] = [
    "ingredient_bind_gene"
] * ingredient_bind_gene_df.shape[0]
ingredient_bind_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_bind_gene.csv"), index=False
)
final = (
    ingredient_a_gene
    - set(ingredient_down_gene)
    - set(ingredient_up_gene)
    - set(ingredient_bind_gene)
)
ingredient_associate_gene_df = pd.DataFrame(final, columns=["source_id", "target_id"])
ingredient_associate_gene_df["Relation_type"] = [
    "ingredient_associate_gene"
] * ingredient_associate_gene_df.shape[0]
ingredient_associate_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/ingredient_associate_gene.csv"),
    index=False,
)

# target-target(overall) primekg
# gene-gene(regulates) REGULATES_GrG bi-directional
# gene-gene(covaries) COVARIES_GcG unidirectinal
# gene-gene(associate) pharmebinet INTERACTS_GiG primekg protein_protein
gene_regulate_gene = extract_specific_relation(
    pharmebinet_relations, pharmebinet_gene_map, pharmebinet_gene_map, "REGULATES_GrG"
)
gene_covary_gene = extract_specific_relation(
    pharmebinet_relations, pharmebinet_gene_map, pharmebinet_gene_map, "COVARIES_GcG"
)

gene_regulate_gene_df = pd.DataFrame(
    set(gene_regulate_gene), columns=["source_id", "target_id"]
)
gene_regulate_gene_df["Relation_type"] = [
    "gene_regulate_gene"
] * gene_regulate_gene_df.shape[0]
gene_regulate_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/gene_regulate_gene.csv"), index=False
)

gene_covary_gene_df = pd.DataFrame(
    set(gene_covary_gene), columns=["source_id", "target_id"]
)
gene_covary_gene_df["Relation_type"] = ["gene_covary_gene"] * gene_covary_gene_df.shape[
    0
]
gene_covary_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/gene_covary_gene.csv"), index=False
)

# 等价于associate
gene_associate_gene = extract_specific_relation(
    pharmebinet_relations, pharmebinet_gene_map, pharmebinet_gene_map, "INTERACTS_GiG"
)
gene_associate_gene2 = extract_specific_relation_from_primekg(
    primekg_relations, primekg_gene_map, primekg_gene_map, "protein_protein"
)
gene_associate_gene.extend(gene_associate_gene2)
gene_associate_gene = set(gene_associate_gene)

remove_rel = set()
for rel in gene_covary_gene:
    if rel in gene_associate_gene:
        remove_rel.add(rel)
    if (rel[1], rel[0]) in gene_associate_gene:
        remove_rel.add((rel[1], rel[0]))

for rel in gene_regulate_gene:
    if rel in gene_associate_gene:
        remove_rel.add(rel)
    if (rel[1], rel[0]) in gene_associate_gene:
        remove_rel.add((rel[1], rel[0]))

gene_associate_gene -= remove_rel
remove_rel = set()
# 如果正反向都在，保留一个
for rel in gene_associate_gene:
    if ((rel[1], rel[0]) in gene_associate_gene) and (
        (rel[1], rel[0]) not in remove_rel
    ):
        remove_rel.add(rel)

gene_associate_gene -= remove_rel

gene_associate_gene_df = pd.DataFrame(
    gene_associate_gene, columns=["source_id", "target_id"]
)
gene_associate_gene_df["Relation_type"] = [
    "gene_associate_gene"
] * gene_associate_gene_df.shape[0]
gene_associate_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/gene_associate_gene.csv"), index=False
)

# target-pathway PARTICIPATES_IN_GpiPW
gene_participate_in_pathway = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_gene_map,
    pharmebinet_pathway_map,
    "PARTICIPATES_IN_GpiPW",
)
gene_participate_in_pathway2 = extract_specific_relation_from_primekg(
    primekg_relations, primekg_gene_map, primekg_pathway_map, "pathway_protein"
)
gene_participate_in_pathway.extend(gene_participate_in_pathway2)
gene_participate_in_pathway_df = pd.DataFrame(
    set(gene_participate_in_pathway), columns=["source_id", "target_id"]
)
gene_participate_in_pathway_df["Relation_type"] = [
    "gene_associate_pathway"
] * gene_participate_in_pathway_df.shape[0]
gene_participate_in_pathway_df.to_csv(
    os.path.join(merge_result_dir, "relation/gene_associate_pathway.csv"), index=False
)

# target-disease(overall) primekg
# disease-gene(downregulate) DOWNREGULATES_DdG
# disease-gene(upregulate) UPREGULATES_DuG
disease_down_gene = extract_specific_relation(
    pharmebinet_relations,
    disease_id_maps["pharmebinet_id"],
    pharmebinet_gene_map,
    "DOWNREGULATES_DdG",
)
disease_up_gene = extract_specific_relation(
    pharmebinet_relations,
    disease_id_maps["pharmebinet_id"],
    pharmebinet_gene_map,
    "UPREGULATES_DuG",
)

disease_associate_gene = extract_specific_relation_from_primekg(
    primekg_relations,
    disease_id_maps["PrimeKG_id"],
    primekg_gene_map,
    "disease_protein",
)
# read cpmcp symmap tcmbank ingredient related target
# cpmcp_relation
cpmcp_disease_data = os.path.join(cpmcp_root_dir, "disease")
for diseas_gene_path in glob.glob(cpmcp_disease_data + "/*/target.json"):
    cpmcp_disease_id = diseas_gene_path.split("/")[-2]
    if cpmcp_disease_id not in disease_id_maps["CPMCP_id"]:
        continue
    TMDB_disease_id = disease_id_maps["CPMCP_id"][cpmcp_disease_id]
    gene_json = json.load(open(diseas_gene_path))
    for gene in gene_json["items"]:
        TMDB_gene_id = cpmcp_gene_map[str(int(gene["id"]))]
        disease_associate_gene.append((TMDB_disease_id, TMDB_gene_id))

# read symmap relation
symmap_gene_data = os.path.join(symmap_root_dir, "target")
for diseas_gene_path in glob.glob(symmap_gene_data + "/*/disease.json"):
    symmap_gene_id = diseas_gene_path.split("/")[-2]
    TMDB_gene_id = symmap_gene_map[symmap_gene_id]
    disease_json = json.load(open(diseas_gene_path))
    for disease in disease_json["data"]:
        if disease["Disease_id"] not in disease_id_maps["SymMap_id"]:
            continue
        TMDB_disease_id = disease_id_maps["SymMap_id"][disease["Disease_id"]]
        disease_associate_gene.append((TMDB_disease_id, TMDB_gene_id))

# read tcmbank relation
tcmbank_disease_data = os.path.join(tcmbank_root_dir, "disease")
for diseas_gene_path in glob.glob(tcmbank_disease_data + "/*.json"):
    tcmbank_disease_id = diseas_gene_path.split("/")[-1][:-5]
    if tcmbank_disease_id not in disease_id_maps["TCMBank_id"]:
        continue
    TMDB_disease_id = disease_id_maps["TCMBank_id"][tcmbank_disease_id]
    gene_json = json.load(open(diseas_gene_path))
    for gene in gene_json["data"]["chart_data"]:
        if gene["record_type"] == "Targets":
            TMDB_gene_id = TCMBank_gene_map[gene["TCMBank_ID"]]
            disease_associate_gene.append((TMDB_disease_id, TMDB_gene_id))

disease_down_gene_df = pd.DataFrame(
    set(disease_down_gene), columns=["source_id", "target_id"]
)
disease_down_gene_df["Relation_type"] = [
    "disease_downregulate_gene"
] * disease_down_gene_df.shape[0]
disease_down_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/disease_downregulate_gene.csv"),
    index=False,
)

disease_up_gene_df = pd.DataFrame(
    set(disease_up_gene), columns=["source_id", "target_id"]
)
disease_up_gene_df["Relation_type"] = [
    "disease_upregulate_gene"
] * disease_up_gene_df.shape[0]
disease_up_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/disease_upregulate_gene.csv"), index=False
)

disease_associate_gene_df = pd.DataFrame(
    set(disease_associate_gene) - set(disease_up_gene) - set(disease_down_gene),
    columns=["source_id", "target_id"],
)
disease_associate_gene_df["Relation_type"] = [
    "disease_associate_gene"
] * disease_associate_gene_df.shape[0]
disease_associate_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/disease_associate_gene.csv"), index=False
)

# anatomy-gene(target)(downregulate) DOWNREGULATES_AdG
# anatomy-gene(target)(upregulate) UPREGULATES_AuG
# anatomy-gene(target))(express) EXPRESSES_AeG
anatomy_down_gene = extract_specific_relation(
    pharmebinet_relations,
    anatomy_id_maps["pharmebinet_id"],
    pharmebinet_gene_map,
    "DOWNREGULATES_AdG",
)
anatomy_up_gene = extract_specific_relation(
    pharmebinet_relations,
    anatomy_id_maps["pharmebinet_id"],
    pharmebinet_gene_map,
    "UPREGULATES_AuG",
)
anatomy_express_gene = extract_specific_relation(
    pharmebinet_relations,
    anatomy_id_maps["pharmebinet_id"],
    pharmebinet_gene_map,
    "EXPRESSES_AeG",
)

anatomy_down_gene_df = pd.DataFrame(
    set(anatomy_down_gene), columns=["source_id", "target_id"]
)
anatomy_down_gene_df["Relation_type"] = [
    "anatomy_downregulate_gene"
] * anatomy_down_gene_df.shape[0]
anatomy_down_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/anatomy_downregulate_gene.csv"),
    index=False,
)

anatomy_up_gene_df = pd.DataFrame(
    set(anatomy_up_gene), columns=["source_id", "target_id"]
)
anatomy_up_gene_df["Relation_type"] = [
    "anatomy_upregulate_gene"
] * anatomy_up_gene_df.shape[0]
anatomy_up_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/anatomy_upregulate_gene.csv"), index=False
)

anatomy_express_gene_df = pd.DataFrame(
    set(anatomy_express_gene), columns=["source_id", "target_id"]
)
anatomy_express_gene_df["Relation_type"] = [
    "anatomy_express_gene"
] * anatomy_express_gene_df.shape[0]
anatomy_express_gene_df.to_csv(
    os.path.join(merge_result_dir, "relation/anatomy_express_gene.csv"), index=False
)

# target-celluar component(overall) LOCATED_IN_GliCC IS_ACTIVE_IN_GiaiCC PART_OF_GpoCC
gene_cell_component = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_gene_map,
    cell_component_id_maps["pharmebinet_id"],
    "LOCATED_IN_GliCC",
)
gene_cell_component.extend(
    extract_specific_relation(
        pharmebinet_relations,
        pharmebinet_gene_map,
        cell_component_id_maps["pharmebinet_id"],
        "IS_ACTIVE_IN_GiaiCC",
    )
)
gene_cell_component.extend(
    extract_specific_relation(
        pharmebinet_relations,
        pharmebinet_gene_map,
        cell_component_id_maps["pharmebinet_id"],
        "PART_OF_GpoCC",
    )
)
# cellcomp_protein
cell_component_gene = extract_specific_relation_from_primekg(
    primekg_relations,
    cell_component_id_maps["PrimeKG_id"],
    primekg_gene_map,
    "cellcomp_protein",
)
gene_cell_component.extend([(rel[1], rel[0]) for rel in cell_component_gene])

gene_cell_component_df = pd.DataFrame(
    set(gene_cell_component), columns=["source_id", "target_id"]
)
gene_cell_component_df["Relation_type"] = [
    "gene2cell_component"
] * gene_cell_component_df.shape[0]
gene_cell_component_df.to_csv(
    os.path.join(merge_result_dir, "relation/gene2cell_component.csv"), index=False
)
# target-biological process(overall) INVOLVED_IN_GiiBP ACTS_UPSTREAM_OF_OR_WITHIN_GauoowBP

gene_biological_process = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_gene_map,
    biological_process_id_maps["pharmebinet_id"],
    "INVOLVED_IN_GiiBP",
)
gene_biological_process.extend(
    extract_specific_relation(
        pharmebinet_relations,
        pharmebinet_gene_map,
        biological_process_id_maps["pharmebinet_id"],
        "ACTS_UPSTREAM_OF_OR_WITHIN_GauoowBP",
    )
)

# bioprocess_protein
biological_process_gene = extract_specific_relation_from_primekg(
    primekg_relations,
    biological_process_id_maps["PrimeKG_id"],
    primekg_gene_map,
    "bioprocess_protein",
)

gene_biological_process.extend([(rel[1], rel[0]) for rel in biological_process_gene])

gene_biological_process_df = pd.DataFrame(
    set(gene_biological_process), columns=["source_id", "target_id"]
)
gene_biological_process_df["Relation_type"] = [
    "gene2biological_process"
] * gene_biological_process_df.shape[0]
gene_biological_process_df.to_csv(
    os.path.join(merge_result_dir, "relation/gene2biological_process.csv"), index=False
)
# target-molecular function(overall) ENABLES_GeMF
gene_molecular_function = extract_specific_relation(
    pharmebinet_relations,
    pharmebinet_gene_map,
    molecular_function_id_maps["pharmebinet_id"],
    "ENABLES_GeMF",
)
molecular_function_gene = extract_specific_relation_from_primekg(
    primekg_relations,
    molecular_function_id_maps["PrimeKG_id"],
    primekg_gene_map,
    "molfunc_protein",
)  # molfunc_protein
gene_molecular_function.extend([(rel[1], rel[0]) for rel in molecular_function_gene])
gene_molecular_function_df = pd.DataFrame(
    set(gene_molecular_function), columns=["source_id", "target_id"]
)
gene_molecular_function_df["Relation_type"] = [
    "gene2molecular_function"
] * gene_molecular_function_df.shape[0]
gene_molecular_function_df.to_csv(
    os.path.join(merge_result_dir, "relation/gene2molecular_function.csv"), index=False
)
