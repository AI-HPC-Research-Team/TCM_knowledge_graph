import pandas as pd
import numpy as np
import ast
from merge_utility import merge_database_by_id
import re


def extract_properties_values(dataframe, properties_name):
    property_values_series = []
    for index, row in dataframe.iterrows():
        properties = ast.literal_eval(row["properties"])
        if properties_name in properties:
            property_values_series.append(properties[properties_name])
        else:
            property_values_series.append(None)
    return property_values_series


symmap_data_dir = "../data/symmap"
cpmcp_data_dir = "../data/CPMCP"
pharmebinet_data_dir = "../data/PharMeBINet"
merge_result_dir = "../merge_result"

symmap_mm_symptom = pd.read_excel(
    os.path.join(symmap_data_dir, "symmap_mm_symptom.xlsx")
)
symmap_mm_symptom.rename(columns={"MM_symptom_id": "SymMap_id"}, inplace=True)
cpmcp_mm_symptom = pd.read_csv(os.path.join(cpmcp_data_dir, "mm_symptom.csv"))
cpmcp_mm_symptom.rename(
    columns={
        "ID": "CPMCP_id",
        "Name": "MM_symptom_name",
        "Definition": "MM_symptom_definition",
    },
    inplace=True,
)
merge_mm_symptom = symmap_mm_symptom.merge(
    cpmcp_mm_symptom, on=["MM_symptom_name", "MM_symptom_definition"], how="outer"
)
merge_mm_symptom.drop(["Suppress", "Version"], inplace=True, axis=1)

pharmebinet_node = pd.read_table(os.path.join(pharmebinet_data_dir, "nodes.tsv"))
pharmebinet_mm_symptom = pharmebinet_node[
    pharmebinet_node.labels == "Phenotype|Symptom"
]
del pharmebinet_node
pharmebinet_mm_symptom.rename(
    columns={"node_id": "pharmebinet_id", "name": "MM_symptom_name"}, inplace=True
)
pharmebinet_mm_symptom["MeSH_id"] = pharmebinet_mm_symptom.identifier.apply(
    lambda x: x if "D" in x else None
)
pharmebinet_mm_symptom["HPO_id"] = pharmebinet_mm_symptom.identifier.apply(
    lambda x: x if "HP" in x else None
)

# properties
# 'comment', 'definition', 'xrefs',
# 'pharmgkb','hpo', 'hetionet' 'hpo_release',
# 'url_HPO', 'alternative_ids', 'property_values',
# 'synonyms', 'broad_synonyms', 'related_synonyms', 'narrow_synonyms'

HPO_ids_series = []
for index, row in pharmebinet_mm_symptom.iterrows():
    HPO_ids = []
    properties = ast.literal_eval(row["properties"])
    if "url_HPO" in properties and re.match(r"HP:\d+", properties["url_HPO"]):
        HPO_ids.append(re.match(r"HP:\d+", properties["url_HPO"]))
    if "alternative_ids" in properties:
        for hpo_id in properties["alternative_ids"]:
            assert "HP" in hpo_id
        HPO_ids.extend(properties["alternative_ids"])
    if isinstance(row["HPO_id"], str):
        HPO_ids.append(row["HPO_id"])
    if HPO_ids == []:
        HPO_ids_series.append(None)
    else:
        HPO_ids_series.append(";".join(HPO_ids))
pharmebinet_mm_symptom["HPO_id"] = HPO_ids_series

pharmebinet_mm_symptom["MM_symptom_definition"] = extract_properties_values(
    pharmebinet_mm_symptom, "definition"
)
pharmebinet_mm_symptom["Comment"] = extract_properties_values(
    pharmebinet_mm_symptom, "comment"
)
pharmebinet_mm_symptom["Xrefs"] = extract_properties_values(
    pharmebinet_mm_symptom, "xrefs"
)
pharmebinet_mm_symptom["Property_Values"] = extract_properties_values(
    pharmebinet_mm_symptom, "property_values"
)
pharmebinet_mm_symptom["Synonyms"] = extract_properties_values(
    pharmebinet_mm_symptom, "synonyms"
)
pharmebinet_mm_symptom["Broad_Synonyms"] = extract_properties_values(
    pharmebinet_mm_symptom, "broad_synonyms"
)
pharmebinet_mm_symptom["Related_Synonyms"] = extract_properties_values(
    pharmebinet_mm_symptom, "related_synonyms"
)
pharmebinet_mm_symptom["Narrow_Synonyms"] = extract_properties_values(
    pharmebinet_mm_symptom, "narrow_synonyms"
)

pharmebinet_mm_symptom.drop(
    ["labels", "license", "properties", "url", "resource", "identifier", "source"],
    inplace=True,
    axis=1,
)

concate_database = pharmebinet_mm_symptom.merge(
    merge_mm_symptom,
    on=["MM_symptom_name", "MM_symptom_definition", "MeSH_id", "HPO_id"],
    how="outer",
)
concate_database.replace(np.nan, None, inplace=True)
concate_database.pharmebinet_id = concate_database.pharmebinet_id.apply(
    lambda x: str(int(x)) if x else None
)
concate_database.SymMap_id = concate_database.SymMap_id.apply(
    lambda x: "SMMS{:05d}".format(int(x)) if x else None
)
concate_database.CPMCP_id = concate_database.CPMCP_id.apply(
    lambda x: "SYM{:05d}".format(int(x)) if x else None
)
concate_database = merge_database_by_id(concate_database, "MeSH_id")
concate_database = merge_database_by_id(concate_database, "HPO_id")
concate_database["TMDB_id"] = [
    "TMMS{:05d}".format(index) for index in range(1, concate_database.shape[0] + 1)
]
concate_database.to_csv(
    os.path.join(merge_result, "entity/mm_symptom.csv"), index=False
)
