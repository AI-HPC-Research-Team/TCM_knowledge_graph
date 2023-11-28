import pandas as pd
import json
import os
import glob
import re
from tqdm import tqdm
import numpy as np
import ast
from merge_utility import merge_database_by_id_group


if __name__ == "__main__":
    symmap_dir = '../data/symmap'
    merge_result_dir = '../merge_result'
    
    syndrome = pd.read_excel(os.path.join(symmap_dir, 'symmap_syndrome.xlsx'))
    syndrome['Syndrome_id'] = syndrome.Syndrome_id.apply(lambda x:'SMSY{:05d}'.format(x))
    syndrome.rename(columns={'Syndrome_id':'SymMap_id'}, inplace=True)
    syndrome.drop(54, inplace=True)
    # Version	Type	Suppress
    syndrome.drop(["Version", "Type", "Suppress"], inplace=True, axis=1)
    # Merge two syndromes with the same syndrome description
    syndrome = merge_database_by_id_group(syndrome, 'Syndrome_definition')
    syndrome['TMDB_id'] = ['TMSY{:05d}'.format(index) for index in range(1, syndrome.shape[0]+1)]
    syndrome.to_csv(os.path.join(merge_result_dir, 'entity/syndrome.csv'), index=False)
    
    # herb2syndrome
        herb = pd.read_csv(os.path.join(merge_result_dir, "entity/medicinal_material.csv"))
    cpmcp_herb_map = {}
    symmap_herb_map = {}
    for index, row in herb.iterrows():
        if isinstance(row["SymMap_id"], str):
            for SymMap_id in row["SymMap_id"].split(";"):
                symmap_herb_map["SMHB{:05d}".format(int(SymMap_id))] = row["TMDB_id"]
    
    
    herb2syndrome = []
    symmap_syndrome_data = os.path.join(symmap_data_dir, "syndrome")
    for syndrome_herb_path in glob.glob(symmap_syndrome_data + "/*/herb.json"):
        symmap_syndrome_id = syndrome_herb_path.split("/")[-2]
        TMDB_syndrome_id = symmap_syndrome_map[symmap_syndrome_id]
        herb_json = json.load(open(syndrome_herb_path))
        for herb in herb_json["data"]:
            TMDB_herb_id = symmap_herb_map[herb["Herb_id"]]
            herb2syndrome.append((TMDB_herb_id, TMDB_syndrome_id))
    herb2syndrome_df = pd.DataFrame(
        set(herb2syndrome), columns=["source_id", "target_id"]
    )
    herb2syndrome_df["Relation_type"] = [
        "herb_treat_syndrome"
    ] * herb2syndrome_df.shape[0]
    herb2syndrome_df.to_csv(
        os.path.join(merge_result_dir, "relation/herb2syndrome.csv"), index=False
    )


