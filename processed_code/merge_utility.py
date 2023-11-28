import pandas as pd
import json
import os
import glob
import re
from tqdm import tqdm
import numpy as np
import math


def calulate_reverse_relation(relation_df):
    """Some relationship types only allow one direction.
        For example, the relation(Entity A->Relation 1->Entity B) conflict with the relation(Entity B->Relation 1->Entity A).
        This function removes conflicting knowledge.
    Args:
        relation_df (DataFame): [source_id, target_id, type]

    Returns:
        DataFrame: does not contain bi-directional relationships
    """
    assert relation_df.duplicated().sum() == 0
    reverse_df = relation_df.copy()
    reverse_df.rename(
        {"source_id": "target_id", "target_id": "source_id"}, axis=1, inplace=True
    )
    concate_database_df = pd.concat([relation_df, reverse_df])

    exist_reverse_count = concate_database_df.duplicated().sum()
    nonexist_reverse_count = relation_df.shape[0] - exist_reverse_count
    print("total:", relation_df.shape[0])
    print("exist_reverse_count:", exist_reverse_count)
    print("nonexist_reverse_count:", nonexist_reverse_count)

    return concate_database_df.duplicated(keep=False)[: relation_df.shape[0]]


def extract_specific_relation(
    pharmebinet_relations, source_id_map, target_id_map, type_name
):
    """Extract specific types of relationships from pharmebinet database.
        Map pharmebinet id into ours

    Args:
        pharmebinet_relations (DataFrame): read relationships from pharmebinet database
        source_id_map (dict): pharmebinet_id-ours mapping
        target_id_map (dict): pharmebinet_id-ours mapping
        type_name (string): relationship type in pharmebinet database

    Returns:
        list: list of entity pairs
    """
    TMDB_relations = []
    for index, row in pharmebinet_relations[
        pharmebinet_relations.type == type_name
    ].iterrows():
        start_id = str(int(row["start_id"]))
        end_id = str(int(row["end_id"]))
        if (start_id in source_id_map) and (end_id in target_id_map):
            TMDB_start_id = source_id_map[start_id]
            TMDB_end_id = target_id_map[end_id]
            TMDB_relations.append((TMDB_start_id, TMDB_end_id))
    return TMDB_relations


def extract_specific_relation_from_primekg(
    primekg_relations, source_id_map, target_id_map, type_name
):
    """Extract specific types of relationships from primekg database.
        Map primekg id into ours

    Args:
        primekg_relations (DataFrame): read relationships from primekg database
        source_id_map (dict): primekg_id-ours mapping
        target_id_map (dict): primekg_id-ours mapping
        type_name (string): relationship type in primekg database

    Returns:
        list: list of entity pairs
    """
    TMDB_relations = []
    for index, row in primekg_relations[
        primekg_relations.relation == type_name
    ].iterrows():
        start_id = str(int(row["x_index"]))
        end_id = str(int(row["y_index"]))
        if (start_id in source_id_map) and (end_id in target_id_map):
            TMDB_start_id = source_id_map[start_id]
            TMDB_end_id = target_id_map[end_id]
            TMDB_relations.append((TMDB_start_id, TMDB_end_id))
    return TMDB_relations


def id_map(entity_df, database_names):
    """get entity id mapping

    Args:
        entity_df (dataframe): entity merge result
        database_names (list): subset of ['pharmebinet_id', 'PrimeKG_id', 'CPMCP_id', 'SymMap_id', 'TCMBank_id']

    Returns:
        dict: {'pharmebinet_id':{...},  'PrimeKG_id':{...}, ...}
    """
    database_maps = {}
    for database_name in database_names:
        database_maps[database_name] = {}

    for index, row in entity_df.iterrows():
        for database_name in database_names:
            if row[database_name]:
                if isinstance(row[database_name], str):
                    for id in row[database_name].split(";"):
                        database_maps[database_name][id] = row["TMDB_id"]
                else:
                    database_maps[database_name][str(int(row[database_name]))] = row[
                        "TMDB_id"
                    ]
    return database_maps


def merge_two_value(v1, v2):
    """Combine two values into a string

    Args:
        v1 (str/list/int/float): assume that different values are combined into strings using ';'
        v2 (str/list/int/float): assume that different values are combined into strings using ';'

    Returns:
        string: Combine different values into a string
    """
    # split the string
    if isinstance(v1, str) or isinstance(v2, str):
        merge_value = []
        if isinstance(v1, str):
            merge_value.extend([item.strip() for item in v1.split(";")])
        if isinstance(v2, str):
            merge_value.extend([item.strip() for item in v2.split(";")])
        if merge_value != []:
            return ";".join(list(set(merge_value)))

    if isinstance(v1, list) or isinstance(v2, list):
        merge_value = []
        if isinstance(v1, list):
            merge_value.extend(v1)
        if isinstance(v2, list):
            merge_value.extend(v2)
        if merge_value != []:
            return ";".join(list(set(merge_value)))

    # if v1 and v2 are numerical, they must be close to each other
    if v1 and v2:
        assert math.isclose(v1, v2, rel_tol=0.2)
        return v1
    elif v1:
        return v1
    else:
        return v2


def merge_two_row(row1, row2):
    # merge two record into one record
    for key, value in row1.items():
        row1[key] = merge_two_value(value, row2[key])
    return row1


def merge_database_by_id(concate_database, columns_name, debug=False):
    """This function merges records with the same identifier and returns the merged result.

    Args:
        concate_database (dataframe): dataframe with duplicate records
        columns_name (string): the column name of identifier
        debug (bool, optional): _description_. Defaults to False.

    Returns:
        DataFrame: dataframe without duplicate records
    """
    # one records may correspond to multiple identifiers and different identifiers are combined with the delimiter ';'
    # Get the mapping of identifiers and indexes
    id_map = {}
    for index, row in concate_database.iterrows():
        if not isinstance(row[columns_name], str):
            continue
        for database_id in row[columns_name].split(";"):
            database_id = database_id.strip()
            if database_id not in id_map:
                id_map[database_id] = [index]
            else:
                id_map[database_id].append(index)

    # Record the index of the merged record (which needs to be deleted later), and the index after the merge.
    merged_id_map = {}
    # Merge according to the order of merging the first ID
    database_ids = list(id_map.keys())
    database_ids.sort(key=lambda x: id_map[x][0])

    for database_id in database_ids:
        row_ids = id_map[database_id]

        # No need to merge
        if len(row_ids) == 1:
            continue
        first_id = row_ids[0]
        # if debug and (first_id == 137):
        #     print(row_ids)
        # find the first id that has not been merged
        while first_id in merged_id_map:
            first_id = merged_id_map[first_id]
        if first_id != row_ids[0]:
            merged_id_map[row_ids[0]] = first_id
        first_row = concate_database.loc[first_id].copy()
        for other_id in row_ids[1:]:
            # find another id that has not been merged
            while other_id in merged_id_map:
                other_id = merged_id_map[other_id]
            # update the first id that has not been merged
            if first_id > other_id:
                temp = first_id
                first_id = other_id
                other_id = temp
                concate_database.loc[other_id] = first_row
                first_row = concate_database.loc[first_id]
            elif first_id == other_id:
                continue
            first_row = merge_two_row(first_row, concate_database.loc[other_id])
            merged_id_map[other_id] = first_id
        concate_database.loc[first_id] = first_row
    concate_database = concate_database[
        ~concate_database.index.isin(merged_id_map.keys())
    ]
    concate_database.reset_index(drop=True, inplace=True)
    return concate_database


def merge_database_by_id_group(concate_database, columns_name):
    """This function merges records with the same identifier and returns the merged result.
       The identifier is the value of the columns. For Instance，"A;B" and "B;A" are different.

    Args:
        concate_database (dataframe): dataframe with duplicate records
        columns_name (string): the column name of identifier

    Returns:
        DataFrame: dataframe without duplicate records
    """
    delete_merge_id = []
    concate_database_copy = concate_database.copy()
    merge_group = concate_database_copy[
        concate_database_copy[columns_name].notnull()
    ].groupby(columns_name)
    for group_id, group in merge_group:
        if group.shape[0] > 1:
            first_row = group.iloc[0]
            first_index = group.index[0]
            for index, row in group.iloc[1:].iterrows():
                first_row = merge_two_row(first_row, row)
                delete_merge_id.append(index)
            concate_database.loc[first_index] = first_row
    concate_database = concate_database[~concate_database.index.isin(delete_merge_id)]
    concate_database.reset_index(drop=True, inplace=True)
    return concate_database


def extract_external_ids(pharmebinet_nodes, external_database_name_list=[]):
    """extract external database id from pharmebinet database

    Args:
        pharmebinet_nodes (dataframe):
        external_database_name_list (list, optional): list of the name of specific database id . Defaults to [].

    Returns:
        dict: the external id of correspoing database.
    """
    external_database_ids = {database: [] for database in external_database_name_list}
    for property_item in pharmebinet_nodes["properties"]:
        # extract omim id
        temp_id_record = {database: [] for database in external_database_name_list}
        if "xrefs" in property_item:
            for id_ref in property_item["xrefs"]:
                database_name, database_id = id_ref.split(":", 1)
                if database_name in external_database_name_list:
                    temp_id_record[database_name].append(database_id)
        for database_name, database_id in temp_id_record.items():
            if database_id != []:
                external_database_ids[database_name].append(";".join(database_id))
            else:
                external_database_ids[database_name].append(None)
    return external_database_ids


def extract_properties(pharmebinet_nodes, property_list=[]):
    """extract specific properties from pharmebinet database

    Args:
        pharmebinet_nodes (dataframe):
        property_list (list, optional): list of the property names. Defaults to [].

    Returns:
        dict: the values of corresponding properties
    """
    external_property = {property_name: [] for property_name in property_list}
    for property_item in pharmebinet_nodes["properties"]:
        for property_name in property_list:
            if property_name in property_item:
                external_property[property_name].append(property_item[property_name])
            else:
                external_property[property_name].append(None)
    return external_property
