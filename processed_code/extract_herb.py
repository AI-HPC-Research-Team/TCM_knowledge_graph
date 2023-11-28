import pandas as pd
import json
import os
import glob
import re
from tqdm import tqdm
import numpy as np
from merge_utility import merge_database_by_id, merge_database_by_id_group


def standardization_english_property(property_series):
    property_series = property_series.apply(lambda x: x if isinstance(x, str) else None)
    property_series = property_series.apply(
        lambda x: x.replace("big", "extremely") if isinstance(x, str) else None
    )
    property_series = property_series.apply(
        lambda x: x.replace("extreme ", "extremely ") if isinstance(x, str) else None
    )
    property_series = property_series.apply(
        lambda x: x.replace("poisonous", "toxic") if isinstance(x, str) else None
    )
    property_series = property_series.apply(
        lambda x: x.replace("minor", "slightly") if isinstance(x, str) else None
    )
    property_series = property_series.apply(
        lambda x: x.replace("punkery", "puckery") if isinstance(x, str) else None
    )
    property_series = property_series.apply(
        lambda x: x.replace("calm", "neutral") if isinstance(x, str) else None
    )
    property_series = property_series.apply(
        lambda x: x.replace("chill", "cold") if isinstance(x, str) else None
    )
    property_series = property_series.apply(
        lambda x: x.replace("mild", "neutral") if isinstance(x, str) else None
    )
    property_series = property_series.apply(
        lambda x: x.replace("puckery", "astringent") if isinstance(x, str) else None
    )
    property_series = property_series.apply(
        lambda x: x.replace("dryness", "hot") if isinstance(x, str) else None
    )
    return property_series


def formalization_dataframe(dataframe):
    # id are in uppercase letters, the others are lowercase
    return dataframe.apply(
        lambda x: x.str.upper() if "_id" in x.name else x.str.lower(), axis=0
    )


if __name__ == "__main__":
    data_root_dir = "../data"
    merge_result_dir = "../merge_result"
    # read symmap database
    symmap_data_dir = os.path.join(data_root_dir, "symmap", "SymMap herb.xlsx")
    symmap_herb = pd.read_excel(symmap_data_dir)

    # merge herb by alias name

    symmap_herb.drop(["Link_herb_id", "Suppress", "Alias"], axis=1, inplace=True)
    symmap_herb.rename(
        columns={
            "Herb_id": "SymMap_id",
            "Chinese_name": "Chinese_Name",
            "Pinyin_name": "Pinyin_Name",
            "Latin_name": "Latin_Name",
            "English_name": "English_Name",
            "Class_Chinese": "Therapeutic_Class_Chinese",
            "Class_English": "Therapeutic_Class_English",
            "UsePart": "Use_Part",
            "HERBDB_ID": "HERBDB_id",
        },
        inplace=True,
    )
    symmap_herb["Latin_Name"] = symmap_herb.Latin_Name.apply(
        lambda x: x.replace("\n", " ") if isinstance(x, str) else None
    )
    # read CPMCP
    cpmcp_data_dir = os.path.join(data_root_dir, "CPMCP", "herb")
    cpmcp_herb = []

    for herb_path in tqdm(glob.glob(cpmcp_data_dir + "/*/[0-9]*")):
        temp_herb = json.load(open(herb_path))
        cpmcp_herb.append(temp_herb)
    cpmcp_herb = pd.DataFrame(cpmcp_herb)
    cpmcp_herb["HERBDB_id"] = cpmcp_herb.externals.apply(lambda x: x["HERB id"])
    cpmcp_herb.drop(
        ["_show", "link_herb_id", "suppress", "externals"], axis=1, inplace=True
    )
    cpmcp_herb.rename(
        columns={
            "class_chinese": "Therapeutic_Class_Chinese",
            "class_english": "Therapeutic_Class_English",
            "id": "CPMCP_id",
            "meridians_chinese": "Meridians_Chinese",
            "meridians_english": "Meridians_English",
            "name_chinese": "Chinese_Name",
            "name_english": "English_Name",
            "name_latin": "Latin_Name",
            "name_pinyin": "Pinyin_Name",
            "properties_chinese": "Properties_Chinese",
            "properties_english": "Properties_English",
            "symmap_id": "SymMap_id",
            "use_part": "Use_Part",
        },
        inplace=True,
    )
    cpmcp_herb.merge(
        symmap_herb,
        on=[
            "Therapeutic_Class_Chinese",
            "Properties_Chinese",
            "Meridians_Chinese",
            "Latin_Name",
            "SymMap_id",
            "Properties_English",
            "Chinese_Name",
            "English_Name",
            "Pinyin_Name",
            "Meridians_English",
            "Therapeutic_Class_English",
            "Use_Part",
        ],
    )
    cpmcp_herb["HERBDB_id"] = cpmcp_herb["HERBDB_id"].apply(
        lambda x: ";".join(x) if x != [] else None
    )
    cpmcp_herb.replace(to_replace=["None", "NaN"], value=None, inplace=True)
    cpmcp_herb["Latin_Name"] = cpmcp_herb.Latin_Name.apply(
        lambda x: x.replace("\r\n", " ") if isinstance(x, str) else None
    )
    merge_herb = cpmcp_herb.merge(
        symmap_herb,
        on=[
            "SymMap_id",
            "HERBDB_id",
            "Therapeutic_Class_Chinese",
            "Therapeutic_Class_English",
            "Properties_Chinese",
            "Properties_English",
            "Meridians_Chinese",
            "Meridians_English",
            "Latin_Name",
            "Pinyin_Name",
            "Chinese_Name",
            "Use_Part",
        ],
        how="outer",
        suffixes=["_cpmcp", "_symmap"],
    )
    merge_herb.rename(columns={"English_Name_cpmcp": "English_Name"}, inplace=True)
    merge_herb.drop("English_Name_symmap", axis=1, inplace=True)
    merge_herb.English_Name = merge_herb.English_Name.apply(
        lambda x: x.replace(",", ";") if isinstance(x, str) else None
    )
    merge_herb.Latin_Name = merge_herb.Latin_Name.apply(
        lambda x: x.replace(",", ";") if isinstance(x, str) else None
    )
    merge_herb.Meridians_Chinese = merge_herb.Meridians_Chinese.apply(
        lambda x: x.replace(",", ";") if isinstance(x, str) else None
    )
    merge_herb.Meridians_English = merge_herb.Meridians_English.apply(
        lambda x: x.replace(",", ";") if isinstance(x, str) else None
    )
    merge_herb["HERBDB_id"] = merge_herb["HERBDB_id"].apply(
        lambda x: x.replace("|", ";") if isinstance(x, str) else None
    )
    merge_herb["SymMap_id"] = merge_herb["SymMap_id"].apply(
        lambda x: str(x) if isinstance(x, int) else None
    )
    merge_herb["CPMCP_id"] = merge_herb["CPMCP_id"].apply(
        lambda x: str(int(x)) if ~np.isnan(x) else None
    )
    merge_herb["TCMID_id"] = merge_herb["TCMID_id"].apply(
        lambda x: str(int(x)) if ~np.isnan(x) else None
    )
    merge_herb["TCM-ID_id"] = merge_herb["TCM-ID_id"].apply(
        lambda x: str(int(x)) if ~np.isnan(x) else None
    )
    merge_herb["TCMSP_id"] = merge_herb["TCMSP_id"].apply(
        lambda x: str(int(x)) if ~np.isnan(x) else None
    )
    merge_herb = formalization_dataframe(merge_herb)
    # extract "toxicity", "herb property", "flavour" from Properties_English
    # toxicity  'nontoxic', 'slightly toxic', 'toxic', 'extremely poisonous',
    # property 'extremely cold', 'extreme cold', 'slightly warm', 'minor warm', 'minor cold',  'calm'-->'neutral',  'cool', 'cold', 'dryness(燥)', 'warm', 'hot', 'neutral', 'mild'-->'neutral', 'big hot', 'big chill' --> 'extreme cold', 'slightly cold', 'sharp'
    # smell 'slightly astringent', 'slightly bitter', 'slightly pungent',  'slightly salty', 'slightly sour', 'slightly sweet', 'astringent', 'bitter', 'Punkery' -->'Puckery', 'pungent', 'salty', 'sour', 'sweet',  'tasteless',
    merge_herb.Properties_English = standardization_english_property(
        merge_herb.Properties_English
    )
    # toxicity 'nontoxic', 'slightly toxic', 'extremely toxic', 'toxic'
    # smell 'slightly astringent', 'slightly bitter', 'slightly pungent', 'slightly salty', 'slightly sour', 'slightly sweet', 'astringent', 'bitter', 'pungent', 'salty', 'sour', 'sweet', 'tasteless'
    # property  'bitter', 'cold', 'cool', 'dryness', 'hot','neutral', 'sharp', 'warm', 'extremely hot', 'slightly cold', 'slightly warm', 'extremely cold'
    toxicity_dict = {
        "nontoxic": "无毒",
        "slightly toxic": "小毒",
        "extremely toxic": "大毒",
        "toxic": "有毒",
    }
    smell_dict = {
        "slightly astringent": "微涩",
        "slightly bitter": "微苦",
        "slightly pungent": "微辛",
        "slightly salty": "微咸",
        "slightly sour": "微酸",
        "slightly sweet": "微甘",
        "astringent": "涩",
        "bitter": "苦",
        "pungent": "辛",
        "salty": "咸",
        "sour": "酸",
        "sweet": "甘",
        "tasteless": "淡",
    }
    property_dict = {
        "cold": "寒",
        "cool": "凉",
        "hot": "热",
        "warm": "温",
        "neutral": "平",
        "sharp": "锐",
        "extremely hot": "大热",
        "slightly cold": "小寒",
        "slightly warm": "微温",
        "extremely cold": "大寒",
    }
    toxicity_en = []
    smell_en = []
    smell_cn = []
    property_en = []
    property_cn = []

    for item in merge_herb.Properties_English:
        if not isinstance(item, str):
            toxicity_en.append(None)
            smell_en.append(None)
            smell_cn.append(None)
            property_en.append(None)
            property_cn.append(None)
            continue

        item = [temp.strip() for temp in item.split(",")]
        smell_en_temp = []
        smell_cn_temp = []
        property_en_temp = []
        property_cn_temp = []
        toxicity_en_temp = None
        for sub_property in item:
            if sub_property in smell_dict:
                smell_en_temp.append(sub_property)
                smell_cn_temp.append(smell_dict[sub_property])
            if sub_property in property_dict:
                property_en_temp.append(sub_property)
                property_cn_temp.append(property_dict[sub_property])
            if (sub_property in toxicity_dict) and toxicity_en_temp is None:
                toxicity_en_temp = sub_property
        else:
            if smell_en_temp == []:
                smell_en.append(None)
                smell_cn.append(None)
            else:
                smell_en.append(smell_en_temp)
                smell_cn.append(smell_cn_temp)

            if property_en_temp == []:
                property_en.append(None)
                property_cn.append(None)
            else:
                property_en.append(property_en_temp)
                property_cn.append(property_cn_temp)

            toxicity_en.append(toxicity_en_temp)

    merge_herb["Toxicity"] = toxicity_en
    merge_herb["Smell_English"] = smell_en
    merge_herb["Smell_Chinese"] = smell_cn
    merge_herb["Properties_English"] = property_en
    merge_herb["Properties_Chinese"] = property_cn

    # read tcmbank
    tcmbank_data_dir = os.path.join(data_root_dir, "TCMBANK", "herb_all.xlsx")
    tcmbank_herb = pd.read_excel(tcmbank_data_dir)
    tcmbank_herb.rename(
        columns={
            "TCMBank_ID": "TCMBank_id",
            "TCM_name": "Chinese_Name",
            "TCM_name_en": "English_Name",
            "Herb_pinyin_name": "Pinyin_Name",
            "Herb_latin_name": "Latin_Name",
            "Properties": "Properties_English",
            "Meridians": "Meridians_English",
            "UsePart": "Use_Part",
            "Indication": "Indications",
            "Clinical_manifestations": "Clinical_Manifestations",
            "Therapeutic_en_class": "Therapeutic_Class_English",
            "Therapeutic_cn_class": "Therapeutic_Class_Chinese",
            "TCM_ID_id": "TCM-ID_id",
            "SymMap_id": "SymMap_id",
            "Herb_ID": "HERBDB_id",
        },
        inplace=True,
    )
    # tcmbank_herb['SymMap_id'] = tcmbank_herb.TCMID_id.apply(lambda x: str(int(x.split(';')[0][4:])) if isinstance(x, str) else None)
    tcmbank_herb["TCM-ID_id"] = tcmbank_herb["TCM-ID_id"].apply(
        lambda x: str(x) if isinstance(x, int) else None
    )
    tcmbank_herb["TCMSP_id"] = tcmbank_herb["TCMSP_id"].apply(
        lambda x: str(x) if isinstance(x, int) else None
    )
    tcmbank_herb.drop(
        ["TCMID_id", "level1_name_en", "level2_name"], axis=1, inplace=True
    )
    tcmbank_herb["English_Name"] = tcmbank_herb["English_Name"].apply(
        lambda x: x.replace(",", ";") if isinstance(x, str) else None
    )
    tcmbank_herb["Latin_Name"] = tcmbank_herb["Latin_Name"].apply(
        lambda x: x.replace(",", ";") if isinstance(x, str) else None
    )
    tcmbank_herb["Toxicity"] = tcmbank_herb["Toxicity"].apply(
        lambda x: x if isinstance(x, str) else None
    )
    tcmbank_herb.drop(index=6101, inplace=True)  # drop duplicated
    # extract "herb property", "flavour" from Properties_English
    tcmbank_herb.loc[5366, "Properties_English"] = None
    tcmbank_herb = formalization_dataframe(tcmbank_herb)
    tcmbank_herb.Properties_English = standardization_english_property(
        tcmbank_herb.Properties_English
    )
    # smell 'sour', 'slightly sour', 'slightly sweet', 'pungent', 'slightly bitter', 'sweet', 'astringent', 'slightly pungent', 'salty', 'bitter',
    # property 'cool','warm', 'cold','hot', 'neutral', 'slightly warm', 'slightly cool', 'slightly cold','extremely cold', 'extremely hot','extremely warm',
    smell_dict = {
        "slightly astringent": "微涩",
        "slightly bitter": "微苦",
        "slightly pungent": "微辛",
        "slightly salty": "微咸",
        "slightly sour": "微酸",
        "slightly sweet": "微甘",
        "astringent": "涩",
        "bitter": "苦",
        "pungent": "辛",
        "salty": "咸",
        "sour": "酸",
        "sweet": "甘",
        "tasteless": "淡",
    }
    property_dict = {
        "cold": "寒",
        "cool": "凉",
        "hot": "热",
        "warm": "温",
        "neutral": "平",
        "sharp": "锐",
        "extremely hot": "大热",
        "slightly cool": "小凉",
        "slightly cold": "小寒",
        "slightly warm": "微温",
        "extremely cold": "大寒",
        "extremely warm": "大温",
    }
    smell_en = []
    smell_cn = []
    property_en = []
    property_cn = []
    for item in tcmbank_herb.Properties_English:
        if not isinstance(item, str):
            smell_en.append(None)
            smell_cn.append(None)
            property_en.append(None)
            property_cn.append(None)
            continue

        item = [temp.strip() for temp in item.split(";")]
        smell_en_temp = []
        smell_cn_temp = []
        property_en_temp = []
        property_cn_temp = []

        for sub_property in item:
            if sub_property in smell_dict:
                smell_en_temp.append(sub_property)
                smell_cn_temp.append(smell_dict[sub_property])
            if sub_property in property_dict:
                property_en_temp.append(sub_property)
                property_cn_temp.append(property_dict[sub_property])
        else:
            if smell_en_temp == []:
                smell_en.append(None)
                smell_cn.append(None)
            else:
                smell_en.append(smell_en_temp)
                smell_cn.append(smell_cn_temp)

            if property_en_temp == []:
                property_en.append(None)
                property_cn.append(None)
            else:
                property_en.append(property_en_temp)
                property_cn.append(property_cn_temp)

    tcmbank_herb["Smell_English"] = smell_en
    tcmbank_herb["Smell_Chinese"] = smell_cn
    tcmbank_herb["Properties_English"] = property_en
    tcmbank_herb["Properties_Chinese"] = property_cn
    # merge by symmap id, TCM-ID, HERB_id

    concate_database = pd.concat([tcmbank_herb, merge_herb], ignore_index=True)
    concate_database.replace(np.nan, None, inplace=True)
    concate_database = merge_database_by_id_group(concate_database, "SymMap_id")
    concate_database = merge_database_by_id_group(concate_database, "TCM-ID_id")
    concate_database = merge_database_by_id(concate_database, "HERBDB_id")
    # concate_database.replace(np.nan, None, inplace=True)
    # merged by alias name
    alias_names_df1 = pd.read_csv(
        os.path.join(merge_result_dir, "alias_name_modified.csv")
    )
    alias_names_df2 = pd.read_csv(
        os.path.join(merge_result_dir, "manual_alias_name.csv")
    )
    alias_names_df = pd.concat([alias_names_df1, alias_names_df2])
    alias_names_df["chinese_name"] = alias_names_df["chinese_name"].apply(
        lambda x: x.split(";")
    )

    alias_names_map = {}
    for alias_names_set in alias_names_df["chinese_name"]:
        for name in alias_names_set:
            alias_names_map[name] = alias_names_set

    chinese_names_series = []
    for chinese_name_item in concate_database["Chinese_Name"]:
        chinese_name_set = set()
        if not chinese_name_item:
            chinese_names_series.append(None)
            continue
        for chinese_name in chinese_name_item.split(";"):
            if chinese_name in alias_names_map:
                chinese_name_set.update(alias_names_map[chinese_name])
        chinese_name_set.update(chinese_name_item.split(";"))
        chinese_names = ";".join(chinese_name_set)
        chinese_names_series.append(chinese_names)
    concate_database["Chinese_Name"] = chinese_names_series
    concate_database = merge_database_by_id(
        concate_database, "Chinese_Name", debug=True
    )
    concate_database["TMDB_id"] = [
        "TMHB{:05d}".format(index) for index in range(1, concate_database.shape[0] + 1)
    ]
    concate_database["Smell_English"] = concate_database["Smell_English"].apply(
        lambda x: ";".join(x) if isinstance(x, list) else x
    )  # 可能为str
    concate_database["Smell_Chinese"] = concate_database["Smell_Chinese"].apply(
        lambda x: ";".join(x) if isinstance(x, list) else x
    )
    concate_database["Properties_Chinese"] = concate_database[
        "Properties_Chinese"
    ].apply(lambda x: ";".join(x) if isinstance(x, list) else x)
    concate_database["Properties_English"] = concate_database[
        "Properties_English"
    ].apply(lambda x: ";".join(x) if isinstance(x, list) else x)
    concate_database.to_csv(
        os.path.join(merge_result_dir, "entity/medicinal_material.csv"), index=False
    )

    # flavour csv  TMDB_id, Chinese_Name, English_Name
    flavour = []
    concate_database["Smell_English"].apply(
        lambda x: flavour.extend(x) if isinstance(x, list) else None
    )
    flavour = list(set(flavour))
    flavour_dict = {
        "bitter": "苦",
        "slightly bitter": "微苦",
        "sour": "微酸",
        "slightly sweet": "微甘",
        "pungent": "辛",
        "slightly pungent": "微辛",
        "slightly astringent": "微涩",
        "slightly salty": "微咸",
        "salty": "咸",
        "slightly sour": "微酸",
        "tasteless": "淡",
        "astringent": "涩",
        "sweet": "甘",
    }
    flavour = pd.DataFrame(
        {"Chinese_Name": flavour_dict.values(), "English_Name": flavour_dict.keys()}
    )
    flavour["TMDB_id"] = [
        "TMFV{:05d}".format(index) for index in range(1, flavour.shape[0] + 1)
    ]
    flavour[["TMDB_id", "English_Name", "Chinese_Name"]].to_csv(
        os.path.join(merge_result_dir, "entity/flavour.csv"), index=False
    )
    # toxicity TMDB_id, Chinese_Name, English_Name
    toxicity = []
    concate_database["Toxicity"].apply(
        lambda x: toxicity.extend([item.strip() for item in x.split(";")])
        if isinstance(x, str)
        else None
    )
    toxicity = list(set(toxicity))
    toxicity_dict = {
        "nontoxic": "无毒",
        "extremely toxic": "极毒",
        "slightly toxic": "微毒",
        "toxic": "毒",
    }
    toxicity = pd.DataFrame(
        {"Chinese_Name": toxicity_dict.values(), "English_Name": toxicity_dict.keys()}
    )
    toxicity["TMDB_id"] = [
        "TMTX{:05d}".format(index) for index in range(1, toxicity.shape[0] + 1)
    ]
    toxicity[["TMDB_id", "English_Name", "Chinese_Name"]].to_csv(
        os.path.join(merge_result_dir, "entity/toxicity.csv"), index=False
    )
    # tropism TMDB_id, Chinese_Name, English_Name
    tropism = []
    concate_database["Meridians_English"].apply(
        lambda x: tropism.extend([item.strip() for item in x.split(";")])
        if isinstance(x, str)
        else None
    )
    tropism = list(set(tropism))
    # ['stomach', 'small intestine', 'liver', 'bladder', 'kidney', 'large intestine', 'spleen', 'heart', 'cardiovascular', 'lung', 'triple energizers', 'gallbladder', 'pericardium', 'three end']
    tropism_dict = {
        "stomach": "胃",
        "small intestine": "小肠",
        "liver": "肝",
        "bladder": "膀胱",
        "kidney": "肾",
        "large intestine": "大肠",
        "spleen": "脾",
        "heart": "心",
        "cardiovascular": "心血管",
        "lung": "肺",
        "triple energizers": "三焦",
        "gallbladder": "胆",
        "pericardium": "心包",
    }
    tropism = pd.DataFrame(
        {"Chinese_Name": tropism_dict.values(), "English_Name": tropism_dict.keys()}
    )
    tropism["TMDB_id"] = [
        "TMTP{:02d}".format(index) for index in range(1, tropism.shape[0] + 1)
    ]
    tropism[["TMDB_id", "English_Name", "Chinese_Name"]].to_csv(
        os.path.join(merge_result_dir, "entity/tropism.csv"), index=False
    )
    # property TMDB_id, Chinese_Name, English_Name
    properties = []
    concate_database["Properties_English"].apply(
        lambda x: properties.extend([item.strip() for item in x.split(";")])
        if isinstance(x, str)
        else None
    )
    properties = list(set(properties))
    properties_dict = {
        "extremely warm": "大温",
        "extremely cold": "大寒",
        "extremely hot": "大热",
        "sharp": "锐",
        "hot": "热",
        "neutral": "平",
        "warm": "温",
        "cool": "凉",
        "cold": "寒",
        "slightly cool": "小凉",
        "slightly cold": "小寒",
        "slightly warm": "小温",
    }
    properties = pd.DataFrame(
        {
            "Chinese_Name": properties_dict.values(),
            "English_Name": properties_dict.keys(),
        }
    )
    properties["TMDB_id"] = [
        "TMPP{:05d}".format(index) for index in range(1, properties.shape[0] + 1)
    ]
    properties[["TMDB_id", "English_Name", "Chinese_Name"]].to_csv(
        os.path.join(merge_result_dir, "entity/properties.csv"), index=False
    )

    # relation
    # herb-flavour
    herb_flavour = []
    flavour_map = dict(zip(flavour.English_Name.tolist(), flavour.TMDB_id.tolist()))
    for index, row in concate_database.iterrows():
        if not isinstance(row["Smell_English"], str):
            continue
        for smell in row["Smell_English"].split(";"):
            herb_flavour.append((row["TMDB_id"], flavour_map[smell]))
    herb_flavour = pd.DataFrame(set(herb_flavour), columns=["source_id", "target_id"])
    herb_flavour["Relation_type"] = ["herb_has_flavour"] * herb_flavour.shape[0]
    herb_flavour.to_csv(
        os.path.join(merge_result_dir, "relation/herb2flavour.csv"), index=False
    )
    # herb-properties
    herb_property = []
    properties_map = dict(
        zip(properties.English_Name.tolist(), properties.TMDB_id.tolist())
    )
    concate_database["Properties_English"] = concate_database[
        "Properties_English"
    ].apply(lambda x: x if (isinstance(x, str) and x != "") else None)
    for index, row in concate_database.iterrows():
        if not isinstance(row["Properties_English"], str):
            continue
        for prop in row["Properties_English"].split(";"):
            if prop == "":
                print(index, row)
            herb_property.append((row["TMDB_id"], properties_map[prop]))
    herb_property = pd.DataFrame(set(herb_property), columns=["source_id", "target_id"])
    herb_property["Relation_type"] = ["herb_has_property"] * herb_property.shape[0]
    herb_property.to_csv(
        os.path.join(merge_result_dir, "relation/herb2property.csv"), index=False
    )
    # herb-tropism
    herb_tropism = []
    tropism_map = dict(zip(tropism.English_Name.tolist(), tropism.TMDB_id.tolist()))
    for index, row in concate_database.iterrows():
        if not isinstance(row["Meridians_English"], str):
            continue
        for trop in row["Meridians_English"].split(";"):
            if trop.strip() == "three end":
                continue
            herb_tropism.append((row["TMDB_id"], tropism_map[trop.strip()]))
    herb_tropism = pd.DataFrame(set(herb_tropism), columns=["source_id", "target_id"])
    herb_tropism["Relation_type"] = ["tropism_of_herb"] * herb_tropism.shape[0]
    herb_tropism.to_csv(
        os.path.join(merge_result_dir, "relation/herb2tropism.csv"), index=False
    )
    # herb-toxicity
    herb_toxicity = []
    toxicity_map = dict(zip(toxicity.English_Name.tolist(), toxicity.TMDB_id.tolist()))
    for index, row in concate_database.iterrows():
        if not isinstance(row["Toxicity"], str):
            continue
        if len(row["Toxicity"].split(";")) > 1:
            herb_toxicity.append((row["TMDB_id"], toxicity_map["toxic"]))
        else:
            tox = row["Toxicity"].split(";")[0]
            herb_toxicity.append((row["TMDB_id"], toxicity_map[tox]))
    herb_toxicity = pd.DataFrame(set(herb_toxicity), columns=["source_id", "target_id"])
    herb_toxicity["Relation_type"] = ["toxicity_of_herb"] * herb_toxicity.shape[0]
    herb_toxicity.to_csv(
        os.path.join(merge_result_dir, "relation/herb2toxicity.csv"), index=False
    )
