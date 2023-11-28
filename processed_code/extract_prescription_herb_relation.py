import numpy as np
import pandas as pd
import os
import json
import re
from tqdm import tqdm
from collections import Counter


merge_result_dir = "../merge_result"


prescription_doses = json.load(
    open("../data/ChatGLM-6B/ptuning/prescription_dose_preditions_correct.json")
)

# Extract medicinal materials and corresponding dosages from prescription descriptions through rules
unit = "钱分两斤厘合握枚条个粒颗升gml对片朵株丸具茎字张瓶铢"
unpredicted_prescription = []
count = 0
for prescription in prescription_doses:
    if not isinstance(prescription["Dose"], dict):
        Dose = {}
        herbs = re.sub(r"\（[^）]*\）|\([^)]*\)", "", prescription["Prescription"])
        herbs = re.sub(r"\（.*|\(.*", "", herbs)
        herbs = re.sub("各等分|不拘多少|辅料为|少许|若干|制成|减半|等分|不以多少|倍用|不拘分两|不限多少", "", herbs)
        herbs = re.split("[、，。\s]|[^五]加", herbs)
        for index, herb in enumerate(herbs):
            if herb == "":
                continue
            dose = re.search(
                "([0-9\.一二三四五六七八九十百半\-\/]+[瓶钱分两公斤厘合握枚条个粒颗升gml对片朵株丸具茎字张铢]+)+[半]*", herb
            )
            if dose and herb[: dose.start()]:
                Dose[herb[: dose.start()]] = dose.group()
            elif dose is None:
                Dose[herb] = np.nan
        prescription["Dose"] = Dose
        unpredicted_prescription.append(prescription)

json.dump(
    unpredicted_prescription, open("rule_extract_dose.json", "w"), ensure_ascii=False
)
prescriptions = pd.read_csv(os.path.join(merge_result_dir, "entity/prescription.csv"))
prescriptions_map = {}
for index, row in tqdm(prescriptions.iterrows()):
    if isinstance(row["Prescription"], str):
        assert row["Prescription"] not in prescriptions_map
        prescriptions_map[row["Prescription"]] = row["TMDB_id"]

herbs = pd.read_csv(os.path.join(merge_result_dir, "entity/medicinal_material.csv"))
herb_map = {}
for index, row in tqdm(herbs.iterrows()):
    if isinstance(row["Chinese_Name"], str):
        for herb_name in row["Chinese_Name"].split(";"):
            herb_map[herb_name] = row["TMDB_id"]

prescription2herb = []
unrecorded_herb = Counter()
incomplete_count = 0
# Map the medicinal materials in the prescription to the prescription ID in the database
nonexist_prescription_count = 0
for one_prescription in tqdm(prescription_doses):
    if one_prescription["Prescription"] not in prescriptions_map:
        nonexist_prescription_count += 1
        continue
    prepscription_id = prescriptions_map[one_prescription["Prescription"]]
    incomplete = False
    for herb, dose in one_prescription["Dose"].items():
        herb = herb.strip()
        herb_sub = re.search("^[\u4e00-\u9fa5]+(?<![（(])", herb)
        if not herb_sub:
            print(herb)
            continue
        herb_sub = herb_sub.group()
        if herb_sub in herb_map:
            prescription2herb.append((prepscription_id, herb_map[herb_sub], dose))
        else:
            herb_sub = re.search(
                "(?![川吴姜干炒煨醋熟制炙真])[\u4e00-\u9fa5]+(?<![粉末屑梢汁])", herb_sub
            )
            if not herb_sub:
                print(herb)
                continue
            herb_sub = herb_sub.group()
            if herb_sub in herb_map:
                prescription2herb.append((prepscription_id, herb_map[herb_sub], dose))
            else:
                unrecorded_herb.update([herb])
                incomplete = True
                continue
    if incomplete:
        incomplete_count += 1

print(
    "incomplete prescription count:{}/{}".format(
        incomplete_count, len(prescription_doses)
    )
)
print("nonexist prscription count: {}".format(nonexist_prescription_count))
print(len(unrecorded_herb))
prescription2herb_df = pd.DataFrame(
    prescription2herb, columns=["source_id", "target_id", "dose"]
)
prescription2herb_df["Relation_type"] = [
    "prescription_consistof_herb"
] * prescription2herb_df.shape[0]
prescription2herb_df.to_csv(
    os.path.join(merge_result_dir, "relation/prescription2medicinal_material.csv"),
    index=False,
)
