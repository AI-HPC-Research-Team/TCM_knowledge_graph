import pandas as pd
import os
import numpy as np
import re
import cn2an
from tqdm import tqdm


merge_result_dir = "../merge_result"

# Convert dosages in prescriptions to standard unit dosages

prescription = pd.read_csv(os.path.join(merge_result_dir, "entity/prescription.csv"))
prescription.replace(np.nan, None, inplace=True)
prescription2herb = pd.read_csv(
    os.path.join(merge_result_dir, "relation/prescription2medicinal_material.csv")
)
prescription2herb.replace(np.nan, None, inplace=True)
herb = pd.read_csv(
    os.path.join(os.path.join(merge_result_dir, "entity/medicinal_material.csv"))
)


# 1. Manually some correct errors in dose identification tasks
with open('correct_word_list.txt', encoding='utf-8') as f:
    words = f.read().split(',')
prescription2herb.replace(
    words,
    None,
    inplace=True,
)
prescription2herb.replace("两半", "一两半", inplace=True)
prescription2herb.replace("钱半", "一钱半", inplace=True)
prescription2herb.replace("斤半", "一斤半", inplace=True)
prescription2herb.replace("分半", "一分半", inplace=True)
prescription2herb.replace("分半", "一分半", inplace=True)
prescription2herb.replace("9两半12两半", "9.5-12.5两", inplace=True)
prescription2herb.replace("2两钱", "2两", inplace=True)

prescription2herb["dose"] = prescription2herb["dose"].apply(
    lambda x: x.replace("小", "") if x else None
)
prescription2herb["dose"] = prescription2herb["dose"].apply(
    lambda x: x.replace("大", "") if x else None
)
prescription2herb["dose"] = prescription2herb["dose"].apply(
    lambda x: x.replace("中", "") if x else None
)
prescription2herb["dose"] = prescription2herb["dose"].apply(
    lambda x: x.replace("～", "-") if x else None
)
prescription2herb["dose"] = prescription2herb["dose"].apply(
    lambda x: x.replace("至钱半", "至1钱半") if x else None
)
prescription2herb["dose"] = prescription2herb["dose"].apply(
    lambda x: x.replace("至两半", "至1两半") if x else None
)


# 2.1 Delete the dose information that is not in the specified unit.
precise_dose_herb = set()
count = 0

prescription_herb_dose = []
for index, row in prescription2herb.iterrows():
    if row["dose"]:
        if (
            re.search("分|两|钱|kg|g|mg|厘|毫|铢|公斤|斤|千克|克|合|ml|斗|升", row["dose"]) is not None
        ) and (re.search("钱匕|分盏|字|厘米", row["dose"]) is None):
            prescription_herb_dose.append(row["dose"])
        else:
            count += 1  # 9423
            prescription_herb_dose.append(None)
    else:
        prescription_herb_dose.append(None)
prescription2herb["dose"] = prescription_herb_dose
# prescription2herb.to_csv(os.path.join(merge_result_dir, 'relation/prescription2herb_v1.csv'), index=False)
# 3. Convert the dose in the specified unit to the dose in the unified unit (ml or g).
numbers = []
units = []
times = {
    "两": 31.25,
    "钱": 3.125,
    "铢": 1.3,
    "分": 0.3125,
    "厘": 0.03125,
    "毫": 0.003125,
    "斤": 500,
    "合": 20,
    "斗": 2000,
    "升": 200,
    "g": 1,
    "ml": 1,
    "kg": 1000,
    "克": 1,
    "千克": 1000,
    "mg": 0.001,
    "公斤": 1000,
}
for index, row in tqdm(prescription2herb.iterrows()):
    dose = row["dose"]
    if row["dose"] is None:
        units.append(None)
        numbers.append(None)
        continue
    half_loc = dose.find("半")
    re_res = list(
        filter(None, re.split("(分|两|钱|kg|g|mg|厘|毫|铢|公斤|斤|千克|克|合|ml|斗|升)", dose))
    )
    if half_loc == 0:
        assert len(re_res) == 2
        numbers.append(0.5 * times[re_res[1]])
        if re_res[1] in [
            "分",
            "两",
            "钱",
            "厘",
            "斤",
            "克",
            "千克",
            "公斤",
            "mg",
            "g",
            "kg",
            "毫",
            "铢",
        ]:
            units.append("g")
        elif re_res[1] in ["ml", "斗", "升", "合"]:
            units.append("ml")
        else:
            raise Exception("Sorry, unexcepted unit")
    else:
        cn_list = re_res[::2]
        unit_list = re_res[1::2]
        if half_loc == -1 and len(cn_list) - len(unit_list) == 1:
            print(row)

        g_unit = True
        ml_unit = True
        for unit in unit_list:
            if (
                unit
                in ["分", "两", "钱", "厘", "斤", "克", "千克", "公斤", "mg", "g", "kg", "毫", "铢"]
                and g_unit
            ):
                ml_unit = False
            elif unit in ["ml", "斗", "升", "合"] and ml_unit:
                g_unit = False
            else:
                raise Exception("Sorry, unit conflict")
        else:
            if ml_unit:
                units.append("ml")
            elif g_unit:
                units.append("g")
        overall_dose = 0
        for cn_id in range(len(unit_list)):
            half = 0
            cn = cn_list[cn_id]
            if "-" in cn:
                low, high = cn.split("-")
                an = (cn2an.cn2an(low, "smart") + cn2an.cn2an(high, "smart")) / 2
            elif "至" in cn:
                cn = cn[1:]
                half = 1
            else:
                an = cn2an.cn2an(cn, "smart")
            overall_dose += times[unit_list[cn_id]] * an
        if half_loc > 0:
            overall_dose += times[unit_list[cn_id]] * 0.5
        overall_dose /= half + 1
        numbers.append(overall_dose)


prescription2herb["number"] = numbers
prescription2herb["unit"] = units
presciption_overall_dose = prescription2herb.number.groupby(
    prescription2herb.source_id
).sum()
# Discard prescriptions with dosage units in milliliters
drop_prescription_id = (
    prescription2herb[prescription2herb["unit"] == "ml"].source_id.unique().tolist()
)
# If a prescription contains medicinal materials without doses,
# the percentage of the medicinal materials in the prescription cannot be calculated.
drop_prescription_id.extend(
    prescription2herb[prescription2herb.number.isnull()].source_id.unique().tolist()
)

herb_percentage = []
for index, row in tqdm(prescription2herb.iterrows()):
    if row["source_id"] in drop_prescription_id:
        herb_percentage.append(None)
        continue
    if row["number"]:
        herb_percentage.append(
            row["number"] / presciption_overall_dose.loc[row["source_id"]]
        )
    else:
        herb_percentage.append(None)
prescription2herb["percentage"] = herb_percentage
# Sum the doses of the same medicine
group_prescription2herb = (
    prescription2herb[["source_id", "target_id", "Relation_type", "percentage"]]
    .groupby(["source_id", "target_id", "Relation_type"])
    .sum()
    .reset_index()
)
group_prescription2herb.replace(0.0, None, inplace=True)
group_prescription2herb.to_csv(
    os.path.join(merge_result_dir, "relation/prescription2medicinal_material.csv"),
    index=False,
)
prescription2herb.to_csv(
    os.path.join(merge_result_dir, "prescription2medicinal_material_with_info.csv"),
    index=False,
)
