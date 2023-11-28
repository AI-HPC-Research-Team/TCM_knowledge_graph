from transformers import AutoTokenizer, AutoModel
from torch.nn import functional as F
import pandas as pd
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import math
from sentence_transformers import SentenceTransformer


def get_sentence_embedding(model, tokenizer, sentences, device):
    inputs = tokenizer(sentences, return_tensors="pt", pad_to_max_length=True)
    inputs = inputs.to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    output = model(input_ids, attention_mask=attention_mask)[0]
    sentence_rep = output.mean(dim=1)
    return sentence_rep.detach().cpu()


data_root_dir = "../data"
symmap_data_dir = "../data/symmap"

os.environ["CURL_CA_BUNDLE"] = ""
device = "cuda:0"

symptom_definition = json.load(
    open(
        os.path.join(data_root_dir, "resource/symptom_definition.json")
    )
)
symptom_definition_map_dict = {}
for group_dict in symptom_definition:
    symptom_definition_map_dict.update(group_dict)

symptom = pd.read_excel(os.path.join(symmap_data_dir, "symmap_tcm_symptom.xlsx"))
symptom["Description"] = symptom.TCM_symptom_name.apply(
    lambda x: symptom_definition_map_dict[x]
    if x in symptom_definition_map_dict
    else np.nan
)
symptom = symptom[symptom["Description"].notnull()]
symptom.reset_index(inplace=True, drop=True)


model = SentenceTransformer("all-MiniLM-L6-v2")


num_samples = symptom.shape[0]
batch_size = 32
batch_num = math.ceil(num_samples / batch_size)
# batch_num = 1
result = []
for group_index in tqdm(range(batch_num)):
    start_index = group_index * batch_size
    end_index = min((group_index + 1) * batch_size, num_samples)
    batch_inputs = symptom.loc[start_index : end_index - 1, "Description"]
    # sentence_embedding = get_sentence_embedding(model, tokenizer, batch_inputs.tolist(), device)
    sentence_embedding = model.encode(batch_inputs.tolist())
    result.append(sentence_embedding)

del model
torch.cuda.empty_cache()
all_sentence_embedding = np.concatenate(result)
all_sentence_embedding = torch.tensor(all_sentence_embedding)

cos_sim_matrix = np.zeros((num_samples, num_samples))
for group_name, group in symptom.groupby(["Symptom_locus", "Symptom_property"]):
    group_embedding = all_sentence_embedding[group.index].to(device)
    cos_sim = F.cosine_similarity(
        group_embedding.unsqueeze(0), group_embedding.unsqueeze(1), dim=-1
    )
    cos_sim_matrix[np.ix_(group.index, group.index)] = cos_sim.cpu().numpy()
# all_sentence_embedding.to(device)
# cos_sim = F.cosine_similarity(all_sentence_embedding.unsqueeze(0), all_sentence_embedding.unsqueeze(1), dim=-1)
cos_sim_df = pd.DataFrame(
    cos_sim_matrix, index=symptom.TCM_symptom_name, columns=symptom.TCM_symptom_name
)
cos_sim_df.to_csv("cos_sim_df.csv")

group = []
merged_symptom_map = {}
for index, symptom_cos_sim in cos_sim_df.iterrows():
    sim_symptom = symptom_cos_sim[index:][symptom_cos_sim[index:] > 0.98].index.tolist()
    group_index = len(group)
    group_index = min(
        [
            merged_symptom_map[one_sym]
            if one_sym in merged_symptom_map
            else group_index
            for one_sym in sim_symptom
        ]
    )
    if group_index == len(group):
        group.append(sim_symptom)
    else:
        group[group_index].extend(sim_symptom)
    for one_sym in sim_symptom:
        merged_symptom_map[one_sym] = group_index

group_df = pd.DataFrame(
    {
        "symptom": list(merged_symptom_map.keys()),
        "group_id": list(merged_symptom_map.values()),
    }
)
group_df = group_df.groupby("group_id").apply(lambda x: ";".join(x.symptom.tolist()))
group_df.to_csv(
    os.path.join(data_root_dir, "resource/threshold_98.csv"), index=False, header=False
)
