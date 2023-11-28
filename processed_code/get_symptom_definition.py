import openai
import pandas as pd
import os
import re
import ast
from openai.embeddings_utils import get_embedding, cosine_similarity
import time
import json
from tqdm import tqdm

if __name__ == "__main__":
    symmap_dir = "../data/symmap"
    data_root_dir = "../data"
    openai.organization = "your orgranization name"
    openai.api_key = "your api key"
    symptom = pd.read_excel(os.path.join(symmap_dir, "symmap_tcm_symptom.xlsx"))
    example = []
    for group_name, group in symptom.groupby("Symptom_definition"):
        temp = []
        if len(group) == 1:
            continue
        for index, row in group.iterrows():
            temp.append((row["TCM_symptom_name"], group_name))
        example.append(temp)
    example_group1 = [item[0] for item in example[-1]]
    example_group2 = [item[0] for item in example[-3]]
    symptom.groupby(["Symptom_locus", "Symptom_property"])
    # Group symptom names according to symptom location and property
    synonymous_symptom_groups = []
    for index, group in symptom.groupby(["Symptom_locus", "Symptom_property"]):
        if len(group) == 1:
            continue
        temp_group = []
        for index, row in group.iterrows():
            temp_group.append(row["TCM_symptom_name"])
        synonymous_symptom_groups.append(temp_group)
    explanation = {}
    for item in example[-1] + example[-3]:
        explanation[item[0]] = item[1]

    message_log = [
        {"role": "system", "content": "You are a traditional Chinese medicine doctor"}
    ]

    group_results = []
    symptom_definition = []
    explanation = {
        "乳房结节": "乳房内有可触及的圆形或椭圆形结节",
        "乳房疼痛": "乳房出现疼痛或不适",
        "乳房有块": "乳房内有可触及的肿块或硬块",
        "乳房胀痛": "乳房出现胀痛或不适",
        "乳房肿块": "乳房内有可触及的肿块或硬块",
        "乳房肿痛": "乳房出现肿痛或不适",
        "乳癖": "乳房内出现结节、囊肿、增生等改变",
        "乳腺增生": "乳房内组织增生，形成类似结节、肿块或囊肿的改变",
        "乳胀痛": "乳房出现胀痛或不适。",
        "乳汁不通": "乳汁无法流出或流量减少。",
        "乳汁不下": "乳汁无法流出或流量减少",
        "乳汁郁积": "乳汁在乳腺内积聚，无法正常流出",
        "乳肿痛": "乳房出现肿痛或不适。",
        "停乳": "哺乳期结束后，乳汁停止分泌。",
    }
    for synonymous_syptoms in tqdm(synonymous_symptom_groups[1:]):
        # time.sleep(45)
        # prompt = "Imagine that you have been given a task to categorize a long list of symptoms into separate groups based on their definitions. This task can be accomplished by following two critical steps: \
        # 1. Provide clear and concise definitions for each symptom, using easy-to-understand language. \
        # 2. Group symptoms that share same definitions or meanings into the same list according the above definitions.The number of group may be greater than 2.  \
        # For example: \n\n \
        # Symptom list: {} \n \
        # Definition: {} \n  \
        # Group result: {}, {} \n \
        # Symptom list: {} ".format(example_group1+example_group2, explanation, example_group1, example_group2, synonymous_syptoms)
        prompt = "Imagine that you have been given a task to give the definitions of a list of symptoms.Please provide a json object of clear and concise definitions for each symptom, using easy-to-understand language. Do not give any pathogen of the symptom. \
        For example: \n\n \
        Symptom list: {} \n \
        Definition: {} \n  \
        Symptom list: {} \n \
        Definiton:".format(
            list(explanation.keys()), explanation, synonymous_syptoms
        )
        message_log = [
            {
                "role": "system",
                "content": "You are a traditional Chinese medicine doctor",
            }
        ]
        message_log.append({"role": "user", "content": prompt})
        time.sleep(60)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message_log,
            max_tokens=2000,
            stop=None,
            temperature=0.7,
        )
        # result = re.search('(?<=result: \n).*',
        #                    response.choices[0].message.content, flags=re.DOTALL).group()
        result = response.choices[0].message.content
        try:
            symptom_definition.append(ast.literal_eval(result))
        except:
            print(synonymous_syptoms)
    with open(
        os.path.join(data_root_dir, "resource/symptom_definition.json"), "w"
    ) as f:
        json.dump(symptom_definition, f)
