# TCMM: Unified Intelligence Platform for Traditional Chinese Medicine Modernization
TCMM integrates six high-quality Traditional Chinese Medicine (TCM) and Western medicine databases to construct a modernized TCM database. It includes 20 types of entities, 46 kinds of relations, and 3,447,023 records, incorporating additional information such as biological processes, pathways, anatomic sites, and side effects. In this section, we release the data processing code for building the TCMM knowledge graph.

## Data Preparation
To proceed with data preparation, download the datasets from the official websites of the six databases and extract them to the designated data storage path. Set the project path using the following command:`export PROJECT_DIR="DirectoryToThisProject"`.

### CPMCP[<sup>1</sup>](#ref-1)
`Origin Data Files` Download the files from [CPMCP](http://cpmcp.top) and place them in the directory `$PROJECT_DIR/data/CPMCP`. 
- cmp.csv
- cpm.csv
- disease.csv
- ingredient.csv
- medicinal_material.csv
- mm_symptom.csv
- target.csv
- tcm_symptpm.csv
  
`Entity relations` Visit the [CPMCP](http://cpmcp.top) website to obtain relationship information related to entities. Organize files according to entity ID. For example, place the ingredient information file related to the HERB entity with ID equal to 5 in the directory `$PROJECT_DIR/data/CPMCP/herb/5`.
### TCMBANK[<sup>2</sup>](#ref-2)
`Origin Data Files` Download the files from [TCMBANK](https://tcmbank.cn/) and place them in the directory `$PROJECT_DIR/data/TCMBANK`.
- disease_all.csv
- herb_all.csv
- ingredient_all.csv
- target_all.csv

`Entity relations` Visit the TCMBANK website to obtain relationship information related to entities. Rename the file according to entity ID and organize files according to entity type. For example, place the ingredient information file `TCMBANKHE000001.json` related to the HERB entity with ID equal to 1 in the directory `$PROJECT_DIR/data/TCMBANK/herb`.

### SymMap[<sup>3</sup>](#ref-3)
The following operations are similar to the CPMCP database.
`Origin Data Files` Download the files from [SymMap](http://www.symmap.org/download/) and place them in the directory `$PROJECT_DIR/data/symmap`. 
- [SymMap herb.xlsx](http://www.symmap.org/static/download/V2.0/SymMap%20v2.0%2C%20SMHB%20file.xlsx)
- [symmap_syndrome.xlsx](http://www.symmap.org/static/download/V2.0/SymMap%20v2.0%2C%20SMSY%20file.xlsx)
- [symmap_tcm_symptom.xlsx](http://www.symmap.org/static/download/V2.0/SymMap%20v2.0%2C%20SMSY%20file.xlsx)
- [symmap_mm_symptom.xlsx](http://www.symmap.org/static/download/V2.0/SymMap%20v2.0%2C%20SMMS%20file.xlsx)
- [Symmap target.xlsx](http://www.symmap.org/static/download/V2.0/SymMap%20v2.0%2C%20SMTT%20file.xlsx)
  
`Entity Relations` Visit the SymMap website to obtain relationship information related to the entities. Organize files according to entity ID. For example, place the ingredient information file `ingredient.json` related to the HERB entity with ID equal to `SMHB00001` in the directory `$PROJECT_DIR/data/symmap/herb/SMHB00001`.

### TCMID 2.0[<sup>4</sup>](#ref-4)
Download the prescription file from [TCMID](http://www.megabionet.org/tcmid).
### PharMeBINet[<sup>5</sup>](#ref-5)
`Origin Data File` Download the compressed package of data from [PharMeBInet](https://zenodo.org/records/7009457). Decompress the file and place the two files `nodes.tsv` and `edges.tsv` into the directory `$PROJECT_DIR/data/PharMeBINet`.

### PrimeKG[<sup>6</sup>](#ref-6)
`Origin Data File` Download the compressed package of data from [PrimeKG](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM). Decompress the file and rename the folder to `$PROJECT_DIR/data/primeKG`.

## Entity Alignment & Relation Extraction
To integrate the above five databases, entities are aligned based on attributes such as entity ID, name, and entity description. Additionally, relationships between entities are integrated, and some relations are supplemented based on rule extraction. Data processing scripts are organized by entity type, and the processed results are stored under the directory `$PROJECT_DIR/merge_result`.
### Herb
`extract_herb.py` This script aligns medicinal material entities from different databases, creates new entity types such as tropism and flavor from medicinal material information, and extracts relationships between medicinal materials and entities of new entity types.
- Alignment: Medicinal materials from CPMCP, SymMap, and TCMBANK are aligned based on database ID and the Chinese name of the medicinal material. Similar medicinal materials are merged based on aliases.
- New Entity Types: Four new entity types (Flavour, Tropism, Toxicity, and Property) are created.
- Relations: Medicinal material information is used to extract relationships between medicinal materials and entities of new types.

`Outputs`  After running `python extract_herb.py`, you will get the following files:
```
merge_result   
└───entity
│   │   toxicity.csv
│   │   tropism.csv
│   │   properties.csv
|   |   flavour.csv
|   |   medicinal_material.csv
└───relation
    │   herb2flavour.csv
    │   herb2property.csv
    │   herb2tropism.csv
    │   herb2toxicity.csv
```

### Prescription
Align prescriptions from CPMCP and TCMID, use the large language model `ChatGLM` [<sup>7</sup>](#ref-7) and rule extraction to extract dosage information of Chinese medicinal materials in prescriptions, and convert dosage into a unified unit with proportional representation.

`extract_prescription.py` Use prescription descriptions as identifiers to align prescriptions.

`extract_prescription_herb_relation.py` Use a large language model to extract entity pairs of medicinal materials and dosages and correct some extraction results. Further extract knowledge from unprocessed prescriptions based on rule extraction.

`process_dose_unit.py` Define the relationship between prescriptions and medicinal materials as a weighted relationship. Convert dosage equivalence into a unified unit and represent the dosage of medicinal materials as proportions in the prescription.

`Outputs`  
```Python
python extract_prescription.py
python extract_prescription_herb_relation.py
python process_dose_unit.py
```
After running the above commands, you will get the following files:
```
merge_result   
└───entity
│   │   prescription.csv
└───relation
    │   prescription2medicinal_material.csv
```

### MM symptom
`extract_mm_symptom.py` Align modern medicine symptoms in CPMCP, SymMap, and PharMeBINet by symptom names and definitions.
`Outputs` After running the command `python extract_mm_symptom.py`, you will get the following files:
```
merge_result   
└───entity
│   │   mm_symptom.csv
```

### Syndrome
`extract_syndrome.py` Merge syndromes from SymMap based on syndrome definitions.
`Outputs` After running the command `python extract_syndrome.py`, you will get the following files:
```
merge_result   
└───entity
│   │   syndrome.csv
└───relation
    │   herb2syndrome.csv
```

### Others 
`extract_other_entities.py` Align entities in PharMeBINet and PrimeKG by external database ID.
`Outputs`  After running python extract_other_entities.py, you will get the following files:
```
merge_result   
└───entity
│   │   cellular_component.csv
|   |   biological_process.csv
│   │   molecular_function.csv
|   |   anatomy.csv
|   |   pathway.csv
|   |   pharmacologic_class.csv
|   |   sideeffect.csv
```
### Disease
`extract_disease.py` Align diseases in CPMCP, TCMBANK, PrimeKG, and PharMeBINet by disease names and external database ID. Merge relationships between diseases and other entity types.
  
`Outputs`  After running python extract_disease.py, you will get the following files:
```
merge_result   
└───entity
│   │   disease.csv
└───relation
    │   disease2mm_symptom.csv
    │   disease_is_a_disease.csv
    │   disease_resemble_disease.csv
    |   pathway2disease.csv
```

### Ingredient
`extract_ingredient.py` Align ingredients in CPMCP, TCMBANK, and PharMeBINet by molecule names and SymMap ID. Merge relationships between ingredients and other entity types.

`Outputs`  After running python extract_ingredient.py, you will get the following files:
```
merge_result   
└───entity
│   │   ingredient.csv
└───relation
    │   herb2ingredient.csv
    │   ingredient_resemble_ingredient.csv
    │   ingredient_associate_ingredient.csv
    |   ingredient_belong_to_pharmacologic_class.csv
    |   ingredient_induce_disease.csv
    |   ingredient_treat_disease.csv
    |   ingredient_contraindicate_disease.csv
    |   ingredient_associate_biological_process.csv
    |   ingredient_associate_cellular_component.csv
    |   ingredient_associate_molecular_function.csv
    |   ingredient_associate_pathway.csv
    |   ingredient_cause_sideeffect.csv
    |   ingredient_might_cause_sideeffect.csv
```

### Target
`extract_target.py` Align targets in CPMCP, SymMap, TCMBANK, PrimeKG, and PharMeBINet by Gene Symbol and external database ID. Merge relationships between targets and other entity types.

`Ouputs` After running python extract_target.py, you will get the following files:
```
merge_result   
└───entity
│   │   gene.csv
└───relation
    │   ingredient_downregulate_gene.csv
    │   ingredient_upregulate_gene.csv
    │   ingredient_bind_gene.csv
    |   ingredient_associate_gene.csv
    |   gene_regulate_gene.csv
    |   gene_covary_gene.csv
    |   gene_associate_gene.csv
    |   gene_associate_pathway.csv
    |   disease_downregulate_gene.csv
    |   disease_upregulate_gene.csv
    |   disease_associate_gene.csv
    |   anatomy_downregulate_gene.csv
    |   anatomy_upregulate_gene.csv
    |   anatomy_express_gene.csv
    |   gene2cell_component.csv
    |   gene2biological_process.csv
    |   gene2molecular_function.csv
```
### TCM symptom
`get_symptom_definition.py` Use ChatGPT[<sup>8</sup>](#ref-8) to provide the definition of each symptom.

`symptom_similarity_using_sentence_embedding.py` Use the SentenceTransformer[<sup>9</sup>](#ref-9) model to calculate the similarity between each symptom definition.

`merge_tcm_symptom.py` Merge symptoms with a similarity greater than 0.98 and relationships between symptoms and other entity types.

`add_more_prescription_symptom_map.py` Supplement more knowledge about the relationships between prescriptions and symptoms based on rule extraction.


`Outputs` 
```Python
python extract_disease.py
python add_more_prescription_symptom_map.py
```
After running the above commands, you will get the following files:
```
merge_result   
└───entity
│   │   disease.csv
└───relation
    │   tcm_symptom2locus.csv
    │   herb2symptom.csv
    │   tcm_symptom2mm_symptom.csv
    |   syndrome2tcm_symptom.csv
    |   prescription2symptom.csv
```
## References
<div id="ref-1"></div>

[1]: Sun C, Huang J, Tang R, et al. CPMCP: a database of Chinese patent medicine and compound prescription[J]. Database, 2022, 2022: baac073.

<div id="ref-2"></div>

[2]: Lv Q, Chen G, He H, et al. TCMBank-the largest TCM database provides deep learning-based Chinese-Western medicine exclusion prediction[J]. Signal Transduction and Targeted Therapy, 2023, 8(1): 127.
<div id="ref-3"></div>

[3]: Wu Y, Zhang F, Yang K, et al. SymMap: an integrative database of traditional Chinese medicine enhanced by symptom mapping[J]. Nucleic acids research, 2019, 47(D1): D1110-D1117.
<div id="ref-4"></div>

[4]: Huang L, Xie D, Yu Y, et al. TCMID 2.0: a comprehensive resource for TCM[J]. Nucleic acids research, 2018, 46(D1): D1117-D1120.

<div id="ref-5"></div>

[5]: Königs C, Friedrichs M, Dietrich T. The heterogeneous pharmacological medical biochemical network PharMeBINet[J]. Scientific Data, 2022, 9(1): 393.
<div id="ref-6"></div>

[6]: Chandak P, Huang K, Zitnik M. Building a knowledge graph to enable precision medicine[J]. Scientific Data, 2023, 10(1): 67.
<div id="ref-7"></div>
[7]: Du Z, Qian Y, Liu X, et al. GLM: General Language Model Pretraining with Autoregressive Blank Infilling[C]//Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2022: 320-335.
<div id="ref-8"></div>
[8]: Ouyang L, Wu J, Jiang X, et al. Training language models to follow instructions with human feedback[J]. Advances in Neural Information Processing Systems, 2022, 35: 27730-27744.
<div id="ref-9"></div>
[9] Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks[C]//Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). 2019: 3982-3992.

