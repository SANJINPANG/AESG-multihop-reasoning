import os

Datasets = ["NELL23K"]

PreDatasetPathRoot = "/home/sy/Experiments/SeQTO/data"
ProDatasetPathRoot = "/home/sy/Experiments/SeQTO/kbc/src/src_data"

def load_mapping(file_path):
    mapping = {}
    with open(file_path, 'r') as file:
        for line in file:
            entity_or_relation, id_ = line.strip().split('\t')
            mapping[entity_or_relation] = id_
    return mapping

for dataset in Datasets:
    PreDatasetPath = os.path.join(PreDatasetPathRoot, dataset)
    ProDatasetPath = os.path.join(ProDatasetPathRoot, dataset)
    

    os.makedirs(ProDatasetPath, exist_ok=True)
    
    datas = ["train", "valid", "test"]
    
    entity2id = load_mapping(os.path.join(PreDatasetPath, "entity2id.txt"))
    relation2id = load_mapping(os.path.join(PreDatasetPath, "relation2id.txt"))

    numentity = len(entity2id)
    numrelations = len(relation2id)

    for data in datas:
        input_file = os.path.join(PreDatasetPath, f"{data}.txt")
        output_file = os.path.join(ProDatasetPath, data)  
        
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                entity1, relation, entity2 = line.strip().split('\t')
                entity1_id = entity2id.get(entity1, 'UNKNOWN')
                relation_id = relation2id.get(relation, 'UNKNOWN')
                entity2_id = entity2id.get(entity2, 'UNKNOWN')

                outfile.write(f"{entity1_id}\t{relation_id}\t{entity2_id}\n")
    

    with open(os.path.join(ProDatasetPath, 'stats.txt'), 'w') as statsfile:
        statsfile.write(f"numentity: {numentity}\n")
        statsfile.write(f"numrelations: {numrelations}\n")





