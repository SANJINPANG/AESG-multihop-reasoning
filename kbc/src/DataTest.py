import torch
from tqdm import tqdm
import numpy as np
from models import ComplEx


if __name__ == "__main__":
    num_entities = 14541  
    num_relations = 237   
    embedding_dim = 1000  
    batch_size = 128      


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    model = ComplEx((num_entities, num_relations, embedding_dim), embedding_dim)
    model.load_state_dict(torch.load("/home/sy/Experiments/SeQTO/kbc/FB15K237/best_valid.model"))
    model.to(device)  
    model.eval()


    test_file = "/home/sy/Experiments/SeQTO/kbc/src/src_data/FB15K237/test"  

    def load_test_data(file_path):
        test_triples = []
        with open(file_path, 'r') as f:
            for line in f:
                head, relation, tail = map(int, line.strip().split('\t'))
                test_triples.append((head, relation, tail))
        return test_triples

    test_triples = load_test_data(test_file)
    test_triples = torch.tensor(test_triples, dtype=torch.long).to(device)  


def evaluate_batch(model, test_triples, num_entities, hits_at_k=(1, 3, 10), batch_size=128, chunk_size=2048):
    ranks = []
    hits = {k: 0 for k in hits_at_k}

    for i in tqdm(range(0, len(test_triples), batch_size), desc="Evaluating"):
        batch = test_triples[i : i + batch_size]
        head = batch[:, 0]
        relation = batch[:, 1]
        tail = batch[:, 2]

        with torch.no_grad():
            lhs = model.embeddings[0](head)  
            rel = model.embeddings[1](relation)  
            rhs = model.embeddings[0].weight  

            lhs_real, lhs_imag = torch.chunk(lhs, 2, dim=-1)
            rel_real, rel_imag = torch.chunk(rel, 2, dim=-1)
            rhs_real, rhs_imag = torch.chunk(rhs, 2, dim=-1)

            rel_real = rel_real.unsqueeze(1)  
            rel_imag = rel_imag.unsqueeze(1) 

            scores = []
            for start in range(0, num_entities, chunk_size):
                end = min(start + chunk_size, num_entities)
                rhs_chunk_real = rhs_real[start:end]
                rhs_chunk_imag = rhs_imag[start:end]

                chunk_scores = (
                    ((lhs_real @ rhs_chunk_real.T).unsqueeze(-1)) * rel_real
                    + ((lhs_imag @ rhs_chunk_imag.T).unsqueeze(-1)) * rel_real
                    + ((lhs_real @ rhs_chunk_imag.T).unsqueeze(-1)) * rel_imag
                    - ((lhs_imag @ rhs_chunk_real.T).unsqueeze(-1)) * rel_imag
                ).sum(dim=-1)
                scores.append(chunk_scores)

            scores = torch.cat(scores, dim=1)

        sorted_indices = torch.argsort(scores, dim=1, descending=True)
        for idx in range(batch.size(0)):
            tail_idx = tail[idx].item()
            rank = (sorted_indices[idx] == tail_idx).nonzero(as_tuple=True)[0].item() + 1  # 1-based rank
            ranks.append(rank)
            for k in hits_at_k:
                if rank <= k:
                    hits[k] += 1

        torch.cuda.empty_cache()

    mrr = np.mean([1.0 / r for r in ranks])

    hits = {k: v / len(test_triples) for k, v in hits.items()}

    return mrr, hits


mrr, hits = evaluate_batch(model, test_triples, num_entities, hits_at_k=(1, 3, 10), batch_size=batch_size)

print(f"MRR: {mrr:.4f}")
for k, v in hits.items():
    print(f"Hit@{k}: {v:.4f}")

