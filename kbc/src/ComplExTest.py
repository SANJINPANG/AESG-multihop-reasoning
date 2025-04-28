import torch
from models import ComplEx 
from tqdm import tqdm

def evaluate_matrix(model, test_data, all_entities, batch_size=1024):
    """
    使用矩阵化操作评估ComplEx模型的MRR和Hit@k指标。
    """
    mrr = 0
    hits_at_k = {1: 0, 3: 0, 10: 0}

    for batch_start in tqdm(range(0, len(test_data), batch_size), desc="Evaluating", unit="batch"):
        batch_end = min(batch_start + batch_size, len(test_data))
        batch = test_data[batch_start:batch_end]
        s, r, o = batch[:, 0], batch[:, 1], batch[:, 2]

        s_emb = model.embeddings[0](s)
        r_emb = model.embeddings[1](r)
        o_emb = model.embeddings[0](o)

        s_re, s_im = s_emb[:, :model.rank], s_emb[:, model.rank:]
        r_re, r_im = r_emb[:, :model.rank], r_emb[:, model.rank:]
        o_re, o_im = o_emb[:, :model.rank], o_emb[:, model.rank:]

        all_entities_emb = model.embeddings[0].weight
        all_entities_re, all_entities_im = all_entities_emb[:, :model.rank], all_entities_emb[:, model.rank:]

        all_scores = (
            (s_re * r_re - s_im * r_im) @ all_entities_re.T +
            (s_re * r_im + s_im * r_re) @ all_entities_im.T
        )

        true_scores = torch.gather(all_scores, 1, o.unsqueeze(1))

        sorted_indices = torch.argsort(all_scores, dim=1, descending=True)
        true_ranks = (sorted_indices == o.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1

        mrr += torch.sum(1.0 / true_ranks).item()

        for k in hits_at_k.keys():
            hits_at_k[k] += torch.sum(true_ranks <= k).item()

    num_samples = len(test_data)
    mrr /= num_samples
    for k in hits_at_k:
        hits_at_k[k] /= num_samples

    return mrr, hits_at_k


def load_stats(stats_file):
    with open(stats_file, 'r') as f:
        stats = {}
        for line in f:
            key, value = line.strip().split(': ')
            stats[key] = int(value)
    return stats['numentity'], stats['numrelations']

def load_test_data(file_path):
    test_data = []
    with open(file_path, 'r') as f:
        for line in f:
            s, r, o = map(int, line.strip().split('\t'))
            test_data.append([s, r, o])
    return torch.tensor(test_data, dtype=torch.long)

if __name__ == "__main__":
    stats_file = "/home/sy/Experiments/SeQTO/kbc/src/src_data/NELL23K/stats.txt"
    numentity, numrelations = load_stats(stats_file)

    sizes = (numentity, numrelations, numentity)
    rank = 1000
    model = ComplEx(sizes, rank)

    model.load_state_dict(torch.load('/home/sy/Experiments/SeQTO/kbc/NELL23K/best_valid.model'))
    model.eval()

    test_file = "/home/sy/Experiments/SeQTO/kbc/src/src_data/NELL23K/test"
    test_data = load_test_data(test_file)

    all_entities = torch.arange(sizes[0])

    mrr, hits = evaluate_matrix(model, test_data, all_entities)

    print(f"MRR: {mrr:.4f}")
    print(f"Hit@1: {hits[1]:.4f}")
    print(f"Hit@3: {hits[3]:.4f}")
    print(f"Hit@10: {hits[10]:.4f}")
