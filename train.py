import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from dataset import Seq2SeqDataset, TestDataset
from model import TransformerModel
import argparse
import numpy as np
import os
from tqdm import tqdm
import logging
import transformers
from iterative_training import Iter_trainer
import math
from kbc.src.models import ComplEx

from model import KGReasoning
from scipy.sparse import csr_matrix

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-dim", default=256, type=int)
    parser.add_argument("--hidden-size", default=512, type=int)
    parser.add_argument("--num-layers", default=6, type=int)
    parser.add_argument("--batch-size", default=1024, type=int)
    parser.add_argument("--test-batch-size", default=16, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--weight-decay", default=0, type=float)
    parser.add_argument("--num-epoch", default=20, type=int)
    parser.add_argument("--save-interval", default=10, type=int)
    parser.add_argument("--save-dir", default="model_1")
    parser.add_argument("--ckpt", default="ckpt_30.pt")
    parser.add_argument("--dataset", default="FB15K237")
    parser.add_argument("--label-smooth", default=0.5, type=float)
    parser.add_argument("--l-punish", default=False, action="store_true") # during generation, add punishment for length
    parser.add_argument("--beam-size", default=128, type=int) # during generation, beam size
    parser.add_argument("--no-filter-gen", default=False, action="store_true") # during generation, not filter unreachable next token
    parser.add_argument("--test", default=False, action="store_true") # for test mode
    parser.add_argument("--encoder", default=False, action="store_true") # only use TransformerEncoder
    parser.add_argument("--trainset", default="6_rev_rule")
    parser.add_argument("--loop", default=False, action="store_true") # add self-loop instead of <eos>
    parser.add_argument("--prob", default=0, type=float) # ratio of replaced token
    parser.add_argument("--max-len", default=3, type=int) # maximum number of hops considered
    parser.add_argument("--iter", default=False, action="store_true") # switch for iterative training
    parser.add_argument("--iter-batch-size", default=128, type=int)
    parser.add_argument("--smart-filter", default=False, action="store_true") # more space consumed, less time; switch on when --filter-gen
    parser.add_argument("--warmup", default=3, type=float) # warmup steps ratio
    parser.add_argument("--self-consistency", default=False, action="store_true") # self-consistency
    parser.add_argument("--output-path", default=False, action="store_true") # output top correct path in a file (for interpretability evaluation)
    
    models = ['CP', 'ComplEx', 'TransE', 'RESCAL', 'TuckER']
    parser.add_argument('--thrshd', type=float, default=0.001, help='thrshd for neural adjacency matrix')
    parser.add_argument('--kbcModel', choices=models, help="Model in {}".format(models))
    parser.add_argument('--neg_scale', type=int, default=1, help='scaling neural adjacency matrix for negation')
    parser.add_argument('--kbc_path', type=str, default=None, help="kbc model path")
    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--fraction', type=int, default=1, help='fraction the entity to save gpu memory usage')
    
    args = parser.parse_args()
    return args


def rank_tail_entities(head_entity, relation, tail_candidates, model):
    if not tail_candidates:
        return tail_candidates, None

    try:
        device = next(model.parameters()).device

        queries = torch.tensor([[head_entity, relation, tail] for tail in tail_candidates], device=device)

        scores = model.score(queries).squeeze()

        sorted_indices = torch.argsort(scores, descending=True)

        sorted_tail_entities = torch.tensor(tail_candidates, device=device)[sorted_indices]

        return sorted_tail_entities, scores[sorted_indices]

    except Exception as e:
        return tail_candidates, None


def evaluate(model, kbcmodel, dataloader, device, args, true_triples=None, valid_triples=None):
    shibaijishuqi = 0
    model.eval()
    beam_size = args.beam_size
    l_punish = args.l_punish
    max_len = 2 * args.max_len + 1
    restricted_punish = -30
    mrr, hit, hit1, hit3, hit10, count = (0, 0, 0, 0, 0, 0)
    vocab_size = len(model.dictionary)
    eos = model.dictionary.eos()
    bos = model.dictionary.bos()
    rev_dict = dict()
    nol_dict = dict()
    lines = []
    for k in model.dictionary.indices.keys():
        v = model.dictionary.indices[k]
        nol_dict[k] = v
        rev_dict[v] = k
    

    output_dir = f"/home/sy/Experiments/SeQTO/models_new/{args.save_dir}"
    output_file = os.path.join(output_dir, "Result_file.txt")


    with tqdm(dataloader, desc="testing") as pbar:
        for samples in pbar:
            pbar.set_description("MRR: %f, Hit@1: %f, Hit@3: %f, Hit@10: %f" % (mrr/max(1, count), hit1/max(1, count), hit3/max(1, count), hit10/max(1, count)))
            batch_size = samples["source"].size(0)
            candidates = [dict() for i in range(batch_size)]
            candidates_path = [dict() for i in range(batch_size)]

            entity_candidates = [dict() for i in range(batch_size)]
            entity_candidates_path = [dict() for i in range(batch_size)] 
            source = samples["source"].unsqueeze(dim=1).repeat(1, beam_size, 1).to(device)
            prefix = torch.zeros([batch_size, beam_size, max_len], dtype=torch.long).to(device)
            prefix[:, :, 0].fill_(model.dictionary.bos())
            lprob = torch.zeros([batch_size, beam_size]).to(device)
            clen = torch.zeros([batch_size, beam_size], dtype=torch.long).to(device)
            tmp_source = samples["source"]
            tmp_prefix = torch.zeros([batch_size, 1], dtype=torch.long).to(device)
            tmp_prefix[:, 0].fill_(model.dictionary.bos())
            logits = model.logits(tmp_source, tmp_prefix).squeeze()
            if args.no_filter_gen:
                logits = F.log_softmax(logits, dim=-1)
            else:
                restricted = torch.ones([batch_size, vocab_size]) * restricted_punish
                index = tmp_source[:, 1].cpu().numpy()
                for i in range(batch_size):
                    if index[i] in true_triples:
                        if args.smart_filter:
                            restricted[i] = true_triples[index[i]]
                        else:
                            idx = torch.LongTensor(true_triples[index[i]]).unsqueeze(0)
                            restricted[i] = -restricted_punish * torch.zeros(1, vocab_size).scatter_(1, idx, 1) + restricted_punish
                logits = F.log_softmax(logits+restricted.to(device), dim=-1) # batch_size * vocab_size
            logits = logits.view(-1, vocab_size)
            argsort = torch.argsort(logits, dim=-1, descending=True)[:, :beam_size]
            prefix[:, :, 1] = argsort[:, :]
            lprob += torch.gather(input=logits, dim=-1, index=argsort)
            clen += 1
            prefix_copy = prefix.clone()
            lprob_copy = lprob.clone()
            clen_copy = clen.clone()
            target = samples["target"].cpu()                 

            for l in range(2, max_len):
                tmp_prefix = prefix.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                tmp_lprob = lprob.unsqueeze(dim=-1).repeat(1, 1, beam_size)  
                tmp_clen = clen.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                bb = batch_size * beam_size
                all_logits = model.logits(source.view(bb, -1), prefix.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                logits = torch.gather(input=all_logits, dim=2, index=clen.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                if args.no_filter_gen:
                    logits = F.log_softmax(logits, dim=-1)
                else:
                    restricted = torch.ones([batch_size, beam_size, vocab_size]) * restricted_punish
                    hid = prefix[:, :, l-2]
                    if l == 2:
                        hid = source[:, :, 1]
                    rid = prefix[:, :, l-1]
                    if l % 2 == 0:
                        index = vocab_size * rid + hid
                    else:
                        index = rid
                    index = index.cpu().numpy()
                    for i in range(batch_size):
                        for j in range(beam_size):
                            if index[i][j] in true_triples:
                                if args.smart_filter:
                                    restricted[i][j] = true_triples[index[i][j]]
                                else:
                                    idx = torch.LongTensor(true_triples[index[i][j]]).unsqueeze(0)
                                    restricted[i][j] = -restricted_punish * torch.zeros(1, vocab_size).scatter_(1, idx, 1) + restricted_punish
                    logits = F.log_softmax(logits+restricted.to(device), dim=-1)
                argsort = torch.argsort(logits, dim=-1, descending=True)[:, :, :beam_size]
                tmp_clen = tmp_clen + 1
                tmp_prefix = tmp_prefix.scatter_(dim=-1, index=tmp_clen.unsqueeze(-1), src=argsort.unsqueeze(-1))
                tmp_lprob += torch.gather(input=logits, dim=-1, index=argsort)
                tmp_prefix, tmp_lprob, tmp_clen = tmp_prefix.view(batch_size, -1, max_len), tmp_lprob.view(batch_size, -1), tmp_clen.view(batch_size, -1)
                if l == max_len-1:
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :(2*beam_size)]
                else:
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :beam_size]
                prefix = torch.gather(input=tmp_prefix, dim=1, index=argsort.unsqueeze(-1).repeat(1, 1, max_len))
                lprob = torch.gather(input=tmp_lprob, dim=1, index=argsort)
                clen = torch.gather(input=tmp_clen, dim=1, index=argsort)
                for i in range(batch_size):
                    for j in range(beam_size):
                        if prefix[i][j][l].item() == eos:
                            candidate = prefix[i][j][l-1].item()
                            if l_punish:
                                prob = lprob[i][j].item() / int(l / 2)
                            else:
                                prob = lprob[i][j].item()
                            lprob[i][j] -= 10000
                            if candidate not in candidates[i]:
                                if args.self_consistency:
                                    candidates[i][candidate] = math.exp(prob)
                                else:
                                    candidates[i][candidate] = prob
                                candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates[i][candidate] += math.exp(prob)
                                else:
                                    candidates[i][candidate] = max(candidates[i][candidate], prob)
                if l == max_len-1:
                    for i in range(batch_size):
                        for j in range(beam_size*2):
                            candidate = prefix[i][j][l].item()
                            if l_punish:
                                prob = lprob[i][j].item() / int(max_len/2)
                            else:
                                prob = lprob[i][j].item()
                            if candidate not in candidates[i]:
                                if args.self_consistency:
                                    candidates[i][candidate] = math.exp(prob)
                                else:
                                    candidates[i][candidate] = prob
                                candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                            else:
                                if prob > candidates[i][candidate]:
                                    candidates_path[i][candidate] = prefix[i][j].cpu().numpy()
                                if args.self_consistency:
                                    candidates[i][candidate] += math.exp(prob)
                                else:                             
                                    candidates[i][candidate] = max(candidates[i][candidate], prob) 

            prefix = prefix_copy
            lprob = lprob_copy
            clen = clen_copy
            
            for l in range(2, max_len):


                if l == 2 :
                    entity_lprob = lprob
                    entity_clen = clen
                    entity_prefix = prefix
                    source_second_column = samples["source"][:, 1].unsqueeze(1)
                    source_second_column = source_second_column.repeat(1,256)
                    prefix_second_column = prefix[:, :, 1]
                    QTOTriple = torch.stack((source_second_column,prefix_second_column), dim=2)
                    for i in range(QTOTriple.shape[0]):
                        for j in range(QTOTriple.shape[1]):
                            relation_id = rev_dict[QTOTriple[i,j,1].item()]
                            if not str(relation_id).startswith('R'):
                                continue
                            relation_id = int(relation_id[1:]) 
                            if relation_id > 236:
                                relation_id -= 237 
                            file_path = 'neural_adj/FB15K237-20'
                            file_name = f'FB15K237-20_{relation_id}.pt'
                            full_path = os.path.join(file_path, file_name)
                            ProbabilityList = torch.load(full_path)
                            entity_id = int(rev_dict[QTOTriple[i, j, 0].item()])
                            Probability = ProbabilityList[0]
                            Probability_dense = Probability.to_dense()
                            row = Probability_dense
                            nonzero_indices = torch.nonzero(row).squeeze()
                            nonzero_values = row[nonzero_indices]

                            if len(nonzero_values) > 0:
                                entValue1, max_index = torch.max(nonzero_values, dim=0)
                                entIndice = nonzero_indices[max_index].item()
                                entIndice = str(entIndice)
                                entIndice = nol_dict[entIndice]
                                entity_prefix[i, j, 2] = entIndice

                    entity_clen += 1
                    for i in range(entity_prefix.shape[0]):
                        for j in range(entity_prefix.shape[1]):
                            if entity_prefix[i, j, 2] == 0:
                                entity_lprob[i][j] -= 1000

                if l >2 and l%2 == 0 and l < 7:
                    tem_entity_prefix = entity_prefix
                    QTOTriple = entity_prefix[:,:,l-2:l+1]
                    for i in range(QTOTriple.shape[0]):
                        for j in range(QTOTriple.shape[1]):
                            relation_id = rev_dict[QTOTriple[i,j,1].item()]
                            entity_id = QTOTriple[i,j,0].item()
                            if entity_id < 476 or str(rev_dict[entity_id]).startswith('R'):
                                continue
                            if not str(relation_id).startswith('R'):
                                continue
                            relation_id = int(relation_id[1:])
                            if relation_id > 236:
                                relation_id -= 237
                            file_path = 'neural_adj/FB15K237-20'
                            file_name = f'FB15K237-20_{relation_id}.pt'
                            full_path = os.path.join(file_path, file_name)
                            ProbabilityList = torch.load(full_path)
                            entity_id = int(rev_dict[entity_id])
                            Probability = ProbabilityList[0]
                            Probability_dense = Probability.to_dense()
                            row = Probability_dense  
                           
                            nonzero_indices = torch.nonzero(row).squeeze()
                            nonzero_values = row[nonzero_indices]

                            if len(nonzero_values) > 0:
                                entValue1, max_index = torch.max(nonzero_values, dim=0)
                                entIndice = nonzero_indices[max_index].item()
                                entIndice = str(entIndice)
                                entIndice = nol_dict[entIndice]
                                tem_entity_prefix[i, j, 2] = entIndice

                    entity_prefix[:,:,l] = tem_entity_prefix[:,:,l]
                    for i in range(entity_prefix.shape[0]):
                        for j in range(entity_prefix.shape[1]):
                            if entity_prefix[i, j, l] == 0:
                                entity_lprob[i][j] -= 1000
                    entity_clen += 1

                if l%2 != 0:
                    tmp_prefix = entity_prefix.unsqueeze(dim=2).repeat(1, 1, beam_size, 1)
                    tmp_lprob = entity_lprob.unsqueeze(dim=-1).repeat(1, 1, beam_size)    
                    tmp_clen = entity_clen.unsqueeze(dim=-1).repeat(1, 1, beam_size)
                    bb = batch_size * beam_size
                    all_logits = model.logits(source.view(bb, -1), entity_prefix.view(bb, -1)).view(batch_size, beam_size, max_len, -1)
                    logits = torch.gather(input=all_logits, dim=2, index=entity_clen.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, vocab_size)).squeeze(2)
                    logits = F.log_softmax(logits, dim=-1)
                    argsort = torch.argsort(logits, dim=-1, descending=True)[:, :, :beam_size]

                    tmp_clen = tmp_clen + 1
                    tmp_prefix = tmp_prefix.scatter_(dim=-1, index=tmp_clen.unsqueeze(-1), src=argsort.unsqueeze(-1))
                    tmp_lprob += torch.gather(input=logits, dim=-1, index=argsort)
                    tmp_prefix, tmp_lprob, tmp_clen = tmp_prefix.view(batch_size, -1, max_len), tmp_lprob.view(batch_size, -1), tmp_clen.view(batch_size, -1)
                    argsort = torch.argsort(tmp_lprob, dim=-1, descending=True)[:, :beam_size]     

                                  
                    entity_prefix = torch.gather(input=tmp_prefix, dim=1, index=argsort.unsqueeze(-1).repeat(1, 1, max_len))
                    entity_lprob = torch.gather(input=tmp_lprob, dim=1, index=argsort)
                    entity_clen = torch.gather(input=tmp_clen, dim=1, index=argsort)

                    
                
                for i in range(batch_size):
                    for j in range(beam_size):
                        if entity_prefix[i][j][l].item() == eos and entity_lprob[i][j] > -100:
                            entity_candidate = entity_prefix[i][j][l-1].item()
                            if l_punish:
                                prob = lprob[i][j].item() / int(l / 2)
                            else:
                                prob = lprob[i][j].item()
                            lprob[i][j] -= 10000
                            if entity_candidate not in entity_candidates[i]:
                                if args.self_consistency:
                                    entity_candidates[i][entity_candidate] = math.exp(prob)
                                else:
                                    entity_candidates[i][entity_candidate] = prob
                                entity_candidates_path[i][entity_candidate] = entity_prefix[i][j].cpu().numpy()
                            else:
                                if prob > entity_candidates[i][entity_candidate]:
                                    entity_candidates_path[i][entity_candidate] = entity_prefix[i][j].cpu().numpy()
                                if args.self_consistency:
                                    entity_candidates[i][entity_candidate] += math.exp(prob)
                                else:
                                    entity_candidates[i][entity_candidate] = max(entity_candidates[i][entity_candidate], prob)
                if l == max_len-1:
                    for i in range(batch_size):
                        for j in range(beam_size):
                            entity_candidate = entity_prefix[i][j][l].item()
                            if l_punish:
                                prob = lprob[i][j].item() / int(max_len/2)
                            else:
                                prob = lprob[i][j].item()
                            if entity_candidate not in entity_candidates[i]:
                                if args.self_consistency:
                                    entity_candidates[i][entity_candidate] = math.exp(prob)
                                else:
                                    entity_candidates[i][entity_candidate] = prob
                                entity_candidates_path[i][entity_candidate] = entity_prefix[i][j].cpu().numpy()
                            else:
                                if prob > entity_candidates[i][entity_candidate]:
                                    entity_candidates_path[i][entity_candidate] = entity_prefix[i][j].cpu().numpy()
                                if args.self_consistency:
                                    entity_candidates[i][entity_candidate] += math.exp(prob)
                                else:                             
                                    entity_candidates[i][entity_candidate] = max(entity_candidates[i][entity_candidate], prob)
                
            target = samples["target"].cpu()




            with open(output_file, "a") as result_file:
                
                for i in range(batch_size):
                    hid = samples["source"][i][1].item()
                    rid = samples["source"][i][2].item()
                    index = vocab_size * rid + hid
                    
                    if index in valid_triples:
                        mask = valid_triples[index]
                        for tid in candidates[i].keys():
                            if tid == target[i].item():
                                continue
                            elif args.smart_filter:
                                if mask[tid].item() == 0:
                                    candidates[i][tid] -= 100000
                            else:
                                if tid in mask:
                                    candidates[i][tid] -= 100000
                    count += 1

                    candidate_ = sorted(zip(candidates[i].items(), candidates_path[i].items()), key=lambda x: x[0][1], reverse=True)
                    entity_candidate_ = sorted(zip(entity_candidates[i].items(), entity_candidates_path[i].items()), key=lambda x: x[0][1], reverse=True)

                    entity_candidate = [pair[0][0] for pair in entity_candidate_]
                    entity_candidate_path = [pair[1][1] for pair in entity_candidate_]

                    candidate = [pair[0][0] for pair in candidate_]
                    candidate_path = [pair[1][1] for pair in candidate_]

                    candidate[10:20] = entity_candidate
                    candidate_path[10:20] = entity_candidate_path

                    candidate = candidate[:20]
                    candidate_path = candidate_path[:20]

                    head_entity = hid
                    if rid < 236:
                        relation = rid

                        rerank_candidate, RerankResult = rank_tail_entities(head_entity, relation, candidate, kbcmodel)
                        if RerankResult == None:
                            shibaijishuqi += 1
                        else:
                            rerank_list = rerank_candidate.tolist()
                            candidate = rerank_list
                            index_map = {value: idx for idx, value in enumerate(candidate)}
                            candidate_path = [candidate_path[index_map[value]] for value in rerank_list]
                            candidate = rerank_list                    

                    candidate = torch.from_numpy(np.array(candidate))
                    ranking = (candidate[:] == target[i]).nonzero()
                    path_token = rev_dict[hid] + " " + rev_dict[rid] + " " + rev_dict[target[i].item()] + '\t'

                    if ranking.nelement() != 0:
                        for j in range(ranking.shape[0]):
                            path = candidate_path[ranking[j]]
                            for token in path[1:-1]:
                                path_token += (rev_dict[token] + ' ')
                            path_token += (rev_dict[path[-1]] + '\t')
                            path_token += str(ranking[j].item())
                            ranking[j] = 1 + ranking[j].item()
                            mrr += (1 / ranking[j])
                            hit += 1
                            if ranking[j] <= 1:
                                hit1 += 1
                            if ranking[j] <= 3:
                                hit3 += 1
                            if ranking[j] <= 10:
                                hit10 += 1
                    else:
                        path_token += "wrong"
                    lines.append(path_token + '\n')


    if args.output_path:
        with open("test_output_squire.txt", "a") as f:
            f.writelines(lines)
    logging.info("[MRR: %f] [Hit@1: %f] [Hit@3: %f] [Hit@10: %f]" % (mrr/count, hit1/count, hit3/count, hit10/count))
    return hit/count, hit1/count, hit3/count, hit10/count

def read_triples(filenames, nrelation, datapath):
    adj_list = [[] for i in range(nrelation)]
    edges_all = set()
    edges_vt = set()
    for filename in filenames:
        with open(filename) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                adj_list[int(r)].append((int(h), int(t)))
    for filename in ['valid1.txt', 'test1.txt']:
        with open(os.path.join(datapath, filename)) as f:
            for line in f.readlines():
                h, r, t = line.strip().split('\t')
                edges_all.add((int(h), int(r), int(t)))
                edges_vt.add((int(h), int(r), int(t)))
    with open(os.path.join(datapath, "train1.txt")) as f:
        for line in f.readlines():
            h, r, t = line.strip().split('\t')
            edges_all.add((int(h), int(r), int(t)))

    return adj_list, edges_all, edges_vt

def read_triples_sparse(filenames, nrelation, nentities, datapath):
    # Initialize sparse matrix list for relations
    adj_matrices = [csr_matrix((nentities, nentities)) for _ in range(nrelation)]
    
    # Temporary lists to store matrix entries
    rows, cols, data = [[] for _ in range(nrelation)], [[] for _ in range(nrelation)], [[] for _ in range(nrelation)]
    
    # Read files to populate the temporary lists
    for filename in filenames:
        with open(filename) as f:
            for line in f:
                h, r, t = map(int, line.strip().split('\t'))
                rows[r].append(h)
                cols[r].append(t)
                data[r].append(1) # assuming binary relation (either edge exists or it doesn't)
    
    # Convert temporary lists into sparse CSR matrices
    for r in range(nrelation):
        adj_matrices[r] = csr_matrix((data[r], (rows[r], cols[r])), shape=(nentities, nentities))
    
    edges_all = set()
    edges_vt = set()
    
    # Process validation and test files
    for filename in ['valid1.txt', 'test1.txt']:
        with open(os.path.join(datapath, filename)) as f:
            for line in f:
                h, r, t = map(int, line.strip().split('\t'))
                edges_all.add((h, r, t))
                edges_vt.add((h, r, t))
    
    # Process training file
    with open(os.path.join(datapath, "train1.txt")) as f:
        for line in f:
            h, r, t = map(int, line.strip().split('\t'))
            edges_all.add((h, r, t))

    return adj_matrices, edges_all, edges_vt


def train(args):

    args.dataset = os.path.join('data', args.dataset)
    save_path = os.path.join('models_new', args.save_dir)
    ckpt_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    logging.basicConfig(level=logging.DEBUG,
                    filename=save_path+'/train.log',
                    filemode='a',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    logging.info(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = Seq2SeqDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, args=args)
    valid_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="valid_triples.txt")
    test_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="test_triples.txt")
    train_valid, eval_valid = train_set.get_next_valid()
    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True)
    
    model = TransformerModel(args, train_set.dictionary).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = len(train_loader)
    total_step_num = len(train_loader) * args.num_epoch
    warmup_steps = total_step_num / args.warmup
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, total_step_num)
    
    # evaluate(model, test_loader, device, args, train_valid, eval_valid)
    if args.iter:
        iter_trainer = Iter_trainer(args.dataset, args.iter_batch_size, 32, 4)
        iter_epoch = []
        max_len = args.max_len
        total = 0
        for i in range(1, max_len+1):
            total += (1/i)
        epochs = 0
        for i in range(1, max_len+1):
            iter_epoch.append(int(args.num_epoch/(total*i)))
            epochs += int(args.num_epoch/(total*i))
        iter_epoch[-1] += (args.num_epoch-epochs)
        curr_iter = -1
        curr_iter_epoch = 0
        logging.info(
                    "[Iter0: %d] [Iter1: %d] [Iter2: %d]"
                    % (iter_epoch[0], iter_epoch[1], iter_epoch[2])
                    )
    steps = 0
    for epoch in range(args.num_epoch):
        if args.iter:
            if curr_iter_epoch == 0: # start next iteration
                curr_iter += 1
                curr_iter_epoch = iter_epoch[curr_iter]
                # label new dataset
                if curr_iter > 0:
                    logging.info("--------Iterating--------")
                    (src_lines, tgt_lines) = iter_trainer.get_iter(model, curr_iter)
                    train_set.src_lines += src_lines
                    train_set.tgt_lines += tgt_lines
                    train_loader = DataLoader(train_set, batch_size=args.batch_size, collate_fn=train_set.collate_fn, shuffle=True)
                # new scheduler
                step_num = len(train_loader) * curr_iter_epoch
                warmup_steps = step_num / args.warmup
                if curr_iter != 0:
                    optimizer = optim.Adam(model.parameters(), lr=args.lr / 5, weight_decay=args.weight_decay) # fine-tuning with smaller lr
                    warmup_steps = 0
                scheduler = transformers.get_linear_schedule_with_warmup(optimizer, warmup_steps, step_num)
            curr_iter_epoch -= 1
        model.train()
        with tqdm(train_loader, desc="training") as pbar:
            losses = []
            for samples in pbar:
                optimizer.zero_grad()
                loss = model.get_loss(**samples)
                loss.backward()
                optimizer.step()
                scheduler.step()
                steps += 1
                losses.append(loss.item())
                pbar.set_description("Epoch: %d, Loss: %0.8f, lr: %0.6f" % (epoch + 1, np.mean(losses), optimizer.param_groups[0]['lr']))
        logging.info(
                "[Epoch %d/%d] [train loss: %f]"
                % (epoch + 1, args.num_epoch, np.mean(losses))
                )
        if (epoch % args.save_interval == 0 and epoch != 0) or (epoch == args.num_epoch - 1):
            torch.save(model.state_dict(), ckpt_path + "/ckpt_{}.pt".format(epoch + 1))
            with torch.no_grad():
                evaluate(model, test_loader, device, args, train_valid, eval_valid)



def load_kbc(model_path, device, nentity, nrelation):
    model = ComplEx(sizes=[nentity, nrelation, nentity], rank=1000, init_size=1e-3)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.to(device)
    return model

def checkpoint(args):
    args.dataset = os.path.join('data', args.dataset)
    save_path = os.path.join('models_new', args.save_dir)
    ckpt_path = os.path.join(save_path, 'checkpoint')
    if not os.path.exists(ckpt_path):
        print("Invalid path!")
        return
    logging.basicConfig(level=logging.DEBUG,
                    filename=save_path+'/test.log',
                    filemode='w',
                    format=
                    '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_set = Seq2SeqDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, args=args)
    test_set = TestDataset(data_path=args.dataset+"/", vocab_file=args.dataset+"/vocab.txt", device=device, src_file="test_triples.txt")
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, collate_fn=test_set.collate_fn, shuffle=True)
    train_valid, eval_valid = train_set.get_next_valid()
    model = TransformerModel(args, train_set.dictionary)
    model.load_state_dict(torch.load(os.path.join(ckpt_path, args.ckpt)))
    model.args = args
    model = model.to(device)




    data_path =args.dataset
    with open('%s/stats.txt'%data_path) as f:
        entrel = f.readlines()
        nentity = int(entrel[0].split(' ')[-1])
        nrelation = int(entrel[1].split(' ')[-1])   
    
    args.nentity = nentity
    args.nrelation = nrelation  
    kbc_model = load_kbc(args.kbc_path, device, args.nentity, args.nrelation)


    with torch.no_grad():
        evaluate(model, kbc_model, test_loader, device, args, train_valid, eval_valid)
    

if __name__ == "__main__":
    args = get_args()
    if args.test:
        checkpoint(args)
    else:
        train(args)
