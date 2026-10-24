"""Finite-space enumeration and an RBF Gaussian-process UCB proposer."""
import itertools,json,math
import torch
from . import search_space,engine


def enumerate_genes(space):
    variables=space['variables'];domains=[search_space.domain(v) for v in variables]
    if math.prod(map(len,domains))>8192:raise ValueError('Enumeration is limited to 8192 combinations; select random, evolutionary, or Bayesian search')
    for values in itertools.product(*domains):yield dict(zip([v['id'] for v in variables],values))


def propose(space,ir,rng,candidates,seen):
    pool=[];hashes=set()
    for _ in range(96):
        genes=search_space.sample(space,ir,rng)
        try:h=engine.fingerprint(search_space.instantiate(ir,space,genes))
        except (ValueError,KeyError,TypeError):continue
        if h in seen or h in hashes:continue
        hashes.add(h);pool.append(genes)
        if len(pool)>=48:break
    if not pool:return None
    if len(candidates)<4:return pool[0]
    engine.rank_candidates(candidates)
    variables=space['variables'];domains=[search_space.domain(v) for v in variables]
    def encode(genes):
        return [float(genes.get(v['id'])==value) for v,domain in zip(variables,domains) for value in domain]
    train=torch.tensor([encode(c['genes']) for c in candidates],dtype=torch.float64);test=torch.tensor([encode(g) for g in pool],dtype=torch.float64)
    y=torch.tensor([c['score'] for c in candidates],dtype=torch.float64);center=y.mean();y=y-center
    kernel=lambda a,b:torch.exp(-torch.cdist(a,b).square()/max(1,len(variables)))
    k=kernel(train,train)+torch.eye(len(train),dtype=torch.float64)*1e-4
    chol=torch.linalg.cholesky(k);cross=kernel(test,train);alpha=torch.cholesky_solve(y[:,None],chol)
    mean=(cross@alpha).flatten()+center;solved=torch.linalg.solve_triangular(chol,cross.T,upper=False)
    std=(1-solved.square().sum(0)).clamp_min(0).sqrt()
    # UCB trades off measured proxy rank and unexplored configurations.
    return pool[int((mean+1.5*std).argmax())]
