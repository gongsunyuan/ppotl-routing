import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh


def spectral_embedding(ug, dim):
    nodes = list(ug.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    for u, v in ug.edges():
        A[idx[u], idx[v]] = 1.0
        A[idx[v], idx[u]] = 1.0
    deg = A.sum(axis=1)
    Dm = np.diag(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = np.eye(n) - Dm @ A @ Dm
    if n <= dim + 1:
        vals, vecs = np.linalg.eigh(L)
    else:
        vals, vecs = eigsh(csr_matrix(L), k=dim + 1, which="SM")
    order = np.argsort(vals)
    emb = vecs[:, order[1:dim + 1]]
    if emb.shape[1] < dim:
        emb = np.hstack([emb, np.zeros((n, dim - emb.shape[1]))])
    norm = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norm, 1e-12)
    return emb


class GRLPSAgent:
    needs_pretrain = True

    def __init__(self, state_dim, action_dim, cfg, device="cpu", seed=0, embedding_matrix=None):
        from .dqn import DQNAgent
        self.emb_dim = cfg["grlps"]["embedding_dim"]
        self.emb_matrix = embedding_matrix
        self.dqn = DQNAgent(state_dim + 2 * self.emb_dim, action_dim, cfg, device, seed)
        self.action_dim = action_dim

    def augment(self, state, src_idx, dst_idx):
        if self.emb_matrix is None:
            z = np.zeros(self.emb_dim)
            return np.concatenate([state, z, z])
        return np.concatenate([state, self.emb_matrix[src_idx], self.emb_matrix[dst_idx]])

    def select_action(self, aug_state, greedy=False):
        return self.dqn.select_action(aug_state, greedy)

    def observe(self, s, a, r, s_next, done):
        self.dqn.observe(s, a, r, s_next, done)

    def save(self, path, norms=None):
        self.dqn.save(path, norms)

    def load(self, path):
        return self.dqn.load(path)
