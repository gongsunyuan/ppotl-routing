"""GML 拓扑加载与前 k 条最短路径（规格 A 节）。

legacy 出处：TRL_Routing/code/routingEnv.py:28-65（read_gml → to_directed →
adjacency_matrix → DiGraph 坍缩、nx.diameter、neighbors）、134-146（k_paths）；
μ 赋值与整数边权在 RoutingEnv.__init__（routingEnv.py:106-108 + 规格 A）。
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx


def load_topology(gml_path: str | Path) -> nx.DiGraph:
    """逐字复刻 legacy 图构建（routingEnv.py:32-36）。

    read_gml 节点 id 为 int 0..N-1（CERNET.gml 北京-西安平行边可能使 read_gml
    返回 MultiGraph——无碍，经 to_directed → adjacency_matrix → DiGraph 后并行边
    自然坍缩为单边，节点=矩阵下标 0..N-1）。
    """
    graph = nx.read_gml(gml_path)
    directed = graph.to_directed()
    adj = nx.adjacency_matrix(directed).todense()
    return nx.DiGraph(adj)


def k_shortest_paths(graph: nx.DiGraph, src: int, dst: int, k: int = 3) -> list[list[int]]:
    """前 k 条（整数权和, 节点字典序）最短路径（legacy routingEnv.py:134-146 稳定化版）。

    shortest_simple_paths(weight="weight") 按权非降 yield；收集至「已收 ≥k 且
    新路径权 > 当前第 k 小权」即止（等权路径须收齐，最终按 (权和, 字典序) 决胜）；
    生成器尽则返回全部（_failure 拓扑可能不足 k 条）。
    """
    collected: list[tuple[int, tuple[int, ...]]] = []
    for path in nx.shortest_simple_paths(graph, src, dst, weight="weight"):
        weight = sum(graph[u][v]["weight"] for u, v in zip(path, path[1:]))
        collected.append((weight, tuple(path)))
        if len(collected) >= k and weight > sorted(w for w, _ in collected)[k - 1]:
            break
    collected.sort()
    return [list(p) for _, p in collected[:k]]
