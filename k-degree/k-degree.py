import os
import sys
from typing import Any

import networkx as nx  # type: ignore
import numpy as np
import plotly.graph_objects as go  # type: ignore

sys.setrecursionlimit(10**6)


def compute_I(d: np.ndarray[Any, Any]) -> int:
    d_i = d[0]
    res = 0
    for d_j in d:
        res += d_i - d_j
    return res


def c_merge(d: np.ndarray[Any, Any], d1: int, k: int) -> int:
    res = d1 - d[k] + compute_I(d[k + 1 : min(len(d), 2 * k)])
    return res


def c_new(d: np.ndarray[Any, Any], k: int) -> int:
    t = d[k : min(len(d), 2 * k - 1)]
    res = compute_I(t)
    return res


def greedy_rec_algorithm(array_degrees: np.ndarray[Any, Any], k_degree: int, pos_init: int, extension: int) -> bool | None:

    if k_degree == 1: return True

    elif k_degree >= 100 or k_degree < 1:
        raise ValueError("k_degree must be between 1 and 100")

    elif extension >= len(array_degrees) or pos_init + k_degree >= len(array_degrees) or k_degree >= len(array_degrees) - 1:
        for i in range(pos_init, len(array_degrees)):
            array_degrees[i] = array_degrees[i-1 if i >= 1 else i]
        return True

    do_graph_anonymization(array_degrees, pos_init, extension)

    C_merge = c_merge(array_degrees[pos_init:], array_degrees[extension], k_degree)
    C_new = c_new(array_degrees[pos_init:], k_degree)

    if C_merge < C_new:
        greedy_rec_algorithm(array_degrees, k_degree, extension, extension + 1)
    else:
        greedy_rec_algorithm(array_degrees, k_degree, extension, extension + k_degree)
    
    return True

def do_graph_anonymization(array_degrees: np.ndarray[Any, Any], pos_init: int, extension: int) -> None:

    if extension - pos_init == 1:
            array_degrees[pos_init] = array_degrees[pos_init - 1]
    else:
        for i in range(pos_init, extension):
            array_degrees[i] = array_degrees[pos_init]
    
                
def construct_graph(
    tab_index: np.ndarray[Any, Any], anonymized_degree: np.ndarray[Any, Any]
) -> nx.Graph | None:
    graph = nx.Graph()
    if sum(anonymized_degree) % 2 == 1:
        return None
    while True:
        if not all(di >= 0 for di in anonymized_degree):
            return None
        if all(di == 0 for di in anonymized_degree):
            return graph
        v: int = np.random.choice((np.where(np.array(anonymized_degree) > 0))[0])  # type: ignore
        dv = anonymized_degree[v]
        anonymized_degree[v] = 0
        for index in np.argsort(anonymized_degree)[-dv:][::-1]:  # type: ignore
            if index == v:
                return None
            if not graph.has_edge(tab_index[v], tab_index[index]):  # type: ignore
                graph.add_edge(tab_index[v], tab_index[index])  # type: ignore
                anonymized_degree[index] = anonymized_degree[index] - 1


def plot_graph(graph: nx.Graph) -> None:
    pos: dict[str, list[np.float64]] = nx.spring_layout(graph)  # type: ignore
    edge_x: list[np.float64 | None] = []
    edge_y: list[np.float64 | None] = []
    for edge in graph.edges():  # type: ignore
        x0, y0 = pos[edge[0]]  # type: ignore
        x1, y1 = pos[edge[1]]  # type: ignore
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )
    node_x: list[np.float64] = []
    node_y: list[np.float64] = []
    for node in graph.nodes():  # type: ignore
        x, y = pos[node]  # type: ignore
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=True,
            colorscale="Rainbow",
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title="Node Connections",
                xanchor="left",
                titleside="right",
            ),
            line_width=2,
        ),
    )
    node_adjacencies: list[int] = []
    node_text: list[str] = []
    for _, adjacencies in enumerate(graph.adjacency()):  # type: ignore
        connections = len(adjacencies[1])  # type: ignore
        node_adjacencies.append(connections)
        node_text.append(f"# of connections: {connections}")
    node_trace.marker.color = node_adjacencies  # type: ignore
    node_trace.text = node_text
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )
    fig.show()  # type: ignore


if __name__ == "__main__":
    k_degree = int(sys.argv[1])
    file_graph = sys.argv[2]
    G = nx.Graph()
    if os.path.exists(file_graph):
        with open(file_graph) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        for line in content:
            names = line.split(",")
            start_node = names[0]
            if start_node not in G:
                G.add_node(start_node)  # type: ignore
            for index in range(1, len(names)):
                node_to_add = names[index]
                if node_to_add not in G:
                    G.add_node(node_to_add)  # type: ignore
                G.add_edge(start_node, node_to_add)  # type: ignore
    d: list[int] = [x[1] for x in G.degree()]  # type: ignore
    array_index = np.argsort(d)[::-1]  # type: ignore
    array_degrees = np.sort(d)[::-1]  # type: ignore
    # print(f"Array of degrees\n{d}")
    # print(f"Array of degrees sorted (array_degrees)\n{array_degrees}")
    array_degrees_greedy = array_degrees
    greedy_rec_algorithm(array_degrees_greedy, k_degree, 0, k_degree)
    print(array_degrees)
    #print(f"Array anonymized\n{array_degrees_greedy}")
    graph_greedy = construct_graph(array_index, array_degrees_greedy)
    print(graph_greedy)
    if graph_greedy:
        plot_graph(graph_greedy)