import logging
import networkx as nx


logger = logging.getLogger('sna')


def get_modularity(G, degree_matrix, m):
    New_A = nx.adj_matrix(G)
    New_deg = calculate_degrees(New_A, G)
    connected_components = nx.connected_components(G)
    Q = 0
    for component in connected_components:
        A = 0
        k = 0
        for node in component:
            A += New_deg[node]
            k += degree_matrix[node]
        Q = Q + (float(A) - float(k*k)/float(2*m))
    Q = Q/float(2*m)
    return Q


def calculate_degrees(adj_matrix, graph):
    deg_dict = {}
    B = adj_matrix.sum(axis=1)
    for cnt, node_id in enumerate(list(graph.nodes())):
        deg_dict[node_id] = B[cnt, 0]
    return deg_dict


def remove_edge(G, k):
    init_ncc = nx.number_connected_components(G)
    current_ncc = init_ncc
    while current_ncc <= init_ncc:
        betweenness_centralities = nx.edge_betweenness_centrality(G, k=k)
        max_centrality = 0
        for edge, value in betweenness_centralities.items():
            if value > max_centrality:
                max_centrality = value
        for  edge, value in betweenness_centralities.items():
            if value == max_centrality:
                G.remove_edge(edge[0], edge[1])
        current_ncc = nx.number_connected_components(G)
    return G


def detect(graph, args):
    number_of_edges = nx.number_of_edges(graph)
    adj_matrix = nx.adj_matrix(graph)
    degree_matrix = calculate_degrees(adj_matrix, graph)

    max_modularity = 0
    previous_m = None
    m_converging = 0
    while True:
        graph = remove_edge(graph, args.k)
        modularity = get_modularity(
            graph, degree_matrix, number_of_edges)
        if previous_m != None and previous_m > modularity:
            m_converging = m_converging + 1
        if modularity > max_modularity:
            max_modularity = modularity
            communities = list(nx.connected_components(graph))
        if graph.number_of_edges() == 0 or args.converging != None and m_converging > args.converging:
            break

    logger.info(f'\nMax modularity: {max_modularity}')
    logger.info(f'\nNumber of communities: {len(communities)}')
