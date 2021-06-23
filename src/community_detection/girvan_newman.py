import logging
import networkx as nx
import src.utils as utils


logger = logging.getLogger('sna')


def get_modularity(G, degree_dict, number_of_edges):
    """Calculate the modularity.

    Args:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
        degree_dict (Dict): A dictionary of node ids as keys and degrees as values.
        number_of_edges (int): Number of edges in the graph.

    Returns:
        Q (float): Graph modularity.
    """

    new_adj_matrix = nx.adj_matrix(G)
    new_degree_dict = utils.calculate_node_degrees(G, new_adj_matrix)
    connected_components = nx.connected_components(G)

    Q = 0
    for component in connected_components:
        A = 0
        k = 0
        for node in component:
            A = A + new_degree_dict[node]
            k = k + degree_dict[node]
        Q = Q + (float(A) - float(k*k)/float(2*number_of_edges))
    Q = Q/float(2*number_of_edges)
    return Q


def remove_edge(G, k):
    """Calculate the modularity.

    Args:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
        k (int): Number of node samples to estimate betweenness.

    Returns:
        G (networkx.classes.graph.Graph): A NetworkX graph object.
    """

    init_ncc = nx.number_connected_components(G)
    current_ncc = init_ncc
    while current_ncc <= init_ncc:
        betweenness_centralities = nx.edge_betweenness_centrality(G, k=k)
        max_centrality = 0
        for edge, value in betweenness_centralities.items():
            if value > max_centrality:
                max_centrality = value
        for edge, value in betweenness_centralities.items():
            if value == max_centrality:
                G.remove_edge(edge[0], edge[1])
        current_ncc = nx.number_connected_components(G)
    return G


def detect(graph_original, args):
    """Perform the Girvan-Newman algorithm.

    Args:
        graph_original (networkx.classes.graph.Graph): A NetworkX graph object.
        args (argparse.Namespace): The provided application arguments.

    Returns:
        number_of_communities (int): Number of communities in the graph.
    """

    graph = graph_original.copy()
    number_of_edges = nx.number_of_edges(graph)
    adj_matrix = nx.adj_matrix(graph)
    degree_dict = utils.calculate_node_degrees(graph, adj_matrix)

    max_modularity = 0
    previous_m = None
    m_converging = 0
    while True:
        graph = remove_edge(graph, args.k)
        modularity = get_modularity(
            graph, degree_dict, number_of_edges)
        if previous_m != None and previous_m > modularity:
            m_converging = m_converging + 1
        if modularity > max_modularity:
            max_modularity = modularity
            communities = list(nx.connected_components(graph))
        if graph.number_of_edges() == 0 or args.converging != None and m_converging > args.converging:
            break

    communities_dict = {}
    for i, community in enumerate(communities):
        for j in community:
            communities_dict[j] = i

    number_of_communities = len(communities)
    return number_of_communities
