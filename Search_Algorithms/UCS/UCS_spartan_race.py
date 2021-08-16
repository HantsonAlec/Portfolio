import networkx as nx
import matplotlib.pyplot as plt

INF = 1000000


def make_graph():
    adys = {}
    adys['a'] = [('g1', 3), ('b', 2), ('f', 4)]
    adys['b'] = [('a', 2), ('d', 4)]
    adys['c'] = [('g1', 2), ('e', 1)]
    adys['d'] = [('b', 4), ('f', 1), ('s', 2)]
    adys['e'] = [('c', 1), ('f', 5)]
    adys['f'] = [('e', 4), ('d', 1), ('s', 2), ('a', 4)]

    adys['g1'] = [('c', 2), ('a', 3)]
    adys['g2'] = [('h', 2), ('j', 2), ('l', 3)]
    adys['g3'] = [('p', 5), ('q', 1)]

    adys['h'] = [('i', 7), ('g2', 2)]
    adys['i'] = [('h', 7), ('j', 8), ('k', 6), ('s', 4)]
    adys['j'] = [('i', 8), ('g2', 2)]

    adys['k'] = [('i', 6), ('l', 1)]
    adys['l'] = [('k', 1), ('g2', 3)]
    adys['m'] = [('s', 3), ('n', 2), ('o', 3)]

    adys['n'] = [('m', 2), ('p', 3)]
    adys['o'] = [('m', 3), ('q', 8)]
    adys['p'] = [('n', 3), ('g3', 5)]

    adys['q'] = [('o', 8), ('g3', 1)]
    adys['s'] = [('f', 2), ('m', 3), ('i', 4), ('d', 2)]
    return adys


# Obtains the element in Q with the minimum distance
def popMinimum(queue, dists):
    min = INF
    minNode = None
    for q in queue:
        if dists[q][1] < min:
            min = dists[q][1]
            minNode = q

    queue.remove(minNode)
    return minNode


# Obtains the adyacencies of u that are not in V
def getAdyacencies(graph, u, visited):
    res = []
    for ady in graph[u]:
        if ady[0] not in visited:
            res.append(ady)

    return res


# Uniform Cost Search Algorithm
def UCS(graph, src, dests):
    queue = []  # Set of nodes being analyzed
    dists = {}  # Node -> (Parent, dist)
    visited = []  # Set of visited vertices
    dists[src] = (None, 0)
    queue.append(src)
    u = src
    while u not in dests:
        u = popMinimum(queue, dists)
        visited.append(u)
        ady_u = getAdyacencies(graph, u, visited)
        for ady in ady_u:  # For each neighbour of the currunt node
            a, w_ua = ady  # a = name of analyzed node, w_ua = cost of analyzed node
            # pa = updated parent of the analyzed node, da = updated cost of the parent analyzed node
            pa, da = dists[a] if a in dists else (None, INF)
            pu, du = dists[u] if u in dists else (
                None, INF)  # pu = updated parent of the node, du = updated cost of the parent node
            if du + w_ua < da:  # check if cost of parent+analyzed node is smaller than updated cost of analyzed parent
                dists[a] = (u, du + w_ua)  # if true update parent and cost
                if a not in queue:
                    queue.append(a)

    return dists


# Show plot
def show_graph(graph):
    g = nx.Graph()
    for src in graph:
        for dest, weight in graph[src]:
            g.add_edge(src, dest, weight=weight)

    pos = nx.spring_layout(g, scale=2)
    nx.draw(g, pos, with_labels=True)
    labels = nx.get_edge_attributes(g, 'weight')
    nx.draw_networkx_edge_labels(g, pos, edge_labels=labels)
    plt.show()


graph = make_graph()
distances = UCS(graph, 's', ['g1', 'g2', 'g3'])
print(distances)
show_graph(graph)
