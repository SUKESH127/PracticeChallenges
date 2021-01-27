def minCostToConnectNodes(N: int, edges: List[List[int]], newEdges : List[List[int]]) -> int:
    # WRITE YOUR BRILLIANT CODE HERE
    num_nodes, base_mst, poss_mst = N, edges, newEdges
    uf = {}

    # create union find for the initial edges given 
    def find(edge):
        uf.setdefault(edge, edge)
        if uf[edge] != edge:
            uf[edge] = find(uf[edge])
        return uf[edge]

    def union(edge1, edge2):
        uf[find(edge1)] = find(edge2)

    for e1, e2 in base_mst:
        if find(e1) != find(e2):
            union(e1, e2)

    # sort the new edges by cost
    # if an edge is not part of the minimum spanning tree, then include it, else continue
    cost_ret = 0
    for c1, c2, cost in sorted(poss_mst, key=lambda x : x[2]):
        if find(c1) != find(c2):
            union(c1, c2)
            cost_ret += cost

    if len({find(c) for c in uf}) == 1 and len(uf) == num_nodes:
        return cost_ret
    else:
        return -1

