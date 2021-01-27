# K Closest points to origin
def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
    points.sort(key = lambda P: P[0]**2 + P[1]**2)
    res = []
    for i in range(K):
        res.append(points[i])
    return res

