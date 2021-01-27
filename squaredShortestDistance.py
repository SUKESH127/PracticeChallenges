# Squared Shortest Distance
class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
def squaredShortestDistance(numRobots, positionX, positionY):
    def dist(p1, p2):
        return (((p1.x-p2.x)**2) + ((p1.y-p2.y)**2))
    def bruteForce(P, n):
        minVal = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                if dist(P[i], P[j]) < minVal:
                    minVal = dist(P[i], P[j])
        return minVal
    # find distance between the closest points of a strip of given size - all points in strip[] are sorted via y
    def stripClosest(strip, size, d):
        minVal = d # upper bound on min distance as d
        # pick each point till difference  
        for i in range(size):
            j = i + 1
            while j < size and (strip[j].y - strip[i].y) < minVal:
                minVal = dist(strip[i], strip[j])
                j += 1
        return minVal
    # recursively find smallest distance 
    def closestUtil(P, Q, n): # P is all points sorted by x coordinate
        if n <= 3:
            return bruteForce(P, n)
        # find middle point
        mid = n // 2
        midPoint = P[mid]
        # calculate smallest distance dl on left of middle point and dr on right
        dl = closestUtil(P[:mid], Q, mid)
        dr = closestUtil(P[mid:], Q, n - mid)
        d = min(dl, dr) # find smaller of both
        # build array strip[] w/ points closer than d to line passing thru middle point
        strip = []
        for i in range(n):
            if abs(Q[i].x - midPoint.x) < d:
                strip.append(Q[i])
        return min(d, stripClosest(strip, len(strip), d)) # get closest points in strip and return min of d and closest distance of strip
    def closest(P, n):
        P.sort(key = lambda point: point.x)
        Q = copy.deepcopy(P)
        Q.sort(key = lambda point: point.y)
        return closestUtil(P, Q, n)
    # P = [Point(2, 3), Point(12, 30), Point(40, 50), Point(5, 1), Point(12, 10), Point(3, 4)] 
    # n = len(P) 
    # print("The smallest distance is", closest(P, n)) 
    P = []
    for i in range(numRobots):
        P.append(Point(positionX[i], positionY[i]))
    n = len(P)
    print("The smallest distance is", closest(P,n))
