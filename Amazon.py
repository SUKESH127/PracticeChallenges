from collections import deque
import math
from typing import List
import copy
from queue import PriorityQueue
from math import floor

# Two Sum
def twoSum(self, nums, target):
    seen = {}
    for i, x in enumerate(nums):
        otherNumInPair = target - x
        if x not in seen:
            seen[x] = i
        if otherNumInPair in seen:
            output = [seen[otherNumInPair], i]
    return output

# Number of Islands
def numIslands(self, grid: List[List[str]]) -> int:
    if not grid:
        return 0
    numIslands = 0
    
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            if grid[i][j] == "1":
                numIslands += dfs(grid, i, j)
    return numIslands

def dfs(grid, i, j):
    if (i < 0) or (i >= len(grid)) or (j < 0) or (j >= len(grid[i])) or (grid[i][j] == "0"):
        return 0
    grid[i][j] = "0"
    dfs(grid, i + 1, j)
    dfs(grid, i - 1, j)
    dfs(grid, i, j + 1)
    dfs(grid, i, j - 1)
    return 1

# Utilization Checks
def finalInstances(instances, averageUtil):
    end = len(averageUtil)
    i = 0
    lim = 2 * (10**8)
    while i < end:
        if averageUtil[i] < 25 and instances > 1:
            instances = math.ceil(instances / 2)
            i += 10
        elif averageUtil[i] > 60 and instances < lim:
            instances *= 2 
            i += 10
        i += 1
    return instances

# Minimum Total Container Size
def minContainerSize(d, P):
    l=len(P)
    k=l-d + 1
    i,j = 0,0
    max_=float('-inf')
    sum_=0
    first,last=-1,-1
    q=deque()
    while j< l:
        q.append(P[j])
        if j<k:
            sum_ += P[j]
        else:
            curr=q.popleft()
            i+=1
            sum_ = sum_ - curr + P[j]
        if sum_ > max_ :
                max_= max(max_,sum_)
                first=i
                last=j
        j+=1
    containerSize=0
    max_=float('-inf')
    for z in range(l):
        if z>=first and z<=last:
            if P[z]>max_:
                max_=P[z]
        else:
            containerSize+=P[z]
    containerSize+=max_
    return containerSize

# Maximum Bounded Array or Winning Sequence     
def maxBounded(num, lower, upper):
    diff = upper - lower
    # initial conditions check 
    if num >= (2 * (diff + 1)) or num < 3:
        return -1
    
    startingNumber = upper - 1
    while ((upper - startingNumber + 1) + diff) < num:
        startingNumber -= 1
    
    res = [0] * num
    i = 0
    while startingNumber < upper:
        res[i] = startingNumber
        startingNumber += 1
        i += 1
    
    res[i] = startingNumber
    startingNumber -= 1
    i += 1

    while i < num:
        res[i] = startingNumber
        startingNumber -= 1
        i += 1
    
    return res

# Cutoff Ranks
def countLevelUpPlayers(cutOffRank, num, scores):     
    if cutOffRank > num or num > (10**5):
        return 0
    rankArray = []
    scores.sort(reverse=True)

    currentRank = 1
    prev = scores[0]
    repeating = 0
    rankArray.append(currentRank)
    for i in range(1, len(scores)):
        if scores[i] == prev:
            repeating += 1
            rankArray.append(currentRank)
        else:
            currentRank += 1
            if repeating > 0:
                currentRank += repeating
                rankArray.append(currentRank)
            else:
                rankArray.append(currentRank)
            repeating = 0
        prev = scores[i]

    output = 0
    for n in rankArray:
        if n <= cutOffRank:
            output += 1
    return output    

# Rover Control
def roverControl(n, m, commands):
    row, col, size = 0, 0, n
    for command in commands:
        if command == "RIGHT" and col < n:
            col += 1
        elif command == "LEFT" and col != 0:
            col -= 1
        elif command == "DOWN" and row < n:
            row += 1
        elif command == "UP" and row != 0:
            row -= 1
    return (row * size) + col

# Schedule Tasks
def processLogFile(logs, threshold):
    transactionMap = {}
    for i in range(len(logs)):
        logArray = logs[i].split()
        idOne = logArray[0]
        idTwo = logArray[1]
        if idOne == idTwo:
            if idOne in transactionMap:
                transactionMap[idOne] += 1
            else:
                transactionMap[idOne] = 1
        else:
            if idOne in transactionMap:
                transactionMap[idOne] += 1
            else:
                transactionMap[idOne] = 1
            
            if idTwo in transactionMap:
                transactionMap[idTwo] += 1
            else:
                transactionMap[idTwo] = 1
    
    output = []       
    ids = transactionMap.keys()
    for ID in ids:
        if transactionMap[ID] >= threshold:
            output.append(ID)
    return output     

# Load the Cargo / Fill the Truck
def loadTheCargo(num, containers, itemSize, itemsPerContainer, cargoSize):                    
    totalCargo = []
    n = num if num <= itemSize else itemSeize
    
    for i in range(n):
        listToAdd = [itemsPerContainer[i]]*containers[i]
        totalCargo += listToAdd
    
    totalCargo.sort(reverse=True)
    if cargoSize > len(totalCargo):
        cargoSize = len(totalCargo)

    outputSum = 0
    for i in range(cargoSize):
        first = totalCargo.pop(0)
        outputSum += first

    return outputSum

# Debt Records/Smallest Negative Trade Deficit 
class debtRecord:
    borrower = ""   #aka importer
    lender = ""   #aka exporter
    amount = 0
    def __init__(self, borrower, lender, amount):
        self.borrower = borrower
        self.lender = lender
        self.amount = amount
def minimumDebtMembers(records: List[debtRecord]) -> List[str]:
    table = {}
    for i in range(len(records)):
        importer, exporter, value = records[i].borrower, records[i].lender, records[i].amount
        # adjust importer's net or initialize it to value
        if importer in table:
            table[importer] -= value
            if table[importer] == 0:
                del table[importer]
        else:
            table[importer] = (-value)
        # adjust exporter's net or initialize it to value
        if exporter in table:
            table[exporter] += value
            if table[exporter] == 0:
                del table[exporter]
        else:
            table[exporter] = value
    output = ["None"]
    minDeficit = float('inf')
    for key in (table.keys()):
        if table[key] < minDeficit:
            output[0] = key
            minDeficit = table[key]
        elif table[key] == minDeficit:
            output.append(key)
    output.sort()
    return output

# K Closest points to origin
def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
    points.sort(key = lambda P: P[0]**2 + P[1]**2)
    res = []
    for i in range(K):
        res.append(points[i])
    return res

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

# Roll Dice
def rollDice(dice):
    def getAllDist(counts, cur): # counts = how often num shows, cur is number 1-7
        rotate = 0
        for i in range(1, 7):
            if i != cur:
                dist = calcDist(i, cur) # get distance for a specific side
                totalRotationsPerSide = counts[i] * dist 
                rotate += totalRotationsPerSide
        return rotate
    # get number of steps for each from/to combination
    def calcDist(fromInt, toInt):
        if fromInt == toInt:
            return 0
        return 2 if (fromInt + toInt == 7) else 1
    # stores how many of each side there
    counts = [0]*7 
    for dic in dice:
        counts[dic] += 1
    # find the minimum number of rotations 
    minimum = float('inf')
    for i in range(1, 7):
        minimum = min(minimum, getAllDist(counts, i))
    return minimum

# Count Teams / Count Review Combinations
def countReview(num, skills, minAssociates, minLevel, maxLevel):
    screened = []
    possible_teams = [[]]
    for associate in skills:
        if minLevel <= associate <= maxLevel:
            screened.append(associate)
    num_teams = 0
    while screened:
        person = screened.pop()
        new_teams = []
        for team in possible_teams:
            new_team = [person] + team
            if len(new_team) >= minAssociates:
                num_teams += 1
            new_teams.append(new_team)
        possible_teams += new_teams
    return num_teams

# Secret Fruit List ????
def matchSecretLists(secretFruitList: List[List[str]], customerPurchasedList: List[str]) -> bool:
    for codeList in secretFruitList:
        i = j = 0
        while i < len(codeList) and j < len(customerPurchasedList):
            if codeList[i] == customerPurchasedList[j] or codeList[i] == "anything":
                i += 1
            j += 1
        if i == len(codeList):
            print("i: " + str(i) + " codeList: " + str(codeList))
            return True
    return False

    # i, j, M, N = 0, 0, len(secretFruitLists), len(customerPurchasedList)
    # result = [False] * M
    # while i < N:
    #     while j < M:
    #         k = 0
    #         while k < len(secretFruitLists[j]):
    #             # end of shopping list, break out of both loops
    #             if i == N:
    #                 j = M
    #                 break
    #             # try to match item in shopping list to code list[j]
    #             elif (
    #                 customerPurchasedList[i] == secretFruitLists[j][k] or secretFruitLists[j][k] == "anything"
    #             ):
    #                 k += 1
    #             else:
    #                 # match not found, reset k
    #                 k = 0
    #             i += 1
    #         # ensure order and mark visited
    #         if j < M and k == len(secretFruitLists[j]):
    #             result[j] = True
    #             j += 1
    #     i += 1
    # return 1 if all(result) else 0

# Top N buzzwords / Top K frequent words ??
def getTopGames(num, topKGames, games, numReviews, reviews):
    MENTIONS, REVIEW_COUNT = 0, 1
    def compare(w1, w2):
        if buzzwordHash[w1][MENTIONS] > buzzwordHash[w2][MENTIONS]:
            return 1
        elif buzzwordHash[w1][MENTIONS] == buzzwordHash[w2][MENTIONS]:
            return 1 if w1 < w2 else -1
        else:
            return -1

    buzzwordHash = {game.lower() : [0,0] for game in games}
    output = list(buzzwordHash.keys())
    if topKGames >= num:
        return output
    
    for string in reviews:
        newReview = True
        for s in string.split():
            word = s.lower()
            singleReview = True
            if word in buzzwordHash:
                if singleReview:
                    buzzwordHash[word][MENTIONS] += 1
                    singleReview = False
                if newReview:
                    buzzwordHash[word][REVIEW_COUNT] += 1
                    newReview = False

    output.sort(reverse=True, key = lambda word: buzzwordHash[word][MENTIONS])
    return [output[i] for i in range(topKGames)]
    

# print(getTopGames(5, 2, ["anacell", "betacellular", "cetracular", "deltacellular", "eurocell"], 5, [
#   "I love anacell Best services; Best services provided by anacell",
#   "betacellular has great services",
#   "deltacellular provides much better services than betacellular",
#   "cetracular is worse than anacell",
#   "Betacellular is better than deltacellular.",
# ]))

# Treasure Island I
def treasureIslandI(matrix):
    def find_treasure(matrix, row, col, steps, minStep):
        rowCheckCondition = (row < 0 or row >= len(matrix))
        colCheckCondition = (col < 0 or col >= len(matrix[0]))
        if rowCheckCondition or colCheckCondition or matrix[row][col] == 'D' or matrix[row][col] == '#':
            return None, minStep
        elif matrix[row][col] == 'X':
            steps += 1
            if minStep > steps:
                minStep = steps
            return None, minStep
        else:
            tmp = matrix[row][col]
            matrix[row][col] = '#'
            steps += 1
            up = find_treasure(matrix, row-1, col, steps, minStep)
            down = find_treasure(matrix, row+1, col, steps, minStep)
            left = find_treasure(matrix, row, col-1, steps, minStep)
            right = find_treasure(matrix, row, col+1, steps, minStep)
            matrix[row][col] = tmp
            correctMove = min(left[1], right[1], up[1], down[1])
            return steps, correctMove
    return (find_treasure(matrix, 0, 0, -1, float('inf')))[1]

# Slowest Key
def slowestKey(keyTimes):
    longest, prev = ['',0], ['',0] # [character, time]
    keysPressed = [chr(keyTimes[i][0] + 97) for i in range(len(keyTimes))]
    for i, c in enumerate(keysPressed):
        duration = keyTimes[i][1] - prev[1] # calculate duration each time
        if duration > longest[1]:
            longest[0], longest[1] = c, duration
        elif duration == longest[1]: # break ties
            if c > prev[0]:
                longest[0], longest[1] = c, duration
            else:
                longest[0], longest[1] = prev[0], prev[1]
        prev = [c, keyTimes[i][1]] # set the previous character and time
    return longest[0]

# Merge Two Sorted Linked Lists
def mergeTwoSortedLists(l1, l2):
    dummyHead = cur = ListNode()
    while l1 != None and l2 != None:
        if l1.val <= l2.val:
            cur.next = l1
            l1 = l1.next
        else:
            cur.next = l2
            l2 = l2.next
        cur = cur.next
    cur.next = l1 if not l2 else l2
    return dummyHead.next

# Multiprocessor System:
def multiprocessorSystem(ability, num, processes):
    pq = PriorityQueue()
    for a in ability:
        pq.put(-a)
    
    time, processed = 0, 0
    while (not pq.empty()) and (processed < processes): 
        power = -(pq.get())
        processed += power
        newPower = floor(power/2)
        if newPower > 0:
            pq.put(-newPower)
        time += 1
    return time

# Labeling/Tagging System ???
def labelingSystem(originalTag, limit):
    def getString(intChar, count):
        repeat = [chr(intChar + 97)]*count
        return "" .join(repeat)

    count = [0] * 26
    for c in originalTag:
        position = ord(c) - 97
        count[position] += 1
    # print(count)

    
    pq = PriorityQueue()
    for i in range(26):
        if count[i] > 0:
            pq.put((-i, [count[i]]))
    if pq.empty():
        return "" 
    
    sb = ""
    pre = pq.get()
    while not pq.empty():
        cur = pq.get()
        print(f"char: {chr((-cur[0]) + 97)} cur[char]: {-cur[0]} cur[count]: {cur[1][0]}")
        if pre[1][0] > limit:
            pre[1][0] -= limit
            pq.put((-pre[0], pre[1]))
            sb += getString(-pre[0], limit)
        else:
            sb += getString(-pre[0], pre[1][0])
        pre = cur

    sb += getString(-pre[0], min(limit, pre[1][0]))
    return sb

    # for c in originalTag:
    #     castedIntRepresentation = - ord(c)
    #     pq.put((castedIntRepresentation,c)) # O(logn) time complexity
    
    # maxString = ""
    # while not pq.empty():
    #     c = (pq.get())[1]
    #     maxString += c
    
    # print(f"maxstring: {maxString}")

    # prev = maxString[0]
    # consecutiveCharCount = 1
    # output = maxString[0]
    # for i in range(1,len(maxString)):
    #     c = maxString[i]
    #     if prev == c:
    #         consecutiveCharCount += 1
    #     else:
    #         consecutiveCharCount = 1
        
    #     if prev == c and consecutiveCharCount > limit:
    #         diff = consecutiveCharCount - limit
    #         maxString = maxString[:(i+1) - diff] + maxString[i+1] + maxString[diff]

    #         if i + 1 < len(maxString) - 1:
    #             s = list(maxString)
    #             s[i], s[i + 1] = s[i + 1], s[i]
    #             maxString = ''.join(s)
    #             consecutiveCharCount -= 1
    #     else:
    #         output += c
    #     prev = c

    # return "maxString"    

# print(labelingSystem("cbddd", 2))

def lruCacheMisses(num: int, pages : List[int], maxCacheSize : int) -> int:
    # WRITE YOUR BRILLIANT CODE HERE
    class LRUCache:
        class DLL:
            def __init__(self, key, val):
                self.key = key
                self.val = val
                self.next = None
                self.prev = None
        def __init__(self, capacity: int):
            self.m = {}
            self.head = self.DLL(0,0)
            self.tail = self.DLL(0,0)
            self.head.next = self.tail
            self.tail.prev = self.head
            self.size = 0
            self.capacity = capacity
        def get(self, key: int) -> int:
            if key in self.m:
                loc = self.m[key]
                loc.prev.next = loc.next
                loc.next.prev = loc.prev
                self.head.next.prev = loc
                loc.next = self.head.next
                self.head.next = loc
                loc.prev = self.head
                return loc.val
            else:
                return -1
        def put(self, key: int, value: int) -> None:
            if key in self.m:
                self.get(key)
                self.m[key].val = value
                return
            self.size += 1
            if self.size > self.capacity:
                lru = self.tail.prev
                del self.m[lru.key]
                self.tail.prev.val = self.tail.val
                self.tail.prev.next = None
                self.tail = self.tail.prev
                self.size -= 1
            new_head = self.DLL(key, value)
            self.head.next.prev = new_head
            new_head.next = self.head.next
            self.head.next = new_head
            new_head.prev = self.head
            self.m[key] = new_head
    cache = LRUCache(maxCacheSize)
    misses = 0
    for page in pages:
        if cache.get(page) == -1:
            misses += 1
        cache.put(page, None)
    return misses

def findSmallestDivisor(s, t):
    if (len(s) % len(t)) != 0:
        return -1
    l2 = len(t)
    for i in range(len(s)):
        if s[i] != t[i%l2]:
            return -1
    for i in range(len(t)):
        j = 0
        while j < len(t):
            if t[j] != t[j%(i+1)]:
                break
            j += 1
        if j == len(t):
            return i + 1
    return -1

# Throttling Gateway
def throttlingGateway(n, requestTime):
    ans = 0
    for i in range(n):
        if i > 2 and (requestTime[i] == requestTime[i-3]):
            ans += 1
        elif i > 19 and (requestTime[i]-requestTime[i-20]) < 10:
            ans += 1
        elif i > 59 and (requestTime[i]-requestTime[i-60]) < 60:
            ans += 1
    return ans

# Earliest Time to complete deliveries
def earliestTime(numOfBuildings, buildingOpenTime, offloadTime):
    buildingOpenTime.sort()
    offloadTime.sort()
    ans = 0
    for i in range(0, len(offloadTime), 4):
        time = buildingOpenTime[len(buildingOpenTime) - floor(i/4) - 1] + offloadTime[i + 3]
        ans = max(ans, time)
    return ans


# Subtree of Another Tree
def isSubtree(self, s, t):
    def checkTree(root1, root2):
        if not root1 or not root2:
            return root1 == root2
        if root1.val != root2.val:
            return False
        return checkTree(root1.left, root2.left) and checkTree(root1.right, root2.right)
    def dfs(s, t):
        if not s:
            return False
        if s.val == t.val and checkTree(s, t):
            return True
        else:
            return dfs(s.left, t) or dfs(s.right, t)
    return dfs(s, t)

def uniqueDeviceNames(num, deviceNames):
    deviceMap = {}
    output = []
    for i, device in enumerate(deviceNames):
        newString = device
        if not device in deviceMap:
            deviceMap[device] = 1
            output.append(device)
        else:
            newString += str(deviceMap[device])
            output.append(newString)
            deviceMap[device] += 1
    return output 

# Search Suggestions System
class TrieNode:
    def __init__(self):
        self.children = dict()
        self.words = list()
        self.n = 0
class Trie:
    def __init__(self):
        self.root = TrieNode()
    def add(self, word):
        node = self.root
        for c in word:
            if c not in node.children:
                node.children[c] = TrieNode()
            node = node.children[c]
            if node.n < 3:
                node.words.append(word)
                node.n += 1
    def findWord(self, prefix):
        node = self.root
        for c in prefix:
            if c not in node.children:
                return ''
            node = node.children[c]
        return node.words
def suggestedProducts(self, A, searchWord):
    A.sort()
    trie = Trie()
    for word in A:
        trie.add(word)
    ans, cur = [], ''
    for c in searchWord:
        cur += c
        ans.append(trie.findWord(cur))
    return ans

def search2DMatrix(matrix, target):
    # an empty matrix obviously does not contain `target` (make this check
    # because we want to cache `width` for efficiency's sake)
    if len(matrix) == 0 or len(matrix[0]) == 0: return False
    # cache these, as they won't change.
    height, width = len(matrix), len(matrix[0])
    # start our "pointer" in the bottom-left
    row, col = height - 1, 0
    while col < width and row >= 0:
        if matrix[row][col] > target:
            row -= 1
        elif matrix[row][col] < target:
            col += 1
        else: # found it
            return True
    return False

# Critical Connections
def criticalConnections(self, n: int, connections: List[List[int]]) -> List[List[int]]:
    def dfs(rank, curr, prev):
        low[curr], result = rank, []
        for neighbor in edges[curr]:
            if neighbor == prev: continue
            if not low[neighbor]:
                result += dfs(rank + 1, neighbor, curr)
            low[curr] = min(low[curr], low[neighbor])
            if low[neighbor] >= rank + 1:
                result.append([curr, neighbor])
        return result

    low, edges = [0] * n, collections.defaultdict(list)
    for u, v in connections:
        edges[u].append(v)
        edges[v].append(u)

    return dfs(1, 0, -1)

# Maximal Square
def maximalSquare(self, matrix: List[List[str]]) -> int:
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    maxSquare = 0
    dp = [[0] * (cols + 1) for i in range(rows + 1)]
    
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if (matrix[i - 1][j - 1] == '1'):
                dp[i][j] = min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1]) + 1
                maxSquare = max(maxSquare, dp[i][j])
    return maxSquare ** 2