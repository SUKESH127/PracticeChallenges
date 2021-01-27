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
