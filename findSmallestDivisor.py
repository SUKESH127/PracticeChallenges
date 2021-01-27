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
