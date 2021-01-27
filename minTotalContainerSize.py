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
