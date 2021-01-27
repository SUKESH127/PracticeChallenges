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
