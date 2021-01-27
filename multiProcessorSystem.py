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
