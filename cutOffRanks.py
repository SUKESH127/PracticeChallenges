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
