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

