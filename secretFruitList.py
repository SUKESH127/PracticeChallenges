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
