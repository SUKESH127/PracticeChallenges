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

