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
