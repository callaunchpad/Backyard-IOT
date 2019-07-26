
import random
list1 = ['Rock','Paper','Scissors']

def intro():
    print("Welcome to Rock, Paper, Scissors")


def rock_rules(a,b):
    if b=='Rock':
        return "A draw"
    elif b=='Paper':
        return "Computer wins"
    else:
        return "Human wins"

def paper_rules(a,b):
    if b=='Rock':
        return "Human wins"
    elif b=='Paper':
        return "A draw"
    else:
        return "Computer wins"

def scissor_rules(a,b):
    if b=='Rock':
        return "Computer wins"
    elif b=='Paper':
        return "Human wins"
    else:
        return "A draw"

def main():
    intro()
    n = eval(input("Best of what? "))
    winshuman = 0
    winscomp = 0
    draw = 0
    while winshuman <= (n//2) and winscomp <= (n//2) and (winshuman + winscomp) < n:
        humanchoice = input("Choose (R)ock, (P)aper, or (S)cissors? ")
        if humanchoice == 'r' or humanchoice == 'R':
            humanchoice = 'Rock'
            computerchoice = random.choice(list1)
            whowon = rock_rules(humanchoice,computerchoice)
            print("Human:",humanchoice,'     ',"Computer:",computerchoice,'       ' +whowon)
            if rock_rules(humanchoice,computerchoice) == 'Human wins':
                winshuman += 1
            elif rock_rules(humanchoice,computerchoice) == 'Computer wins':
                winscomp +=1
            else:
                draw += 1
        elif humanchoice == 'p' or humanchoice == 'P':
            humanchoice = 'Paper'
            computerchoice = random.choice(list1)
            whowon = paper_rules(humanchoice,computerchoice)
            print("Human:",humanchoice,'     ',"Computer:",computerchoice,'       ' +whowon)
            if paper_rules(humanchoice,computerchoice) == 'Human wins':
                winshuman +=1
            elif paper_rules(humanchoice,computerchoice) == 'Computer wins':
                winscomp +=1
            else:
                draw +=1
        else:
            humanchoice = 'Scissors'
            computerchoice = random.choice(list1)
            whowon = scissor_rules(humanchoice, computerchoice)
            print("Human:", humanchoice, '     ', "Computer:", computerchoice, '       ' + whowon)
            if scissor_rules(humanchoice, computerchoice) == 'Human wins':
                winshuman += 1
            elif scissor_rules(humanchoice, computerchoice) == 'Computer wins':
                winscomp += 1
            else:
                draw +=1
    print("Final Score: Human",winshuman,'    ','Computer',winscomp)
    if winshuman > winscomp:
        print("Human Wins!")
    elif winscomp > winshuman:
        print("Computer Wins!")
    else:
        print("It's a draw!")



main()







