import random

x = [str(i) for i in range(0,10)]
answer = random.sample(x,4)
answer = ''.join(answer)

# TEST section
# answer = "5492"
# print(answer)

play = "y"
answerBool = False

while play:
    while not answerBool:
        countA = 0
        countB = 0
        userInput = input("請輸入四個數字: ")
        if userInput == "n":
            break
        if len(userInput) != 4:
            print("請輸入四個數字")
            continue
        if not userInput.isdigit():
            print("請輸入數字")
            continue
        if len(set(userInput)) != 4:
            print("請輸入不重複的數字")
            continue

        if userInput == answer:
            print("恭喜你答對了")
            answerBool = True
            break
        else:
            for i in range(4):
                if userInput[i] == answer[i]:
                    countA += 1
                if userInput[i] in answer and userInput[i] != answer[i]:
                    countB += 1
            print(str(countA) + "A" + str(countB) + "B")

    play = input("還要要玩嗎? y/n: ")
    if play == "n":
        break



"""
完成時間順序
023
006
024 我
004
007
007 019 020
002 010 013 014 016 025
"""