import random

def draw_card():
    """隨機抽一張牌，A 計為 11，但稍後可以變為 1"""
    card = random.choice(["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"])
    if card in ["J", "Q", "K"]:
        return 10
    elif card == "A":
        return 11
    else:
        return int(card)

def calculate_score(cards):
    """計算手牌總點數，並處理 A 的轉換"""
    score = sum(cards)
    ace_count = cards.count(11)  # 計算 A 的數量
    while score > 21 and ace_count > 0:
        score -= 10  # A 從 11 變 1
        ace_count -= 1
    return score

def computer_strategy(computer_cards):
    """電腦的抽牌邏輯，若總點數小於 17 則抽牌"""
    while calculate_score(computer_cards) < 17:
        computer_cards.append(draw_card())
    return computer_cards

def blackjack():
    print("歡迎來到 21 點遊戲！")
    user_cards = [draw_card(), draw_card()]
    computer_cards = [draw_card(), draw_card()]
    
    while True:
        user_score = calculate_score(user_cards)
        computer_score = calculate_score(computer_cards)
        
        print(f"你的牌: {user_cards} (總點數: {user_score})")
        print(f"電腦的第一張牌: {computer_cards[0]}")
        
        if user_score == 21:
            print("你獲勝！黑傑克！")
            return
        elif user_score > 21:
            print("你爆牌了，電腦獲勝！")
            return
        
        action = input("請選擇是否要抽牌？(y/n): ").lower()
        if action == 'y':
            user_cards.append(draw_card())
        else:
            break
    
    # 電腦開始行動
    computer_cards = computer_strategy(computer_cards)
    computer_score = calculate_score(computer_cards)
    
    print(f"你的最終牌組: {user_cards} (總點數: {user_score})")
    print(f"電腦的最終牌組: {computer_cards} (總點數: {computer_score})")
    
    if computer_score > 21:
        print("電腦爆牌了，你獲勝！")
    elif user_score > computer_score:
        print("你獲勝！")
    elif user_score < computer_score:
        print("電腦獲勝！")
    else:
        print("平手！")

if __name__ == "__main__":
    blackjack()
