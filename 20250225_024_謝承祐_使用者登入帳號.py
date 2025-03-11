class userLogin:
    def __init__(self, pwd1, pwd2):
        self.pwd1 = pwd1
        self.pwd2 = pwd2
    
    def authenticate(self, pwd1):
        return pwd1 == self.pwd1
    
    def changePassword(self, pwd2, newPw):
        if pwd2 == self.pwd2:
            self.pwd1 = newPw
            return True
        return False

# Interactive section
pwd1 = input("請設定密碼: ")
pwd2 = input("請設定第二組密碼: ")
obj = userLogin(pwd1, pwd2)

while True:
    print("\n選擇操作:")
    print("1. 登入")
    print("2. 修改密碼")
    print("3. 離開")
    choice = input("請輸入選項 (1/2/3): ")
    
    if choice == "1":
        pwd = input("請輸入密碼: ")
        if obj.authenticate(pwd):
            print("登入成功！")
        else:
            print("密碼錯誤！")
    elif choice == "2":
        secondPwd = input("請輸入第二組密碼以驗證身份: ")
        newPwd = input("請輸入新密碼: ")
        if obj.changePassword(secondPwd, newPwd):
            print("密碼修改成功！")
        else:
            print("第二組密碼錯誤，無法修改密碼！")
    elif choice == "3":
        print("程式結束。")
        break
    else:
        print("無效選項，請重新輸入。")

# TEST
# obj = userLogin("12345", "67890")
# print(obj.authenticate("12345"))  # True
# print(obj.authenticate("23456"))  # False
# print(obj.changePassword("99999", "11111"))  # False
# print(obj.changePassword("67890", "11111"))  # True
# print(obj.authenticate("12345"))  # False
# print(obj.authenticate("11111"))  # True

"""
完成時間順序
023
024 我
006
"""
