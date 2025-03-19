class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        count = 0
        lst = []
        word = ''
        patternSet = set()
        sSet = set()
        while count < len(s):
            if s[count] == ' ' and word != '':
                lst.append(word)
                word = ''
            if s[count] != ' ':
                word += s[count]
            count += 1
        if word != '':
            lst.append(word)

        longS = ''
        for i in lst:
            longS += i
        
        for i in pattern:
            patternSet.add(i)

        for i in lst:
            sSet.add(i)
        
        for i in range(len(patternSet)):
            longS.replace(list(sSet)[i], list(patternSet)[i])
            
        return True if pattern == longS else False

    def wordPattern2(self, pattern: str, s: str) -> bool:
        lst = []
        lst2 = []

        G = s.split()
        for i in G:
            if i not in lst2:
                lst2.append(i)
        # print(G)

        if len(pattern) != len(G):
            return False

        for i in pattern:
            if i not in lst:
                lst.append(i)
        if len(lst) == len(lst2):
            for i in range(len(lst)):
                s = s.replace(' ', '')
                s = s.replace(lst2[i], lst[i])
        else:
            return False
        print(lst2)
        print(lst)
        print(s)     
        return True if s == pattern else False

print(Solution().wordPattern2('abba', 'dog cat cat dog'))