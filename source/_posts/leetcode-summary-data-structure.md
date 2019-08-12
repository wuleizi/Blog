---
title: Leetcode 数据结构相关整理
date: 2018-08-27 00:53:21
tags: [算法, 总结, Leetcode, OJ]
---
> 这里总结一些leetcode上比较经典的与数据结构相关的例题与思路，因为部分数据结构更偏向与搜索或贪心类型，该类题目就不再本部分总结...
<!-- more -->
# 字符串和数组

> 字符串和数组是比较典型的线性表结构，由于比较好访问，所以通常会在该类数据结构上设计搜索和dp类型的题目。
> 搜索和动规类型的题目会另做讨论，本部分主要总结比较典型反映字符串和数组属性的题目，例如线性表操作或双指针等。

## 字符串

### 表达式求值

表达式求值是一类比较考察细节的题目，一般会涉及括号，四则运算，小数和空格，所以在写的时候要注意思路清晰，以下提供一个模板：

#### 模板
``` cpp
#include <iostream>
#include <math.h>
using namespace std;

double getNum(string s, int& index) {
    int n = s.size();
    int cnt = -1, ans = 0;
    while (index < n && (isdigit(s[index]) || s[index] == '.')) {
        if (s[index] == '.') cnt ++;
        else {
            ans = ans * 10 + (s[index] - '0');
            if (cnt >= 0) cnt ++;
        }
        index ++;
    }
    cnt = cnt < 0 ? 0 : cnt;
    return ans / pow(10, cnt);
}

double helper(string s, int& index) {
    double ret = 0.0, cur_ret = 0.0;
    int n = s.size();
    char op = '+';
    while (index < n && s[index] != ')') {
        if (s[index] == ' ') {
            index ++;
            continue;
        }
        if (isdigit(s[index]) || s[index] == '(') {
            double temp = 0.0;
            if (s[index] == '(') {
                index ++;
                temp = helper(s, index);
                index ++;
            }
            else {
                temp = getNum(s, index);
            }
            switch(op) {
                case '+' : cur_ret += temp; break;
                case '-' : cur_ret -= temp; break;
                case '*' : cur_ret *= temp; break;
                case '/' : cur_ret /= temp; break;
            }
        }
        else {
            if (s[index] == '+' || s[index] == '-') {
                ret += cur_ret;
                cur_ret = 0.0;
            }
            op = s[index++];
        }
    }
    return ret + cur_ret;
}

int main() {
    string s;
    cin >> s;
    int index = 0;
    cout << helper(s, index) << endl;
    return 0;
}


```

#### Basic Calculator
> [Leetcode 224](https://leetcode.com/problems/basic-calculator/description/)

```
Implement a basic calculator to evaluate a simple expression string.

The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces .

Example 1:
Input: "1 + 1"
Output: 2

Example 2:
Input: " 2-1 + 2 "
Output: 3

Example 3:
Input: "(1+(4+5+2)-3)+(6+8)"
Output: 23
```

``` cpp
class Solution {
public:
    int getNum(string s, int& index) {
        int ret = 0, n = s.size();
        while (index < n && isdigit(s[index])) {
            ret = ret * 10 + (s[index++] - '0');
        }
        return ret;
    }
    int helper(string s, int& index) {
        int ret = 0, cur_ret = 0;
        char op = '+';
        int n = s.size();
        while (index < n && s[index] != ')') {
            if (s[index] == ' ') {
                index ++;
                continue;
            }
            if (isdigit(s[index]) || s[index] == '(') {
                int temp = 0;
                if (s[index] == '(') {
                    index ++;
                    temp = helper(s, index);
                    index ++;
                }
                else temp = getNum(s, index);
                switch(op) {
                    case '+': cur_ret += temp; break;
                    case '-': cur_ret -= temp; break;
                    case '*': cur_ret *= temp; break;
                    case '/': cur_ret /= temp; break;
                }
            }
            else {
                if (s[index] == '+' || s[index] == '-') {
                    ret += cur_ret;
                    cur_ret = 0;
                }
                op = s[index ++];
            }
        }
        return ret + cur_ret;
    }
    int calculate(string s) {
        int index = 0;
        return helper(s, index);
    }
};
```
#### Basic Calculator II
> [Leetcode 227](https://leetcode.com/problems/basic-calculator-ii/description/)

```
Implement a basic calculator to evaluate a simple expression string.

The expression string contains only non-negative integers, +, -, *, / operators and empty spaces . The integer division should truncate toward zero.

Example 1:
Input: "3+2*2"
Output: 7

Example 2:
Input: " 3/2 "
Output: 1

Example 3:
Input: " 3+5 / 2 "
Output: 5
```

``` cpp
class Solution {
public:
    int getNum(string s, int& index) {
        int ret = 0, n = s.size();
        while (index < n && isdigit(s[index])) {
            ret = ret * 10 + (s[index++] - '0');
        }
        return ret;
    }
    int helper(string s, int& index) {
        int ret = 0, cur_ret = 0;
        char op = '+';
        int n = s.size();
        while (index < n && s[index] != ')') {
            if (s[index] == ' ') {
                index ++;
                continue;
            }
            if (isdigit(s[index]) || s[index] == '(') {
                int temp = 0;
                if (s[index] == '(') {
                    index ++;
                    temp = helper(s, index);
                    index ++;
                }
                else temp = getNum(s, index);
                switch(op) {
                    case '+': cur_ret += temp; break;
                    case '-': cur_ret -= temp; break;
                    case '*': cur_ret *= temp; break;
                    case '/': cur_ret /= temp; break;
                }
            }
            else {
                if (s[index] == '+' || s[index] == '-') {
                    ret += cur_ret;
                    cur_ret = 0;
                }
                op = s[index ++];
            }
        }
        return ret + cur_ret;
    }
    int calculate(string s) {
        int index = 0;
        return helper(s, index);
    }
};
```

### 字符串操作

#### Multiply Strings
> [Leetcode 43](https://leetcode.com/problems/multiply-strings/description/)字符串乘法

Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.

```
Example 1:
Input: num1 = "2", num2 = "3"
Output: "6"

Example 2:
Input: num1 = "123", num2 = "456"
Output: "56088"
```

``` cpp
class Solution {
public:
    string multiply(string num1, string num2) {
        int n1 = num1.size(), n2 = num2.size();
        string ret(n1 + n2, '0');
        reverse(num1.begin(), num1.end());
        reverse(num2.begin(), num2.end());
        for (int i = 0; i < n1; i++) {
            int a = num1[i] - '0';
            int c = 0;
            for (int j = 0; j < n2; j++) {
                int b = num2[j] - '0';
                int ans = a * b + c + (ret[i + j] - '0');
                c = ans / 10;
                ret[i + j] = '0' + (ans % 10);
            }
            ret[n2 + i] = c + '0';
        }
        reverse(ret.begin(), ret.end());
        while (!ret.empty() && ret[0] == '0') ret.erase(ret.begin());
        return ret.empty() ? "0" : ret;
    }
};
```
#### Add Two Numbers II
> [Leetcode 445](https://leetcode.com/problems/add-two-numbers-ii/description/)无符号字符串加法

You are given two non-empty linked lists representing two non-negative integers. The most significant digit comes first and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Follow up:
What if you cannot modify the input lists? In other words, reversing the lists is not allowed.
```
Example:
Input: (7 -> 2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 8 -> 0 -> 7
```

``` cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        string num1, num2;
        ListNode* cur = l1;
        while (cur) {
            num1.push_back('0' + cur->val);
            cur = cur->next;
        }
        cur = l2;
        while (cur) {
            num2.push_back('0' + cur->val);
            cur = cur->next;
        }
        string ret;
        int c = 0;
        int i = num1.size() - 1, j = num2.size() - 1;
        while (i >= 0 && j >= 0) {
            int a = num1[i--] - '0';
            int b = num2[j--] - '0';
            int ans = c + a + b;
            ret = to_string(ans % 10) + ret;
            c = ans / 10;
        }
        while (i >= 0) {
            int a = num1[i--] - '0';
            int ans = c + a;
            ret = to_string(ans % 10) + ret;
            c = ans / 10;
        }
        while (j >= 0) {
            int a = num2[j--] - '0';
            int ans = c + a;
            ret = to_string(ans % 10) + ret;
            c = ans / 10;
        }
        if (c) ret = to_string(c) + ret;
        ListNode* head = NULL;
        for (i = ret.size() - 1; i >= 0; i--) {
            ListNode* cur = new ListNode(ret[i] - '0');
            cur->next = head;
            head = cur;
        }
        return head;
    }
};
```

#### 带符号字符串加减法模板
``` cpp
#include <iostream>
#include <string.h>
using namespace std;


// 此处由于是默认加法所以没有对减法进行符号处理，否则这里也需要像加法一样对其进行符号判断
string INT_SUB(string a, string b) {
    int syn = 1;
    if (a < b) {
        string temp = a;
        a = b;
        b = temp;
        syn *= -1;
    }
    int i = a.size() - 1, j = b.size() - 1;
    int c = 0;
    string ret;
    while (i >= 0) {
        int x = a[i--] - '0';
        int y = b[j--] - '0';
        int ans = (10 + x - y - c) % 10;
        c = x < y ? 1 : 0;
        ret = (char)('0' + ans) + ret;
    }
    while (ret.empty() && ret[0] == '0') ret.erase(ret.begin());
    if (ret.empty()) return "0";
    return syn == -1 ? "-" + ret : ret;
}


string INT_ADD(string a, string b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    if (a[0] == '+') a.erase(a.begin());
    if (b[0] == '+') b.erase(b.begin());
    int syn = 1;
    string ret;
    if (a[0] == '-') {
        if (b[0] == '-') {
            syn = -1;
            ret = INT_ADD(a.substr(1), b.substr(1));
        }
        else {
            ret = INT_SUB(b, a.substr(1));
        }
    }
    else {
        if (b[0] == '-') {
            ret = INT_SUB(a, b.substr(1));
        }
        else {
            int i = a.size() - 1, j = b.size() - 1;
            int c = 0;
            while (i >= 0 && j >= 0) {
                int x = a[i--] - '0';
                int y = b[j--] - '0';
                int ans = x + y + c;
                ret = (char)('0' + ans % 10) + ret;
                c = ans / 10;
            }
            while (i >= 0) {
                int x = a[i--] - '0';
                int ans = x + c;
                ret = (char)('0' + ans % 10) + ret;
                c = ans / 10;
            }
            while (j >= 0) {
                int x = b[j--] - '0';
                int ans = x + c;
                ret = (char)('0' + ans % 10) + ret;
                c = ans / 10;
            }
            if (c) ret = (char)('0' + c) + ret;
        }
    }
    if (ret[0] == '-') {
        if (syn == 1) return ret;
        else return ret.substr(1);
    }
    else {
        if (syn == 1) return ret;
        else return ret != "0" ? "-" + ret : ret;
    }
}

int main() {
    string a, b;
    cin >> a >> b;
    cout << INT_ADD(a, b) << endl;
    return 0;
}

```


### 字符串双指针题目

双指针有两种用法，第一种是用于更新原字符串，另一种是用于搜索。一般而言，快指针不光可以表示原本字符串上的位置，同时还可以表示扩展字符串后的位置。

### Longest Substring with At Least K Repeating Characters
> [Leetcode 395](https://leetcode.com/problems/longest-substring-with-at-least-k-repeating-characters/description/)

Find the length of the longest substring T of a given string (consists of lowercase letters only) such that every character in T appears no less than k times.

Example 1:
```
Input:
s = "aaabb", k = 3

Output:
3

The longest substring is "aaa", as 'a' is repeated 3 times.
```
Example 2:
```
Input:
s = "ababbc", k = 2

Output:
5

The longest substring is "ababb", as 'a' is repeated 2 times and 'b' is repeated 3 times.
```


``` cpp
class Solution {
public:
    int longestSubstring(string s, int k) {
        int ret = 0, n = s.size();
        for (int h = 1; h <= 26; h++) {
            vector<int> m(26, 0);
            int i = 0, j = 0, unique = 0, ans = 0;
            while (j < n) {
                if (unique <= h) {
                    int index = s[j++] - 'a';
                    m[index] ++;
                    if (m[index] == k) ans ++;
                    if (m[index] == 1) unique ++;
                }
                else {
                    int index = s[i++] - 'a';
                    m[index] --;
                    if (m[index] == k - 1) ans --;
                    if (m[index] == 0) unique --;
                }
                if (unique == h && unique == ans) ret = max(ret, j - i);
            }
        
        }
        return ret;
    }
};
```




#### Reverse Words in a String
> [Leetcode 151](https://leetcode.com/problems/reverse-words-in-a-string/description/)

```
Given an input string, reverse the string word by word.

Example:  
Input: "the sky is blue",
Output: "blue is sky the".

Note:

A word is defined as a sequence of non-space characters.
Input string may contain leading or trailing spaces. However, your reversed string should not contain leading or trailing spaces.
You need to reduce multiple spaces between two words to a single space in the reversed string.
Follow up: For C programmers, try to solve it in-place in O(1) space.
```

``` cpp
class Solution {
public:
    void reverseWords(string &s) {
        // 更新原本字符串
        if (s.empty()) return;
        reverse(s.begin(), s.end());
        int i = 0, cur = 0;
        int n = s.size();
        while (i < n) {
            if (s[i] != ' ') {
                if (cur) s[cur++] = ' ';
                int j = i;
                while (j < n && s[j] != ' ') s[cur++] = s[j++];
                reverse(s.begin() + cur - (j - i), s.begin() + cur);
                i = j;
            }
            i++;
        }
        s.erase(s.begin() + cur, s.end());
    }
};
```

#### Minimum Window Substring
> [Leetcode 76](https://leetcode.com/problems/minimum-window-substring/description/)

```
Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

Example:
Input: S = "ADOBECODEBANC", T = "ABC"
Output: "BANC"

Note:
If there is no such window in S that covers all characters in T, return the empty string "".
If there is such window, you are guaranteed that there will always be only one unique minimum window in S.
```

``` cpp
class Solution {
public:
    string minWindow(string s, string t) { 
        // ans保存未满足的字符数
        unordered_map<char, int> m;
        for (auto i : t) m[i] ++;
        int ans = m.size(), n = s.size();
        int cnt = INT_MAX, i = 0, j = 0;
        string ret;
        while (j <= n && i <= j) {
            if (j < n && ans > 0) {
                if (m.find(s[j]) != m.end()) {
                    m[s[j]]--;
                    if (!m[s[j]]) ans--;
                }
                j ++;
            }
            else {
                if (m.find(s[i]) != m.end()) {
                    if (!m[s[i]]) ans ++;
                    m[s[i]] ++;
                }
                i++;
            }
            if (!ans) {
                if (j - i < cnt) {
                    cnt = j - i;
                    ret = s.substr(i, cnt);
                }
            }
        }
        return ret;
    }
};
```

#### Decoded String at Index
> [Leetcode 884](https://leetcode.com/problems/decoded-string-at-index/description/)

An encoded string S is given.  To find and write the decoded string to a tape, the encoded string is read one character at a time and the following steps are taken:

If the character read is a letter, that letter is written onto the tape.
If the character read is a digit (say d), the entire current tape is repeatedly written d-1 more times in total.
Now for some encoded string S, and an index K, find and return the K-th letter (1 indexed) in the decoded string.

 
```
Example 1:
Input: S = "leet2code3", K = 10
Output: "o"
Explanation: 
The decoded string is "leetleetcodeleetleetcodeleetleetcode".
The 10th letter in the string is "o".

Example 2:
Input: S = "ha22", K = 5
Output: "h"
Explanation: 
The decoded string is "hahahaha".  The 5th letter is "h".

Example 3:
Input: S = "a2345678999999999999999", K = 1
Output: "a"
Explanation: 
The decoded string is "a" repeated 8301530446056247680 times.  The 1st letter is "a".
```

``` cpp
class Solution {
public:
    string decodeAtIndex(string S, int K) {
        int n = S.size();
        // 此题因为反复更新K，也可以认为是栈类型的题目
        // 但是由于栈通常一般时间复杂度为O(n)，所以放到双指针类型中
        while (K >= 0) {
            long long cur = 0, pre = 0;
            for (int i = 0; i < n; i++) {
                if (isdigit(S[i])) {
                    cur = pre * (S[i] - '0');
                    if (cur >= K) {
                        // 更新K
                        K = ((K - 1) % pre) + 1;
                        break;
                    }
                }
                else {
                    cur ++;
                    if (cur >= K) {
                        // 如果超过了原本就超过K则K就会被更新
                        // 这里就是在当前字符串段上
                        return S.substr(i, 1);
                    }
                }
                pre = cur;
            }
        }
    }
};
```

#### Longest Palindromic Substring
> [Leetcode 5](https://leetcode.com/problems/longest-palindromic-substring/description/)

Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

```
Example 1:
Input: "babad"
Output: "bab"
Note: "aba" is also a valid answer.

Example 2:
Input: "cbbd"
Output: "bb"
```

``` cpp
// 最长回文数的最常见算法为O(n2)，还有优化版本，但是需要讲解数学逻辑，之后更新...
class Solution {
public:
    string longestPalindrome(string s) {
        if (s.empty()) return s;
        int n = s.size(), len = 0, index = 0;
        for (int i = 0; i < n; i++) {
            if (n - i <= len / 2) break;
            int l = i, r = i;
            while (r < n - 1 && s[r + 1] == s[r]) r++;
            while (l > 0 && r < n - 1 && s[l - 1] == s[r + 1]) l--, r++;
            if (len < r - l + 1) {
                index = l, len = r - l + 1;
            }
        }
        return s.substr(index, len);
    }
};
```

### 字符串处理
此部分会不断更新...

#### Validate IP Address
> [Leetcode 468](https://leetcode.com/problems/validate-ip-address/description/)

```
Write a function to check whether an input string is a valid IPv4 address or IPv6 address or neither.

IPv4 addresses are canonically represented in dot-decimal notation, which consists of four decimal numbers, each ranging from 0 to 255, separated by dots ("."), e.g.,172.16.254.1;

Besides, leading zeros in the IPv4 is invalid. For example, the address 172.16.254.01 is invalid.

IPv6 addresses are represented as eight groups of four hexadecimal digits, each group representing 16 bits. The groups are separated by colons (":"). For example, the address 2001:0db8:85a3:0000:0000:8a2e:0370:7334 is a valid one. Also, we could omit some leading zeros among four hexadecimal digits and some low-case characters in the address to upper-case ones, so 2001:db8:85a3:0:0:8A2E:0370:7334 is also a valid IPv6 address(Omit leading zeros and using upper cases).

However, we don't replace a consecutive group of zero value with a single empty group using two consecutive colons (::) to pursue simplicity. For example, 2001:0db8:85a3::8A2E:0370:7334 is an invalid IPv6 address.

Besides, extra leading zeros in the IPv6 is also invalid. For example, the address 02001:0db8:85a3:0000:0000:8a2e:0370:7334 is invalid.

Note: You may assume there is no extra space or special characters in the input string.

Example 1:
Input: "172.16.254.1"
Output: "IPv4"
Explanation: This is a valid IPv4 address, return "IPv4".

Example 2:
Input: "2001:0db8:85a3:0:0:8A2E:0370:7334"
Output: "IPv6"
Explanation: This is a valid IPv6 address, return "IPv6".

Example 3:
Input: "256.256.256.256"
Output: "Neither"
Explanation: This is neither a IPv4 address nor a IPv6 address.
```

``` cpp
// 此方法主要是用于练习字符串处理，如果想要寻找更加高效的算法请在leetcode discuss区寻找
class Solution {
public:
    bool helper4(string s) {
        vector<string> nums;
        string ans;
        int cnt = 0;
        for (int i = 0; i < s.size(); i++) {
            if (isdigit(s[i]) || s[i] == '.') {
                if (s[i] == '.') {
                    nums.push_back(ans);
                    ans = "";
                    cnt ++;
                }
                else {
                    ans.push_back(s[i]);
                }
            }
            else return false;
        }
        nums.push_back(ans);
        for (string i : nums) {
            if (i.empty() || (i.size() > 1 && i[0] == '0')) return false;
            int temp = atoi(i.c_str());
            if (temp < 0 || temp > 255) return false;
        }
        return nums.size() == 4 && cnt == 3;
        
    }
    bool helper6(string s) {
        vector<string> nums;
        string ans;
        int cnt = 0;
        for (auto i : s) {
            if (i == ':') {
                nums.push_back(ans);
                ans = "";
                cnt ++;
            }
            else {
                ans.push_back(i);
            }
        }
        nums.push_back(ans);
        for (auto i : nums) {
            int n = i.size();
            if (n < 1 || n > 4) return false;
            for (auto j : i) {
                if ((j <= '9' && j >= '0') || (j <= 'F' && j >= 'A') || (j <= 'f' && j >= 'a'));
                else return false;
            }
        }
        return nums.size() == 8 && cnt == 7;
        
    }
    string validIPAddress(string IP) {
        if (helper4(IP)) return "IPv4";
        else if (helper6(IP)) return "IPv6";
        else return "Neither";
    }
};
```

## 数组

### 数组操作

#### Increasing Triplet Subsequence
> [Leetcode 334](https://leetcode.com/problems/increasing-triplet-subsequence/description/)

Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.

Formally the function should:

Return true if there exists i, j, k 
such that arr[i] < arr[j] < arr[k] given 0 ≤ i < j < k ≤ n-1 else return false.
Note: Your algorithm should run in O(n) time complexity and O(1) space complexity.
```
Example 1:
Input: [1,2,3,4,5]
Output: true

Example 2:
Input: [5,4,3,2,1]
Output: false
```

``` cpp
class Solution {
public:
    bool increasingTriplet(vector<int>& nums) {
        int c1 = INT_MAX, c2 = INT_MAX;
        for (auto i : nums) {
            if (i <= c1) c1 = i;
            else if (i <= c2) c2 = i;
            else return true;
        }
        return false;
    }
};
```

#### Contiguous Array
> [Leetcode 525](https://leetcode.com/problems/contiguous-array/description/)

Given a binary array, find the maximum length of a contiguous subarray with equal number of 0 and 1.
```
Example 1:
Input: [0,1]
Output: 2
Explanation: [0, 1] is the longest contiguous subarray with equal number of 0 and 1.

Example 2:
Input: [0,1,0]
Output: 2
Explanation: [0, 1] (or [1, 0]) is a longest contiguous subarray with equal number of 0 and 1.
```
Note: The length of the given binary array will not exceed 50,000.

``` cpp
class Solution {
public:
    int findMaxLength(vector<int>& nums) {
        // 前缀和的变形
        for (auto &i : nums) {
            if (!i) i = -1;
        }
        unordered_map<int, int> m;
        m[0] = -1;
        int ans = 0, ret = 0, n = nums.size();
        for (int i = 0; i < n; i++) {
            ans += nums[i];
            if (m.find(ans) != m.end()) ret = max(ret, i - m[ans]);
            else m[ans] = i;
        }
        return ret;
    }
};
```

#### 4Sum II
> [Leetcode 454](https://leetcode.com/problems/4sum-ii/description/)

Given four lists A, B, C, D of integer values, compute how many tuples (i, j, k, l) there are such that A[i] + B[j] + C[k] + D[l] is zero.

To make problem a bit easier, all A, B, C, D have same length of N where 0 ≤ N ≤ 500. All integers are in the range of -228 to 228 - 1 and the result is guaranteed to be at most 2<sup>31</sup> - 1.
```
Example:

Input:
A = [ 1, 2]
B = [-2,-1]
C = [-1, 2]
D = [ 0, 2]

Output:
2

Explanation:
The two tuples are:
1. (0, 0, 0, 1) -> A[0] + B[0] + C[0] + D[1] = 1 + (-2) + (-1) + 2 = 0
2. (1, 1, 0, 0) -> A[1] + B[1] + C[0] + D[0] = 2 + (-1) + (-1) + 0 = 0
```

``` cpp
class Solution {
public:
    void helper(vector<int>& A, vector<int>& B, unordered_map<int, int>& ret) {
        for (auto i : A) {
            for (auto j : B) {
                ret[i + j]++;
            }
        }
    }
    int fourSumCount(vector<int>& A, vector<int>& B, vector<int>& C, vector<int>& D) {
        unordered_map<int, int> a, b;
        helper(A, B, a);
        helper(C, D, b);
        int ret = 0;
        for (auto i : a) {
            if (b.find(-i.first) != b.end()) ret += i.second * b[-i.first];
        }
        return ret;
    }
};
```

#### Largest Number

> [Leetcode 179](https://leetcode.com/problems/largest-number/description/)

Given a list of non negative integers, arrange them such that they form the largest number.
```
Example 1:
Input: [10,2]
Output: "210"

Example 2:
Input: [3,30,34,5,9]
Output: "9534330"
```
Note: The result may be very large, so you need to return a string instead of an integer.

``` cpp
class Solution {
public:
    bool static cmp(const int& a, const int& b) {
        string a1 = to_string(a);
        string b1 = to_string(b);
        return a1 + b1 > b1 + a1;
    }
    string largestNumber(vector<int>& nums) {
        string ret;
        if (nums.empty()) return ret;
        sort(nums.begin(), nums.end(), cmp);
        for (auto i : nums) {
            ret += to_string(i);
        }
        while (!ret.empty() && ret[0] == '0') ret.erase(ret.begin()); 
        return ret.empty() ? "0" : ret;
    }
};
```

#### First Missing Positive

> [Leetcode 41](https://leetcode.com/problems/first-missing-positive/description/)

Given an unsorted integer array, find the smallest missing positive integer.
```
Example 1:
Input: [1,2,0]
Output: 3

Example 2:
Input: [3,4,-1,1]
Output: 2

Example 3:
Input: [7,8,9,11,12]
Output: 1
```

``` cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) swap(nums[nums[i] - 1], nums[i]);
        }
        for (int i = 0; i < n; i++) {
            if (i + 1 != nums[i]) return i + 1;
        }
        return n + 1;
    }
};
```

#### Find the Duplicate Number

> [Leetcode 287](https://leetcode.com/problems/find-the-duplicate-number/description/)

Given an array nums containing n + 1 integers where each integer is between 1 and n (inclusive), prove that at least one duplicate number must exist. Assume that there is only one duplicate number, find the duplicate one.
```
Example 1:
Input: [1,3,4,2,2]
Output: 2

Example 2:
Input: [3,1,3,4,2]
Output: 3
```

``` cpp
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        if (nums.empty()) return -1;
        for (auto &i : nums) {
            if (nums[abs(i) - 1] < 0) return abs(i);
            nums[abs(i) - 1] *= -1;
        }
        return -1;
    }
};
```


#### Find All Duplicates in an Array

> [Leetcode 442](https://leetcode.com/problems/find-all-duplicates-in-an-array/description/)

Given an array of integers, 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.

Find all the elements that appear twice in this array.

Could you do it without extra space and in O(n) runtime?
```
Example:
Input:
[4,3,2,7,8,2,3,1]

Output:
[2,3]
```

``` cpp
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> ret;
        for (auto &i : nums) {
            if (nums[abs(i) - 1] < 0) ret.push_back(abs(i));
            nums[abs(i) - 1] *= -1;
        }
        return ret;
    }
};
```


#### Longest Substring Without Repeating Characters

> [Leetcode 3](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/)

Given a string, find the length of the longest substring without repeating characters.
```
Example 1:
Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", which the length is 3.

Example 2:
Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.

Example 3:
Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 
             Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

``` cpp
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if (s.empty()) return 0;
        unordered_map<char, int> m;
        int index = -1, n = s.size(), ret = 0;
        for (int i = 0; i < n; i++) {
            auto c = s[i];
            if (m.find(c) != m.end() && m[c] > index) index = m[c]; 
            ret = max(ret, i - index);
            m[c] = i;
        }
        return ret;
    }
};
```


#### 连续子数组的最大值

> [牛客网](https://www.nowcoder.com/practice/459bd355da1549fa8a49e350bf3df484?tpId=13&tqId=11183&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。给一个数组，返回它的最大连续子序列的和，你会不会被他忽悠住？(子向量的长度至少是1)

``` cpp
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        int ans = 0, n = array.size();
        int ret = INT_MIN;
        for (int i = 0; i < n; i++) {
            ans += array[i];
            ret = max(ret, ans);
            if (ans <= 0) ans = 0;
        }
        return ret;
    }
};
```

#### 整数中1出现的次数（从1到n整数中1出现的次数）

> [牛客网](https://www.nowcoder.com/practice/bd7f978302044eee894445e244c7eee6?tpId=13&tqId=11184&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)
> 因为此题细节较多，为了多加练习，在此处也做整理

求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数（从1 到 n 中1出现的次数）。

``` cpp
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n) {
        string s = to_string(n);
        int len = s.size(), ret = 0;
        for (int i = 0; i < len; i++) {
            int c = s[len - 1 - i] - '0';
            if (c == 0) {
                ret += n / (int)pow(10, i + 1) * pow(10, i);
            }
            else if (c == 1) {
                ret += n / (int)pow(10, i + 1) * pow(10, i) + (n % (int)pow(10, i)) + 1;
            }
            else {
                ret += (n / (int)pow(10, i + 1) + 1) * pow(10, i);
            }
        }
        return ret;
    }
};
```

### 双指针

#### Minimum Size Subarray Sum
> [Leetcode 209](https://leetcode.com/problems/minimum-size-subarray-sum/description/)

Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.
```
Example: 

Input: s = 7, nums = [2,3,1,2,4,3]
Output: 2
Explanation: the subarray [4,3] has the minimal length under the problem constraint.
```

Follow up:
If you have figured out the O(n) solution, try coding another solution of which the time complexity is O(n log n). 

``` cpp
class Solution {
public:
    // 因为是正整数，和一直是递增的，所以使用双指针就可以了
    int minSubArrayLen(int s, vector<int>& nums) {
        int ans = 0, i = 0, j = 0, n = nums.size();
        int ret = INT_MAX;
        while (j < n) {
            ans += nums[j++];
            while (i < j && ans - nums[i] >= s) ans -= nums[i++];
            if (ans >= s) ret = min(ret, j - i);
        }
        return ret == INT_MAX ? 0 : ret;
    }
};
```

#### 3Sum
> [Leetcode 15](https://leetcode.com/problems/3sum/description/)

Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

Note:

The solution set must not contain duplicate triplets.
```
Example:

Given array nums = [-1, 0, 1, 2, -1, -4],

A solution set is:
[
  [-1, 0, 1],
  [-1, -1, 2]
]
```

``` cpp
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> ret;
        int n = nums.size();
        if (n < 3) return ret;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n - 2; i++) {
            if (i == 0 || nums[i] != nums[i - 1]) {
                int j = i + 1, k = n - 1;
                while (j < k) {
                    int ans = nums[i] + nums[j] + nums[k];
                    if (!ans) {
                        ret.push_back(vector<int>{nums[i], nums[j], nums[k]});
                        while (j < k && nums[j + 1] == nums[j]) j++;
                        while (j < k && nums[k - 1] == nums[k]) k--;
                        j ++, k --;
                    }
                    else if (ans > 0) k --;
                    else j ++;
                }    
            }
            
        }
        return ret;
    }
};
```


#### 3Sum Closest
> [Leetcode 16](https://leetcode.com/problems/3sum-closest/description/)

Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.
```
Example:

Given array nums = [-1, 2, 1, -4], and target = 1.
```
The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

``` cpp
// 3Sum的扩展版本
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        int n = nums.size();
        if (n < 3) return -1;
        sort(nums.begin(), nums.end());
        int ret = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < n - 2; i++) {
            int j = i + 1, k = n - 1;
            while (j < k) {
                int ans = nums[i] + nums[j] + nums[k];
                if (ans == target) return target;
                if (abs(ans - target) < abs(ret - target)) ret = ans;
                if (ans > target) k--;
                else j ++;
            }
        }
        return ret;
    }
};
```

#### 丑数
> [牛客网](https://www.nowcoder.com/practice/6aa9e04fc3794f68acf8778237ba065b?tpId=13&tqId=11186&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

把只包含质因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含质因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

``` cpp
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if (index < 1) return 0;
        vector<int> ret(index, 0);
        int c2 = 0, c3 = 0, c5 = 0;
        ret[0] = 1;
        for (int i = 1; i < index; i++) {
            ret[i] = min(ret[c2] * 2, min(ret[c3] * 3, ret[c5] * 5));
            if (ret[c2] * 2 == ret[i]) c2 ++;
            if (ret[c3] * 3 == ret[i]) c3 ++;
            if (ret[c5] * 5 == ret[i]) c5 ++;
        }
        return ret[index - 1];
    }
};
```



#### Super Ugly Number
> [Leetcode 313](https://leetcode.com/problems/super-ugly-number/description/)

Write a program to find the nth super ugly number.

Super ugly numbers are positive numbers whose all prime factors are in the given prime list primes of size k.
```
Example:

Input: n = 12, primes = [2,7,13,19]
Output: 32 
Explanation: [1,2,4,7,8,13,14,16,19,26,28,32] is the sequence of the first 12 
             super ugly numbers given primes = [2,7,13,19] of size 4.
```

``` cpp
class Solution {
public:
    int nthSuperUglyNumber(int n, vector<int>& primes) {
        int len = primes.size();
        vector<int> cnt(len + 1, 0);
        vector<int> ret(n, INT_MAX);
        ret[0] = 1;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < len; j++) ret[i] = min(ret[i], primes[j] * ret[cnt[j]]);
            for (int j = 0; j < len; j++) {
                if (ret[i] == primes[j] * ret[cnt[j]]) cnt[j] ++;
            }
        }
        return ret[n - 1];
    }
};
```

#### Remove Duplicates from Sorted Array II

> [Leetcode 80](https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/description/)

Given a sorted array nums, remove the duplicates in-place such that duplicates appeared at most twice and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
```
Example 1:
Given nums = [1,1,1,2,2,3],
Your function should return length = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
It doesn't matter what you leave beyond the returned length.

Example 2:
Given nums = [0,0,1,1,1,1,2,3,3],
Your function should return length = 7, with the first seven elements of nums being modified to 0, 0, 1, 1, 2, 3 and 3 respectively.
It doesn't matter what values are set beyond the returned length.
```

Clarification:

Confused why the returned value is an integer but your answer is an array?

Note that the input array is passed in by reference, which means modification to the input array will be known to the caller as well.

Internally you can think of this:
```
// nums is passed in by reference. (i.e., without making a copy)
int len = removeDuplicates(nums);

// any modification to nums in your function would be known by the caller.
// using the length returned by your function, it prints the first len elements.
for (int i = 0; i < len; i++) {
    print(nums[i]);
}
```

``` cpp
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int index = 0, n = nums.size();
        for (int i = 0; i < n; i++) {
            if (index < 2 || nums[index - 2] != nums[i]) nums[index ++] = nums[i]; 
        }
        return index;
    }
};
```

#### Trapping Rain Water

> [Leetcode 42](https://leetcode.com/problems/trapping-rain-water/description/)

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.


The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped. Thanks Marcos for contributing this image!

```
Example:
Input: [0,1,0,2,1,0,1,3,2,1,2,1]
Output: 6
```

``` cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int ret = 0, n = height.size();
        int level = 0, l = 0, r = n - 1;
        while (l < r) {
            int lower = height[height[l] < height[r] ? l ++ : r --];
            level = max(level, lower);
            ret += level - lower;
        }
        return ret;
    }
};
```

#### 构建乘积数组

> [牛客网](https://www.nowcoder.com/practice/94a4d381a68b47b7a8bed86f2975db46?tpId=13&tqId=11204&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

``` cpp
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int len = A.size();
        vector<int> B(len, 1);
        for (int i = 1; i < len; i++) {
            B[i] = A[i - 1] * B[i - 1];
        }
        int ans = 1;
        for (int i = len - 1; i>= 0; i--) {
            B[i] *= ans;
            ans *= A[i];
        }
        return B;
    }
};
```



# 队列

队列通常的应用一般包括单调队列等，这里只列出一部分单调队列题目，之后不断补充...

## Shortest Subarray with Sum at Least K
> [Leetcode 862](https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/description/)

Return the length of the shortest, non-empty, contiguous subarray of A with sum at least K.

If there is no non-empty subarray with sum at least K, return -1.

 
```
Example 1:
Input: A = [1], K = 1
Output: 1

Example 2:
Input: A = [1,2], K = 4
Output: -1

Example 3:
Input: A = [2,-1,2], K = 3
Output: 3
 

Note:

1 <= A.length <= 50000
-10 ^ 5 <= A[i] <= 10 ^ 5
1 <= K <= 10 ^ 9
```

``` cpp
class Solution {
public:
    int shortestSubarray(vector<int>& A, int K) {
        // 此算法是利用了双指针的思路，不过需要维护一个单调队列
        // 才能需要保证更新长度过程中是满足条件的，否则有升有降只能使用平方级的时间复杂度
        int n = A.size(), ret = n + 1;
        vector<int> B(n + 1, 0);
        for (int i = 0; i < n; i++) {
            B[i + 1] = B[i] + A[i];
        }
        deque<int> d;
        for (int i = 0; i <= n; i++) {
            // 因为是有序的，所以此处还可以使用二分查找
            while (!d.empty() && B[i] - B[d.front()] >= K) {
                ret = min(ret, i - d.front());
                d.pop_front();
            }
            while (!d.empty() && B[d.back()] >= B[i]) d.pop_back();
            d.push_back(i);
        }
        return ret == n + 1 ? -1 : ret;
    }
};
```

## Sliding Window Maximum
> [Leetcode 239](https://leetcode.com/problems/sliding-window-maximum/description/)

Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.

```
Example:

Input: nums = [1,3,-1,-3,5,3,6,7], and k = 3
Output: [3,3,5,5,6,7] 
Explanation: 

Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
```
Note: 
You may assume k is always valid, 1 ≤ k ≤ input array's size for non-empty array.

Follow up:
Could you solve it in linear time?

``` cpp
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        int n = nums.size();
        vector<int> ret;
        deque<int> d;
        for (int i = 0; i < n; i++) {
            while (!d.empty() && nums[i] > nums[d.back()]) d.pop_back();
            d.push_back(i);
            if (i >= k - 1) ret.push_back(nums[d.front()]);
            if (d.front() <= i - k + 1) d.pop_front();
        }
        return ret;
    }
};
```



# 链表

## LRU Cache（双端链表list）

> [Leetcode 146](https://leetcode.com/problems/lru-cache/description/)

Design and implement a data structure for Least Recently Used (LRU) cache. It should support the following operations: get and put.

get(key) - Get the value (will always be positive) of the key if the key exists in the cache, otherwise return -1.
put(key, value) - Set or insert the value if the key is not already present. When the cache reached its capacity, it should invalidate the least recently used item before inserting a new item.

Follow up:
Could you do both operations in O(1) time complexity?
```
Example:

LRUCache cache = new LRUCache( 2 /* capacity */ );

cache.put(1, 1);
cache.put(2, 2);
cache.get(1);       // returns 1
cache.put(3, 3);    // evicts key 2
cache.get(2);       // returns -1 (not found)
cache.put(4, 4);    // evicts key 1
cache.get(1);       // returns -1 (not found)
cache.get(3);       // returns 3
cache.get(4);       // returns 4
```
> list是stl的双端链表，链表指针可以用O(1)时间删除，但是随机访问时间慢

``` cpp
typedef list<int>::iterator Iter;
class LRUCache {
public:
    int cap;
    unordered_map<int, pair<Iter, int>> m;
    list<int> d;
    
    LRUCache(int capacity) {
        cap = capacity;
    }
    
    void helper(int key) {
        d.erase(m[key].first);
        m.erase(key);
    }
    
    int get(int key) {
        if (m.find(key) == m.end()) return -1;
        int val = m[key].second;
        helper(key);
        d.push_front(key);
        m[key] = {d.begin(), val};
        return val;
    }
    
    void put(int key, int value) {
        if (m.find(key) != m.end()) helper(key);
        else if (m.size() == cap) helper(d.back());
        d.push_front(key);
        m[key] = {d.begin(), value};
    }
};
```

## Insertion Sort List

> [Leetcode 147](https://leetcode.com/problems/insertion-sort-list/description/)

Sort a linked list using insertion sort.

![](https://upload.wikimedia.org/wikipedia/commons/0/0f/Insertion-sort-example-300px.gif)

Algorithm of Insertion Sort:

Insertion sort iterates, consuming one input element each repetition, and growing a sorted output list.
At each iteration, insertion sort removes one element from the input data, finds the location it belongs within the sorted list, and inserts it there.
It repeats until no input elements remain.
```
Example 1:
Input: 4->2->1->3
Output: 1->2->3->4

Example 2:
Input: -1->5->3->4->0
Output: -1->0->3->4->5
```

``` cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    void helper(ListNode* &ret, ListNode* head) {
        while (head) {
            ListNode* temp = head;
            head = head->next;
            temp->next = NULL;
            ListNode* cur = ret;
            while (cur->next && cur->next->val < temp->val) {
                cur = cur->next;
            }
            temp->next = cur->next;
            cur->next = temp;
        }
    }
    ListNode* insertionSortList(ListNode* head) {
        if (!head) return NULL;
        ListNode* ret = new ListNode(-1);
        helper(ret, head);
        return ret->next;
    }
};
```

## Sort List

> [Leetcode 148](https://leetcode.com/problems/sort-list/description/)

Sort a linked list in O(n log n) time using constant space complexity.
```
Example 1:
Input: 4->2->1->3
Output: 1->2->3->4

Example 2:
Input: -1->5->3->4->0
Output: -1->0->3->4->5
```

``` cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
// 归并排序
class Solution {
public:
    ListNode* merge(ListNode* l, ListNode* r) {
        auto ret = new ListNode(-1);
        auto cur = ret;
        while (l && r) {
            if (l->val < r->val) {
                cur->next = l;
                cur = cur->next;
                l = l->next;
            }
            else {
                cur->next = r;
                cur = cur->next;
                r = r->next;
            }
        }
        while (l) cur->next = l, cur = cur->next, l = l->next;
        while (r) cur->next = r, cur = cur->next, r = r->next;
        cur->next = NULL;
        return ret->next;
    }
    
    ListNode* sortList(ListNode* head) {
        if (!head || !head->next) return head;
        auto fast = head, slow = head;
        while (fast->next && fast->next->next) {
            slow = slow->next;
            fast = fast->next->next;
        }
        auto mid = slow->next;
        slow->next = NULL;
        auto l = sortList(head);
        auto r = sortList(mid);
        return merge(l, r);
    }
};
```

``` cpp
// 快排版本
class Solution {
public:
    ListNode* partition(ListNode* start, ListNode* end) {
        ListNode* base = start;
        start = start->next;
        ListNode* l = new ListNode(-1);
        ListNode* curl = l;
        ListNode* r = new ListNode(-1);
        ListNode* curr = r;
        while (start != end) {
            if (base->val > start->val) 
                curl->next = start, curl = curl->next;
            else 
                curr->next = start, curr = curr->next;
            start = start->next;
        }
        curr->next = end;
        curl->next = base;
        base->next = r->next;
        return l->next;
    }
    ListNode* helper(ListNode* start, ListNode* end) {
        if (start == end) return start;
        auto mid = start;
        start = partition(start, end);
        auto l = helper(start, mid);
        auto r = helper(mid->next, end);
        mid->next = r;
        return l;
    }
    ListNode* sortList(ListNode* head) {
        if (!head) return head;
        return helper(head, NULL);
    }
};
```

## Remove Duplicates from Sorted List II

> [Leetcode 82](https://leetcode.com/problems/remove-duplicates-from-sorted-list-ii/description/)


Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.
```
Example 1:
Input: 1->2->3->3->4->4->5
Output: 1->2->5

Example 2:
Input: 1->1->1->2->3
Output: 2->3
```

``` cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if (!head || !head->next) return head;
        int val = head->val;
        ListNode* cur = head->next;
        if (cur->val != val) {
            head->next = deleteDuplicates(cur);
            return head;
        }
        while (cur && cur->val == val) cur = cur->next;
        return deleteDuplicates(cur);
    }
};
```

## Reorder List

> [Leetcode 143](https://leetcode.com/problems/reorder-list/description/)

Given a singly linked list L: L0→L1→…→Ln-1→Ln,
reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…

You may not modify the values in the list's nodes, only nodes itself may be changed.
```
Example 1:
Given 1->2->3->4, reorder it to 1->4->2->3.

Example 2:
Given 1->2->3->4->5, reorder it to 1->5->2->4->3.
```

``` cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    void reorderList(ListNode* head) {
        if (!head || !head->next) return;
        ListNode* fast = head, *slow = head;
        while (fast->next && fast->next->next) {
            fast = fast->next->next;
            slow = slow->next;
        }
        auto mid = slow->next;
        slow->next = NULL;
        ListNode* cur = NULL;
        while (mid) {
            ListNode* temp = mid->next;
            mid->next = cur;
            cur = mid;
            mid = temp;
        }
        mid = cur;
        cur = head;
        while (mid) {
            ListNode* temp = cur->next;
            cur->next = mid;
            mid = mid->next;
            cur->next->next = temp;
            cur = temp;
        }
    }
};
```

## Reverse Linked List II

> [Leetcode 92](https://leetcode.com/problems/reverse-linked-list-ii/description/)

Reverse a linked list from position m to n. Do it in one-pass.

Note: 1 ≤ m ≤ n ≤ length of list.
```
Example:
Input: 1->2->3->4->5->NULL, m = 2, n = 4
Output: 1->4->3->2->5->NULL
```

``` cpp
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        if (m == n) return head;
        ListNode* ret = new ListNode(-1);
        ret->next = head;
        auto cur = ret;
        // 因为从1开始数，所以m-1
        for (int i = 0; i < m - 1; i++) cur = cur->next;
        auto start = cur->next;
        for (int i = 0; i < n - m; i++) {
            auto temp = start->next;
            start->next = temp->next;
            temp->next = cur->next;
            cur->next = temp;
        }
        return ret->next;
    }
};
```

# 栈

## 132 Pattern

 > [Leetcode 456](https://leetcode.com/problems/132-pattern/description/)
 
 Given a sequence of n integers a1, a2, ..., an, a 132 pattern is a subsequence ai, aj, ak such that i < j < k and ai < ak < aj. Design an algorithm that takes a list of n numbers as input and checks whether there is a 132 pattern in the list.

Note: n will be less than 15,000.
```
Example 1:
Input: [1, 2, 3, 4]
Output: False
Explanation: There is no 132 pattern in the sequence.

Example 2:
Input: [3, 1, 4, 2]
Output: True
Explanation: There is a 132 pattern in the sequence: [1, 4, 2].

Example 3:
Input: [-1, 3, 2, 0]
Output: True
Explanation: There are three 132 patterns in the sequence: [-1, 3, 2], [-1, 3, 0] and [-1, 2, 0].
```

``` cpp
class Solution {
public:
    bool find132pattern(vector<int>& nums) {
        stack<pair<int, int>> s;
        for (auto i : nums) {
            if (s.empty() || s.top().second > i) {
                s.push(make_pair(i, i));
            } 
            else if (s.top().second < i) {
                int m = s.top().second;
                if (s.top().first > i) return true;
                while (!s.empty() && s.top().first <= i) s.pop();
                if (!s.empty() && s.top().second < i) return true;
                s.push({i, m});
            }
        }
        return false;
    }
};
```

## Next Greater Element II
> [Leetcode 503](https://leetcode.com/problems/next-greater-element-ii/description/)

Given a circular array (the next element of the last element is the first element of the array), print the Next Greater Number for every element. The Next Greater Number of a number x is the first greater number to its traversing-order next in the array, which means you could search circularly to find its next greater number. If it doesn't exist, output -1 for this number.

```
Example 1:
Input: [1,2,1]
Output: [2,-1,2]
Explanation: The first 1's next greater number is 2; 
The number 2 can't find next greater number; 
The second 1's next greater number needs to search circularly, which is also 2.
Note: The length of given array won't exceed 10000.
```

``` cpp
class Solution {
public:
    vector<int> nextGreaterElements(vector<int>& nums) {
        if (nums.empty()) return vector<int>();
        int len = nums.size();
        stack<int> s;
        vector<int> ret(len, -1);
        for (int i = 0; i < len * 2; i++) {
            while (!s.empty() && nums[s.top()] < nums[i % len]) {
                ret[s.top()] = nums[i % len];
                s.pop();
            }
            if (i < len) s.push(i);
        }
        return ret;
    }
};
```

## Binary Search Tree Iterator
> [Leetcode 173](https://leetcode.com/problems/binary-search-tree-iterator/description/)

Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

Calling next() will return the next smallest number in the BST.

Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.

``` cpp
/**
 * Definition for binary tree
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class BSTIterator {
public:
    stack<TreeNode*> s;
    void helper(TreeNode* cur) {
        while (cur) {
            s.push(cur);
            cur = cur->left;
        }
    }
    BSTIterator(TreeNode *root) {
        helper(root);
    }

    /** @return whether we have a next smallest number */
    bool hasNext() {
        return !s.empty();
    }

    /** @return the next smallest number */
    int next() {
        auto cur = s.top();
        s.pop();
        helper(cur->right);
        return cur->val;
    }
};

/**
 * Your BSTIterator will be called like this:
 * BSTIterator i = BSTIterator(root);
 * while (i.hasNext()) cout << i.next();
 */
```

## Verify Preorder Serialization of a Binary Tree
> [Leetcode 331](https://leetcode.com/problems/verify-preorder-serialization-of-a-binary-tree/description/)

One way to serialize a binary tree is to use pre-order traversal. When we encounter a non-null node, we record the node's value. If it is a null node, we record using a sentinel value such as #.
```
     _9_
    /   \
   3     2
  / \   / \
 4   1  #  6
/ \ / \   / \
# # # #   # #
```
For example, the above binary tree can be serialized to the string "9,3,4,#,#,1,#,#,2,#,6,#,#", where # represents a null node.

Given a string of comma separated values, verify whether it is a correct preorder traversal serialization of a binary tree. Find an algorithm without reconstructing the tree.

Each comma separated value in the string must be either an integer or a character '#' representing null pointer.

You may assume that the input format is always valid, for example it could never contain two consecutive commas such as "1,,3".

```
Example 1:
Input: "9,3,4,#,#,1,#,#,2,#,6,#,#"
Output: true

Example 2:
Input: "1,#"
Output: false

Example 3:
Input: "9,#,#,1"
Output: false
```

``` cpp
class Solution {
public:
    bool isValidSerialization(string preorder) {
        if (preorder.empty()) return false;
        vector<string> s;
        string ans;
        for (auto c : preorder) {
            if (c == ',') {
                s.push_back(ans);
                ans = "";
            }
            else ans.push_back(c);
        }
        s.push_back(ans);
        int d = 0;
        for (int i = 0; i < s.size() - 1; i++) {
            string temp = s[i];
            if (temp == "#") {
                if (!d) return false;
                else d--;
            }
            else d++;
        }
        return !d && s.back() == "#";
    }
};
```

# HashTable
> 哈希表通常用来记录一些中间状态从而实现O(1)，这里提供一些比较典型的例子，之后不断补充...

## Max Points on a Line
> [Leetcode 149](https://leetcode.com/problems/max-points-on-a-line/description/)

Given n points on a 2D plane, find the maximum number of points that lie on the same straight line.
```
Example 1:

Input: [[1,1],[2,2],[3,3]]
Output: 3
Explanation:
^
|
|        o
|     o
|  o  
+------------->
0  1  2  3  4
Example 2:

Input: [[1,1],[3,2],[5,3],[4,1],[2,3],[1,4]]
Output: 4
Explanation:
^
|
|  o
|     o        o
|        o
|  o        o
+------------------->
0  1  2  3  4  5  6
```

``` cpp
/**
 * Definition for a point.
 * struct Point {
 *     int x;
 *     int y;
 *     Point() : x(0), y(0) {}
 *     Point(int a, int b) : x(a), y(b) {}
 * };
 */
class Solution {
public:
    int maxPoints(vector<Point>& points) {
        int ret = 0;
        int len = points.size();
        if (len < 3) return len;
        for (int i = 0; i < len; i++) {
            map<pair<int, int>, int> m;
            int vertical = 0, overlap = 0, local_ret = 0;
            for (int j = i + 1; j < len; j++) {
                int a = points[i].x - points[j].x;
                int b = points[i].y - points[j].y;
                if (!a && !b) overlap ++;
                else if (!a) vertical ++;
                else {
                    int k = GCD(a, b);
                    auto ans = make_pair(a / k, b / k);
                    m[ans] ++;
                    local_ret = max(local_ret, m[ans]);
                }
                local_ret = max(local_ret, vertical);
            }
            ret = max(ret, local_ret + overlap + 1);
        }
        return ret;
    }
    int GCD(int a, int b) {
        if (!b) return a;
        else return GCD(b, a % b);
    }
};
```

# 桶
## Contains Duplicate III
> [Leetcode 220](https://leetcode.com/problems/contains-duplicate-iii/description/)

Given an array of integers, find out whether there are two distinct indices i and j in the array such that the absolute difference between nums[i] and nums[j] is at most t and the absolute difference between i and j is at most k.

```
Example 1:
Input: nums = [1,2,3,1], k = 3, t = 0
Output: true

Example 2:
Input: nums = [1,0,1,1], k = 1, t = 2
Output: true

Example 3:
Input: nums = [1,5,9,1,5,9], k = 2, t = 3
Output: false
```

``` cpp
class Solution {
public:
    long long helper(long long num, long long w) {
        return num >= 0 ? num / w : ((num + 1) / w - 1);
    }
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        if (t < 0) return false;
        int n = nums.size();
        long long w = t + 1;
        unordered_map<long long, int> m;
        for (int i = 0; i < n; i++) {
            long long ID = helper(nums[i], w);
            if (m.find(ID) != m.end()) return true;
            // 要注意不能用t比较大小否则int会溢出
            if (m.find(ID - 1) != m.end() && abs(nums[i] - nums[m[ID - 1]]) < w) return true;
            if (m.find(ID + 1) != m.end() && abs(nums[i] - nums[m[ID + 1]]) < w) return true;
            m[ID] = i;
            if (i >= k) {
                m.erase(helper(nums[i - k], w));
            }
        }
        return false;
    }
};
```


# 树

## 最近公共祖先
``` cpp
#include <iostream>
#include <vector>
#include <climits>

using namespace std;

struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x): val(x), left(NULL), right(NULL) {}
};



TreeNode* build() {
    int x;
    cin >> x;
    if (x == -1) return NULL;
    TreeNode* ret = new TreeNode(x);
    ret->left = build();
    ret->right = build();
    return ret;
}


int helper(TreeNode* root, int& x, int& y) {
    if (!root) return -1;
    if (root->val == x) {
        x = INT_MIN;
        helper(root->left, x, y);
        helper(root->right, x, y);
        if (y == INT_MIN) return root->val;
    }
    else if (root->val == y) {
        y = INT_MIN;
        helper(root->left, x, y);
        helper(root->right, x, y);
        if (x == INT_MIN) return root->val;
    }
    else {
        int ans = 0;
        int l = helper(root->left, x, y);
        ans = (x == INT_MIN) + (y == INT_MIN);
        if ((x == INT_MIN) && (y == INT_MIN)) return l;
        
        int r = helper(root->right, x, y);
        if ((x == INT_MIN) && (y == INT_MIN)) {
            if (ans) return root->val;
            else return r;
        }
    }
    return -1;
}

int main() {
    TreeNode* root = build();
    cout << "Tree Finish" << endl;
    while (true) {
        int x, y;
        cin >> x >> y;
        cout << helper(root, x, y) << endl;
    }
    return 0;
}
```

## 指定节点距离叶子节点最近的距离

此题是之前面试中一道题，题目意思是如果指定一个值（若存在则唯一存在），寻找距离此节点最近叶子节点的距离，因为此题的细节比较多，所以这里也列出来

例如：按前序遍历的树 1,2,3,4,5,#,#,#,6,7,#,#,8,9,#,10,#,#,11,#,#,#,12,#,#

``` cpp
#include <iostream>
#include <vector>
#include <algorithm>
#include <climits>
using namespace std;
struct TreeNode {
    int val;
    TreeNode* left, *right;
    TreeNode(int x): val(x), left(NULL), right(NULL) {};
};


TreeNode* build(int& index, vector<int>& nums) {
    int n = nums.size();
    if (index == n) return NULL;
    int val = nums[index++];
    if (val == -1) return NULL;
    TreeNode* ret = new TreeNode(val);
    ret->left = build(index, nums);
    ret->right = build(index, nums);
    return ret;
}

void helper(int& ret, int& l, int& r, int& m, int target, TreeNode* root) {
    l = r = m = 0;
    if (!root) return;
    int val = root->val;
    if (val == target) {
        m = 1;
    }
    if (!root->left && !root->right) {
        if (m) ret = 0;
        return;
    }
    int ll, lr, rl, rr, ml, mr;
    helper(ret, ll, lr, ml, target, root->left);
    helper(ret, rl, rr, mr, target, root->right);
    if (m) {
        if (!root->left) ret = min(ret, min(rl, rr) + 1);
        if (!root->right) ret = min(ret, min(ll, lr) + 1);
        if (root->left && root->right) {
            int ans = min(min(ll, lr) + 1, min(rl, rr) + 1);
            ret = min(ret, ans);
        }
        return;
    }
    if (ml) {
        if (root->right) {
            ret = min(ret, min(rl, rr) + ll + 2);
            //cout << "ml " << ret<< endl;
        }
        l = r = ll + 1;
        m = 1;
        return;
    }
    else if (mr) {
        if (root->left) {
            ret = min(ret, min(ll, lr) + rr + 2);
            //cout << "mr " << ret<< endl;
        }
        l = r = rr + 1;
        m = 1;
        return;
    }
    else {
        l = !root->left ? INT_MAX : min(ll, lr) + 1;
        r = !root->right ? INT_MAX : min(rr, rl) + 1;
        //cout << "mid: " << val << " " << l << " " << r << endl;
    }

}

int main() {
    int n;
    cin >> n;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    int index = 0;
    TreeNode* root = build(index, nums);
    for (int i = 0; i < 12; i++) {
        int ret = INT_MAX;
        int m = 0, l, r;
        helper(ret, l, r, m, i + 1, root);
        cout << i + 1 << " ";
        if (m) cout << ret << endl;
        else cout << -1 << endl;
    }
    return 0;
}


/*
25
1 2 3 4 5 -1 -1 -1 6 7 -1 -1 8 9 -1 10 -1 -1 11 -1 -1 -1 12 -1 -1


1 1
2 2
3 2
4 1
5 4
6 1
7 3
8 1
9 1
10 3
11 3
12 3
*/

/*
25
1 2 3 4 5 -1 -1 -1 6 7 -1 -1 8 9 -1 10 -1 -1 11 -1 -1 -1 12 -1 -1
*/

```

## Populating Next Right Pointers in Each Node II
> [Leetcode 117](https://leetcode.com/problems/populating-next-right-pointers-in-each-node-ii/description/)

Given a binary tree
```
struct TreeLinkNode {
  TreeLinkNode *left;
  TreeLinkNode *right;
  TreeLinkNode *next;
}
```
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

```
Note:

You may only use constant extra space.
Recursive approach is fine, implicit stack space does not count as extra space for this problem.
Example:

Given the following binary tree,

     1
   /  \
  2    3
 / \    \
4   5    7
After calling your function, the tree should look like:

     1 -> NULL
   /  \
  2 -> 3 -> NULL
 / \    \
4-> 5 -> 7 -> NULL
```

``` cpp
/**
 * Definition for binary tree with next pointer.
 * struct TreeLinkNode {
 *  int val;
 *  TreeLinkNode *left, *right, *next;
 *  TreeLinkNode(int x) : val(x), left(NULL), right(NULL), next(NULL) {}
 * };
 */
class Solution {
public:
    void connect(TreeLinkNode *root) {
        if (!root) return;
        TreeLinkNode* pre = root;
        pre->next = NULL;
        while (pre) {
            while (pre && !pre->left && !pre->right) pre = pre->next;
            if (!pre) continue;
            TreeLinkNode* head = pre->left ? pre->left : pre->right;
            TreeLinkNode* cur = head;
            while (pre) {
                if (pre->left) {
                    cur->next = pre->left;
                    cur = cur->next;
                }
                if (pre->right) {
                    cur->next = pre->right;
                    cur = cur->next;
                }
                pre = pre->next;
            }
            cur->next = NULL;
            pre = head;
        }
    }
};
```

## Count Complete Tree Nodes
> [Leetcode 222](https://leetcode.com/problems/count-complete-tree-nodes/description/)

Given a complete binary tree, count the number of nodes.

Note:

Definition of a complete binary tree from Wikipedia:
In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.
```
Example:

Input: 
    1
   / \
  2   3
 / \  /
4  5 6

Output: 6
```

``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int countNodes(TreeNode* root) {
        // 本题的重点是不能遍历所有，
        // 所以如果是完全树，则直接返回该树的节点数，
        // 如果不是，则返回左子树加右子树加自己的节点数
        if (!root) return 0;
        int h1 = 0, h2 = 0;
        TreeNode* l = root, *r = root;
        while (l) {
            h1 ++;
            l = l->left;
        }
        while (r) {
            h2 ++;
            r = r->right;
        }
        if (h1 == h2) return pow(2, h1) - 1;
        return 1 + countNodes(root->left) + countNodes(root->right);
    }
};
```

## Delete Node in a BST

> [Leetcode 450](https://leetcode.com/problems/delete-node-in-a-bst/description/)

Given a root node reference of a BST and a key, delete the node with the given key in the BST. Return the root node reference (possibly updated) of the BST.

Basically, the deletion can be divided into two stages:

Search for a node to remove.
If the node is found, delete the node.
Note: Time complexity should be O(height of tree).
```
Example:

root = [5,3,6,2,4,null,7]
key = 3

    5
   / \
  3   6
 / \   \
2   4   7

Given key to delete is 3. So we find the node with value 3 and delete it.

One valid answer is [5,4,6,2,null,null,7], shown in the following BST.

    5
   / \
  4   6
 /     \
2       7

Another valid answer is [5,2,6,null,4,null,7].

    5
   / \
  2   6
   \   \
    4   7
```

``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        if (!root) return NULL;
        if (root->val == key) {
            if (!root->left) return root->right;
            if (!root->right) return root->left;
            TreeNode* cur = root->right;
            while (cur->left) cur = cur->left;
            swap(root->val, cur->val);
        }
        root->left = deleteNode(root->left, key);
        root->right = deleteNode(root->right, key);
        return root;
    }
};
```

## Serialize and Deserialize BST
> [Leetcode 449](https://leetcode.com/problems/serialize-and-deserialize-bst/description/)

Serialization is the process of converting a data structure or object into a sequence of bits so that it can be stored in a file or memory buffer, or transmitted across a network connection link to be reconstructed later in the same or another computer environment.

Design an algorithm to serialize and deserialize a binary search tree. There is no restriction on how your serialization/deserialization algorithm should work. You just need to ensure that a binary search tree can be serialized to a string and this string can be deserialized to the original tree structure.

The encoded string should be as compact as possible.

``` cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Codec {
public:
    void helper(TreeNode* root, string& ret) {
        if (!root) ret += ",#";
        else {
            ret += "," + to_string(root->val);
            helper(root->left, ret);
            helper(root->right, ret);
        }
        
    }
    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        string ret;
        helper(root, ret);
        return ret.substr(1);
    }
    TreeNode* helper1(string& data, int& index) {
        string ans;
        while (index < data.size() && data[index] != ',') {
            ans.push_back(data[index++]);
        }
        if (index < data.size()) index ++;
        if (ans == "#") return NULL;
        int syn = 1;
        if (ans[0] == '-') {
            ans.erase(ans.begin());
            syn = -1;
        }
        int val = 0;
        for (auto i : ans) {
            val = val * 10 + (i - '0');
        }
        val *= syn;
        TreeNode* ret = new TreeNode(val);
        ret->left = helper1(data, index);
        ret->right = helper1(data, index);
        return ret;
    }
    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        int index = 0;
        return helper1(data, index);
    }
};

// Your Codec object will be instantiated and called as such:
// Codec codec;
// codec.deserialize(codec.serialize(root));
```

# 前缀树

## Implement Trie (Prefix Tree)
> [Leetcode 208](https://leetcode.com/problems/implement-trie-prefix-tree/description/)

Implement a trie with insert, search, and startsWith methods.

```
Example:

Trie trie = new Trie();

trie.insert("apple");
trie.search("apple");   // returns true
trie.search("app");     // returns false
trie.startsWith("app"); // returns true
trie.insert("app");   
trie.search("app");     // returns true
```
``` cpp
struct TrieNode {
    TrieNode* child[26];
    bool isKey;
    TrieNode() : isKey(false) {
        memset(child, NULL, sizeof(child));
    }
};
class Trie {
public:
    TrieNode* root;
    /** Initialize your data structure here. */
    Trie() {
        root = new TrieNode();
    }
    
    /** Inserts a word into the trie. */
    void insert(string word) {
        TrieNode* cur = root;
        for (auto i : word) {
            int c = i - 'a';
            if (!cur->child[c]) cur->child[c] = new TrieNode();
            cur = cur->child[c];
        }
        cur->isKey = true;
    }
    
    /** Returns if the word is in the trie. */
    bool search(string word) {
        TrieNode* cur = root;
        for (auto i : word) {
            int c = i - 'a';
            if (!cur->child[c]) return false;
            cur = cur->child[c];
        }
        return cur->isKey;
    }
    
    /** Returns if there is any word in the trie that starts with the given prefix. */
    bool startsWith(string prefix) {
        TrieNode* cur = root;
        for (auto i : prefix) {
            int c = i - 'a';
            if (!cur->child[c]) return false;
            cur = cur->child[c];
        }
        return true;
    }
};

/**
 * Your Trie object will be instantiated and called as such:
 * Trie obj = new Trie();
 * obj.insert(word);
 * bool param_2 = obj.search(word);
 * bool param_3 = obj.startsWith(prefix);
 */
```

## Maximum XOR of Two Numbers in an Array
> [Leetcode 421](https://leetcode.com/problems/maximum-xor-of-two-numbers-in-an-array/description/)

Given a non-empty array of numbers, a0, a1, a2, … , an-1, where 0 ≤ ai < 231.

Find the maximum result of ai XOR aj, where 0 ≤ i, j < n.

Could you do this in O(n) runtime?

```
Example:

Input: [3, 10, 5, 25, 2, 8]

Output: 28

Explanation: The maximum result is 5 ^ 25 = 28.
```

``` cpp
struct TrieNode {
    TrieNode* child[2];
    TrieNode()  {
        memset(child, NULL, sizeof(child));
    }
};
class Solution {
public:
    
    int findMaximumXOR(vector<int>& nums) {
        TrieNode* root = new TrieNode();
        for (auto n : nums) {
            TrieNode* cur = root;
            // 必须是从左往右，因为不是这样的话就不是最大，这里用了贪心
            for (int i = 31; i >= 0; i--) {
                int c = ((n >> i) & 1);
                if (!cur->child[c]) cur->child[c] = new TrieNode();
                cur = cur->child[c];
            }
        }
        int ret = 0;
        for (auto n : nums) {
            TrieNode* cur = root;
            int ans = 0;
            for (int i = 31; i >= 0; i--) {
                int c = ((n >> i) & 1);
                if (cur->child[!c]) {
                    ans <<= 1;
                    ans |= 1;
                    cur = cur->child[!c];
                }
                // 只要有一条路肯定能走到底，所以不用判断为空的情况
                else {
                    ans <<= 1;
                    cur = cur->child[c];
                }
            }
            for (int i = 0; i < 32; i++) {
                cout << ((2147483648 >> i) & 1);
            }
            cout << endl;
            ret = max(ret, ans);
        }
        return ret;
    }
};
```

## Word Search II
> [Leetcode 212](https://leetcode.com/problems/word-search-ii/description/)

Given a 2D board and a list of words from the dictionary, find all words in the board.

Each word must be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once in a word.

```
Example:

Input: 
words = ["oath","pea","eat","rain"] and board =
[
  ['o','a','a','n'],
  ['e','t','a','e'],
  ['i','h','k','r'],
  ['i','f','l','v']
]

Output: ["eat","oath"]
```

``` cpp
struct TrieNode {
    TrieNode* child[26];
    bool isKey;
    TrieNode() : isKey(false) {
        memset(child, NULL, sizeof(child));
    }
};
class Solution {
public:
    void helper(TrieNode* root, vector<vector<char>>& board, int x, int y, string ans, unordered_set<string>& ret) {
        ans = ans + board[x][y];
        if (root->isKey) {
            ret.insert(ans);
        }
        board[x][y] = '\0';
        int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
        int m = board.size(), n = board[0].size();
        for (int i = 0; i < 4; i++) {
            int X = x + a[i], Y = y + b[i];
            if (X >= 0 && X < m && Y >= 0 && Y < n && board[X][Y] != '\0') {
                int c = board[X][Y] - 'a';
                if (root->child[c]) {
                    helper(root->child[c], board, X, Y, ans, ret);
                }
            }
        }
        board[x][y] = ans.back();
    }
    vector<string> findWords(vector<vector<char>>& board, vector<string>& words) {
        if (board.empty() || board[0].empty()) return vector<string>();
        TrieNode* root = new TrieNode();
        for (auto s : words) {
            auto cur = root;
            for (auto i : s) {
                int c = i - 'a';
                if (!cur->child[c]) cur->child[c] = new TrieNode();
                cur = cur->child[c];
            }
            cur->isKey = true;
        }
        unordered_set<string> ret_set;
        for (int i = 0; i < board.size(); i++) {
            for (int j = 0; j < board[0].size(); j++) {
                int c = board[i][j] - 'a';
                if (root->child[c]) helper(root->child[c], board, i, j, "", ret_set);
            }
        }
        vector<string> ret;
        for (auto s : ret_set) {
            ret.push_back(s);
        }
        return ret;
    }
};
```

# 堆
## TopK

### Kth Largest Element in an Array
> [Leetcode 215](https://leetcode.com/problems/kth-largest-element-in-an-array/description/)

Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.
```
Example 1:
Input: [3,2,1,5,6,4] and k = 2
Output: 5

Example 2:
Input: [3,2,3,1,2,4,5,5,6] and k = 4
Output: 4
```

``` cpp
class Solution {
public:
    void heapfy(vector<int>& nums, int index, int max) {
        int left = index * 2 + 1;
        int right = left + 1;
        int smallest = index;
        if (left < max && nums[left] < nums[smallest]) smallest = left;
        if (right < max && nums[right] < nums[smallest]) smallest = right;
        if (index != smallest) {
            swap(nums[index], nums[smallest]);
            heapfy(nums, smallest, max);
        }
    }
    
    int helper(vector<int>& nums, int k) {
        int n = nums.size();
        if (n < k) return -1;
        vector<int> ans;
        for (int i = 0; i < k; i++) ans.push_back(nums[i]);
        for (int i = k / 2; i >= 0; i--) heapfy(ans, i, k);
        for (int i = k; i < n; i++) {
            if (nums[i] > ans[0]) {
                ans[0] = nums[i];
                heapfy(ans, 0, k);
            }
        }
        return ans[0];
    }
    
    int findKthLargest(vector<int>& nums, int k) {
        return helper(nums, k);
    }
};
```

### Top K Frequent Elements
> [Leetcode 347](https://leetcode.com/problems/top-k-frequent-elements/description/)

Given a non-empty array of integers, return the k most frequent elements.
```
Example 1:
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]

Example 2:
Input: nums = [1], k = 1
Output: [1]
```

``` cpp
class Solution {
public:
    void heapfy(vector<int>& nums, int index, int max, unordered_map<int, int>& m) {
        int left = index * 2 + 1;
        int right = left + 1;
        int smallest = index;
        if (left < max && m[nums[left]] < m[nums[smallest]]) smallest = left;
        if (right < max && m[nums[right]] < m[nums[smallest]]) smallest = right;
        if (smallest != index) {
            swap(nums[smallest], nums[index]);
            heapfy(nums, smallest, max, m);
        }
    }
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> m;
        vector<int> ans;
        for (auto i : nums) {
            m[i] ++;
            if (m[i] == 1) ans.push_back(i);
        }
        int len = ans.size();
        if (len < k) return ans;
        vector<int> ret;
        for (int i = 0; i < k; i++) ret.push_back(ans[i]);
        for (int i = k / 2 - 1; i >= 0; i--) heapfy(ret, i, k, m);
        
        for (int i = k; i < len; i++) {
            if (m[ans[i]] > m[ret[0]]) {
                ret[0] = ans[i];
                for (int j = k / 2 - 1; j >= 0; j--) heapfy(ret, j, k, m);
            }
        }
        return ret;
        
    }
};
```

### Find K Pairs with Smallest Sums
> [Leetcode 373](https://leetcode.com/problems/find-k-pairs-with-smallest-sums/description/)

You are given two integer arrays nums1 and nums2 sorted in ascending order and an integer k.

Define a pair (u,v) which consists of one element from the first array and one element from the second array.

Find the k pairs (u1,v1),(u2,v2) ...(uk,vk) with the smallest sums.
```
Example 1:
Given nums1 = [1,7,11], nums2 = [2,4,6],  k = 3
Return: [1,2],[1,4],[1,6]
The first 3 pairs are returned from the sequence:
[1,2],[1,4],[1,6],[7,2],[7,4],[11,2],[7,6],[11,4],[11,6]

Example 2:
Given nums1 = [1,1,2], nums2 = [1,2,3],  k = 2
Return: [1,1],[1,1]
The first 2 pairs are returned from the sequence:
[1,1],[1,1],[1,2],[2,1],[1,2],[2,2],[1,3],[1,3],[2,3]

Example 3:
Given nums1 = [1,2], nums2 = [3],  k = 3 
Return: [1,3],[2,3]
All possible pairs are returned from the sequence:
[1,3],[2,3]
```

``` cpp
class Solution {
public:
    void heapfy(vector<pair<int, int>>& nums, vector<int>& A, vector<int>& B, int index, int max) {
        int left = index * 2 + 1;
        int right = left + 1;
        int smallest = index;
        if (left < max && A[nums[left].first] + B[nums[left].second] < A[nums[smallest].first] + B[nums[smallest].second])
            smallest = left;
        if (right < max && A[nums[right].first] + B[nums[right].second] < A[nums[smallest].first] + B[nums[smallest].second])
            smallest = right;
        if (index != smallest) {
            swap(nums[index], nums[smallest]);
            heapfy(nums, A, B, smallest, max);
        }
    }
    vector<pair<int, int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
        if (nums1.empty() || nums2.empty()) return vector<pair<int, int>>();
        vector<pair<int, int>> ret;
        vector<pair<int, int>> ans;
        ans.push_back(make_pair(0, 0));
        for (int i = 0; i < k && !ans.empty(); i++) {
            auto t = ans[0];
            ans.erase(ans.begin());
            ret.push_back(make_pair(nums1[t.first], nums2[t.second]));
            if (t.first == 0 && t.second + 1 < nums2.size()) ans.push_back(make_pair(0, t.second + 1));
            if (t.first + 1 < nums1.size()) ans.push_back(make_pair(t.first + 1, t.second));
            int len = ans.size();
            for (int j = len / 2; j >= 0; j--) heapfy(ans, nums1, nums2, j, len);
        }
        return ret;
    }
};
```

## Find Median from Data Stream
> [Leetcode 295](https://leetcode.com/problems/find-median-from-data-stream/description/)

Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle value.

For example,
[2,3,4], the median is 3

[2,3], the median is (2 + 3) / 2 = 2.5

Design a data structure that supports the following two operations:

void addNum(int num) - Add a integer number from the data stream to the data structure.
double findMedian() - Return the median of all elements so far.
 
```
Example:

addNum(1)
addNum(2)
findMedian() -> 1.5
addNum(3) 
findMedian() -> 2
```

``` cpp
class MedianFinder {
public:
    /** initialize your data structure here. */
    priority_queue<int> small;
    priority_queue<int, vector<int>, greater<int>> large;
    MedianFinder() {
        
    }
    
    void addNum(int num) {
        small.push(num);
        large.push(small.top());
        small.pop();
        if (small.size() < large.size()) {
            small.push(large.top());
            large.pop();
        }
    }
    
    double findMedian() {
        if (large.size() == small.size()) return (large.top() + small.top()) / 2.0;
        return small.top();
    }
};

/**
 * Your MedianFinder object will be instantiated and called as such:
 * MedianFinder obj = new MedianFinder();
 * obj.addNum(num);
 * double param_2 = obj.findMedian();
 */
```

## Sliding Window Median
> [Leetcode 480](https://leetcode.com/problems/sliding-window-median/description/)

Median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value. So the median is the mean of the two middle value.

```
Examples: 
[2,3,4] , the median is 3

[2,3], the median is (2 + 3) / 2 = 2.5

Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Your job is to output the median array for each window in the original array.

For example,
Given nums = [1,3,-1,-3,5,3,6,7], and k = 3.

Window position                Median
---------------               -----
[1  3  -1] -3  5  3  6  7       1
 1 [3  -1  -3] 5  3  6  7       -1
 1  3 [-1  -3  5] 3  6  7       -1
 1  3  -1 [-3  5  3] 6  7       3
 1  3  -1  -3 [5  3  6] 7       5
 1  3  -1  -3  5 [3  6  7]      6
Therefore, return the median sliding window as [1,-1,-1,3,5,6].
```

``` cpp
class Solution {
public:
    vector<double> medianSlidingWindow(vector<int>& nums, int k) {
        int len = nums.size();
        if (len < k) return vector<double>();
        vector<double> ret;
        unordered_map<int, int> m;
        priority_queue<int> small;
        priority_queue<int, vector<int>, greater<int>> large;
        for (int i = 0; i < k; i++) {
            small.push(nums[i]);
        }
        for (int i = 0; i < k / 2; i++) {
            large.push(small.top());
            small.pop();
        }
        for (int i = k; i <= len; i++) {
            if (k & 1) ret.push_back(small.top());
            // 这里要用double，否则可能会溢出
            else ret.push_back(((double)small.top() + (double)large.top()) / 2.0);
            
            if (i == len) continue;
            
            int blance = 0;
            int ans = nums[i];
            if (ans > small.top()) {
                large.push(ans);
                blance ++;
            }
            else {
                small.push(ans);
                blance --;
            }
            
            ans = nums[i - k];
            if (ans > small.top()) {
                if (large.top() == ans) large.pop();
                else m[ans]++;
                blance --;
            }
            else {
                if (small.top() == ans) small.pop();
                else m[ans]++;
                blance ++;
            }
            
            if (blance > 0) {
                small.push(large.top());
                large.pop();
            }
            if (blance < 0) {
                large.push(small.top());
                small.pop();
            }
            while (!small.empty() && m[small.top()]) {
                m[small.top()] --;
                small.pop();
            }
            while (!large.empty() && m[large.top()]) {
                m[large.top()] --;
                large.pop();
            }
        }
        return ret;
    }
};
```

## Design Twitter
> [Leetcode 355](https://leetcode.com/problems/design-twitter/description/)

Design a simplified version of Twitter where users can post tweets, follow/unfollow another user and is able to see the 10 most recent tweets in the user's news feed. Your design should support the following methods:

postTweet(userId, tweetId): Compose a new tweet.
getNewsFeed(userId): Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
follow(followerId, followeeId): Follower follows a followee.
unfollow(followerId, followeeId): Follower unfollows a followee.
```
Example:

Twitter twitter = new Twitter();

// User 1 posts a new tweet (id = 5).
twitter.postTweet(1, 5);

// User 1's news feed should return a list with 1 tweet id -> [5].
twitter.getNewsFeed(1);

// User 1 follows user 2.
twitter.follow(1, 2);

// User 2 posts a new tweet (id = 6).
twitter.postTweet(2, 6);

// User 1's news feed should return a list with 2 tweet ids -> [6, 5].
// Tweet id 6 should precede tweet id 5 because it is posted after tweet id 5.
twitter.getNewsFeed(1);

// User 1 unfollows user 2.
twitter.unfollow(1, 2);

// User 1's news feed should return a list with 1 tweet id -> [5],
// since user 1 is no longer following user 2.
twitter.getNewsFeed(1);
```

``` cpp
struct tweet {
    int id;
    int time;
    tweet(int id, int time) : id(id), time(time) {};
};
class Twitter {
public:
    unordered_map<int, unordered_set<int>> fo;
    unordered_map<int, vector<tweet>> po;
    int current;
    /** Initialize your data structure here. */
    Twitter() {
        current = 0;
    }
    
    /** Compose a new tweet. */
    void postTweet(int userId, int tweetId) {
        current ++;
        po[userId].push_back(tweet(tweetId, current));
    }
    
    void heapfy(vector<tweet>& nums, int index, int max) {
        int left = index * 2 + 1;
        int right = left + 1;
        int smallest = index;
        if (left < max && nums[left].time > nums[smallest].time) smallest = left;
        if (right < max && nums[right].time > nums[smallest].time) smallest = right;
        if (smallest != index) {
            swap(nums[smallest], nums[index]);
            heapfy(nums, smallest, max);
        }
    }
    
    /** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
    vector<int> getNewsFeed(int userId) {
        // 此处在实际使用中应该使用多路归并的思路，但是这里为了简便就将所有数据放到堆中然后选出top10
        vector<tweet> ans;
        for (auto i : fo[userId]) {
            for (auto j : po[i]) {
                ans.push_back(j);
            }
        }
        for (auto i : po[userId]) {
            ans.push_back(i);
        }
        int len = ans.size();
        for (int i = len / 2 - 1; i >= 0; i--) heapfy(ans, i, len);
        vector<int> ret;
        for (int i = len - 1; i >= 0 && ret.size() < 10; i--) {
            ret.push_back(ans[0].id);
            swap(ans[i], ans[0]);
            // 此处注意堆排序调整只需要调整一个路径就可以
            heapfy(ans, 0, i);
        }
        return ret;
    }
    
    /** Follower follows a followee. If the operation is invalid, it should be a no-op. */
    void follow(int followerId, int followeeId) {
        if (followerId == followeeId) return;
        fo[followerId].insert(followeeId);
    }
    
    /** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
    void unfollow(int followerId, int followeeId) {
        fo[followerId].erase(followeeId);
    }
};

/**
 * Your Twitter object will be instantiated and called as such:
 * Twitter obj = new Twitter();
 * obj.postTweet(userId,tweetId);
 * vector<int> param_2 = obj.getNewsFeed(userId);
 * obj.follow(followerId,followeeId);
 * obj.unfollow(followerId,followeeId);
 */
```

实际中应该使用的是对已经排序了的数组使用多路归并，思路是堆中存放的是路角标和列角标的pair：

``` cpp
#include <vector>
#include <utility>
#include <iostream>
using namespace std;

void heapfy(vector<pair<int, int>>& ans, vector<vector<int>>& nums, int index, int max) {
    int left = index * 2 + 1;
    int right = left + 1;
    int smallest = index;
    if (left < max && nums[ans[left].first][ans[left].second] < nums[ans[smallest].first][ans[smallest].second]) smallest = left;
    if (right < max && nums[ans[right].first][ans[right].second] < nums[ans[smallest].first][ans[smallest].second]) smallest = right;
    if (index != smallest) {
        swap(ans[index], ans[smallest]);
        heapfy(ans, nums, smallest, max);
    }
}

vector<int> helper(vector<vector<int>>& nums) {
    int n = nums.size();
    vector<int> ret;
    vector<pair<int, int>> ans;
    int N = 0;
    for (int i = 0; i < n; i++) {
        N += nums[i].size();
        ans.push_back(make_pair(i, 0));
    }
    for (int i = n /2; i >= 0; i--) {
        heapfy(ans, nums, i, n);
    }
    while (ret.size() < N) {
        auto temp = ans[0];
        ret.push_back(nums[temp.first][temp.second]);
        cout << temp.first << ":" << temp.second << ": "<< ret.back() << endl;
        ans.erase(ans.begin());
        if (temp.second + 1 < nums[temp.first].size()) {
            ans.push_back(make_pair(temp.first, temp.second + 1));
        }
        int len = ans.size();
        for (int i = len / 2; i >= 0; i--) heapfy(ans, nums, i, len);
    }
    return ret;

}

int main() {
    int n;
    cin >> n;
    vector<vector<int>> nums;
    for (int i = 0; i < n; i++) {
        int t;
        cin >> t;
        vector<int> ans(t, 0);
        for (int j = 0; j < t; j++) {
            cin >> ans[j];
        }
        nums.push_back(ans);
    }
    auto ret = helper(nums);
    for (auto i : ret) {
        cout << i << " ";
    }
    return 0;
}

```


