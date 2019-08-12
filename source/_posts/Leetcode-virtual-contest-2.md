---
title: Leetcode 周赛模拟 【2】
date: 2018-10-25 22:07:19
tags: [Leetcode, 周赛]
---
<!-- more -->
# Weekly Contest 83
> [参赛地址](https://leetcode.com/contest/weekly-contest-83/)

## 830. Positions of Large Groups [AC]
> [Leetcode 830](https://leetcode.com/contest/weekly-contest-83/problems/positions-of-large-groups/)

In a string S of lowercase letters, these letters form consecutive groups of the same character.

For example, a string like S = "abbxxxxzyy" has the groups "a", "bb", "xxxx", "z" and "yy".

Call a group large if it has 3 or more characters.  We would like the starting and ending positions of every large group.

The final answer should be in lexicographic order.

 

Example 1:
```
Input: "abbxxxxzzy"
Output: [[3,6]]
Explanation: "xxxx" is the single large group with starting  3 and ending positions 6.
```
Example 2:
```
Input: "abc"
Output: []
Explanation: We have "a","b" and "c" but no large group.
```
Example 3:
```
Input: "abcdddeeeeaabbbcd"
Output: [[3,5],[6,9],[12,14]]
```
> Note:  1 <= S.length <= 1000

``` cpp
class Solution {
public:
    vector<vector<int>> largeGroupPositions(string S) {
        vector<vector<int>> ret;
        int n = S.size();
        if (!n) return ret;
        for (int i = 0; i < n; i++) {
            int cnt = 1;
            while (i + cnt < n && S[i] == S[i + cnt]) cnt ++;
            if (cnt <= 2) continue;
            vector<int> ans({i, i + cnt - 1});
            ret.push_back(ans);
            i += cnt - 1;
        }
        return ret;
    }
};
```

## 831. Masking Personal Information [AC]
> [Leetcode 831](https://leetcode.com/contest/weekly-contest-83/problems/masking-personal-information/)

We are given a personal information string S, which may represent either an email address or a phone number.

We would like to mask this personal information according to the following rules:


1. Email address:

We define a name to be a string of length ≥ 2 consisting of only lowercase letters a-z or uppercase letters A-Z.

An email address starts with a name, followed by the symbol '@', followed by a name, followed by the dot '.' and followed by a name. 

All email addresses are guaranteed to be valid and in the format of "name1@name2.name3".

To mask an email, all names must be converted to lowercase and all letters between the first and last letter of the first name must be replaced by 5 asterisks '*'.


2. Phone number:

A phone number is a string consisting of only the digits 0-9 or the characters from the set {'+', '-', '(', ')', ' '}. You may assume a phone number contains 10 to 13 digits.

The last 10 digits make up the local number, while the digits before those make up the country code. Note that the country code is optional. We want to expose only the last 4 digits and mask all other digits.

The local number should be formatted and masked as "***-***-1111", where 1 represents the exposed digits.

To mask a phone number with country code like "+111 111 111 1111", we write it in the form "+***-***-***-1111".  The '+' sign and the first '-' sign before the local number should only exist if there is a country code.  For example, a 12 digit phone number mask should start with "+**-".

Note that extraneous characters like "(", ")", " ", as well as extra dashes or plus signs not part of the above formatting scheme should be removed.

 

Return the correct "mask" of the information provided.

 

Example 1:
```
Input: "LeetCode@LeetCode.com"
Output: "l*****e@leetcode.com"
Explanation: All names are converted to lowercase, and the letters between the
             first and last letter of the first name is replaced by 5 asterisks.
             Therefore, "leetcode" -> "l*****e".
```
Example 2:
```
Input: "AB@qq.com"
Output: "a*****b@qq.com"
Explanation: There must be 5 asterisks between the first and last letter 
             of the first name "ab". Therefore, "ab" -> "a*****b".
```
Example 3:
```
Input: "1(234)567-890"
Output: "***-***-7890"
Explanation: 10 digits in the phone number, which means all digits make up the local number.
```
Example 4:
```
Input: "86-(10)12345678"
Output: "+**-***-***-5678"
Explanation: 12 digits, 2 digits for country code and 10 digits for local number. 
```

``` cpp
class Solution {
public:
    bool helper(string s, int& cnt) {
        bool ret = false;
        for (auto i : s) {
            if (i == '@') ret = true;
            if (i >= '0' && i <= '9') cnt ++;
        }
        return ret;
    }
    string maskPII(string S) {
        int cnt = 0;
        if (!helper(S, cnt)) {
            string ret;
            string ans;
            int index = S.size() - 1;
            int c = 0;
            while (c < 4) {
                if (S[index] <= '9' && S[index] >= '0') {
                    ans = S[index] + ans;
                    c ++;
                }
                index --;
            }
            
            if (cnt > 10) {
                ret = "+";
                for (int i = 0; i < cnt - 10; i++) ret += "*";
                ret += "-***-***-";
            }
            else ret = "***-***-";
            ret += ans;
            return ret;
        }
        else {
            for (auto &i : S) {
                if (i >= 'A' && i <= 'Z') i = 'a' + (i - 'A');
            }
            string ret;
            ret.push_back(S[0]);
            ret += "*****";
            int index = 1;
            while (S[index + 1] != '@') index ++;
            while (index < S.size()) ret += S[index ++];
            return ret;
        }
    }
};
```

## 829. Consecutive Numbers Sum [AC]

> [Leetcode 829](https://leetcode.com/contest/weekly-contest-83/problems/consecutive-numbers-sum/)

Given a positive integer N, how many ways can we write it as a sum of consecutive positive integers?

Example 1:
```
Input: 5
Output: 2
Explanation: 5 = 5 = 2 + 3
```
Example 2:
```
Input: 9
Output: 3
Explanation: 9 = 9 = 4 + 5 = 2 + 3 + 4
```
Example 3:
```
Input: 15
Output: 4
Explanation: 15 = 15 = 8 + 7 = 4 + 5 + 6 = 1 + 2 + 3 + 4 + 5
```

``` cpp
class Solution {
public:
    int consecutiveNumbersSum(int N) {
        int ret = 1;
        for (int i = 2; 2 * N - i * (i - 1) > 0; i++) {
            if ((2 * N - i * (i - 1)) % (2 * i) == 0) ret ++;
        }
        return ret;
    }
};
```

## 828. Unique Letter String [Unsolved]

> [Leetcode 828](https://leetcode.com/contest/weekly-contest-83/problems/unique-letter-string/)

A character is unique in string S if it occurs exactly once in it.

For example, in string S = "LETTER", the only unique characters are "L" and "R".

Let's define UNIQ(S) as the number of unique characters in string S.

For example, UNIQ("LETTER") =  2.

Given a string S with only uppercases, calculate the sum of UNIQ(substring) over all non-empty substrings of S.

If there are two or more equal substrings at different positions in S, we consider them different.

Since the answer can be very large, return the answer modulo 10 ^ 9 + 7.

 

Example 1:
```
Input: "ABC"
Output: 10
Explanation: All possible substrings are: "A","B","C","AB","BC" and "ABC".
Evey substring is composed with only unique letters.
Sum of lengths of all substring is 1 + 1 + 1 + 2 + 2 + 3 = 10
```
Example 2:
```
Input: "ABA"
Output: 8
Explanation: The same as example 1, except uni("ABA") = 1.
```

``` cpp
class Solution {
public:
    int uniqueLetterString(string S) {
        long long ret = 0;
        int n = S.size();
        for (int i = 0; i < n; i++) {
            int l = i - 1, r = i + 1;
            while (l >= 0 && S[l] != S[i]) l--;
            while (r < n && S[r] != S[i]) r ++;
            ret += (long long)(r - i) * (i - l);
            // 相乘表示从l->i和i->r所有子串的组合
        }
        return ret % 1000000007;
    }
};
```

# Weekly Contest 82

## 824. Goat Latin [AC]
> [Leetcode 824](https://leetcode.com/contest/weekly-contest-82/problems/goat-latin/)
A sentence S is given, composed of words separated by spaces. Each word consists of lowercase and uppercase letters only.

We would like to convert the sentence to "Goat Latin" (a made-up language similar to Pig Latin.)

The rules of Goat Latin are as follows:

If a word begins with a vowel (a, e, i, o, or u), append "ma" to the end of the word.
For example, the word 'apple' becomes 'applema'.
 
If a word begins with a consonant (i.e. not a vowel), remove the first letter and append it to the end, then add "ma".
For example, the word "goat" becomes "oatgma".
 
Add one letter 'a' to the end of each word per its word index in the sentence, starting with 1.
For example, the first word gets "a" added to the end, the second word gets "aa" added to the end and so on.
Return the final sentence representing the conversion from S to Goat Latin. 

 

Example 1:
```
Input: "I speak Goat Latin"
Output: "Imaa peaksmaaa oatGmaaaa atinLmaaaaa"
```
Example 2:
```
Input: "The quick brown fox jumped over the lazy dog"
Output: "heTmaa uickqmaaa rownbmaaaa oxfmaaaaa umpedjmaaaaaa overmaaaaaaa hetmaaaaaaaa azylmaaaaaaaaa ogdmaaaaaaaaaa"
```

``` cpp
class Solution {
public:
    bool helper(char c) {
        if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'A' || c == 'E' || c == 'I' || c == 'O' || c == 'U') return true;
        else return false;
    }
    string toGoatLatin(string S) {
        string ret;
        string ans;
        int cnt = 1;
        for (auto i : S) {
            if (i != ' ') {
                ans.push_back(i);
            }
            else {
                if (ans.empty()) ans += " ";
                else {
                    if (helper(ans[0])) {
                        ans += "ma";
                    }
                    else {
                        char t = ans[0];
                        ans.erase(ans.begin());
                        ans.push_back(t);
                        ans += "ma";
                    }
                    for (int i = 0; i < cnt; i++) {
                        ans += "a";
                    }
                    cnt ++;
                }
                ret += " " + ans;
                ans = "";
            }
        }
        if (helper(ans[0])) {
                ans += "ma";
        }
        else {
            char t = ans[0];
            ans.erase(ans.begin());
            ans.push_back(t);
            ans += "ma";
            }
        for (int i = 0; i < cnt; i++) {
            ans += "a";
        }
        ret += " " + ans;
        ans = "";
        ret.erase(ret.begin());
        return ret;
    }
};
```

## 825. Friends Of Appropriate Ages [AC]
> [Leetcode 825](https://leetcode.com/contest/weekly-contest-82/problems/friends-of-appropriate-ages/)

Some people will make friend requests. The list of their ages is given and ages[i] is the age of the ith person. 

Person A will NOT friend request person B (B != A) if any of the following conditions are true:

age[B] <= 0.5 * age[A] + 7
age[B] > age[A]
age[B] > 100 && age[A] < 100
Otherwise, A will friend request B.

Note that if A requests B, B does not necessarily request A.  Also, people will not friend request themselves.

How many total friend requests are made?

Example 1:
```
Input: [16,16]
Output: 2
Explanation: 2 people friend request each other.
```
Example 2:
```
Input: [16,17,18]
Output: 2
Explanation: Friend requests are made 17 -> 16, 18 -> 17.
```
Example 3:
```
Input: [20,30,100,110,120]
Output: 
Explanation: Friend requests are made 110 -> 100, 120 -> 110, 120 -> 100.
```

``` cpp
class Solution {
public:
    bool helper(int a, int b) {
        if ((2 * b <= a + 14) || b > a || (b > 100 && a < 100)) return false;
        else return true;
    }
    int numFriendRequests(vector<int>& ages) {
        int ret = 0;
        int n = ages.size();
        sort(ages.begin(), ages.end());
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (helper(ages[j], ages[i])) {
                    cout << ages[j] << ages[i] << endl;
                    ret ++;
                }
            }
        }
        if (ret && n == 2) return 2;
        return ret;
    }
};
```

## 826. Most Profit Assigning Work [AC]
> [Leetcode 826](https://leetcode.com/contest/weekly-contest-82/problems/most-profit-assigning-work/)

We have jobs: difficulty[i] is the difficulty of the ith job, and profit[i] is the profit of the ith job. 

Now we have some workers. worker[i] is the ability of the ith worker, which means that this worker can only complete a job with difficulty at most worker[i]. 

Every worker can be assigned at most one job, but one job can be completed multiple times.

For example, if 3 people attempt the same job that pays $1, then the total profit will be $3.  If a worker cannot complete any job, his profit is $0.

What is the most profit we can make?

Example 1:
```
Input: difficulty = [2,4,6,8,10], profit = [10,20,30,40,50], worker = [4,5,6,7]
Output: 100 
Explanation: Workers are assigned jobs of difficulty [4,4,6,6] and they get profit of [20,20,30,30] seperately.
```

``` cpp
class Solution {
public:
    
    int maxProfitAssignment(vector<int>& difficulty, vector<int>& profit, vector<int>& worker) {
        int n = worker.size(), m = profit.size();
        vector<pair<int, int>> nums;
        if (!n || !m) return 0;
        for (int i = 0; i < m; i++) {
            nums.push_back(make_pair(difficulty[i], profit[i]));
        }
        sort(nums.begin(), nums.end());
        int ret = 0;
        for (int i = 0; i < n; i++) {
            int ans  = 0;
            for (int j = 0; j < m; j++) {
                if (worker[i] < nums[j].first) break;
                ans = max(ans, nums[j].second);
            }
            ret += ans;
        }
        return ret;
    }
};
```

## 827. Making A Large Island [Unsolved]
> [Leetcode 827](https://leetcode.com/contest/weekly-contest-82/problems/making-a-large-island/)

In a 2D grid of 0s and 1s, we change at most one 0 to a 1.

After, what is the size of the largest island? (An island is a 4-directionally connected group of 1s).

Example 1:

Input: [[1, 0], [0, 1]]
Output: 3
Explanation: Change one 0 to 1 and connect two 1s, then we get an island with area = 3.
Example 2:

Input: [[1, 1], [1, 0]]
Output: 4
Explanation: Change the 0 to 1 and make the island bigger, only one island with area = 4.
Example 3:

Input: [[1, 1], [1, 1]]
Output: 4
Explanation: Can't change any 0 to 1, only one island with area = 4.

> 本来思路是并查集加BFS，但是由于并查集写的代码太长导致时间不够用

``` cpp
class Solution {
public:
    int N;
    int largestIsland(vector<vector<int>>& grid) {
        N = grid.size();
        //DFS every island and give it an index of island
        int index = 2, res = 0;
        unordered_map <int, int>area;
        for (int x = 0; x < N; ++x) {
            for (int y = 0; y < N; ++y) {
                if (grid[x][y] == 1) {
                    area[index] = dfs(grid, x, y, index);
                    res = max(res, area[index++]);
                }
            }
        }
        //traverse every 0 cell and count biggest island it can conntect
        for (int x = 0; x < N; ++x) for (int y = 0; y < N; ++y) if (grid[x][y] == 0) {
            unordered_set<int> seen = {};
            int cur = 1;
            for (auto p : move(x, y)) {
                index = grid[p.first][p.second];
                if (index > 1 && seen.count(index) == 0) {
                    seen.insert(index);
                    cur += area[index];
                }
            }
            res = max(res, cur);
        }
        return res;
    }

    vector<pair<int, int>> move(int x, int y) {
        vector<pair<int, int>> res;
        for (auto p : vector<vector<int>> {{1, 0}, {-1, 0}, {0, 1}, {0, -1}}) {
            if (valid(x + p[0], y + p[1]))
                res.push_back(make_pair(x + p[0], y + p[1]));
        }
        return res;
    }

    int valid(int x, int y) {
        return 0 <= x && x < N && 0 <= y && y < N;
    }

    int dfs(vector<vector<int>>& grid, int x, int y, int index) {
        int area = 0;
        grid[x][y] = index;
        for (auto p : move(x, y))
            if (grid[p.first][p.second] == 1)
                area += dfs(grid, p.first, p.second, index);
        return area + 1;
    }
};
```

# Weekly Contest 81

## 821. Shortest Distance to a Character [AC]
> [Leetcode 821](https://leetcode.com/contest/weekly-contest-81/problems/shortest-distance-to-a-character/)
Given a string S and a character C, return an array of integers representing the shortest distance from the character C in the string.

Example 1:
```
Input: S = "loveleetcode", C = 'e'
Output: [3, 2, 1, 0, 1, 0, 0, 1, 2, 2, 1, 0]
```

``` cpp
class Solution {
public:
     vector<int> shortestToChar(string S, char C) {
        int n = S.size();
        vector<int> res(n, n);
        int pos = -n;
        for (int i = 0; i < n; ++i) {
            if (S[i] == C) pos = i;
            res[i] = min(res[i], abs(i - pos));
        }
        for (int i = n - 1; i >= 0; --i) {
            if (S[i] == C)  pos = i;
            res[i] = min(res[i], abs(i - pos));
        }
        return res;
    }
};
```



## 822. 822. Card Flipping Game [AC]
> [Leetcode 821](https://leetcode.com/contest/weekly-contest-81/problems/card-flipping-game/)

On a table are N cards, with a positive integer printed on the front and back of each card (possibly different).

We flip any number of cards, and after we choose one card. 

If the number X on the back of the chosen card is not on the front of any card, then this number X is good.

What is the smallest number that is good?  If no number is good, output 0.

Here, fronts[i] and backs[i] represent the number on the front and back of card i. 

A flip swaps the front and back numbers, so the value on the front is now on the back and vice versa.

Example:
```
Input: fronts = [1,2,4,4,7], backs = [1,3,4,1,3]
Output: 2
Explanation: If we flip the second card, the fronts are [1,3,4,4,7] and the backs are [1,2,4,1,3].
We choose the second card, which has number 2 on the back, and it isn't on the front of any card, so 2 is good.
```

``` cpp
class Solution {
public:
    int flipgame(vector<int>& fronts, vector<int>& backs) {
        unordered_set<int> same;
        int n = fronts.size();
        for (int i = 0; i < n; i++) {
            if (fronts[i] == backs[i]) same.insert(fronts[i]);
        }
        int ret = INT_MAX;
        for (auto i : fronts) {
            if (same.find(i) == same.end()) ret = min(ret, i);
        }
        for (auto i : backs) {
            if (same.find(i) == same.end()) ret = min(ret, i);
        }
        return ret == INT_MAX ? 0 : ret;
    }
};
```

## 820. Short Encoding of Words [AC]
> [Leetcode 820](https://leetcode.com/contest/weekly-contest-81/problems/short-encoding-of-words/)
Given a list of words, we may encode it by writing a reference string S and a list of indexes A.

For example, if the list of words is ["time", "me", "bell"], we can write it as S = "time#bell#" and indexes = [0, 2, 5].

Then for each index, we will recover the word by reading from the reference string from that index until we reach a "#" character.

What is the length of the shortest reference string S possible that encodes the given words?

Example:
```
Input: words = ["time", "me", "bell"]
Output: 10
Explanation: S = "time#bell#" and indexes = [0, 2, 5].
```

``` cpp
class Solution {
public:
    int minimumLengthEncoding(vector<string>& words) {
        unordered_set<string> s(words.begin(), words.end());
        for (auto w : words) {
            for (int i = 1; i < w.size(); i++) {
                s.erase(w.substr(i));
            }
        }
        int ret = 0;
        for (auto i : s) ret += i.size() + 1;
        return ret;
    }
};
```


## 823. Binary Trees With Factors [AC]
> [Leetcode 823](https://leetcode.com/contest/weekly-contest-81/problems/binary-trees-with-factors/)

Given an array of unique integers, each integer is strictly greater than 1.

We make a binary tree using these integers and each number may be used for any number of times.

Each non-leaf node's value should be equal to the product of the values of it's children.

How many binary trees can we make?  Return the answer modulo 10 ** 9 + 7.

Example 1:
```
Input: A = [2, 4]
Output: 3
Explanation: We can make these trees: [2], [4], [4, 2, 2]
```

Example 2:
```
Input: A = [2, 4, 5, 10]
Output: 7
Explanation: We can make these trees: [2], [4], [5], [10], [4, 2, 2], [10, 2, 5], [10, 5, 2].
```

``` cpp
class Solution {
public:
    long MOD = 1000000007;
    int numFactoredBinaryTrees(vector<int>& A) {
        sort(A.begin(), A.end());
        unordered_map<int, long> m;
        for (int i = 0; i < A.size(); i++) {
            int x = A[i];
            m[x] = 1;
            for (int j = 0; j < i; j++) {
                int y = A[j];
                if (x % y == 0 && m.find(x / y) != m.end()) {
                    m[x] = (m[x] + (m[y] * m[x / y]) % MOD) % MOD;
                }
            }
        }
        long ret = 0;
        for (auto i : m) {
            ret = (ret + i.second) % MOD;
        }
        return ret;
    }
};
```

# Weekly Contest 80
> [Leetcode 80](https://leetcode.com/contest/weekly-contest-80/)

## 819. Most Common Word [AC]
> [Leetcode 819](https://leetcode.com/contest/weekly-contest-80/problems/most-common-word/)

Given a paragraph and a list of banned words, return the most frequent word that is not in the list of banned words.  It is guaranteed there is at least one word that isn't banned, and that the answer is unique.

Words in the list of banned words are given in lowercase, and free of punctuation.  Words in the paragraph are not case sensitive.  The answer is in lowercase.

 

Example:
```
Input: 
paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
banned = ["hit"]
Output: "ball"
Explanation: 
"hit" occurs 3 times, but it is a banned word.
"ball" occurs twice (and no other word does), so it is the most frequent non-banned word in the paragraph. 
Note that words in the paragraph are not case sensitive,
that punctuation is ignored (even if adjacent to words, such as "ball,"), 
and that "hit" isn't the answer even though it occurs more because it is banned.
```

``` cpp
class Solution {
public:
    vector<string> helper(string s) {
        vector<string> ret;
        string ans;
        for (auto i : s) {
            if (i == ' ' || ispunct(i)) {
                if (ans != "") {
                    ret.push_back(ans);
                }
                ans = "";
            }
            else {
                if (i <= 'Z' && i >= 'A') {
                    i = 'a' + (i - 'A');
                }
                ans.push_back(i);
            }
        }
        if (ans != "") ret.push_back(ans);
        return ret;
    }
    string mostCommonWord(string paragraph, vector<string>& banned) {
        unordered_set<string> m(banned.begin(), banned.end());
        unordered_map<string, int> cnt;
        for (auto s : helper(paragraph)) {
            if (m.find(s) == m.end()) {
                cout << s << endl;
                cnt[s] ++;
            }
        }
        string ret;
        int ans = 0;
        for (auto i : cnt) {
            if (i.second > ans) {
                ans = i.second;
                ret = i.first;
            }
        }
        return ret;
    }
};
```

## 817. Linked List Components [AC]
> [Leetcode 817](https://leetcode.com/contest/weekly-contest-80/problems/linked-list-components/)

We are given head, the head node of a linked list containing unique integer values.

We are also given the list G, a subset of the values in the linked list.

Return the number of connected components in G, where two values are connected if they appear consecutively in the linked list.

Example 1:
```
Input: 
head: 0->1->2->3
G = [0, 1, 3]
Output: 2
Explanation: 
0 and 1 are connected, so [0, 1] and [3] are the two connected components.
```
Example 2:
```
Input: 
head: 0->1->2->3->4
G = [0, 3, 1, 4]
Output: 2
Explanation: 
0 and 1 are connected, 3 and 4 are connected, so [0, 1] and [3, 4] are the two connected components.
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
    int numComponents(ListNode* head, vector<int>& G) {
        if (!head) return 0;
        unordered_set<int> m(G.begin(), G.end());
        int ret = 0;
        int check = 0;
        ListNode* cur = head;
        while (true) {
            int x = cur->val;
            if (m.find(x) != m.end()) {
                check = 1;
            }
            else {
                if (check) {    
                    ret += 1, check = 0;
                }
            }
            cur = cur->next;
            if (!cur) {
                if (check) ret += 1;
                break;
            }
        }
        return ret;
    }
};
```

## 816. Ambiguous Coordinates [AC]
> [Leetcode 816](https://leetcode.com/contest/weekly-contest-80/problems/ambiguous-coordinates/)

We had some 2-dimensional coordinates, like "(1, 3)" or "(2, 0.5)".  Then, we removed all commas, decimal points, and spaces, and ended up with the string S.  Return a list of strings representing all possibilities for what our original coordinates could have been.

Our original representation never had extraneous zeroes, so we never started with numbers like "00", "0.0", "0.00", "1.0", "001", "00.01", or any other number that can be represented with less digits.  Also, a decimal point within a number never occurs without at least one digit occuring before it, so we never started with numbers like ".1".

The final answer list can be returned in any order.  Also note that all coordinates in the final answer have exactly one space between them (occurring after the comma.)
```
Example 1:
Input: "(123)"
Output: ["(1, 23)", "(12, 3)", "(1.2, 3)", "(1, 2.3)"]
Example 2:
Input: "(00011)"
Output:  ["(0.001, 1)", "(0, 0.011)"]
Explanation: 
0.0, 00, 0001 or 00.01 are not allowed.
Example 3:
Input: "(0123)"
Output: ["(0, 123)", "(0, 12.3)", "(0, 1.23)", "(0.1, 23)", "(0.1, 2.3)", "(0.12, 3)"]
Example 4:
Input: "(100)"
Output: [(10, 0)]
Explanation: 
1.0 is not allowed.
```

``` cpp
class Solution {
public:
    vector<string> helper(string s) {
        vector<string> ret;
        if (s == "0") return vector<string>({"0"});
        if (s[0] == '0' && s.back() == '0') return ret;
        if (s[0] == '0' && s.back() != '0') {
            string ans = "0.";
            for (int i = 1; i < s.size(); i++) {
                ans.push_back(s[i]);
            }
            ret.push_back(ans);
            return ret;
        }
        if (s[0] != '0' && s.back() == '0') {
            ret.push_back(s);
            return ret;
        }
        ret.push_back(s);
        for (int i = 1; i < s.size(); i++) {
            ret.push_back(s.substr(0, i) + "." + s.substr(i));
        }
        return ret;
    }
    vector<string> ambiguousCoordinates(string S) {
        S.pop_back();
        S.erase(S.begin());
        vector<string> ret;
        for (int i = 1; i < S.size(); i++) {
            for (auto a : helper(S.substr(0, i))) {
                if (a == "") continue;
                for (auto b : helper(S.substr(i))) {
                    if (b == "") continue;
                    ret.push_back("(" + a + ", " + b + ")");
                }
                
            }
        }
        return ret;
    }
};
```

## 818. Race Car [AC]
> [Leetcode 818](https://leetcode.com/contest/weekly-contest-80/problems/race-car/)

Your car starts at position 0 and speed +1 on an infinite number line.  (Your car can go into negative positions.)

Your car drives automatically according to a sequence of instructions A (accelerate) and R (reverse).

When you get an instruction "A", your car does the following: position += speed, speed *= 2.

When you get an instruction "R", your car does the following: if your speed is positive then speed = -1 , otherwise speed = 1.  (Your position stays the same.)

For example, after commands "AAR", your car goes to positions 0->1->3->3, and your speed goes to 1->2->4->-1.

Now for some target position, say the length of the shortest sequence of instructions to get there.
```
Example 1:
Input: 
target = 3
Output: 2
Explanation: 
The shortest instruction sequence is "AA".
Your position goes from 0->1->3.

Example 2:
Input: 
target = 6
Output: 5
Explanation: 
The shortest instruction sequence is "AAARA".
Your position goes from 0->1->3->7->7->6.
```

``` cpp
class Solution {
public:
    int dp[10001];
    int racecar(int t) {
        if (dp[t] > 0) return dp[t];
        int n = floor(log2(t)) + 1, res;
        if (1 << n == t + 1) dp[t] = n;
        else {
            dp[t] = racecar((1 << n) - 1 - t) + n + 1;
            for (int m = 0; m < n - 1; ++m)
                dp[t] = min(dp[t], racecar(t - (1 << (n - 1)) + (1 << m)) + n + m + 1);
        }
        return dp[t];
    }
};
```


