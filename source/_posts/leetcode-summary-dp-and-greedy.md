---
title: Leetcode DP及贪心算法总结
date: 2018-09-05 00:39:56
tags: [算法, 总结, Leetcode, OJ]
---
> 这里总结一些DP类型的题目，因为贪心一定程度可以认为是一维的DP，所以也做总结
<!-- more -->
# 贪心

## Jump Game
> [Leetcode 55](https://leetcode.com/problems/jump-game/description/)

Given an array of non-negative integers, you are initially positioned at the first index of the array.

Each element in the array represents your maximum jump length at that position.

Determine if you are able to reach the last index.

Example 1:
```
Input: [2,3,1,1,4]
Output: true
Explanation: Jump 1 step from index 0 to 1, then 3 steps to the last index.
```

Example 2:
```
Input: [3,2,1,0,4]
Output: false
Explanation: You will always arrive at index 3 no matter what. Its maximum
             jump length is 0, which makes it impossible to reach the last index.
```

``` cpp
class Solution {
public:
    bool canJump(vector<int>& nums) {
        if (nums.empty()) return true;
        int n = nums.size(), reach = 0, index = 0;
        for (index = 0; index < n && index <= reach; index ++) {
            reach = max(reach, index + nums[index]);
        }
        return reach >= n - 1;
    }
};
```

## Majority Element II
> [Leetcode 229](https://leetcode.com/problems/majority-element-ii/description/)

Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.

Note: The algorithm should run in linear time and in O(1) space.

Example 1:
```
Input: [3,2,3]
Output: [3]
```

Example 2:
```
Input: [1,1,1,3,3,2,2,2]
Output: [1,2]
```

``` cpp
class Solution {
public:
    vector<int> majorityElement(vector<int>& nums) {
        int n = nums.size();
        int c1 = 0, c2 = 0, i1 = 0, i2 = 1;
        for (auto i : nums) {
            if (i == i1) c1 ++;
            else if (i == i2) c2 ++;
            else if (!c1) i1 = i, c1 = 1;
            else if (!c2) i2 = i, c2 = 1;
            else c1--, c2--;
        }
        c1 = 0, c2 = 0;
        for (auto i : nums) {
            if (i == i1) c1++;
            if (i == i2) c2++;
        }
        vector<int> ret;
        if (c1 > n / 3) ret.push_back(i1);
        if (c2 > n / 3) ret.push_back(i2);
        return ret;
    }
};
```

## Queue Reconstruction by Height
> [Leetcode 406](https://leetcode.com/problems/queue-reconstruction-by-height/description/)

Suppose you have a random list of people standing in a queue. Each person is described by a pair of integers (h, k), where h is the height of the person and k is the number of people in front of this person who have a height greater than or equal to h. Write an algorithm to reconstruct the queue.

Note:
The number of people is less than 1,100.


Example
```
Input:
[[7,0], [4,4], [7,1], [5,0], [6,1], [5,2]]

Output:
[[5,0], [7,0], [5,2], [6,1], [4,4], [7,1]]
```

``` cpp
class Solution {
public:
    bool static cmp(pair<int, int>& a, pair<int, int>& b) {
        return a.first == b.first ? a.second < b.second : a.first > b.first;
    }
    vector<pair<int, int>> reconstructQueue(vector<pair<int, int>>& people) {
        sort(people.begin(), people.end(), cmp);
        vector<pair<int, int>> ret;
        for (auto i : people) {
            ret.insert(ret.begin() + i.second, i);
        }
        return ret;
    }
};
```

## Minimum Number of Arrows to Burst Balloons
> [Leetcode 452](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/description/)

There are a number of spherical balloons spread in two-dimensional space. For each balloon, provided input is the start and end coordinates of the horizontal diameter. Since it's horizontal, y-coordinates don't matter and hence the x-coordinates of start and end of the diameter suffice. Start is always smaller than end. There will be at most 104 balloons.

An arrow can be shot up exactly vertically from different points along the x-axis. A balloon with xstart and xend bursts by an arrow shot at x if xstart ≤ x ≤ xend. There is no limit to the number of arrows that can be shot. An arrow once shot keeps travelling up infinitely. The problem is to find the minimum number of arrows that must be shot to burst all balloons.

Example:
```
Input:
[[10,16], [2,8], [1,6], [7,12]]

Output:
2

Explanation:
One way is to shoot one arrow for example at x = 6 (bursting the balloons [2,8] and [1,6]) and another arrow at x = 11 (bursting the other two balloons).
```

``` cpp
class Solution {
public:
    // 扫描线算法
    bool static cmp(pair<int, int>& a, pair<int, int>& b) {
        return a.first == b.first ? a.second < b.second : a.first < b.first;
    }
    int findMinArrowShots(vector<pair<int, int>>& points) {
        if (points.empty()) return 0;
        int ret = 1, n = points.size();
        sort(points.begin(), points.end(), cmp);
        int limit = points[0].second;
        for (auto i : points) {
            if (i.first > limit) {
                ret ++;
                limit = i.second;
            }
            else {
                limit = min(limit, i.second);
            }
        }
        return ret;
    }
};
```
## Remove K Digits
> [Leetcode 402](https://leetcode.com/problems/remove-k-digits/description/)

Given a non-negative integer num represented as a string, remove k digits from the number so that the new number is the smallest possible.

Note:
- The length of num is less than 10002 and will be ≥ k.
- The given num does not contain any leading zero.

Example 1:
```
Input: num = "1432219", k = 3
Output: "1219"
Explanation: Remove the three digits 4, 3, and 2 to form the new number 1219 which is the smallest.
```
Example 2:
```
Input: num = "10200", k = 1
Output: "200"
Explanation: Remove the leading 1 and the number is 200. Note that the output must not contain leading zeroes.
```
Example 3:
```
Input: num = "10", k = 2
Output: "0"
Explanation: Remove all the digits from the number and it is left with nothing which is 0.
```
``` cpp
class Solution {
public:
// 这里用了栈，其实也可以用两个指针实现
    string removeKdigits(string num, int k) {
        int n = num.size();
        if (n <= k) return "0";
        stack<char> s;
        for (auto i : num) {
            while (!s.empty() && k && s.top() > i) {
                k--;
                s.pop();
            }
            s.push(i);
        }
        for (int i = 0; i < k; i++) {
            s.pop();
        }
        string ret;
        while (!s.empty()) {
            ret = s.top() + ret;
            s.pop();
        }
        while (!ret.empty() && ret[0] == '0') ret.erase(ret.begin());
        return ret.empty() ? "0" : ret;
    }
};
```
# 动态规划

## 矩阵链相乘
> [算法导论](https://blog.csdn.net/tangbo713/article/details/40662435)

``` cpp
#include <iostream>
#include <vector>
#include <utility>
#include <climits>
#include <math.h>
using namespace std;

int helper(vector<pair<int, int>>& nums) {
    int n = nums.size();
    vector<int> p(n + 1, 0);
    for (int i = 0; i < n; i++) {
        p[i] = nums[i].first;
    }
    p[n] = nums.back().second;

    vector<vector<int>> dp(n, vector<int>(n, 0));
    for (int i = 2; i <= n; i++) {
        for (int l = 0; l <= n - i; l ++) {
            int r = l + i - 1;
            dp[l][r] = INT_MAX;
            for (int k = l; k < r; k++) {
                int ans = dp[l][k] + dp[k + 1][r] + p[l] * p[k + 1] * p[r + 1];
                dp[l][r] = min(dp[l][r], ans);
            }
        }
    }
    return dp[0][n - 1];
}

int main() {
    int n;
    cin >> n;
    vector<pair<int, int>> nums;
    for (int i = 0; i < n; i++) {
        int x, y;
        cin >> x >> y;
        nums.push_back({x, y});
    }
    cout << helper(nums) << endl;
    return 0;
}

```

## Gas Station
> [Leetcode 134](https://leetcode.com/problems/gas-station/description/)

There are N gas stations along a circular route, where the amount of gas at station i is gas[i].

You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.

Return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1.

Note:
- If there exists a solution, it is guaranteed to be unique.
- Both input arrays are non-empty and have the same length.
- Each element in the input arrays is a non-negative integer.

Example 1:
```
Input: 
gas  = [1,2,3,4,5]
cost = [3,4,5,1,2]

Output: 3

Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.
```

Example 2:
```
Input: 
gas  = [2,3,4]
cost = [3,4,3]

Output: -1

Explanation:
You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 0. Your tank = 4 - 3 + 2 = 3
Travel to station 1. Your tank = 3 - 3 + 3 = 3
You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
Therefore, you can't travel around the circuit once no matter where you start.
```

``` cpp
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int ans = 0, c = 0, g = 0;
        int ret = 0;
        for (int i = 0; i < gas.size(); i++) {
            ans += gas[i] - cost[i];
            g += gas[i];
            c += cost[i];
            if (ans < 0) {
                ans = 0;
                ret = i + 1;
            }
        }
        return g >= c ? ret : -1;
    }
};
```

## Counting Bits
> [Leetcode 338](https://leetcode.com/problems/counting-bits/description/)

Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.

Example 1:
```
Input: 2
Output: [0,1,1]
```
Example 2:
```
Input: 5
Output: [0,1,1,2,1,2]
```

``` cpp
class Solution {
public:
    vector<int> countBits(int num) {
        vector<int> dp(num + 1, 0);
        for (int i = 1; i <= num; i++) {
            dp[i] = dp[i & (i - 1)] + 1;
        }
        return dp;
    }
};
```

## House Robber II
> [Leetcode 213](https://leetcode.com/problems/house-robber-ii/description/)

You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed. All houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

Example 1:
```
Input: [2,3,2]
Output: 3
Explanation: You cannot rob house 1 (money = 2) and then rob house 3 (money = 2),
             because they are adjacent houses.
```
Example 2:
```
Input: [1,2,3,1]
Output: 4
Explanation: Rob house 1 (money = 1) and then rob house 3 (money = 3).
             Total amount you can rob = 1 + 3 = 4.
```

``` cpp
class Solution {
public:
    int helper(vector<int>& nums, int left, int right) {
        int pre = 0, ret = 0;
        for (int i = left; i <= right; i++) {
            int temp = max(pre + nums[i], ret);
            pre = ret, ret = temp;
        }
        return ret;
    }
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n < 2) return n == 0 ? 0 : nums[0];
        return max(helper(nums, 0, n - 2), helper(nums, 1, n - 1));
    }
};
```

## Largest Divisible Subset
> [Leetcode 368](https://leetcode.com/problems/largest-divisible-subset/description/)

Given a set of distinct positive integers, find the largest subset such that every pair (Si, Sj) of elements in this subset satisfies:

Si % Sj = 0 or Sj % Si = 0.

If there are multiple solutions, return any subset is fine.

Example 1:
```
Input: [1,2,3]
Output: [1,2] (of course, [1,3] will also be ok)
```
Example 2:
```
Input: [1,2,4,8]
Output: [1,2,4,8]
```

``` cpp
class Solution {
public:
    vector<int> largestDivisibleSubset(vector<int>& nums) {
        int n = nums.size();
        vector<int> ret;
        sort(nums.begin(), nums.end(), greater<int>());
        vector<int> dp(n, 0);
        vector<int> parents(n, -1);
        int index = -1, m = 0;
        for (int i = 0; i < n; i++) {
            dp[i] = 1;
            parents[i] = i;
            for (int j = 0; j < i; j ++) {
                if (nums[j] % nums[i] == 0 && dp[i] < dp[j] + 1) {
                    dp[i] = dp[j] + 1;
                    parents[i] = j;
                }
            }
            if (m < dp[i]) {
                m = dp[i];
                index = i;
            }
        }
        for (int i = 0; i < m; i++) {
            ret.push_back(nums[index]);
            index = parents[index];
        }
        reverse(ret.begin(), ret.end());
        return ret;
    }
};
```
## Best Time to Buy and Sell Stock with Cooldown
> [Leetcode 309](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/description/)

Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times) with the following restrictions:

You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
After you sell your stock, you cannot buy stock on next day. (ie, cooldown 1 day)

Example:
```
Input: [1,2,3,0,2]
Output: 3 
Explanation: transactions = [buy, sell, cooldown, buy, sell]
```

``` cpp
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int buy(INT_MIN), sell(0), pre_sell(0);
        for (auto i : prices) {
            int pre_buy = buy;
            buy = max(buy, pre_sell - i);
            pre_sell = sell;
            sell = max(sell, pre_buy + i);
        }
        return sell;
    }
};
```

## 最长公共子序列

``` cpp
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;

string helper(string a, string b) {
    int m = a.size(), n = b.size();
    vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
    vector<vector<int>> parents(m + 1, vector<int>(n + 1, 0));
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (a[i - 1] == b[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1] + 1;
                parents[i][j] = 1;
            }
            else {
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                if (dp[i - 1][j] > dp[i][j - 1]) parents[i][j] = 2;
                else parents[i][j] = 3;
            }
        }
    }
    string ret;
    while (m && n) {
        if (parents[m][n] == 1) {
            ret = a[m - 1] + ret;
            m--, n--;
        }
        else if (parents[m][n] == 2) {
            m--;
        }
        else n--;
    }
    return ret;

}

int main() {
    string a, b;
    cin >> a >> b;
    cout << helper(a, b) << endl;
    return 0;
}

```



## Word Break
> [Leetcode 139](https://leetcode.com/problems/word-break/description/)

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words.

Note:

- The same word in the dictionary may be reused multiple times in the segmentation.
- You may assume the dictionary does not contain duplicate words.

Example 1:
```
Input: s = "leetcode", wordDict = ["leet", "code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
```
Example 2:
```
Input: s = "applepenapple", wordDict = ["apple", "pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
             Note that you are allowed to reuse a dictionary word.
```
Example 3:
```
Input: s = "catsandog", wordDict = ["cats", "dog", "sand", "and", "cat"]
Output: false
```

``` cpp
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        if (wordDict.empty()) return false;
        int n = s.size();
        vector<bool> dp(n + 1, false);
        dp[0] = true;
        for (int i = 0; i <= n; i++) {
            if (!dp[i]) continue;
            for (int j = i + 1; j <= n; j++) {
                string ans = s.substr(i, j - i);
                if (find(wordDict.begin(), wordDict.end(), ans) != wordDict.end()) {
                    dp[j] = true;
                }
            }
        }
        return dp[n];
        
    }
};
```

## Wiggle Subsequence
> [Leetcode 376](https://leetcode.com/problems/wiggle-subsequence/description/)

A sequence of numbers is called a wiggle sequence if the differences between successive numbers strictly alternate between positive and negative. The first difference (if one exists) may be either positive or negative. A sequence with fewer than two elements is trivially a wiggle sequence.

For example, [1,7,4,9,2,5] is a wiggle sequence because the differences (6,-3,5,-7,3) are alternately positive and negative. In contrast, [1,4,7,2,5] and [1,7,4,5,5] are not wiggle sequences, the first because its first two differences are positive and the second because its last difference is zero.

Given a sequence of integers, return the length of the longest subsequence that is a wiggle sequence. A subsequence is obtained by deleting some number of elements (eventually, also zero) from the original sequence, leaving the remaining elements in their original order.

Examples:
```
Input: [1,7,4,9,2,5]
Output: 6
The entire sequence is a wiggle sequence.

Input: [1,17,5,10,13,15,10,5,16,8]
Output: 7
There are several subsequences that achieve this length. One is [1,17,10,13,10,16,8].

Input: [1,2,3,4,5,6,7,8,9]
Output: 2
```

``` cpp
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int f = 1, d = 1, n = nums.size();
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] > nums[i - 1]) f = d + 1;
            if (nums[i] < nums[i - 1]) d = f + 1;
        }
        return min(max(f, d), n);
    }
};
```

## Arithmetic Slices
> [Leetcode 413](https://leetcode.com/problems/arithmetic-slices/description/)

A sequence of number is called arithmetic if it consists of at least three elements and if the difference between any two consecutive elements is the same.
```
For example, these are arithmetic sequence:

1, 3, 5, 7, 9
7, 7, 7, 7
3, -1, -5, -9

The following sequence is not arithmetic.

1, 1, 2, 5, 7

A zero-indexed array A consisting of N numbers is given. A slice of that array is any pair of integers (P, Q) such that 0 <= P < Q < N.

A slice (P, Q) of array A is called arithmetic if the sequence:
A[P], A[p + 1], ..., A[Q - 1], A[Q] is arithmetic. In particular, this means that P + 1 < Q.

The function should return the number of arithmetic slices in the array A.

```

Example:
```
A = [1, 2, 3, 4]

return: 3, for 3 arithmetic slices in A: [1, 2, 3], [2, 3, 4] and [1, 2, 3, 4] itself.
```

``` cpp
class Solution {
public:
    int numberOfArithmeticSlices(vector<int>& nums) {
        int n = nums.size();
        if (n < 3) return 0;
        int cur = nums[1] - nums[0] == nums[2] - nums[1] ? 1 : 0;
        int ret = cur;
        for (int i = 3; i < n; i++) {
            if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) cur ++;
            else cur = 0;
            ret += cur;
        }
        return ret;
    }
};
```

## Regular Expression Matching
> [Leetcode 10](https://leetcode.com/problems/regular-expression-matching/description/)

Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '*'.
```
'.' Matches any single character.
'*' Matches zero or more of the preceding element.
The matching should cover the entire input string (not partial).
```
Note:

- s could be empty and contains only lowercase letters a-z.
- p could be empty and contains only lowercase letters a-z, and characters like . or *.

Example 1:
```
Input:
s = "aa"
p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".
```
Example 2:
```
Input:
s = "aa"
p = "a*"
Output: true
Explanation: '*' means zero or more of the precedeng element, 'a'. Therefore, by repeating 'a' once, it becomes "aa".
```

Example 3:
```
Input:
s = "ab"
p = ".*"
Output: true
Explanation: ".*" means "zero or more (*) of any character (.)".
```

Example 4:
```
Input:
s = "aab"
p = "c*a*b"
Output: true
Explanation: c can be repeated 0 times, a can be repeated 1 time. Therefore it matches "aab".
```

Example 5:
```
Input:
s = "mississippi"
p = "mis*is*p*."
Output: false
```

``` cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        if (s.empty() && p.empty()) return true;
        int m = s.size(), n = p.size();
        bool dp[m + 1][n + 1];
        memset(dp, false, sizeof(dp));
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            dp[0][i] = p[i - 1] == '*' && dp[0][i - 2];
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (p[j - 1] == '*') {
                    char c = p[j - 2];
                    if (c != '.' && s[i - 1] != c) dp[i][j] = dp[i][j - 2];
                    else dp[i][j] = (dp[i][j - 2] || dp[i - 1][j] || dp[i - 1][j - 2]);
                    // .*
                    // 不匹配：dp[i][j - 2] 
                    // 匹配.：dp[i - 1][j - 2] 
                    // 匹配.*：dp[i - 1][j] 因为i-1不一定是由.匹配的可能是由*匹配的，所以用dp[i - 1][j]
                }
                else 
                    dp[i][j] = (s[i - 1] == p[j - 1] || p[j - 1] == '.') && dp[i - 1][j - 1];
            }
        }
        return dp[m][n];
    }
};
```

## Wildcard Matching
> [Leetcode 44](https://leetcode.com/problems/wildcard-matching/description/)

Given an input string (s) and a pattern (p), implement wildcard pattern matching with support for '?' and '*'.

'?' Matches any single character.
'*' Matches any sequence of characters (including the empty sequence).
The matching should cover the entire input string (not partial).

Note:

s could be empty and contains only lowercase letters a-z.
p could be empty and contains only lowercase letters a-z, and characters like ? or *.

Example 1:
```
Input:
s = "aa"
p = "a"
Output: false
Explanation: "a" does not match the entire string "aa".
```
Example 2:
```
Input:
s = "aa"
p = "*"
Output: true
Explanation: '*' matches any sequence.
```
Example 3:
```
Input:
s = "cb"
p = "?a"
Output: false
Explanation: '?' matches 'c', but the second letter is 'a', which does not match 'b'.
```
Example 4:
```
Input:
s = "adceb"
p = "*a*b"
Output: true
Explanation: The first '*' matches the empty sequence, while the second '*' matches the substring "dce".
```
Example 5:
```
Input:
s = "acdcb"
p = "a*c?b"
Output: false
```

``` cpp
class Solution {
public:
    bool isMatch(string s, string p) {
        if (s.empty() && p.empty()) return true;
        int m = s.size(), n = p.size();
        vector<vector<bool>> dp(m + 1, vector<bool>(n + 1, false));
        dp[0][0] = true;
        for (int i = 1; i <= n; i++) {
            dp[0][i] = p[i - 1] == '*' && dp[0][i - 1];
        }
        for (int i = 1; i <= m; i++) {
            char c = s[i - 1];
            for (int j = 1; j <= n; j++) {
                char t = p[j - 1];
                if (t == '?' || t == c) dp[i][j] = dp[i - 1][j - 1];
                if (t == '*') dp[i][j] = dp[i - 1][j - 1] || dp[i][j - 1] || dp[i - 1][j];
            }
        }
        return dp[m][n];
    }
};
```

## Length of Longest Fibonacci Subsequence
> [Leetcode 873](https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/description/)

A sequence X_1, X_2, ..., X_n is fibonacci-like if:

- n >= 3
- X_i + X_{i+1} = X_{i+2} for all i + 2 <= n

Given a strictly increasing array A of positive integers forming a sequence, find the length of the longest fibonacci-like subsequence of A.  If one does not exist, return 0.

(Recall that a subsequence is derived from another sequence A by deleting any number of elements (including none) from A, without changing the order of the remaining elements.  For example, [3, 5, 8] is a subsequence of [3, 4, 5, 6, 7, 8].)

 

Example 1:
```
Input: [1,2,3,4,5,6,7,8]
Output: 5
Explanation:
The longest subsequence that is fibonacci-like: [1,2,3,5,8].
```

Example 2:
```
Input: [1,3,7,11,12,14,18]
Output: 3
Explanation:
The longest subsequence that is fibonacci-like:
[1,11,12], [3,11,14] or [7,11,18].
```

``` cpp
class Solution {
public:
    int lenLongestFibSubseq(vector<int>& nums) {
        int n = nums.size();
        if (n < 3) return 0;
        unordered_map<int, int> m;
        for (int i = 0; i < n; i++) {
            m[nums[i]] = i;
        }
        vector<vector<int>> dp(n + 1, vector<int>(n + 1, 0));
        int ret = 2;
        // 从后向前更新，将阶段的结果用于后面的过程
        for (int i = n - 2; i >= 0; i--) {
            for (int j = i + 1; j <= n; j++) {
                dp[i][j] = 2;
                int ans = nums[i] + nums[j];
                if (!m.count(ans)) continue;
                int k = m[ans];
                dp[i][j] = dp[j][k] + 1;
                ret = max(dp[i][j], ret);
            }
        }
        return ret >= 3 ? ret : 0;
    }
};
```


# 背包
## Ones and Zeroes
> [Leetcode 474](https://leetcode.com/problems/ones-and-zeroes/description/)

In the computer world, use restricted resource you have to generate maximum benefit is what we always want to pursue.

For now, suppose you are a dominator of m 0s and n 1s respectively. On the other hand, there is an array with strings consisting of only 0s and 1s.

Now your task is to find the maximum number of strings that you can form with given m 0s and n 1s. Each 0 and 1 can be used at most once.

Note:
- The given numbers of 0s and 1s will both not exceed 100
- The size of given string array won't exceed 600.

Example 1:
```
Input: Array = {"10", "0001", "111001", "1", "0"}, m = 5, n = 3
Output: 4
Explanation: This are totally 4 strings can be formed by the using of 5 0s and 3 1s, which are “10,”0001”,”1”,”0”
```
Example 2:
```
Input: Array = {"10", "0", "1"}, m = 1, n = 1
Output: 2
Explanation: You could form "10", but then you'd have nothing left. Better form "0" and "1".
```

``` cpp
class Solution {
public:
    void helper(string s, int& one, int& zero) {
        one = zero = 0;
        for (auto i : s) {
            if (i == '0') zero ++;
            else one ++;
        }
    }
    int findMaxForm(vector<string>& strs, int m, int n) {
        vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));
        for (auto s : strs) {
            int o, z;
            helper(s, o, z);
            for (int i = m; i >= z; i--) {
                for (int j = n; j >= o; j--) {
                    dp[i][j] = max(dp[i][j], dp[i - z][j - o] + 1);
                }
            }
        }
        return dp[m][n];
    }
};
```

## Partition Equal Subset Sum
> [Leetcode 416](https://leetcode.com/problems/partition-equal-subset-sum/description/)

Given a non-empty array containing only positive integers, find if the array can be partitioned into two subsets such that the sum of elements in both subsets is equal.

Note:
- Each of the array element will not exceed 100.
- The array size will not exceed 200.

Example 1:
```
Input: [1, 5, 11, 5]

Output: true

Explanation: The array can be partitioned as [1, 5, 5] and [11].
```
Example 2:
```
Input: [1, 2, 3, 5]

Output: false

Explanation: The array cannot be partitioned into equal sum subsets.
```

``` cpp
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        if (nums.size() < 2) return false;
        int ans = 0;
        for (auto i : nums) ans += i;
        if (ans % 2) return false;
        int target = ans / 2;
        vector<bool> dp(target + 1, false);
        dp[0] = true;
        for (auto i : nums) {
            for (int j = target; j >= i; j--) {
                dp[j] = dp[j] || dp[j - i];
            }
        } 
        return dp[target];
    }
};
```

## 最长回文子序列
> [参考资料](https://www.cnblogs.com/AndyJee/p/4465696.html)

``` cpp
#include <iostream>
#include <vector>
#include <math.h>
using namespace std;

int helper(string s) {
    if (s.empty()) return 0;
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) {
        dp[i][i] = 1;
    }
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            if (s[i] == s[j]) dp[i][j] = dp[i + 1][j - 1] + 2;
            else dp[i][j] = max(dp[i + 1][j], dp[i][j - 1]);
        }
    }
    return dp[0][n - 1];
}


int main() {
    string s;
    cin >> s;
    cout << helper(s) << endl;
    return 0;
}
```

## 回文子序列个数
``` cpp
#include <iostream>
#include <vector>

using namespace std;

int helper(string s) {
    int n = s.size();
    if (!n) return 0;
    vector<vector<int>> dp(n, vector<int>(n, 0));
    for (int i = n - 1; i >= 0; i--) {
        for (int j = i + 1; j < n; j++) {
            if (s[i] == s[j]) {
                dp[i][j] = dp[i + 1][j] + dp[i][j + 1] + 1;
            }
            else {
                dp[i][j] = dp[i + 1][j] + dp[i][j - 1] - dp[i + 1][j - 1];
            }
        }
    }
    return dp[0][n - 1];
}

int main() {
    string s;
    cin >> s;
    cout << helper(s) << endl;
    return 0;
}
```

