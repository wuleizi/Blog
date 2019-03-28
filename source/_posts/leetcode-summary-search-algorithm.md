---
title: Leetcode 搜索算法相关整理
date: 2018-09-03 01:34:27
tags: [算法, 总结, Leetcode, OJ]
---

> 这里总结一些leetcode上比较经典的搜索类题目
<!-- more -->
# 快速搜索相关

## Kth Largest Element in an Array
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
    int partition(vector<int>& nums, int left, int right) {
        int base = nums[left];
        while (left < right) {
            // 挖坑填数方法
            // 第一次的left被保存到了base中，之后直接将该位置赋值为right
            // 则下次再找到就会赋值到right位置上，此时right原来的数保存在了原来left中，原来left的值也保存下来
            // 剩下的值就是缺一个第一次的left，最后赋值到left位置就可以了
            while (left < right && nums[right] <= base) right--;
            
            if (left < right) nums[left] = nums[right];
            while (left < right && nums[left] > base) left ++;
            if (left < right) nums[right] = nums[left];
        }
        nums[left] = base;
        return left;
    }
    int qsearch(vector<int>& nums, int left, int right, int k) {
        // 要注意只剩下一个数据的情况
        if (left <= right) {
            int mid = partition(nums, left, right);
            if (mid == k - 1) return nums[mid];
            if (mid < k - 1) return qsearch(nums, mid + 1, right, k);
            else return qsearch(nums, left, mid - 1, k);
        }
        return -1;
        
    }
    int findKthLargest(vector<int>& nums, int k) {
        int len = nums.size();
        if (len < k || k < 1) return -1;
        return qsearch(nums, 0, len - 1, k);
    }
};
```

## Wiggle Sort II
> [Leetcode 324](https://leetcode.com/problems/wiggle-sort-ii/description/)

Given an unsorted array nums, reorder it such that nums[0] < nums[1] > nums[2] < nums[3]....
```
Example 1:
Input: nums = [1, 5, 1, 1, 6, 4]
Output: One possible answer is [1, 4, 1, 5, 1, 6].

Example 2:
Input: nums = [1, 3, 2, 2, 3, 1]
Output: One possible answer is [2, 3, 1, 3, 1, 2].
```

``` cpp
class Solution {
public:
    void wiggleSort(vector<int>& nums) {
        vector<int> ans(nums);
        sort(ans.begin(), ans.end());
        int n = nums.size();
        int left = 0, right = (n + 1) / 2, index = 0;
        for (int i = n - 1; i >= 0; i--) {
            nums[i] = ans[i & 1 ? right++ : left++];
        }
    }
};
```

Follow Up: Can you do it in O(n) time and/or in-place with O(1) extra space?

主要思想是先用快速找到中间的数，然后利用快搜中partition的思想将前半数据放到奇数位上，后半段的数放入偶数位，解释可以参考[Discuss](https://leetcode.com/problems/wiggle-sort-ii/discuss/77682/Step-by-step-explanation-of-index-mapping-in-Java)
``` cpp
void wiggleSort(vector<int>& nums) {
    int n = nums.size();
    
    // Find a median.
    auto midptr = nums.begin() + n / 2;
    nth_element(nums.begin(), midptr, nums.end());
    int mid = *midptr;
    
    // Index-rewiring.
    #define A(i) nums[(1+2*(i)) % (n|1)]

    // 3-way-partition-to-wiggly in O(n) time with O(1) space.
    int i = 0, j = 0, k = n - 1;
    while (j <= k) {
        if (A(j) > mid)
            swap(A(i++), A(j++));
        else if (A(j) < mid)
            swap(A(j), A(k--));
        else
            j++;
    }
}
```

# 并查集

## Longest Consecutive Sequence
> [Leetcode 128](https://leetcode.com/problems/longest-consecutive-sequence/description/)

Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

Your algorithm should run in O(n) complexity.

Example:
```
Input: [100, 4, 200, 1, 3, 2]
Output: 4
Explanation: The longest consecutive elements sequence is [1, 2, 3, 4]. Therefore its length is 4.
```

``` cpp
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_map<int, int> m;
        int ret = 0;
        for (auto i : nums) {
            int l = 0, r = 0;
            // 此题借用了并查集的思想，将边缘的数据进行合并
            // 找到就跳过，防止存在中间结果
            if (m.find(i) != m.end()) continue;
            if (m.find(i - 1) != m.end()) l = m[i - 1];
            if (m.find(i + 1) != m.end()) r = m[i + 1];
            int ans = l + r + 1;
            ret = max(ret, ans);
            m[i] = m[i - l] = m[i + r] = ans;
        }
        return ret;
    }
};
```

# BFS

BFS因为占用的空间比较大且一般时间比较长，所以经常用在需要全部数据都要检索的题目上，一般的类型包括：
- 图上任意两点之间的距离
- 图的拓扑排序
- 树与层数相关的题目，例如树形图的直径，寻找根节点等
- 可能还有其他的类型，之后会来补充...

## Course Schedule （拓扑排序）
> [Leetcode 207](https://leetcode.com/problems/course-schedule/description/)

There are a total of n courses you have to take, labeled from 0 to n-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?
```
Example 1:
Input: 2, [[1,0]] 
Output: true
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0. So it is possible.

Example 2:
Input: 2, [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.
```

``` cpp
class Solution {
public:
    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
        int ret = 0;
        unordered_map<int, vector<int>> adj;
        vector<int> cnt(numCourses, 0);
        for (auto i : prerequisites) {
            cnt[i.first]++;
            adj[i.second].push_back(i.first);
        }
        vector<int> next;
        for (int i = 0; i < numCourses; i++) {
            if (!cnt[i]) {
                ret ++;
                next.push_back(i);
            }
        }
        while (!next.empty()) {
            vector<int> cur;
            for (auto i : next) {
                for (auto j : adj[i]) {
                    cnt[j]--;
                    if (!cnt[j]) {
                        ret ++;
                        cur.push_back(j);
                    }
                }
            }
            next = cur;
        }
        return ret == numCourses;
    }
};
```


## Minimum Height Trees
For a undirected graph with tree characteristics, we can choose any node as the root. The result graph is then a rooted tree. Among all possible rooted trees, those with minimum height are called minimum height trees (MHTs). Given such a graph, write a function to find all the MHTs and return a list of their root labels.

Format
The graph contains n nodes which are labeled from 0 to n - 1. You will be given the number n and a list of undirected edges (each edge is a pair of labels).

You can assume that no duplicate edges will appear in edges. Since all edges are undirected, [0, 1] is the same as [1, 0] and thus will not appear together in edges.


Example 1 :
```
Input: n = 4, edges = [[1, 0], [1, 2], [1, 3]]

        0
        |
        1
       / \
      2   3 

Output: [1]
```

Example 2 :
```
Input: n = 6, edges = [[0, 3], [1, 3], [2, 3], [4, 3], [5, 4]]

     0  1  2
      \ | /
        3
        |
        4
        |
        5 

Output: [3, 4]
```

``` cpp
class Solution {
public:
    vector<int> findMinHeightTrees(int n, vector<pair<int, int>>& edges) {
        if (n == 1) return vector<int>({0});
        vector<unordered_set<int>> adj(n, unordered_set<int>());
        for (auto i : edges) {
            int x = i.first, y = i.second;
            adj[x].insert(y);
            adj[y].insert(x);
        }
        vector<int> cur;
        for (int i = 0; i < n; i++) {
            if (adj[i].size() == 1) 
                cur.push_back(i);
        }
        while (true) {
            vector<int> next;
            for (auto i : cur) {
                for (auto j : adj[i]) {
                // 此处是细节，和拓扑排序不一样
                // 因为此树形图是无向图，所以必须将反向的边删除
                    adj[j].erase(i);
                    if (adj[j].size() == 1) 
                        next.push_back(j);
                }
            }
            if (next.empty()) break;
            else cur = next;
        }
        return cur;
    }
};
```

## Reconstruct Itinerary
> [Leetcode 332](https://leetcode.com/problems/reconstruct-itinerary/description/)

Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.

Note:

If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
All airports are represented by three capital letters (IATA code).
You may assume all tickets form at least one valid itinerary.

Example 1:
```
Input: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
Output: ["JFK", "MUC", "LHR", "SFO", "SJC"]
```

Example 2:
```
Input: [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"].
             But it is larger in lexical order.
```

``` cpp
// 欧拉环路
class Solution {
public:
    vector<string> findItinerary(vector<pair<string, string>> tickets) {
        unordered_map<string, vector<string>> m;
        for (auto i : tickets) {
            m[i.first].push_back(i.second);
        }
        for (auto &i : m) {
            sort(i.second.begin(), i.second.end());
        }
        
        vector<string> ret;
        stack<string> s;
        s.push("JFK");
        while (!s.empty()) {
            string ans = s.top();
            if (m[ans].empty()) {
                s.pop();
                ret.push_back(ans);
            }
            else {
                s.push(m[ans][0]);
                m[ans].erase(m[ans].begin());
            }
        }
        reverse(ret.begin(), ret.end());
        return ret;
    }
};
```

## Populating Next Right Pointers in Each Node
> [Leetcode 116](https://leetcode.com/problems/populating-next-right-pointers-in-each-node/description/)

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

Note:

You may only use constant extra space.
Recursive approach is fine, implicit stack space does not count as extra space for this problem.
You may assume that it is a perfect binary tree (ie, all leaves are at the same level, and every parent has two children).

Example:
```
Given the following perfect binary tree,

     1
   /  \
  2    3
 / \  / \
4  5  6  7
After calling your function, the tree should look like:

     1 -> NULL
   /  \
  2 -> 3 -> NULL
 / \  / \
4->5->6->7 -> NULL
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
        while (pre->left) {
            TreeLinkNode* temp = pre;
            while (pre) {
                pre->left->next = pre->right;
                if (pre->next) pre->right->next = pre->next->left;
                else pre->right->next = NULL;
                pre = pre->next;
            }
            pre = temp->left;
        }
    }
};
```

# DFS

## Binary Tree Maximum Path Sum

> [Leetcode 124](https://leetcode.com/problems/binary-tree-maximum-path-sum/description/)

Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

Example 1:
```
Input: [1,2,3]

       1
      / \
     2   3

Output: 6
```
Example 2:
```
Input: [-10,9,20,null,null,15,7]

   -10
   / \
  9  20
    /  \
   15   7

Output: 42
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
    int helper(TreeNode* root, int& ret) {
        if (!root) return 0;
        // 此题用了贪心的思路，如果路径上的和为负，则就可以删掉这段路径
        // 因此可以满足任意两点之间路径和最大的值，而不用考虑是不是从叶子节点开始
        int l = max(0, helper(root->left, ret));
        int r = max(0, helper(root->right, ret));
        ret = max(ret, l + r + root->val);
        return max(l, r) + root->val;
    }
    int maxPathSum(TreeNode* root) {
        int ret = INT_MIN;
        helper(root, ret);
        return ret;
    }
};
```

## Kth Smallest Element in a BST
> [Leetcode 230](https://leetcode.com/problems/kth-smallest-element-in-a-bst/description/)

Given a binary search tree, write a function kthSmallest to find the kth smallest element in it.

Note: 
You may assume k is always valid, 1 ≤ k ≤ BST's total elements.

Example 1:
```
Input: root = [3,1,4,null,2], k = 1
   3
  / \
 1   4
  \
   2
Output: 1
```

Example 2:
```
Input: root = [5,3,6,2,4,null,null,1], k = 3
       5
      / \
     3   6
    / \
   2   4
  /
 1
Output: 3
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
    int helper(TreeNode* root, int& k) {
        if (!root) return 0;
        int x = helper(root->left, k);
        return !k ? x : (!--k ? root->val : helper(root->right, k));
    }
    int kthSmallest(TreeNode* root, int k) {
        if (!root) return -1;
        return helper(root, k);
    }
};
```
## Generate Parentheses
> [Leetcode 22](https://leetcode.com/problems/generate-parentheses/description/)

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

For example, given n = 3, a solution set is:
```
[
  "((()))",
  "(()())",
  "(())()",
  "()(())",
  "()()()"
]
```

``` cpp
class Solution {
public:
    void helper(int i, int j, int n, string ans, vector<string>& ret) {
        if (j != i) helper(i, j + 1, n, ans + ")", ret);
        if (i != n) helper(i + 1, j, n, ans + "(", ret);
        if (j == n) ret.push_back(ans);
    }
    
    vector<string> generateParenthesis(int n) {
        vector<string> ret;
        if (n < 1) return ret;
        helper(0, 0, n, "", ret);
        return ret;
    }
};
```
## House Robber III
> [Leetcode 337](https://leetcode.com/problems/house-robber-iii/description/)

The thief has found himself a new place for his thievery again. There is only one entrance to this area, called the "root." Besides the root, each house has one and only one parent house. After a tour, the smart thief realized that "all houses in this place forms a binary tree". It will automatically contact the police if two directly-linked houses were broken into on the same night.

Determine the maximum amount of money the thief can rob tonight without alerting the police.

Example 1:
```
Input: [3,2,3,null,3,null,1]

     3
    / \
   2   3
    \   \ 
     3   1

Output: 7 
Explanation: Maximum amount of money the thief can rob = 3 + 3 + 1 = 7.
```
Example 2:
```
Input: [3,4,5,1,3,null,1]

     3
    / \
   4   5
  / \   \ 
 1   3   1

Output: 9
Explanation: Maximum amount of money the thief can rob = 4 + 5 = 9.
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
    int helper(TreeNode* root, int& l, int& r) {
        if (!root) return 0;
        int ll = 0, lr = 0, rl = 0, rr = 0;
        l = helper(root->left, ll, lr);
        r = helper(root->right, rl, rr);
        return max(ll + lr + rl + rr + root->val, l + r);
    }
    int rob(TreeNode* root) {
        int l, r;
        return helper(root, l, r);
    }
};
```
## Decode String
> [Leetcode 394](https://leetcode.com/problems/decode-string/description/)

Given an encoded string, return it's decoded string.

The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there won't be input like 3a or 2[4].

Examples:
```
s = "3[a]2[bc]", return "aaabcbc".
s = "3[a2[c]]", return "accaccacc".
s = "2[abc]3[cd]ef", return "abcabccdcdcdef".
```

``` cpp
class Solution {
public:
    string helper(string s, int& index) {
        string ret;
        int n = s.size();
        while (index < n && s[index] != ']') {
            if (isdigit(s[index])) {
                int cnt = 0;
                while (s[index] != '[') cnt = cnt * 10 + (s[index++] - '0');
                index ++;
                string ans = helper(s, index);
                index ++;
                for (int i = 0; i < cnt; i++) ret += ans;
            }
            else {
                ret.push_back(s[index++]);
            }
        }
        return ret;
    }
    string decodeString(string s) {
        int index = 0;
        return helper(s, index);
    }
};
```

## Matchsticks to Square
> [Leetcode 473](https://leetcode.com/problems/matchsticks-to-square/description/)

Remember the story of Little Match Girl? By now, you know exactly what matchsticks the little match girl has, please find out a way you can make one square by using up all those matchsticks. You should not break any stick, but you can link them up, and each matchstick must be used exactly one time.

Your input will be several matchsticks the girl has, represented with their stick length. Your output will either be true or false, to represent whether you could make one square using all the matchsticks the little match girl has.

Example 1:
```
Input: [1,1,2,2,2]
Output: true
Explanation: You can form a square with length 2, one side of the square came two sticks with length 1.
```

Example 2:
```
Input: [3,3,3,3,4]
Output: false

Explanation: You cannot find a way to form a square with all the matchsticks.
```

``` cpp
class Solution {
public:
    bool helper(vector<int>& ans, vector<int>& nums, int index, int target) {
        int n = nums.size();
        if (index == n) {
            return ans[0] == target && ans[1] == target && ans[2] == target;
        }
        for (int i = 0; i < 4; i++) {
            if (ans[i] + nums[index] > target) continue;
            ans[i] += nums[index];
            if (helper(ans, nums, index + 1, target)) return true;
            ans[i] -= nums[index];
        }
        return false;
        
    }
    bool makesquare(vector<int>& nums) {
        int target = 0;
        for (auto i : nums) target += i;
        if (target % 4 != 0 || nums.size() < 4) return false;
        sort(nums.rbegin(), nums.rend());
        vector<int> ans(4, 0);
        return helper(ans, nums, 0, target / 4);
    }
};
```

follow up: Partition to K Equal Sum Subsets

> [Leetcode 698](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/description/)

Given an array of integers nums and a positive integer k, find whether it's possible to divide this array into k non-empty subsets whose sums are all equal.

Example 1:
```
Input: nums = [4, 3, 2, 3, 5, 2, 1], k = 4
Output: True
Explanation: It's possible to divide it into 4 subsets (5), (1, 4), (2,3), (2,3) with equal sums.
```

``` cpp
class Solution {
public:
    bool helper(vector<int> ans, int index, vector<int>& nums, int target) {
        int k = ans.size(), n = nums.size();
        if (index == n) {
            for (int i = 0; i < k; i++) {
                if (ans[i] != target) return false;
            }
            return true;
        }
        for (int i = 0; i < k; i++) {
            if (ans[i] + nums[index] > target) continue;
            ans[i] += nums[index];
            if (helper(ans, index + 1, nums, target)) return true;
            ans[i] -= nums[index];
        }
        return false;
    }
    bool canPartitionKSubsets(vector<int>& nums, int k) {
        int ans = 0, n = nums.size();
        for (auto i : nums) ans += i;
        if (n < k || ans % k) return false;
        sort(nums.begin(), nums.end(), greater<int>());
        return helper(vector<int>(k, 0), 0, nums, ans / k);
    }
};
```

## Subsets II
> [Leetcode 90](https://leetcode.com/problems/subsets-ii/description/)

DescriptionHintsSubmissionsDiscussSolution
Given a collection of integers that might contain duplicates, nums, return all possible subsets (the power set).

Note: The solution set must not contain duplicate subsets.

Example:
```
Input: [1,2,2]
Output:
[
  [2],
  [1],
  [1,2,2],
  [2,2],
  [1,2],
  []
]
```

``` cpp
class Solution {
public:
    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        vector<vector<int>> ret;
        ret.push_back(vector<int>());
        if (nums.empty()) return ret;
        sort(nums.begin(), nums.end());
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            int index = 0;
            while (i + index < n && nums[i] == nums[i + index]) index ++;
            int len = ret.size();
            for (int j = 0; j < len; j++) {
                auto temp = ret[j];
                for (int k = 0; k < index; k++) {
                    temp.push_back(nums[i]);
                    ret.push_back(temp);
                }
            }
            i += index - 1;
        }
        return ret;
    }
};
```

## Word Search
> [Leetcode 79](https://leetcode.com/problems/word-search/description/)

DescriptionHintsSubmissionsDiscussSolution
Given a 2D board and a word, find if the word exists in the grid.

The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

Example:
```
board =
[
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
]

Given word = "ABCCED", return true.
Given word = "SEE", return true.
Given word = "ABCB", return false.
```

``` cpp
class Solution {
public:
    bool helper(vector<vector<char>>& board, string word, int index, int x, int y) {
        // 尽量在helper函数中做相等判断，否则还需要在外层函数中实现相等判断和访问覆盖操作
        if (word[index] != board[x][y]) return false;
        int len = word.size(), m = board.size(), n = board[0].size();
        if (index == len - 1) return true;
        int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
        board[x][y] = '\0';
        for (int i = 0; i < 4; i++) {
            int X = x + a[i], Y = y + b[i];
            if (X >= 0 && X < m && Y >= 0 && Y < n && helper(board, word, index + 1, X, Y)) return true;
        }
        board[x][y] = word[index];
        return false;
    }
    bool exist(vector<vector<char>>& board, string word) {
        if (board.empty() || board[0].empty()) return false;
        if (word.empty()) return true;
        int m = board.size(), n = board[0].size();
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (helper(board, word, 0, i, j)) return true;
            }
        }
        return false;
    }
};
```

## Longest Increasing Path in a Matrix

> [Leetcode 329](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/description/)

Given an integer matrix, find the length of the longest increasing path.

From each cell, you can either move to four directions: left, right, up or down. You may NOT move diagonally or move outside of the boundary (i.e. wrap-around is not allowed).

Example 1:
```
Input: nums = 
[
  [9,9,4],
  [6,6,8],
  [2,1,1]
] 
Output: 4 
Explanation: The longest increasing path is [1, 2, 6, 9].
```

Example 2:
```
Input: nums = 
[
  [3,4,5],
  [3,2,6],
  [2,2,1]
] 
Output: 4 
Explanation: The longest increasing path is [3, 4, 5, 6]. Moving diagonally is not allowed.
```

``` cpp
class Solution {
public:
    int helper(vector<vector<int>>& matrix, vector<vector<int>>& ma, int x, int y) {
        if (ma[x][y]) return ma[x][y];
        int ans = matrix[x][y];
        int m = matrix.size(), n = matrix[0].size();
        int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
        for (int i = 0; i < 4; i++) {
            int X = x + a[i], Y = y + b[i];
            if (X >= 0 && X < m && Y >= 0 && Y < n) {
                int local = 1;
                if (matrix[X][Y] > ans) {
                    local += helper(matrix, ma, X, Y);
                }
                ma[x][y] = max(local, ma[x][y]);
            }
        }
        return ma[x][y];
    }
    
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        int ret = 1;
        if (matrix.empty() || matrix[0].empty()) return 0;
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<int>> ma(m, vector<int>(n, 0));
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                ret = max(ret, helper(matrix, ma, i, j));
            }
        }
        return ret;
    }
};
```


## Word Break II
> [Leetcode 140](https://leetcode.com/problems/word-break-ii/description/)

Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences.

Note:

The same word in the dictionary may be reused multiple times in the segmentation.
You may assume the dictionary does not contain duplicate words.

Example 1:
```
Input:
s = "catsanddog"
wordDict = ["cat", "cats", "and", "sand", "dog"]
Output:
[
  "cats and dog",
  "cat sand dog"
]
```

Example 2:
```
Input:
s = "pineapplepenapple"
wordDict = ["apple", "pen", "applepen", "pine", "pineapple"]
Output:
[
  "pine apple pen apple",
  "pineapple pen apple",
  "pine applepen apple"
]
Explanation: Note that you are allowed to reuse a dictionary word.
```

Example 3:
```
Input:
s = "catsandog"
wordDict = ["cats", "dog", "sand", "and", "cat"]
Output:
[]
```

``` cpp
class Solution {
public:
    void merge(vector<string> ans, string word, vector<string>& ret) {
        for (auto i : ans) {
            ret.push_back(word + " " + i);
        }
    }
    vector<string> helper(string s, unordered_set<string>& dict, unordered_map<string, vector<string>>& m) {
        if (m.find(s) != m.end()) return m[s];
        vector<string> ret;
        if (dict.find(s) != dict.end()) ret.push_back(s);
        int n = s.size();
        for (int i = 0; i < n; i++) {
            string word = s.substr(0, i);
            if (dict.find(word) == dict.end()) continue;
            string rem = s.substr(i);
            merge(helper(rem, dict, m), word, ret);
        }
        m[s] = ret;
        return ret;
        
    }
    vector<string> wordBreak(string s, vector<string>& wordDict) {
        unordered_set<string> dict;
        unordered_map<string, vector<string>> m;
        for (auto i : wordDict) {
            dict.insert(i);
        }
        return helper(s, dict, m);
    }
};
```

## Add and Search Word - Data structure design
> [Leetcode 211](https://leetcode.com/problems/add-and-search-word-data-structure-design/description/)

Design a data structure that supports the following two operations:

void addWord(word)
bool search(word)
search(word) can search a literal word or a regular expression string containing only letters a-z or .. A . means it can represent any one letter.

Example:
```
addWord("bad")
addWord("dad")
addWord("mad")
search("pad") -> false
search("bad") -> true
search(".ad") -> true
search("b..") -> true
```

``` cpp
struct TrieNode {
    TrieNode* child[26];
    bool isKey;
    TrieNode() : isKey(false) {
        memset(child, NULL, sizeof(child));
    }
};
class WordDictionary {
public:
    TrieNode* root;
    /** Initialize your data structure here. */
    WordDictionary() {
        root = new TrieNode();
    }
    
    /** Adds a word into the data structure. */
    void addWord(string word) {
        TrieNode* cur = root;
        for (auto i : word) {
            int c = i - 'a';
            if (!cur->child[c]) cur->child[c] = new TrieNode();
            cur = cur->child[c];
        }
        cur->isKey = true;
    }
    bool helper(string s, int index, TrieNode* cur) {
        int n = s.size();
        for (int i = index; i < n; i++) {
            if (s[i] == '.') {
                for (int j = 0; j < 26; j++) {
                    if (cur->child[j] && helper(s, i + 1, cur->child[j])) return true;
                }
                return false;
            }
            int c = s[i] - 'a';
            if (!cur->child[c]) return false;
            cur = cur->child[c];
        }
        return cur && cur->isKey;
    }
    /** Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter. */
    bool search(string word) {
        return helper(word, 0, root);
    }
};

/**
 * Your WordDictionary object will be instantiated and called as such:
 * WordDictionary obj = new WordDictionary();
 * obj.addWord(word);
 * bool param_2 = obj.search(word);
 */
```

## Reconstruct Itinerary
> [Leetcode 332](https://leetcode.com/problems/reconstruct-itinerary/description/)

Given a list of airline tickets represented by pairs of departure and arrival airports [from, to], reconstruct the itinerary in order. All of the tickets belong to a man who departs from JFK. Thus, the itinerary must begin with JFK.

Note:

If there are multiple valid itineraries, you should return the itinerary that has the smallest lexical order when read as a single string. For example, the itinerary ["JFK", "LGA"] has a smaller lexical order than ["JFK", "LGB"].
All airports are represented by three capital letters (IATA code).
You may assume all tickets form at least one valid itinerary.

Example 1:
```
Input: [["MUC", "LHR"], ["JFK", "MUC"], ["SFO", "SJC"], ["LHR", "SFO"]]
Output: ["JFK", "MUC", "LHR", "SFO", "SJC"]
```

Example 2:
```
Input: [["JFK","SFO"],["JFK","ATL"],["SFO","ATL"],["ATL","JFK"],["ATL","SFO"]]
Output: ["JFK","ATL","JFK","SFO","ATL","SFO"]
Explanation: Another possible reconstruction is ["JFK","SFO","ATL","JFK","ATL","SFO"].
             But it is larger in lexical order.
```


``` cpp
// dfs 版本
class Solution {
public:
    void helper(vector<string>& ret, unordered_map<string, vector<string>>& m, int n) {
        if (ret.size() == n) return;
        string ans = ret.back();
        for (int i = 0; i < m[ans].size(); i++) {
            string temp = m[ans][i];
            ret.push_back(temp);
            m[ans].erase(m[ans].begin() + i);
            helper(ret, m, n);
            if (ret.size() == n) return;
            ret.pop_back();
            m[ans].insert(m[ans].begin() + i, temp);
        }
    }
    vector<string> findItinerary(vector<pair<string, string>> tickets) {
        vector<string> ret;
        int n = tickets.size() + 1;
        unordered_map<string, vector<string>> m;
        for (auto i : tickets) {
            m[i.first].push_back(i.second);
        }
        for (auto& i : m) {
            sort(i.second.begin(), i.second.end());
        }
        ret.push_back("JFK");
        helper(ret, m, n);
        if (ret.size() == n) return ret;
        else return vector<string>();
    }
};
```

## Construct Binary Tree from Inorder and Postorder Traversal
> [Leetcode 106](https://leetcode.com/problems/construct-binary-tree-from-inorder-and-postorder-traversal/description/)

Given inorder and postorder traversal of a tree, construct the binary tree.

Note:
You may assume that duplicates do not exist in the tree.

For example, given
```
inorder = [9,3,15,20,7]
postorder = [9,15,7,20,3]
Return the following binary tree:

    3
   / \
  9  20
    /  \
   15   7
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
typedef vector<int>::iterator Iter;
class Solution {
public:
    TreeNode* helper(Iter ibegin, Iter iend, Iter pbegin, Iter pend) {
        if (ibegin == iend) return NULL;
        int val = *(pend - 1);
        Iter mid = find(ibegin, iend, val);
        TreeNode* ret = new TreeNode(val);
        ret->left = helper(ibegin, mid, pbegin, pbegin + (mid - ibegin));
        ret->right = helper(mid + 1, iend, pbegin + (mid - ibegin), pend - 1);
        return ret;
    }
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        return helper(inorder.begin(), inorder.end(), postorder.begin(), postorder.end());
    }
};
```

## Restore IP Addresses
> [Leetcode 93](https://leetcode.com/problems/restore-ip-addresses/description/)

Given a string containing only digits, restore it by returning all possible valid IP address combinations.

Example:
```
Input: "25525511135"
Output: ["255.255.11.135", "255.255.111.35"]
```

``` cpp
class Solution {
public:
    void helper(string s, int index, vector<string>& ret, vector<int> ans) {
        if (ans.size() == 4 && index == s.size()) {
            ret.push_back(to_string(ans[0]) + "." + to_string(ans[1]) + "." + to_string(ans[2]) + "." + to_string(ans[3]));
            return;
        }
        if (ans.size() < 4 && index < s.size()) {
            int temp = 0, n = s.size();
            for (int i = 0; i < 3 && index + i < n ; i++) {
                if (i && !temp) return;
                temp = temp * 10 + (s[index + i] - '0');
                if (temp <= 255) {
                    ans.push_back(temp);
                    helper(s, index + i + 1, ret, ans);
                    ans.pop_back();
                }
            }
        }
    }
    vector<string> restoreIpAddresses(string s) {
        vector<string> ret;
        helper(s, 0, ret, vector<int>());
        return ret;
    }
};
```

## Different Ways to Add Parentheses
> [Leetcode 241](https://leetcode.com/problems/different-ways-to-add-parentheses/description/)

Given a string of numbers and operators, return all possible results from computing all the different possible ways to group numbers and operators. The valid operators are +, - and *.

Example 1:
```
Input: "2-1-1"
Output: [0, 2]
Explanation: 
((2-1)-1) = 0 
(2-(1-1)) = 2
```

Example 2:
```
Input: "2*3-4*5"
Output: [-34, -14, -10, -10, 10]
Explanation: 
(2*(3-(4*5))) = -34 
((2*3)-(4*5)) = -14 
((2*(3-4))*5) = -10 
(2*((3-4)*5)) = -10 
(((2*3)-4)*5) = 10
```

``` cpp
class Solution {
public:
    vector<int> diffWaysToCompute(string input) {
        vector<int> ret;
        for (int i = 0; i < input.size(); i++) {
            if (ispunct(input[i])) {
                auto c = input[i];
                for (auto a : diffWaysToCompute(input.substr(0, i))) {
                    for (auto b : diffWaysToCompute(input.substr(i + 1))) {
                        ret.push_back(c == '+' ? a + b : (c == '-' ? a - b : a * b));
                    }
                }
            }
        }
        return ret.empty() ? vector<int>({stoi(input)}) : ret;
    }
};
```


## Palindrome Partitioning
> [Leetcode 131](https://leetcode.com/problems/palindrome-partitioning/description/)

Given a string s, partition s such that every substring of the partition is a palindrome.

Return all possible palindrome partitioning of s.

Example:
```
Input: "aab"
Output:
[
  ["aa","b"],
  ["a","a","b"]
]
```

``` cpp
class Solution {
public:
    void helper(vector<vector<string>>& ret, vector<string> ans, int index, string s) {
        int n = s.size();
        if (index == n) {
            ret.push_back(ans);
            return;
        }
        for (int i = index; i < n; i++) {
            int l = index, r = i;
            while (l < r && s[l] == s[r]) l++, r--;
            if (l >= r) {
                ans.push_back(s.substr(index, i - index + 1));
                helper(ret, ans, i + 1, s);
                ans.pop_back();
            }
            
        }
    }
    vector<vector<string>> partition(string s) {
        vector<vector<string>> ret;
        if (s.empty()) return ret;
        helper(ret, vector<string>(), 0, s);
        return ret;
        
    }
};
```


## Permutations II
> [Leetcode](https://leetcode.com/problems/permutations-ii/description/)

Given a collection of numbers that might contain duplicates, return all possible unique permutations.

Example:
```
Input: [1,1,2]
Output:
[
  [1,1,2],
  [1,2,1],
  [2,1,1]
]
```

``` cpp
class Solution {
public:
    void helper(int index, vector<int> ans, vector<vector<int>>& ret) {
        int n = ans.size();
        if (index == n - 1) {
            ret.push_back(ans);
            return;
        }
        for (int i = index; i < n; i++) {
            if (index != i && ans[i] == ans[index]) continue;
            swap(ans[i], ans[index]);
            helper(index + 1, ans, ret);
        }
    }
    vector<vector<int>> permuteUnique(vector<int>& nums) {
        vector<vector<int>> ret;
        if (nums.empty()) return ret;
        sort(nums.begin(), nums.end());
        helper(0, nums, ret);
        return ret;
    }
};
```

## Permutation Sequence
> [Leetcode 60](https://leetcode.com/problems/permutation-sequence/description/)

The set [1,2,3,...,n] contains a total of n! unique permutations.

By listing and labeling all of the permutations in order, we get the following sequence for n = 3:
```
"123"
"132"
"213"
"231"
"312"
"321"
Given n and k, return the kth permutation sequence.
```

Note:

Given n will be between 1 and 9 inclusive.

Given k will be between 1 and n! inclusive.

Example 1:
```
Input: n = 3, k = 3
Output: "213"
```
Example 2:
```
Input: n = 4, k = 9
Output: "2314"
```

``` cpp
class Solution {
public:
    string getPermutation(int n, int k) {
        vector<int> nums(n, 0);
        vector<int> fac(n, 1);
        string ret;
        for (int i = 0; i < n; i++) {
            nums[i] = i + 1;
        }
        for (int i = 1; i < n; i++) {
            fac[i] = fac[i - 1] * i;
        }
        for (int i = 0; i < n - 1; i++) {
            int index = (k - 1) / fac[n - 1 - i];
            ret += to_string(nums[index]);
            nums.erase(nums.begin() + index);
            k -= index * fac[n - 1 - i];
        }
        ret += to_string(nums[0]);
        return ret;
    }
};
```


# 二分查找
> 二分查找一般用来简化查找逻辑，将O(n)降低成O(logn)，但是由于左右边界更新的细节比较多，每个题都需要单独推导分析

## Missing Number
> [Leetcode 268](https://leetcode.com/problems/missing-number/description/)

Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.

Example 1:
```
Input: [3,0,1]
Output: 2
```
Example 2:
```
Input: [9,6,4,2,3,5,7,0,1]
Output: 8
```

``` cpp
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int left = 0, right = nums.size();
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == mid) left = mid + 1;
            else right = mid;
        }
        return left;
    }
};
```

## Search a 2D Matrix
> [Leetcode 74](https://leetcode.com/problems/search-a-2d-matrix/description/)

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:

- Integers in each row are sorted from left to right.
- The first integer of each row is greater than the last integer of the previous row.

Example 1:
```
Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 3
Output: true
```

Example 2:
```
Input:
matrix = [
  [1,   3,  5,  7],
  [10, 11, 16, 20],
  [23, 30, 34, 50]
]
target = 13
Output: false
```

``` cpp
// 线性扫描
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        int m = matrix.size(), n = matrix[0].size();
        int i = 0, j = n - 1;
        while (i < m && j >= 0) {
            if (matrix[i][j] == target) return true;
            if (matrix[i][j] > target) j--;
            else i ++;
        }
        return false;
    }
};
```

``` cpp
// 二分法
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        int m = matrix.size(), n = matrix[0].size();
        int low = 0, high = m - 1;
        while (low <= high) {
            int mid = low + (high - low) / 2;
            if (matrix[mid][0] > target) high = mid - 1;
            else if (matrix[mid][n - 1] < target) low = mid + 1;
            else {
                int left = 0, right = n - 1;
                while (left <= right) {
                    int m = left + (right - left) / 2;
                    int x = matrix[mid][m];
                    if (x == target) return true;
                    if (x < target) left = m + 1;
                    else right = m - 1;
                }
                return false;
            }
        }
        return false;
    }
};
```


## Search a 2D Matrix II
> [Leetcode 240](https://leetcode.com/problems/search-a-2d-matrix-ii/description/)

Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
- Integers in each row are sorted in ascending from left to right.
- Integers in each column are sorted in ascending from top to bottom.

Example:
```
Consider the following matrix:

[
  [1,   4,  7, 11, 15],
  [2,   5,  8, 12, 19],
  [3,   6,  9, 16, 22],
  [10, 13, 14, 17, 24],
  [18, 21, 23, 26, 30]
]
Given target = 5, return true.

Given target = 20, return false.
```

``` cpp
// 其实也就是二分查找的思想，右侧路过的值就是查找空间的上界，左侧路过的值是下界
// 查找不到就是左侧和右侧不重叠会始终找不到，从而超过边界退出循环

// 此处不是像惯性思维的i,j封闭了左侧封锁了搜索空间
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if (matrix.empty() || matrix[0].empty()) return false;
        int m = matrix.size(), n = matrix[0].size();
        int i = 0, j = n - 1;
        while (i < m && j >= 0) {
            int x = matrix[i][j];
            if (x == target) return true;
            if (x < target) i++;
            else j --;
        }
        return false;
    }
};
```

## Kth Smallest Element in a Sorted Matrix
> [Leetcode 378](https://leetcode.com/problems/kth-smallest-element-in-a-sorted-matrix/description/)

Given a n x n matrix where each of the rows and columns are sorted in ascending order, find the kth smallest element in the matrix.

Note that it is the kth smallest element in the sorted order, not the kth distinct element.

Example:
```
matrix = [
   [ 1,  5,  9],
   [10, 11, 13],
   [12, 13, 15]
],
k = 8,

return 13.
```

``` cpp
class Solution {
public:
    int kthSmallest(vector<vector<int>>& matrix, int k) {
        if (matrix.empty() || matrix[0].empty()) -1;
        int m = matrix.size(), n = matrix[0].size();
        int left = matrix[0][0], right = matrix[m - 1][n - 1];
        while (left < right) {
            int mid = left + (right - left) / 2;
            int cnt = 0;
            for (int i = 0; i < m; i++) {
                cnt += (int)(upper_bound(matrix[i].begin(), matrix[i].end(), mid) - matrix[i].begin());
            }
            if (cnt < k) {
                // 如果不满足等于k，则left会一直向后扩直至取到矩阵中的值
                left = mid + 1;
            }
            else right = mid;
        }
        return left;
    }
};
```

## Median of Two Sorted Arrays
> [Leetcode 4](https://leetcode.com/problems/median-of-two-sorted-arrays/description/)

There are two sorted arrays nums1 and nums2 of size m and n respectively.

Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).

You may assume nums1 and nums2 cannot be both empty.

Example 1:
```
nums1 = [1, 3]
nums2 = [2]

The median is 2.0
```
Example 2:
```
nums1 = [1, 2]
nums2 = [3, 4]

The median is (2 + 3)/2 = 2.5
```

``` cpp
typedef vector<int>::iterator Iter;
class Solution {
public:
    int helper(Iter l, int m, Iter r, int n, int k) {
        if (!m) return r[k - 1];
        if (!n) return l[k - 1];
        if (k == 1) return min(l[0], r[0]);
        int i = min(m, k / 2), j = min(n, k / 2);
        if (l[i - 1] < r[j - 1]) 
            return helper(l + i, m - i, r, n, k - i);
        else 
            return helper(l, m, r + j, n - j, k - j);
    }
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int n1 = nums1.size(), n2 = nums2.size();
        int a = helper(nums1.begin(), n1, nums2.begin(), n2, (n1 + n2 + 1) / 2);
        int b = helper(nums1.begin(), n1, nums2.begin(), n2, (n1 + n2 + 2) / 2);
        return (a + b) / 2.0;
    }
};
```

## Find Right Interval
> [Leetcode 436](https://leetcode.com/problems/find-right-interval/description/)

Given a set of intervals, for each of the interval i, check if there exists an interval j whose start point is bigger than or equal to the end point of the interval i, which can be called that j is on the "right" of i.

For any interval i, you need to store the minimum interval j's index, which means that the interval j has the minimum start point to build the "right" relationship for interval i. If the interval j doesn't exist, store -1 for the interval i. Finally, you need output the stored value of each interval as an array.

Note:
You may assume the interval's end point is always bigger than its start point.
You may assume none of these intervals have the same start point.

Example 1:
```
Input: [ [1,2] ]

Output: [-1]

Explanation: There is only one interval in the collection, so it outputs -1.
```
Example 2:
```
Input: [ [3,4], [2,3], [1,2] ]

Output: [-1, 0, 1]

Explanation: There is no satisfied "right" interval for [3,4].
For [2,3], the interval [3,4] has minimum-"right" start point;
For [1,2], the interval [2,3] has minimum-"right" start point.
```
Example 3:
```
Input: [ [1,4], [2,3], [3,4] ]

Output: [-1, 2, -1]

Explanation: There is no satisfied "right" interval for [1,4] and [3,4].
For [2,3], the interval [3,4] has minimum-"right" start point.
```

``` cpp
/**
 * Definition for an interval.
 * struct Interval {
 *     int start;
 *     int end;
 *     Interval() : start(0), end(0) {}
 *     Interval(int s, int e) : start(s), end(e) {}
 * };
 */
class Solution {
public:
// map的upper_bound和lower_bound是使用二分查找
    vector<int> findRightInterval(vector<Interval>& intervals) {
        int n = intervals.size();
        vector<int> ret(n, -1);
        map<int, int> m;
        for (int i = 0; i < n; i++) {
            m[intervals[i].start] = i;
        }
        for (int i = 0; i < n; i++) {
            auto cur = m.lower_bound(intervals[i].end);
            if (cur != m.end()) ret[i] = cur->second;
        }
        return ret;
    }
};
```

## Find First and Last Position of Element in Sorted Array
> [Leetcode 34](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/description/)

Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

Example 1:
```
Input: nums = [5,7,7,8,8,10], target = 8
Output: [3,4]
```

Example 2:
```
Input: nums = [5,7,7,8,8,10], target = 6
Output: [-1,-1]
```

``` cpp
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        // 此题是推导更新公式最典型的题目
        vector<int> ret(2, -1);
        if (nums.empty()) return ret;
        int n = nums.size();
        int left = 0, right = n - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < target) left = mid + 1;
            else right = mid;
        }
        if (nums[left] != target) return ret;
        else ret[0] = left;
        right = n - 1;
        while (left <= right) {
            // 等号是防止单元素
            int mid = left + (right - left) / 2;
            if (nums[mid] > target) right = mid - 1;
            else {
                ret[1] = mid;
                left = mid + 1;
            }
        }
        return ret;
    }
};
```

## Find Minimum in Rotated Sorted Array
> [Leetcode 153](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/description/)

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e.,  [0,1,2,4,5,6,7] might become  [4,5,6,7,0,1,2]).

Find the minimum element.

You may assume no duplicate exists in the array.

Example 1:
```
Input: [3,4,5,1,2] 
Output: 1
```

Example 2:
```
Input: [4,5,6,7,0,1,2]
Output: 0
```

``` cpp
class Solution {
public:
    int findMin(vector<int>& nums) {
        if (nums.empty()) return -1;
        int n = nums.size();
        int left = 0, right = n - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] > nums[right]) left = mid + 1;
            else right = mid;
        }
        return nums[left];
    }
};
```

## Search in Rotated Sorted Array
> [Leetcode 33](https://leetcode.com/problems/search-in-rotated-sorted-array/description/)

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

Example 1:
```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
```

Example 2:
```
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
```

``` cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if (nums.empty()) return -1;
        int n = nums.size();
        int left = 0, right = n - 1;
        while (left <= right) {
            // 更新过程可能存在1，【2】这种情况，
            // 所以需要在更新了left之后指向同一个数依旧生效
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            if (nums[mid] > nums[right]) {
                if (nums[mid] > target && nums[right] < target) right = mid - 1;
                else left = mid + 1;
            }
            else {
                // 因为是nums[mid] <= nums[right]，所以可能存在nums[right] == target
                if (nums[mid] < target && nums[right] >= target) left = mid + 1;
                else right = mid - 1;
            }
        }
        return -1;
    }
};
```

## Search in Rotated Sorted Array II
> [Leetcode 81](https://leetcode.com/problems/search-in-rotated-sorted-array-ii/description/)

Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,0,1,2,2,5,6] might become [2,5,6,0,0,1,2]).

You are given a target value to search. If found in the array return true, otherwise return false.

Example 1:
```
Input: nums = [2,5,6,0,0,1,2], target = 0
Output: true
```
Example 2:
```
Input: nums = [2,5,6,0,0,1,2], target = 3
Output: false
```
Follow up:

- This is a follow up problem to Search in Rotated Sorted Array, where nums may contain duplicates.
- Would this affect the run-time complexity? How and why?

``` cpp
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if (nums.empty()) return false;
        int n = nums.size();
        int left = 0, right = n - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return true;
                               
            // 与上一题的区别在于，因为需要根据mid和right的大小判断target处于哪个区间，
            // 所以需要更新一下右边界
            if (nums[mid] == nums[right]) right --;
            else if (nums[mid] > nums[right]) {
                if (nums[mid] > target && nums[right] < target) right = mid - 1;
                else left = mid + 1;
            }
            else {
                if (nums[mid] < target && nums[right] >= target) left = mid + 1;
                else right = mid - 1;
            }
        }
        return false;
    }
};
```

## Koko Eating Bananas (*)
> [Leetcode 875](https://leetcode.com/problems/koko-eating-bananas/description/)

Koko loves to eat bananas.  There are N piles of bananas, the i-th pile has piles[i] bananas.  The guards have gone and will come back in H hours.

Koko can decide her bananas-per-hour eating speed of K.  Each hour, she chooses some pile of bananas, and eats K bananas from that pile.  If the pile has less than K bananas, she eats all of them instead, and won't eat any more bananas during this hour.

Koko likes to eat slowly, but still wants to finish eating all the bananas before the guards come back.

Return the minimum integer K such that she can eat all the bananas within H hours.

 

Example 1:
```
Input: piles = [3,6,7,11], H = 8
Output: 4
```
Example 2:
```
Input: piles = [30,11,23,4,20], H = 5
Output: 30
```
Example 3:
```
Input: piles = [30,11,23,4,20], H = 6
Output: 23
```

Note:
- 1 <= piles.length <= 10^4
- piles.length <= H <= 10^9
- 1 <= piles[i] <= 10^9


``` cpp
// 二分的实际应用，用来寻找最优值
class Solution {
public:
    int minEatingSpeed(vector<int>& piles, int H) {
        int mx = 0;
        for (auto i : piles) {
            mx = max(mx, i);
        }
        int left = 1, right = mx;
        while (left < right) {
            int i = (left + right) / 2;
            long long ans = 0;
            for (auto j : piles) {
                int x = j / i;
                ans += (j % i == 0) ? x * i : (x + 1) * i;
            }
            int h = ans / i;
            if (h > H) left = i + 1;
            else right = i;
        }
        return left;
    }
};
```

# 特殊题目
> 还未归类的题目放到这里，此部分待编辑

## 数组中的逆序对 (*)
> [牛客网](https://www.nowcoder.com/practice/96bd6684e04a44eb80e6a68efc0ec6c5?tpId=13&tqId=11188&tPage=1&rp=1&ru=/ta/coding-interviews&qru=/ta/coding-interviews/question-ranking)

在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

> 思路可以参考[牛客网](https://www.nowcoder.com/questionTerminal/96bd6684e04a44eb80e6a68efc0ec6c5)

``` cpp
class Solution {
public:
    int InversePairs(vector<int> data) {
       int length=data.size();
        if(length<=0)
            return 0;
       //vector<int> copy=new vector<int>[length];
       vector<int> copy;
       for(int i=0;i<length;i++)
           copy.push_back(data[i]);
       long long count=InversePairsCore(data,copy,0,length-1);
       //delete[]copy;
       return count%1000000007;
    }
    long long InversePairsCore(vector<int> &data,vector<int> &copy,int start,int end)
    {
       if(start==end)
          {
            copy[start]=data[start];
            return 0;
          }
       int length=(end-start)/2;
       long long left=InversePairsCore(copy,data,start,start+length);
       long long right=InversePairsCore(copy,data,start+length+1,end); 
        
       int i=start+length;
       int j=end;
       int indexcopy=end;
       long long count=0;
       while(i>=start&&j>=start+length+1)
          {
             if(data[i]>data[j])
                {
                  copy[indexcopy--]=data[i--];
                  count=count+j-start-length;          //count=count+j-(start+length+1)+1;
                }
             else
                {
                  copy[indexcopy--]=data[j--];
                }          
          }
       for(;i>=start;i--)
           copy[indexcopy--]=data[i];
       for(;j>=start+length+1;j--)
           copy[indexcopy--]=data[j];       
       return left+right+count;
    }
};
```


