---
title: 【面经】HotStar部分编程算法题
date: 2019-04-21 22:37:33
tags: [算法, 总结, 面经]
---

> 本文收集网络上面试部分编程面经以供复习

<!-- more -->


## Alien Language(待付费)
> 验证地址[Leetcode 269](https://leetcode.com/problems/alien-dictionary)
``` cpp
#include <iostream>
#include <vector>
#include <unordered_set>
#include <unordered_map>
using namespace std;

string helper(vector<string>& nums) {
    int n = nums.size();
    unordered_map<char, unordered_set<char>> adj;
    unordered_map<char, int> cnt;
    unordered_set<char> s;
    for (int i = 0; i < n - 1; i++) {
        int n1 = nums[i].size(), n2 = nums[i + 1].size();
        for (int j = 0; j < n1 && j < n2; j++) {
            if (nums[i][j] != nums[i + 1][j]) {
                adj[nums[i][j]].insert(nums[i + 1][j]);
                cnt[nums[i + 1][j]] ++;
                break;
            }
        }
    }
    for (auto i : nums) {
        for (auto j : i) {
            s.insert(j);
        }
    }

    string cur;
    for (auto i : s) {
        if (cnt.find(i) == cnt.end()) cur.push_back(i);
    }

    string ret;
    while (true) {
        string next;
        for (auto i : cur) {
            ret.push_back(i);
            for (auto j : adj[i]) {
                cnt[j]--;
                if (!cnt[j]) next.push_back(j);
            }
        }
        if (next.empty()) break;
        else cur = next;
    }
    return ret;
}

int main() {
    int n;
    cin >> n;
    vector<string> nums;
    for (int i = 0; i < n; i++) {
        string s;
        cin >> s;
        nums.push_back(s);
    }

    cout << helper(nums) << endl;

    return 0;
}
```

## Days of Our Lives
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/days-of-our-lives/0)
``` cpp
#include <iostream>
#include <vector>

using namespace std;

vector<long long> helper(long long n, long long k) {
    long long ans = n / 7;
    vector<long long> ret(7, ans);
    int c = n % 7;
    for (int i = 0; i < c; i++) {
        ret[(k - 1 + i) % 7] ++;
    }
    return ret;
}


int main() {
	//code
	int t;
	cin >> t;
	for (int i = 0; i < t; i++) {
	    long long n, k;
	    cin >> n >> k;
	    auto ans = helper(n, k);
	    for (int j = 0; j < 6; j++) {
	        cout << ans[j] << " ";
	    }
	    cout << ans[6] << endl;
	}
	return 0;
}
```

## Connect Nodes at Same Level
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/connect-nodes-at-same-level/1)
``` cpp
void connect(Node *p)
{
   // Your Code Here
   if (!p) return;
   Node* pre = p;
   pre->nextRight = NULL;
   while (true) {
       while (pre && !pre->left && !pre->right) pre = pre->nextRight;
       if (!pre) return;
       auto temp = pre->left ? pre->left : pre->right;
       auto cur = temp;
       while (pre) {
           if (pre->left) cur->nextRight = pre->left, cur = cur->nextRight;
           if (pre->right) cur->nextRight = pre->right, cur = cur->nextRight;
           pre = pre->nextRight;
       }
       cur->nextRight = NULL;
       pre = temp;
   }
}
```

## Maximum difference between node and its ancestor
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/maximum-difference-between-node-and-its-ancestor/1)

``` cpp
int helper(Node* root, int& ret) {
    if (!root) return INT_MAX;
    int ans = INT_MAX;
    if (root->left) ans = min(ans, helper(root->left, ret));
    if (root->right) ans = min(ans, helper(root->right, ret));
    if (ans == INT_MAX) return root->data;
    ret = max(ret, root->data - ans);
    return min(root->data, ans);
}
int maxDiff(Node* root)
{
    // Your code here 
    if (!root) return INT_MIN;
    int ret = INT_MIN;
    helper(root, ret);
    return ret;
}
```

## Vertical Order Traversal of a Binary Tree
> 验证地址[Leetcode 987](https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/)

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
    bool static cmp(const pair<int, int>& a, const pair<int, int>& b) {
        return a.first == b.first ? a.second < b.second : a.first < b.first;
    }
    vector<vector<int>> verticalTraversal(TreeNode* root) {
        vector<vector<int>> ret;
        if (!root) return ret;
        vector<pair<TreeNode*, pair<int, int>>> cur;
        cur.push_back({root, {0, 0}});
        map<int, vector<pair<int, int>>> m;
        while (true) {
            vector<pair<TreeNode*, pair<int, int>>> next;
            for (auto i : cur) {
                m[i.second.first].push_back({i.second.second, i.first->val});
                if (i.first->left) 
                    next.push_back({i.first->left, {i.second.first - 1, i.second.second + 1}});
                if (i.first->right) 
                    next.push_back({i.first->right, {i.second.first + 1, i.second.second + 1}});
            }
            if (next.empty()) break;
            cur = next;
        }
        for (auto i : m) {
            sort(i.second.begin(), i.second.end(), cmp);
            vector<int> temp;
            for (auto j : i.second) {
                temp.push_back(j.second);
            }
            ret.push_back(temp);
        }
        return ret;
    }
};
```

## Top View of Binary Tree
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/top-view-of-binary-tree/1)

``` cpp
void topView(struct Node *root)
{
    // Your code here
    unordered_map<int, int> m;
    if (!root) return;
    vector<pair<Node*, int>> cur;
    cur.push_back({root, 0});
    while (true) {
        vector<pair<Node*, int>> next;
        for (auto i : cur) {
            if (m.find(i.second) == m.end()) {
                cout << i.first->data << " ";
                m[i.second] = i.first->data;
            }
                
            if (i.first->left) {
                next.push_back({i.first->left, i.second - 1});
            }
            if (i.first->right) {
                next.push_back({i.first->right, i.second + 1});
            }
        }
        if (next.empty()) break;
        cur = next;
    }
    
}
```

## Smallest window in a string containing all the characters of another string
> 验证地址[geeksforgeeks](https://practice.geeksforgeeks.org/problems/smallest-window-in-a-string-containing-all-the-characters-of-another-string/0)

``` cpp
#include <iostream>
#include <unordered_map>
#include <climits>
using namespace std;

string helper(string s, string p) {
    unordered_map<char, int> m;
    for (auto i : p) m[i] ++;
    int ans = m.size();
    int n= s.size();
    int i = 0, j = 0;
    int len = INT_MAX;
    string ret = "-1";
    while (i <= j && j <= n) {
        if (j < n && ans > 0) {
            if (m.find(s[j]) != m.end()) {
                m[s[j]] --;
                if (!m[s[j]]) ans --;
            }
            j++;
            
        }
        else {
            if (m.find(s[i]) != m.end()) {
                if (!m[s[i]]) ans ++;
                m[s[i]] ++;
            }
            i++;
        }
        if (!ans) {
            if (j - i < len) {
                
                len = j - i;
                ret = s.substr(i, len);
            }
        }
    }
    return ret;
}

int main() {
	//code
	int t;
	cin >> t;
	for (auto i = 0; i < t; i++) {
	    string s, p;
	    cin >> s >> p;
	    cout << helper(s, p) << endl;
	}
	return 0;
}
```

## Simplify the directory path (Unix like)
> 题目参考自[geeksforgeeks](https://www.geeksforgeeks.org/simplify-directory-path-unix-like/)
```
"/a/./"   --> means stay at the current directory 'a'
"/a/b/.." --> means jump to the parent directory
              from 'b' to 'a'
"////"    --> consecutive multiple '/' are a  valid  
              path, they are equivalent to single "/".

Input : /home/
Output : /home

Input : /a/./b/../../c/
Output : /c

Input : /a/..
Output : /

Input : /a/../
Ouput : /

Input : /../../../../../a
Ouput : /a

Input : /a/./b/./c/./d/
Ouput : /a/b/c/d

Input : /a/../.././../../.
Ouput : /

Input : /a//b//c//////d
Ouput : /a/b/c/d
```


``` cpp
// 栈版本
#include <iostream>
#include <vector>
#include <stack>

using namespace std;

string helper(string s) {
    if (s.empty()) return "/";
    if (s.back() != '/') s.push_back('/');

    stack<string> st;
    string ans;
    for (auto i : s) {
        if (i == '/') {
            if (ans == "" || ans == ".") {
                ans = "";
            }
            else if (ans == "..") {
                if (!st.empty()) st.pop();
                ans = "";
            }
            else {
                st.push(ans);
                ans = "";
            }
        }
        else {
            ans.push_back(i);
        }
    }
    string ret;
    while (!st.empty()) {
        ret = "/" + st.top() + ret;
        st.pop();
    }
    return ret.empty() ? "/" : ret;
}

int main() {
    while (true) {
        string s;
        cin >> s;
        cout << helper(s) << endl;
    }
    return 0;
}
```

``` cpp
// 原地压缩版
#include <iostream>


using namespace std;

int helper(string& s) {
    int n = s.size();
    int index = 0;
    string ans;
    for (int i = 0; i <= n; i++) {
        if (i == n || s[i] == '/') {
            if (ans == ".") {
                while (index > 0 && s[index - 1] != '/') index--;
                index --;
            }
            if (ans == "..") {
                while (index > 0 && s[index - 1] != '/') index--;
                if (index) index --;
                while (index > 0 && s[index - 1] != '/') index--;
                if (index) index --;
            }
            
            ans = "";
        }
        else {
            s[index++] = '/';
            while (i < n && s[i] != '/') {
                ans.push_back(s[i]);
                s[index ++] = s[i++];
            }
            i --;
        }
    }
    if (!index) {
        s = "/";
        return 1;
    }
    return index;
}

int main() {
    while (true) {
        string s;
        cin >> s;
        int len = helper(s);
        cout << s.substr(0, len) << endl;
    }
    return 0;
}
```




