---
title: 剑指offer部分题目思路总结
date: 2018-06-13 23:21:30
tags: [算法, OJ, 总结]
---

> 这部分主要是在[牛客网](https://www.nowcoder.com/ta/coding-interviews?page=1)上进行的验证，部分简单的题目没有进行总结，如果有别的思路，欢迎[联系我](http://wuzequn.com/About)进行交流。

<!-- more -->
## 变态跳台阶

> 一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

可以发现：则f(1)=1,f(2)=2,f(3)=4,f(4)=8,我们隐约感觉到f(n)=2^(n-1)，但是需要证明下，同样根据我们根据上篇文章中跳台阶的思路，可以得到f(n)=f(n-1)+f(n-2)+....+f(1)+1,而f(n-1)=f(n-2)+....+f(1)+1,两个式子相减，得到f(n) = 2f(n-1),很明显可以得到f(n)=2^(n-1)。

``` cpp
class Solution {
public:
    int jumpFloorII(int number) {
        return 1 << (number - 1);
    }
};

```

## 矩形覆盖
> 我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

``` cpp
class Solution {
public:
    int rectCover(int number) {
        if (number <= 2) return number;
        int pre = 1, cur = 2;
        for (int i = 3; i <= number; i++) {
            cur += pre;
            pre = cur - pre;
        }
        return cur;
    }
};
```

## 二进制中1的个数
> 输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

``` cpp
class Solution {
public:
     int  NumberOf1(int n) {
         n = ((n & 0xAAAAAAAA) >> 1) + (n & 0x55555555);
         n = ((n & 0xCCCCCCCC) >> 2) + (n & 0x33333333);
         n = ((n & 0xF0F0F0F0) >> 4) + (n & 0x0F0F0F0F);
         n = ((n & 0xFF00FF00) >> 8) + (n & 0x00FF00FF);
         n = ((n & 0xFFFF0000) >> 16) + (n & 0x0000FFFF);
         return n;
     }
};
```

## 浮点数快速幂
> 给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

``` cpp
class Solution {
public:
    double Power(double base, int exponent) {
        int syn = exponent > 0 ? 1 : -1;
        double ret = 1.0;
        long long N = abs((long long) exponent); // 注意可能取负溢出
        while (N) {
            if (N & 1) ret *= base;
            base *= base;
            N >>= 1;
        }
        return syn == 1 ? ret : 1/ ret;
    }
};
```
## 调整数组顺序使奇数位于偶数前面（冒泡排序相关）
> 输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

``` cpp
class Solution {
public:
    // 排序算法的另外一种运用，因为要使用稳定的排序，所以可以利用冒泡排序
    // 冒泡排序的两种写法：https://blog.csdn.net/shuaizai88/article/details/73250615
    
    // 第一种：
    /*
    // 向下沉
    void reOrderArray(vector<int> &nums) {
        int n = nums.size();
        for (int i = 0; i < n - 1; i++) {
            for (int j = 0;j < n - 1 - i; j++) {
                if (nums[j] % 2 == 0 && nums[j + 1] % 2)
                    swap(nums[j], nums[j + 1]);
            }
        }
    }
    */
    // 第二种：向上飘
    void reOrderArray(vector<int> &nums) {
        int n = nums.size();
        for (int i = 0; i < n; i ++) {
            for (int j = n - 1;j > i; j --) {
                if (nums[j] % 2 && nums[j - 1] % 2 == 0)
                    swap(nums[j], nums[j - 1]);
            }
        }
        
    }
};
```
如果要是不要求保证原始顺序不变，就用双指针
``` cpp
#include <iostream>
#include <vector>

using namespace std;

void helper(vector<int>& nums) {
    int n = nums.size();
    int left = 0, right = n - 1;
    while (left < right) {
        while (left < right && nums[left] % 2) left++;
        while (left < right && nums[right] % 2 == 0) right --;
        if (left < right) {
            swap(nums[left], nums[right]);
            left ++, right --;
        }
    }
}

int main() {
    int n;
    cin >> n;
    vector<int> nums(n, 0);
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }
    helper(nums);
    for (auto i : nums) {
        cout << i << " ";
    }
    cout << endl;
    return 0;
}
```

## 翻转单链表
> 输入一个链表，反转链表后，输出链表的所有元素。

``` cpp
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
            val(x), next(NULL) {
    }
};*/
class Solution {
public:
    // 递归方法 空间O(n)
    ListNode* ReverseList(ListNode* pHead) {
        if (!pHead || !pHead->next) return pHead;
        ListNode* ret = ReverseList(pHead->next);
        pHead->next->next = pHead;
        pHead->next = NULL;
        return ret;
    }
    // 迭代 空间O(1)
    ListNode* ReverseList(ListNode* pHead) {
        if (!pHead || !pHead->next) return pHead;
        ListNode* cur = pHead;
        ListNode* ret = NULL;
        while (cur) {
            ListNode* temp = cur->next;
            cur->next = ret;
            ret = cur;
            cur = temp;
        }
        pHead->next = NULL;
        return ret;
    }
};
```

## 树的子结构
> 输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

``` cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};*/
class Solution {
public:
    bool helper(TreeNode* A, TreeNode* B) {
        if (B == NULL) return true;
        if (A == NULL) return false;
        if (A->val == B->val) {
            return helper(A->left, B->left) && helper(A->right, B->right);
        }
        else return false;
    }
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2) {
        if (pRoot1 == NULL || pRoot2 == NULL) return false;
        return helper(pRoot1, pRoot2) || 
            HasSubtree(pRoot1->left, pRoot2) || HasSubtree(pRoot1->right, pRoot2);
    }
};
```

## 二叉树中和为某个值的路径
> 输入一颗二叉树和一个整数，打印出二叉树中结点值的和为输入整数的所有路径。路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。

``` cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};*/
class Solution {
public:
    void helper(vector<vector<int>>& ret, vector<int> ans, int sum, TreeNode* root, int target) {
        if (!root) return;
        sum += root->val;
        ans.push_back(root->val);
        if (!root->left && !root->right && sum == target) {
            ret.push_back(ans);
        }
        helper(ret, ans, sum, root->left, target);
        helper(ret, ans, sum, root->right, target);
    }
    vector<vector<int> > FindPath(TreeNode* root,int expectNumber) {
        vector<vector<int>> ret;
        vector<int> ans;
        helper(ret, ans, 0, root, expectNumber);
        return ret;
    }
};
```

## 复杂链表的复制
> 输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

``` cpp
/*
struct RandomListNode {
    int label;
    struct RandomListNode *next, *random;
    RandomListNode(int x) :
            label(x), next(NULL), random(NULL) {
    }
};
*/
class Solution {
public:
    RandomListNode* Clone(RandomListNode* pHead){
        if (!pHead) return NULL;
        RandomListNode* cur = pHead;
        while (cur) {
            RandomListNode* temp = new RandomListNode(cur->label);
            temp->next = cur->next;
            cur->next = temp;
            cur = temp->next;
        }
        cur = pHead;
        while (cur) {
            RandomListNode* temp = cur->next;
            if (cur->random) {
                temp->random = cur->random->next;
            }
            cur = temp->next;
        }
        RandomListNode* ret = pHead->next;
        cur = pHead;
        RandomListNode* c1 = NULL, *c2 = NULL;
        while (cur) {
            if (!c1) {
                c1 = cur;
                c2 = cur->next;
            }
            else {
                c1->next = cur;
                c2->next = cur->next;
                c1 = c1->next;
                c2 = c2->next;
            }
            cur = cur->next->next;
            c1->next = NULL;
            c2->next = NULL;
        }
        return ret;
    }
};
```

## 二叉树与双向链表
> 输入一棵二叉搜索树，将该二叉搜索树转换成一个排序的双向链表。要求不能创建任何新的结点，只能调整树中结点指针的指向。

``` cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};*/
class Solution {
public:
    void helper(TreeNode* root, TreeNode* &left, TreeNode* &right) {
        left = right = NULL;
        TreeNode* l, *r;
        if (root->left) {
            helper(root->left, l, r);
            left = l, r->right = root, root->left = r;
        }
        else {
            left = root;
        }
        if (root->right) {
            helper(root->right, l, r);
            right = r, root->right = l, l->left = root;
        }
        else {
            right = root;
        }
        
    }
    TreeNode* Convert(TreeNode* pRootOfTree) {
        if (!pRootOfTree) return NULL;
        TreeNode* left, *right;
        helper(pRootOfTree, left, right);
        return left;
    }
};
```

## 字符串的排列
> 输入一个字符串,按字典序打印出该字符串中字符的所有排列。例如输入字符串abc,则打印出由字符a,b,c所能排列出来的所有字符串abc,acb,bac,bca,cab和cba。

``` cpp
class Solution {
public:
    void helper(string ans, int index, vector<string>& ret) {
        int len = ans.size();
        if (index == len - 1) ret.push_back(ans);
        // 因为是有顺序的，所以等于是依次把后面的数提到前面来
        // 比如：1234,2134,3124,4123
        // 所以保证了后面的有序性，从而保证了唯一性，充分利用了递归的思想，每一层只做每一层的事情，不回溯
        for (int i = index; i < len; i++) {
            if (i != index && ans[index] == ans[i]) continue;
            swap(ans[i], ans[index]);
            helper(ans, index + 1, ret);
        }
    }
    vector<string> Permutation(string str) {
        vector<string> ret;
        if (str.empty()) return ret;
        sort(str.begin(), str.end());
        helper(str, 0, ret);
        return ret;
    }
};
```

## 数字中出现次数超过一半
> 数组中有一个数字出现的次数超过数组长度的一半，请找出这个数字。例如输入一个长度为9的数组{1,2,3,2,2,2,5,4,2}。由于数字2在数组中出现了5次，超过数组长度的一半，因此输出2。如果不存在则输出0。

``` cpp
class Solution {
public:
    int MoreThanHalfNum_Solution(vector<int> numbers) {
        if (numbers.empty()) return 0;
        int ans = numbers[0];
        int count = 0;
        for (auto i : numbers) {
            if (ans == i) count ++;
            else count --;
            if (count < 0) {
                count = 0;
                ans = i;
            }
        }
        count = 0;
        for (auto i : numbers) {
            if (ans == i) count ++;
        }
        return count > numbers.size() / 2 ? ans : 0;
    }
};
```

## 最小的k个数
> 输入n个整数，找出其中最小的K个数。例如输入4,5,1,6,2,7,3,8这8个数字，则最小的4个数字是1,2,3,4,。

``` cpp
// 堆排序思路
class Solution {
public:
    void heapfy(vector<int>& nums, int index, int max) {
        int left = index * 2 + 1;
        int right = left + 1;
        int small = index;
        if (left < max && nums[left] < nums[small]) small = left;
        if (right < max && nums[right] < nums[small]) small = right;
        if (small != index) {
            swap(nums[index], nums[small]);
            heapfy(nums, small, max);
        }
    }
    void helper(vector<int>& nums, vector<int>& ret, int k) {
        int len = nums.size();
        for (int i = len / 2; i >= 0; i--) heapfy(nums, i, len);
        for (int i = len - 1; i >= 0 && ret.size() < k; i--) {
            ret.push_back(nums[0]);
            swap(nums[0], nums[i]);
            heapfy(nums, 0, i);
        }
    }
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        int len = input.size();
        vector<int> ret;
        if (k > len) return ret;
        helper(input, ret, k);
        return ret;
    }
};
```

## 连续子数组的最大和
> HZ偶尔会拿些专业问题来忽悠那些非计算机专业的同学。今天测试组开完会后,他又发话了:在古老的一维模式识别中,常常需要计算连续子向量的最大和,当向量全为正数的时候,问题很好解决。但是,如果向量中包含负数,是否应该包含某个负数,并期望旁边的正数会弥补它呢？例如:{6,-3,-2,7,-15,1,2,2},连续子向量的最大和为8(从第0个开始,到第3个为止)。你会不会被他忽悠住？(子向量的长度至少是1)

> 思路：dp[i] = dp[i - 1] + a[i] (>=0), 0 (<0)

``` cpp
class Solution {
public:
    int FindGreatestSumOfSubArray(vector<int> array) {
        if (array.empty()) return 0;
        int ret = array[0], ans = array[0];
        for (int i = 1; i < array.size(); i++) {
            ans += array[i];
            ret = max(ret, ans);
            if (ans < 0) ans = 0;
        }
        return ret;
    }
};
```
## 整数中1出现的次数(*)
> 求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？为此他特别数了一下1~13中包含1的数字有1、10、11、12、13因此共出现6次,但是对于后面问题他就没辙了。ACMer希望你们帮帮他,并把问题更加普遍化,可以很快的求出任意非负整数区间中1出现的次数。

``` cpp
class Solution {
public:
    int NumberOf1Between1AndN_Solution(int n) {
        int count = 0;
        int len = floor(log10(n)) + 1;
        char str[100];
        sprintf(str, "%d", n);
        for (int i = 0; i < len; i++) {
            int temp = str[len - 1 - i] - '0';
            // 每次加的是该位为1的一共有多少个数
            if (temp == 0) count += n / (int)pow(10, i + 1) * pow(10, i);
            // 比如123'0'1, 就有123*10个十位为1的数, 从00010到12210一共有123*10个数
            else if (temp == 1) count += n / (int)pow(10, i + 1) * pow(10, i) + n % (int)pow(10, i) + 1;
            // 比如123'1'1, 从0到12210就有123*10个数，大于12310就有12311 % 10 = 1个数，再加上12311这一个
            else count += (n / (int)pow(10, i + 1) + 1) * pow(10, i);
            // 比如123'4'1, 就有124*10个十位为1的数，例如0-12341共124*10十位为1的数
        }
        return count;
    }
};
```


## 把数组排成最小的数
> 输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

``` cpp
class Solution {
public:
    bool static cmp(const int& a, const int& b) {
        return to_string(a) + to_string(b) < to_string(b) + to_string(a);
    }
    string PrintMinNumber(vector<int> numbers) {
        string ret;
        if (numbers.empty()) return ret;
        sort(numbers.begin(), numbers.end(), cmp);
        for (auto i : numbers) {
            ret += to_string(i);
        }
        return ret;
    }
};
```

## 丑数
> 把只包含因子2、3和5的数称作丑数（Ugly Number）。例如6、8都是丑数，但14不是，因为它包含因子7。 习惯上我们把1当做是第一个丑数。求按从小到大的顺序的第N个丑数。

``` cpp
class Solution {
public:
    int GetUglyNumber_Solution(int index) {
        if (index <= 0) return false;
        if (index == 1) return true;
        vector<int> dp(index, 1);
        int t2 = 0, t3 = 0, t5 = 0;
        for (int i = 1; i < index; i++) {
            dp[i] = min(dp[t2] * 2, min(dp[t3] * 3, dp[t5] * 5));
            // 这里就是标明没有放入数组的最小的乘以(2,3,5)数的角标
            if (dp[i] == dp[t2] * 2) t2 ++;
            if (dp[i] == dp[t3] * 3) t3 ++;
            if (dp[i] == dp[t5] * 5) t5 ++;
        }
        return dp[index - 1];
    }
};
```

## 数组中的逆序对(*)
> 在数组中的两个数字，如果前面一个数字大于后面的数字，则这两个数字组成一个逆序对。输入一个数组,求出这个数组中的逆序对的总数P。并将P对1000000007取模的结果输出。 即输出P%1000000007

思路1：字典树和二叉搜索树的思路，记录左右子树分别有多少节点和自己落了多少个数，落到左子树说明右边和根节点都是自己的逆序对，最差情况下是O(n^2)。

``` cpp
struct Node {
    int val;
    struct Node* left;
    struct Node* right;
    int lcount;
    int rcount;
    int cnt;
    Node(int x): val(x), left(NULL), right(NULL), lcount(0), rcount(0), cnt(1) {}
};
class Solution {
public:
    long long MOD = 1000000007;
    long long ret = 0;
    void helper(Node* root, int val) {
        if (!root) return;
        int temp = root->val;
        if (temp == val) root->cnt ++;
        else if (temp > val) {
            ret = (((ret % MOD + root->rcount) % MOD) + root->cnt) % MOD;
            root->lcount ++;
            if (root->left)
                helper(root->left, val);
            else 
                root->left = new Node(val);
        } 
        else {
            root-> rcount ++;
            if (root->right) helper(root->right, val);
            else root->right = new Node(val);
        }
    }
    int InversePairs(vector<int> data) {
        if(data.size()<=1)  return 0;
        Node* root = new Node(data[0]);
        for (int i = 1; i < data.size(); i++) 
            helper(root, data[i]);
        return ret;
    }
};
```
思路2：归并排

``` cpp
class Solution {
public:
    long long MOD = 1000000007;
    int InversePairs(vector<int> data) {
        if(data.size()<=1)  return 0;
         
        vector<int>  copy(data);
        return InversePairsCore(data,copy,0,data.size()-1);
    }
private:
    int InversePairsCore(vector<int> &data,vector<int> &copy, int begin, int end)
    {//合并data的两半段到辅助数组copy中有序
        if(begin==end)
        {
            copy[end]=data[end];
            return 0;
        }
        else
        {
            int mid=begin+(end-begin)/2;
             
            int left=InversePairsCore(copy,data,begin,mid);//使data的左半段有序
            int right=InversePairsCore(copy,data,mid+1,end);//使data的右半段有序
             
            int cnt=0;
            int cpIndex=end;
            int pre=mid;
            int post=end;
            //合并两个有序段，到copy数组
            while(pre>=begin && post>=mid+1)
            {
                if(data[pre]>data[post])//每次比较的是两个有序序列
                {
                    cnt=(cnt+(post-mid-1+1)) % MOD;
                    copy[cpIndex--]=data[pre];
                    pre--;
                }
                else
                {
                    copy[cpIndex--]=data[post];
                    post--;
                }
            }
             
            for(;pre>=begin;--pre)
                copy[cpIndex--]=data[pre];
            for(;post>=mid+1;--post)
                copy[cpIndex--]=data[post];
             
            return ((left+right)%MOD+cnt) % MOD;
        }
    }
};
```

## 数字在排序数组中的次数
> 统计一个数字在排序数组中出现的次数。

> leetcode中的[search for a range](https://leetcode.com/problems/search-for-a-range/description/)
> 牛客网的判定稍微有问题, 最后的返回值改成left - low + 1可以过，但是实际是有错误的。

``` cpp
class Solution {
public:
    int GetNumberOfK(vector<int> data ,int k) {
        int len = data.size();
        if (len == 0) return 0;
        int left = 0, right = len - 1;
        while (left < right) {
            int mid = left + ((right - left) >> 1);
            // 手推，左端不相等向右移
            if (data[mid] < k) left = mid + 1;
            // 小于等于更新右边界
            else right = mid;
            // 因为有left = mid + 1，所以不会出现left = right - 1的时候left不更新的情况。
        }
        if (data[left] != k) return 0;
        right = len - 1;
        int low = left, ret;
        while (left < right) {
            // 手推，大于的时候右边界向左移
            // 等于的时候保存当前值，然后将左值向右移
            int mid = left + ((right - left) >> 1);
            if (data[mid] > k) right = mid - 1, ret = mid;
            else left = mid + 1;
        }
        return ret - low + 1;
        
    }
};
```


## singel number iii
> 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。


``` cpp
class Solution {
public:
    void FindNumsAppearOnce(vector<int> data,int* num1,int *num2) {
        int ans = 0;
        for (auto i : data) {
            ans ^= i;
        }
        // 此处位操作比较关键
        int c = ans ^ (ans & (ans - 1));
        *num1 = 0, * num2 = 0;
        for (auto i : data) {
            if (i & c) *num1 ^= i;
            else *num2 ^= i;
        }
    }
};
```

## 和为s的连续正数序列
> 小明很喜欢数学,有一天他在做数学作业时,要求计算出9~16的和,他马上就写出了正确答案是100。但是他并不满足于此,他在想究竟有多少种连续的正数序列的和为100(至少包括两个数)。没多久,他就得到另一组连续正数和为100的序列:18,19,20,21,22。现在把问题交给你,你能不能也很快的找出所有和为S的连续正数序列? Good Luck!

> 这道题的思路可以参考[leetcode 523](523. Continuous Subarray Sum)，也可以[参考](https://www.cnblogs.com/felixfang/articles/3577624.html)

``` cpp
class Solution {
public:
    vector<vector<int> > FindContinuousSequence(int sum) {
        vector<vector<int>> ret;
        // 双指针法
        if (sum < 3) return ret;
        int ans = 3;
        int left = 1, right = 2;
        while (left < ((sum + 1) >> 1) && right < sum) {
            while (ans > sum) {
                ans -= left;
                left ++;
            }
            if (ans == sum) {
                vector<int> temp;
                for (int i = left; i <= right; i++)
                    temp.push_back(i);
                ret.push_back(temp);
            }
            right ++;
            ans += right;
        }
        return ret;
    }
};


```

我的思路比较数理一点，用sum表示要求的和，比如sum为15的时候，7，8满足条件，7，8之所以满足，是因为 15/2 = 7.5，所以正好左右各取一个数：7和8，就使得和为15。

4,5,6 之所以满足条件，是因为15/3 = 5，正好5可以放在中间，左右再拿一个4和6，所以满足。

因此，对于sum，如果我们想确定它有没有长度为n的连续序列使得这个序列的和等于sum，我们只要算算sum%n，若n是奇数，sum%n == 0，那么就意味着存在这样的序列。而且这个序列的中间那个数就是 sum/n；若n是偶数，sum%n == n/2，也就是说sum除以n的结果是一个以 .5 结尾的数，<b>余数是除数的一半</b>, 那么就意味着这样的序列存在，向两边各延伸n/2就是答案。

这种思路的代码会更简单，但是适用范围很窄，如果把可选的数字换成只能从一个递增数组中选择，就只能用窗口思想了。

``` cpp
class Solution {
public:
    void helper(int left, int n, vector<vector<int>>& ret) {
        vector<int> temp;
        for (int i = 0; i < n; i++) {
            temp.push_back(i + left);
        }
        ret.push_back(temp);
    }
    vector<vector<int> > FindContinuousSequence(int sum) {
        vector<vector<int>> ret;
        if (sum < 3) return ret;
        for(int i = 2;i * i <= sum * 2; i++){
             if (((i & 1) && sum % i == 0) || (sum % i) * 2 == i) {
                 int start = (sum / i) - (i - 1) / 2;
                 helper(start, i, ret);
             }    
        }
        reverse(ret.begin(), ret.end());
        return ret;
    }
};
```
## 左旋转字符串
> 汇编语言中有一种移位指令叫做循环左移（ROL），现在有个简单的任务，就是用字符串模拟这个指令的运算结果。对于一个给定的字符序列S，请你把其循环左移K位后的序列输出。例如，字符序列S=”abcXYZdef”,要求输出循环左移3位后的结果，即“XYZdefabc”。是不是很简单？OK，搞定它！

``` cpp
class Solution {
public:
    string LeftRotateString(string str, int n) {
        int len = str.size();
        if (!len) return "";
        n = (n + len) % len;
        reverse(str.begin(), str.begin() + n);
        reverse(str.begin() + n, str.end());
        reverse(str.begin(), str.end());
        return str;
    }
};
```

## 翻转单词顺序
> 牛客最近来了一个新员工Fish，每天早晨总是会拿着一本英文杂志，写些句子在本子上。同事Cat对Fish写的内容颇感兴趣，有一天他向Fish借来翻看，但却读不懂它的意思。例如，“student. a am I”。后来才意识到，这家伙原来把句子单词的顺序翻转了，正确的句子应该是“I am a student.”。Cat对一一的翻转这些单词顺序可不在行，你能帮助他么？

``` cpp
class Solution {
public:
    string ReverseSentence(string str) {
        if (str.empty()) return str;
        reverse(str.begin(), str.end());
        int index = 0, len = str.size();
        for (int i = 0; i < len; i++) {
            if (str[i] != ' ') {
                if (index != 0) 
                    str[index++] = ' '; // 在单词前面加一个空格
                int j = i;
                while (j < len && str[j] != ' ')
                    str[index ++] = str[j++];
                reverse(str.begin() + index - (j - i), str.begin() + index);
                i = j; // 加一之后就是空格后一个
            }
        }
        // str.erase(str.begin() + storeIndex, str.end()); 
        // leetcode还要加上这句
        return str;
    }
};
```

## 扑克牌顺子
> LL今天心情特别好,因为他去买了一副扑克牌,发现里面居然有2个大王,2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！“红心A,黑桃3,小王,大王,方片5”,“Oh My God!”不是顺子.....LL不高兴了,他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。LL决定去买体育彩票啦。 现在,要求你使用这幅牌模拟上面的过程,然后告诉我们LL的运气如何。为了方便起见,你可以认为大小王是0。

``` cpp
class Solution {
public:
    bool IsContinuous( vector<int> numbers ) {
        if (numbers.empty()) return false;
        int zeros = 0;
        int gap = 0;
        int len = numbers.size();
        sort(numbers.begin(), numbers.end());
        for (int i = 0; i < len; i++) {
            if (numbers[i] == 0) zeros ++;
            if (i != 0 && numbers[i] != numbers[i - 1] + 1 && numbers[i - 1] != 0) 
                gap += numbers[i] - numbers[i - 1] - 1;
            if (i != 0 && numbers[i] != 0 && numbers[i] == numbers[i - 1])
                return false;
        }
        return zeros >= gap;
    }
};
```
## 孩子们的游戏(约瑟夫环)【着重看推导】
> 每年六一儿童节,牛客都会准备一些小礼物去看望孤儿院的小朋友,今年亦是如此。HF作为牛客的资深元老,自然也准备了一些小游戏。其中,有个游戏是这样的:首先,让小朋友们围成一个大圈。然后,他随机指定一个数m,让编号为0的小朋友开始报数。每次喊到m-1的那个小朋友要出列唱首歌,然后可以在礼品箱中任意的挑选礼物,并且不再回到圈中,从他的下一个小朋友开始,继续0...m-1报数....这样下去....直到剩下最后一个小朋友,可以不用表演,并且拿到牛客名贵的“名侦探柯南”典藏版(名额有限哦!!^_^)。请你试着想下,哪个小朋友会得到这份礼品呢？(注：小朋友的编号是从0到n-1)

> 推导过程[约瑟夫环详解](https://blog.csdn.net/tingyun_say/article/details/52343897)

``` cpp
class Solution {
public:
    int LastRemaining_Solution(int n, int m) {
        if (!n) return -1;
        int ret = 0;
        for (int i = 2; i <= n; i++) {
            ret = (ret + m) % i;
        }
        return ret;
    }
};
```

## 1+2+3+..+n
> 求1+2+3+...+n，要求不能使用乘除法、for、while、if、else、switch、case等关键字及条件判断语句（A?B:C）。

``` cpp
class Solution {
public:
    int Sum_Solution(int n) {
        int ans = n;
        // 与的短路作用，如果n==0就结束
        ans && (ans += Sum_Solution(n - 1));
        return ans;
    }
};
```
## 不用加减乘除做加法
> 写一个函数，求两个整数之和，要求在函数体内不得使用+、-、*、/四则运算符号。

首先看十进制是如何做的： 5+7=12，三步走
第一步：相加各位的值，不算进位，得到2。
第二步：计算进位值，得到10. 如果这一步的进位值为0，那么第一步得到的值就是最终结果。

第三步：重复上述两步，只是相加的值变成上述两步的得到的结果2和10，得到12。


``` cpp
class Solution {
public:
    int Add(int num1, int num2) {
        while (num2) {
            int temp = num1 ^ num2;
            num2 = (num1 & num2) << 1;
            num1 = temp;
        }
        return num1;
    }
};
```

## 数组中重复的数字
> 在一个长度为n的数组里的所有数字都在0到n-1的范围内。 数组中某些数字是重复的，但不知道有几个数字是重复的。也不知道每个数字重复几次。请找出数组中任意一个重复的数字。 例如，如果输入长度为7的数组{2,3,1,0,2,5,3}，那么对应的输出是第一个重复的数字2。

``` cpp
class Solution {
public:
    // Parameters:
    //        numbers:     an array of integers
    //        length:      the length of array numbers
    //        duplication: (Output) the duplicated number in the array number
    // Return value:       true if the input is valid, and there are some duplications in the array number
    //                     otherwise false
    bool duplicate(int numbers[], int length, int* duplication) {
        for (int i = 0; i < length; i++) {
            if (numbers[abs(numbers[i])] < 0) {
                *duplication = abs(numbers[i]);
                return true;
            }
            numbers[numbers[i]] *= -1;
        }
        return false;
    }
};
```

## 构建乘积数组
> 给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。不能使用除法。

``` cpp
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int len = A.size();
        if (len == 0) return vector<int>();
        vector<int> ret(len, 1);
        for (int i = 1; i < len; i++) {
            ret[i] = ret[i - 1] * A[i - 1];
        }
        int temp = 1;
        for (int i = len - 1; i >= 0; i--) {
            ret[i] *= temp;
            temp *= A[i];
        }
        return ret;
    }
};
```

## 删除链表中重复的节点
> 在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

``` cpp
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* deleteDuplication(ListNode* pHead) {
        if (!pHead) return NULL;
        int val = pHead->val;
        if (pHead->next) {
            if (pHead->next->val == val) {
                while (pHead && pHead->val == val) pHead = pHead->next;
                return deleteDuplication(pHead);
            }
            else {
                pHead->next = deleteDuplication(pHead->next);
                return pHead;
            }
        }
        else return pHead;
    }
};
```
## 正则匹配(重要)
> 请实现一个函数用来匹配包括'.'和'*'的正则表达式。模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。 在本题中，匹配是指字符串的所有字符匹配整个模式。例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，但是与"aa.a"和"ab*a"均不匹配

``` cpp
class Solution {
public:
    bool match(char* s, char* p) {
        int m = 0, n = 0;
        for (; s[m] != '\0'; m++);
        for (; p[n] != '\0'; n++);
        bool dp[m + 1][n + 1];
        for (int i = 0; i <= m; i++) {
            for (int j = 0; j <= n; j++)
                dp[i][j] = false;
        }
        dp[0][0] = true;
        for (int i = 1; i <= n; i++)
            // 默认所有p都合法的，所以不会越界，不然要首先检查一下pattern是否满足
            if (p[i-1] == '*' && dp[0][i-2]) dp[0][i] = true;
         
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                // 第j个字符为'*'
                if (p[j-1] == '*') {
                    // 获得第j-1字符
                    char ch = p[j-2];
                    if (ch != '.' && ch != s[i-1]) dp[i][j] = dp[i][j-2];
                    // 如果出现无法追加的情况（不相等），只能不匹配ch*
                    else dp[i][j] = (dp[i][j - 2] || dp[i - 1][j] || dp[i - 1][j - 2]);
                    // 剩下或者相等或者是.都可以追加成功
                    // 前两种为相等的情况：不匹配ch*; 继续追加；
                    // 然后只剩下.*的情况了，这种只需要看前面符不符合就可以
                }
                else {
                    if (p[j-1] == '.' || p[j-1] == s[i-1]) dp[i][j] = dp[i-1][j-1];
                }
            }
         }
         return dp[m][n];
    }
};
```

## 表示数值的字符串
> 请实现一个函数用来判断字符串是否表示数值（包括整数和小数）。例如，字符串"+100","5e2","-123","3.1416"和"-1E-16"都表示数值。 但是"12e","1a3.14","1.2.3","+-5"和"12e+4.3"都不是。

``` cpp
// 方法一
class Solution {
public:
    bool isNumeric(char* string) {
        int len = 0;
        while (string[len] != '\0') len ++;
        if (!len) return false;
        int m1 = 0, m2 = 0, m3 = 0;
        
        for (int i = 0; i < len; i++) {
            if (m3) {
                if (string[i] > '9' || string[i] < '0')
                    return false;
            }
            else if (m2) {
                if (string[i] == 'e' || string[i] == 'E') {
                    if (i == len - 1) return false;
                    char temp = string[i + 1];
                    if (temp != '+' && temp != '-' && temp < '0' && temp > '9')
                        return false;
                    if (temp == '+' || temp == '-') i ++;
                    m3 = 1;
                }
                else if (string[i] > '9' || string[i] < '0') {
                    return false;
                }
            }
            else if (m1) {
                if (string[i] == 'e' || string[i] == 'E') {
                    if (i == len - 1) return false;
                    char temp = string[i + 1];
                    if (temp != '+' && temp != '-' && temp < '0' && temp > '9')
                        return false;
                    if (temp == '+' || temp == '-') i ++;
                    m3 = 1;
                }
                else if ((string[i] > '9' || string[i] < '0') && string[i] != '.') {
                    return false;
                }
                else if (string[i] == '.') m2 = 1;
            }
            else {
                char temp = string[i];
                if (temp != '+' && temp != '-' && temp < '0' && temp > '9') return false;
                m1 = 1;
            }
        }
        return true;
    }

};

// 方法二

class Solution {
public:
    bool isNumeric(char* str) {
        // 标记符号、小数点、e是否出现过
        bool sign = false, decimal = false, hasE = false;
        for (int i = 0; i < strlen(str); i++) {
            if (str[i] == 'e' || str[i] == 'E') {
                if (i == strlen(str)-1) return false; // e后面一定要接数字
                if (hasE) return false;  // 不能同时存在两个e
                hasE = true;
            } else if (str[i] == '+' || str[i] == '-') {
                // 第二次出现+-符号，则必须紧接在e之后
                if (sign && str[i-1] != 'e' && str[i-1] != 'E') return false;
                // 第一次出现+-符号，且不是在字符串开头，则也必须紧接在e之后
                if (!sign && i > 0 && str[i-1] != 'e' && str[i-1] != 'E') return false;
                sign = true;
            } else if (str[i] == '.') {
              // e后面不能接小数点，小数点不能出现两次
                if (hasE || decimal) return false;
                decimal = true;
            } else if (str[i] < '0' || str[i] > '9') // 不合法字符
                return false;
        }
        return true;
    }
};
```

## 字符流中第一个出现的不重复的字符
> 请实现一个函数用来找出字符流中第一个只出现一次的字符。例如，当从字符流中只读出前两个字符"go"时，第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时，第一个只出现一次的字符是"l"。

``` cpp
class Solution
{
public:
    int store[256];
    int index = 0;
    Solution() {
        memset(store, -1, sizeof(store));
    }
  //Insert one char from stringstream
    void Insert(char ch) {
        if (store[ch] == -1) store[ch] = index;
        else store[ch] = -2;
        index ++;
    }
  //return the first appearence once char in current stringstream
    char FirstAppearingOnce() {
        int ret_index = INT_MAX;
        char ret = '\0';
        for (int i = 0; i < 256; i++) {
            if (store[i] >= 0 && ret_index > store[i]) {
                ret_index = store[i];
                ret = (char) i;
            }
        }
        if (ret == '\0') return '#';
        else return ret;
    }

};
```

## 链表中的入口节点
> 一个链表中包含环，请找出该链表的环的入口结点。

``` cpp
/*
struct ListNode {
    int val;
    struct ListNode *next;
    ListNode(int x) :
        val(x), next(NULL) {
    }
};
*/
class Solution {
public:
    ListNode* EntryNodeOfLoop(ListNode* pHead) {
        if (!pHead) return NULL;
        if (!pHead->next) return NULL;
        ListNode* p1 = pHead;
        ListNode* p2 = pHead;
        while (p2 != NULL && p2->next != NULL) {
            p1 = p1->next;
            p2 = p2->next->next;
            if (p1 == p2) {
                p1 = pHead;
                while (p1 != p2) {
                    p1 = p1->next;
                    p2 = p2->next;
                }
                return p1;
            }
        }
        return NULL;
    }
};
```

## 二叉树的下一个节点
> 给定一个二叉树和其中的一个结点，请找出中序遍历顺序的下一个结点并且返回。注意，树中的结点不仅包含左右子结点，同时包含指向父结点的指针。

``` cpp
/*
struct TreeLinkNode {
    int val;
    struct TreeLinkNode *left;
    struct TreeLinkNode *right;
    struct TreeLinkNode *next;
    TreeLinkNode(int x) :val(x), left(NULL), right(NULL), next(NULL) {
        
    }
};
*/
class Solution {
public:
    TreeLinkNode* GetNext(TreeLinkNode* pNode) {
        if (pNode == NULL) return NULL;
        TreeLinkNode* pre = pNode->next;
        if (pre == NULL) { // 根节点
            TreeLinkNode* temp = pNode->right;
            if (!temp) return temp;
            while (temp->left) {
                temp = temp->left;
            }
            return temp;
        }
        if (pre->left == pNode) { // 自己是左节点
            if (pNode->right == NULL)
                return pre;
            return pNode->right;
        }
        
        else { // 自己是右节点
            if (pNode->right != NULL)
                return pNode->right;
            while (pre->next != NULL) {
                TreeLinkNode* temp = pre->next;
                if (temp->left == pre)
                    return temp;
                pre = temp;
            }
        }
        return NULL;
    }
};
```
## 按之字形顺序打印二叉树

> 请实现一个函数按照之字形打印二叉树，即第一行按照从左到右的顺序打印，第二层按照从右至左的顺序打印，第三行按照从左到右的顺序打印，其他行以此类推。

``` cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
// 用栈而不是队列做bfs
public:
    vector<vector<int> > Print(TreeNode* pRoot) {
        vector<vector<int>> ret;
        if (!pRoot) return ret;
        stack<TreeNode*> cur;
        int dir = 1;
        cur.push(pRoot);
        while (true) {
            vector<int> ans;
            stack<TreeNode*> next;
            while (!cur.empty()) {
                ans.push_back(cur.top()->val);
                auto temp = cur.top();
                cur.pop();
                if (dir) {
                    if (temp->left) next.push(temp->left);
                    if (temp->right) next.push(temp->right);
                }
                else {
                    if (temp->right) next.push(temp->right);
                    if (temp->left) next.push(temp->left);
                }
            }
            ret.push_back(ans);
            if (next.empty()) break;
            dir ^= 1;
            cur = next;
        }
        return ret;
    }
    
};
```

## 序列化二叉树
> 请实现两个函数，分别用来序列化和反序列化二叉树

``` cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    void helper(TreeNode* root, vector<int>& buf) {
        if (!root) {
            buf.push_back(INT_MAX);
            return;
        }
        buf.push_back(root->val);
        helper(root->left, buf);
        helper(root->right, buf);
        
    }
    char* Serialize(TreeNode *root) {
        vector<int> buf;
        helper(root, buf);
        int len = buf.size();
        int* temp = new int[len];
        for (int i = 0; i < len; i++) {
            temp[i] = buf[i];
        }
        return (char*) temp;
    }
    TreeNode* helper1(int*& s) {
        if (*s == INT_MAX) {
            s++;
            return NULL;
        }
        TreeNode* ret = new TreeNode(*s);
        s ++;
        ret->left = helper1(s);
        ret->right = helper1(s);
        return ret;
    }
    TreeNode* Deserialize(char *str) {
        int *p = (int*) str;
        return helper1(p);
    }
};
```

## 二叉搜索树的第k个节点
> 给定一颗二叉搜索树，请找出其中的第k大的结点。例如， 5 / \ 3 7 /\ /\ 2 4 6 8 中，按结点数值大小顺序第三个结点的值为4。

``` cpp
/*
struct TreeNode {
    int val;
    struct TreeNode *left;
    struct TreeNode *right;
    TreeNode(int x) :
            val(x), left(NULL), right(NULL) {
    }
};
*/
class Solution {
public:
    TreeNode* helper(TreeNode* root, int& ans) {
        if (root) {
            auto temp = helper(root->left, ans);
            return !ans ? temp : (ans -- == 1 ? root : helper(root->right, ans));
        }
        return NULL;
        
    }
    TreeNode* KthNode(TreeNode* pRoot, int k) {
        return helper(pRoot, k);
    }

    
};
```

## 数据流中的中位数
> 如何得到一个数据流中的中位数？如果从数据流中读出奇数个数值，那么中位数就是所有数值排序之后位于中间的数值。如果从数据流中读出偶数个数值，那么中位数就是所有数值排序之后中间两个数的平均值。

``` cpp
class Solution {
public:
    priority_queue<int> small, large;
    // 优先级队列
    void Insert(int num) {
        small.push(num);
        large.push(-small.top());
        small.pop();
        if (small.size() < large.size()) {
            small.push(-large.top());
            large.pop();
        }
    }

    double GetMedian() { 
        return small.size() > large.size() ? small.top() : (small.top() - large.top()) / 2.0;
    }

};
```


## 滑动窗口最大值
> 给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。例如，如果输入数组{2,3,4,2,6,2,5,1}及滑动窗口的大小3，那么一共存在6个滑动窗口，他们的最大值分别为{4,4,6,6,6,5}； 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}， {2,3,[4,2,6],2,5,1}， {2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。

``` cpp
class Solution {
public:
    vector<int> maxInWindows(const vector<int>& num, unsigned int size)
    {
        int len = (long long)size;
        int n = num.size();
        if (!len) return vector<int>();
        deque<int> d;
        vector<int> ret;
        for (int i = 0; i < n; i++) {
            while (!d.empty() && num[d.back()] <= num[i]) d.pop_back();
            d.push_back(i);
            while (!d.empty() && d.front() <= i - len) d.pop_front();
            if (i >= len - 1) ret.push_back(num[d.front()]);
        }
        return ret;
        
    }
};
```

## 矩阵中的路径
> 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则该路径不能再进入该格子。 例如 a b c e s f c s a d e e 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。

``` cpp
class Solution {
public:
    int m, n;
    bool helper(char* matrix, const char* s, int x, int y) {
        if (*s == '\0') return true;
        if (x < 0 || x >= m || y >= n || y < 0) return false;
        char temp = matrix[x * n + y];
        if (*s != temp) return false;
        matrix[x * n + y] = '\0';
        if (helper(matrix, s + 1, x + 1, y) || 
            helper(matrix, s + 1, x, y + 1) ||
            helper(matrix, s + 1, x - 1, y) ||
            helper(matrix, s + 1, x, y - 1))
            return true;
        matrix[x * n + y] = temp;
        return false;
    }
    bool hasPath(char* matrix, int rows, int cols, char* str) {
        m = rows, n = cols;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (helper(matrix, str, i, j))
                    return true;
            }
        }
        return false;
    }


};
```


## 机器人的运动范围
> 地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？

``` cpp
class Solution {
public:
    int m, n, k;
    bool check(int x, int y) {
        int ans = 0;
        while (x) {
            ans += x % 10;
            x /= 10;
        }
        while (y) {
            ans += y % 10;
            y /=10;
        }
        return ans > k;
    }
    void helper(int x, int y, vector<vector<bool>>& visited, int& ret) {
        if (x < 0 || x >= m || y < 0 || y >= n || visited[x][y] || check(x, y)) return;
        ret ++;
        visited[x][y] = true;
        int a[4] = {0, 0, -1, 1}, b[4] = {-1, 1, 0, 0};
        for (int i = 0; i < 4; i++) {
            helper(x + a[i], y + b[i], visited, ret);
        }
    }
    int movingCount(int threshold, int rows, int cols) {
        m = rows, n = cols, k = threshold;
        if (m < 0 || n < 0) {
            return 0;
        }
        vector<vector<bool>> visited(m, vector<bool>(n, false));
        int ret = 0;
        helper(0, 0, visited, ret);
        return ret;
    }
};
```

