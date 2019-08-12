---
title: Leetcode 位操作相关整理
date: 2018-08-11 15:46:16
tags: [算法, 总结, Leetcode, OJ]
---
> 这里总结一些leetcode上比较经典的位操作例题与思路，分类和题目正在更新...
<!-- more -->
## 四则运算

### Pow(x, n)
> [Leetcode 50](https://leetcode.com/problems/powx-n/description/)

Implement pow(x, n), which calculates x raised to the power n (x<sup>n</sup>).

```
Input: 2.00000, 10
Output: 1024.00000
```
#### 代码
``` cpp
class Solution {
public:
    double myPow(double x, int n) {
        double ret = 1.0;
        // 注意要整型溢出
        long N = abs((long) n);
        while (N) {
            if (N & 1) ret *= x;
            N >>= 1;
            x *= x;
        }
        return n > 0 ? ret : 1 / ret;
    }
};
```
## Divide Two Integers
> [Leetcode 29](https://leetcode.com/problems/divide-two-integers/description/)

Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero.

Example 1:
```
Input: dividend = 10, divisor = 3
Output: 3
```

Example 2:
```
Input: dividend = 7, divisor = -3
Output: -2
```

``` cpp
// 第一种思路是使用log
class Solution {
public:
    int divide(int dividend, int divisor) {
        if (!divisor || (dividend == INT_MIN && divisor == -1)) return INT_MAX;
        double t1 = log(fabs(dividend));
        double t2 = log(fabs(divisor));
        long long ret = exp(t1 - t2);
        int syn = (dividend > 0) ^ (divisor > 0);
        return syn > 0 ? -ret : ret;
    }
};
```


``` cpp
// 第二种思路是dividend = 2^k1*divisor + 2^k2*divisor + ...
class Solution {
public:
    int divide(int dividend, int divisor) {
        if (!divisor || (dividend == INT_MIN && divisor == -1)) return INT_MAX;
        long long ret = 0;
        long long m = abs((long long)dividend);
        long long n = abs((long long)divisor);
        int syn = (dividend > 0) ^ (divisor > 0) ? -1 : 1;
        while (m >= n) {
            long long s = n, ans = 1;
            while (m >= (s << 1)) {
                s <<= 1;
                ans <<= 1;
            }
            m -= s;
            ret += ans;
        }
        return syn * ret;
    }
};
```


### Sum of Two Integers
> [Leetcode 371](https://leetcode.com/problems/sum-of-two-integers/description/)

Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.

Given a = 1 and b = 2, return 3.

#### 代码
``` cpp
class Solution {
public:
    int getSum(int a, int b) {
        while (b) {
            // 加法器原理
            int temp = a ^ b;
            b = (a & b) << 1;
            a = temp;
        }
        return a;
    }
};
```



## 前缀相关

### Power of Two
> [Leetcode 231](https://leetcode.com/problems/power-of-two/description/)

Given an integer, write a function to determine if it is a power of two.
```
Input: 1
Output: true 
Explanation: 20 = 1
```

#### 代码
``` cpp
class Solution {
public:
    bool isPowerOfTwo(int n) {
        if (n <= 0) return false;
        return !(n & (n - 1));
    }
};
```



### Bitwise AND of Numbers Range
> [Leetcode 201](https://leetcode.com/problems/bitwise-and-of-numbers-range/description/)

Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.

```
Input: [5,7]
Output: 4

Input: [0,1]
Output: 0
```

#### 代码
``` cpp
class Solution {
public:
    int rangeBitwiseAnd(int m, int n) {
        // 看相同的前缀
        return n == m ? m : (rangeBitwiseAnd(m / 2, n / 2) << 1);
    }
};
```



### Number Complement
> [Leetcode 476](https://leetcode.com/problems/number-complement/description/)

Given a positive integer, output its complement number. The complement strategy is to flip the bits of its binary representation.

``` 
Input: 5
Output: 2
Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010. So you need to output 2.
```

#### 代码
``` cpp
class Solution {
public:
    int findComplement(int num) {
        return ~num & ((1 << (int)log2(num)) - 1);
    }
};
```




## 位个数和位移动操作

### Number of 1 Bits
> [Leetcode 191](https://leetcode.com/problems/number-of-1-bits/description/)

Write a function that takes an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

#### 代码
``` cpp
class Solution {
public:
    int hammingWeight(uint32_t n) {
        // 此题在剑指offer也有
        n = ((n & 0xAAAAAAAA) >> 1) + (n & 0x55555555);
        n = ((n & 0xCCCCCCCC) >> 2) + (n & 0x33333333);
        n = ((n & 0xF0F0F0F0) >> 4) + (n & 0x0F0F0F0F);
        n = ((n & 0xFF00FF00) >> 8) + (n & 0x00FF00FF);
        n = ((n & 0xFFFF0000) >> 16) + (n & 0x0000FFFF);
        return n;
    }
};
```




### Reverse Bits
> [Leetcode 190](https://leetcode.com/problems/reverse-bits/description/)

Reverse bits of a given 32 bits unsigned integer.

```
Input: 43261596
Output: 964176192
Explanation: 43261596 represented in binary as 00000010100101000001111010011100, 
             return 964176192 represented in binary as 00111001011110000010100101000000.
```

``` cpp
class Solution {
public:
    uint32_t reverseBits(uint32_t n) {
        n = ((n & 0xAAAAAAAA) >> 1) | ((n & 0x55555555) << 1);
        n = ((n & 0xCCCCCCCC) >> 2) | ((n & 0x33333333) << 2);
        n = ((n & 0xF0F0F0F0) >> 4) | ((n & 0x0F0F0F0F) << 4);
        n = ((n & 0xFF00FF00) >> 8) | ((n & 0x00FF00FF) << 8);
        n = ((n & 0xFFFF0000) >> 16) | ((n & 0x0000FFFF) << 16);
        return n;
    }
};
```



### Total Hamming Distance
> [Leetcode 477](https://leetcode.com/problems/total-hamming-distance/description/)

The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

Now your job is to find the total Hamming distance between all pairs of the given numbers.

```
Input: 4, 14, 2

Output: 6

Explanation: In binary representation, the 4 is 0100, 14 is 1110, and 2 is 0010 (just
showing the four bits relevant in this case). So the answer will be:
HammingDistance(4, 14) + HammingDistance(4, 2) + HammingDistance(14, 2) = 2 + 2 + 2 = 6.
```

#### 代码
``` cpp
class Solution {
public:
    int totalHammingDistance(vector<int>& nums) {
        // 此题主要思想是将每一位分开计算
        int ret = 0, n = nums.size();
        for (int i = 0; i < 31; i++) {
            int cnt = 0;
            for (auto j : nums) {
                cnt += (j >> i) & 1;
            }
            ret += cnt * (n - cnt);
        }
        return ret;
    }
};
```



## 重复使用抑或操作

在同一个整型上重复偶数次抑或同一个数会导致该数消失，可以使用抑或操作进行一些变化或引导一些变化

### Missing Number
> [Leetcode 268](https://leetcode.com/problems/missing-number/description/)

Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.
```
Example 1:
Input: [3,0,1]
Output: 2

Example 2:
Input: [9,6,4,2,3,5,7,0,1]
Output: 8
```

#### 代码
``` cpp
class Solution {
public:
    // 从1到n都被抑或一遍，然后再被覆盖一遍就会消失，被抑或一遍的就会剩下。
    int missingNumber(vector<int>& nums) {
        int n = nums.size();
        int ans = n;
        for (int i = 0; i < n; i++) {
            ans ^= i;
            ans ^= nums[i];
        }
        return ans;
    }
};
```




### Single Number
> [Leetcode 136](https://leetcode.com/problems/single-number/description/)

Given a non-empty array of integers, every element appears twice except for one. Find that single one.
```
Example 1:
Input: [2,2,1]
Output: 1

Example 2:
Input: [4,1,2,1,2]
Output: 4
```

#### 代码
``` cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int ans = 0;
        for (auto i : nums) {
            ans ^= i;
        }
        return ans;
    }
};
```




### Single Number II
> [Leetcode 137](https://leetcode.com/problems/single-number-ii/description/)

Given a non-empty array of integers, every element appears three times except for one, which appears exactly once. Find that single one.

```
Example 1:
Input: [2,2,3,2]
Output: 3

Example 2:
Input: [0,1,0,1,0,1,99]
Output: 99
```

#### 代码
``` cpp
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        /*
        此题比较难理解，核心思想是利用状态机将每一位经过抑或操作后按照一定的条件变化
        
        例如用三位数表示一位数分别经过1,2,3次抑或后的状态，即001->010->100
        因为除了一个数都出现三次，所以只需要用两个数保存状态即可
        
        通过总结规律发现，[0][1]->[1][0]->[0][0]，
        two保存所有经过两次抑或后的数的结果，one保存经过一次抑或操作后的数的结果
        two一次由抑或之后和~one做与就可以完成状态的转换，one也是一样的
        
        因为只有一个数出现一次，所以one就是最终结果。
        */
        int one = 0, two = 0;
        for (auto i : nums) {
            one = ~two & (i ^ one);
            two = ~one & (i ^ two);
        }
        return one;
    }
};
```



### Single Number III
> [Leetcode 260](https://leetcode.com/problems/single-number-iii/description/)

Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once.

```
Example:

Input:  [1,2,1,3,2,5]
Output: [3,5]
```

#### 代码
``` cpp
class Solution {
public:
    vector<int> singleNumber(vector<int>& nums) {
        int ans = 0;
        for (auto i : nums) {
            ans ^= i;
        }
        // 此步比较重要，选取一位为1的数，说明该位出现过一次
        int m = (ans & (ans - 1)) ^ ans;
        int a = 0, b = 0;
        for (auto i : nums) {
            if (i & m) a ^= i;
            else b ^= i;
        }
        return vector<int>{a, b};
    }
};
```



## 与位操作相关的数学题

与位操作相关的数学题通常需要找规律或者先证明，此部分也会不断补充...

### Integer Replacement
> [Leetcode 397](https://leetcode.com/problems/integer-replacement/description/)

```
Given a positive integer n and you can do operations as follow:

If n is even, replace n with n/2.
If n is odd, you can replace n with either n + 1 or n - 1.
What is the minimum number of replacements needed for n to become 1?

Example 1:
Input:
8
Output:
3
Explanation:
8 -> 4 -> 2 -> 1

Example 2:
Input:
7
Output:
4
Explanation:
7 -> 8 -> 4 -> 2 -> 1
or
7 -> 6 -> 3 -> 2 -> 1
```

#### 代码
``` cpp
class Solution {
public:
    int integerReplacement(int N) {
        long long n = N;
        int ret = 0;
        while (n != 1) {
            ret ++;
            if (n & 1) {
                if (n & 2 && n != 3) n++;
                else n --;
            }
            else n >>= 1;
        }
        return ret;
    }
};
```





