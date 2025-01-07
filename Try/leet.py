# Definition for singly-linked list.
from typing import Optional
from collections import deque
from collections import defaultdict

# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
#
# def create_tree_from_list(lst):
#     if not lst:
#         return None
#     root = TreeNode(lst[0])
#     queue = deque([root])
#     i = 1
#     while i < len(lst):
#         current = queue.popleft()
#         if lst[i] is not None:
#             current.left = TreeNode(lst[i])
#             queue.append(current.left)
#         i += 1
#         if i < len(lst) and lst[i] is not None:
#             current.right = TreeNode(lst[i])
#             queue.append(current.right)
#         i += 1
#     return root
#
#
# class Solution:
#     def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
#
#         def f(node, ans:list):
#             if node is None:
#                 return []
#             if node.left is None and node.right is None:
#                 ans.append(node.val)
#             f(node.left, ans)
#             f(node.right, ans)
#             return ans
#
#         leaf1 = []
#         leaf2 = []
#         print(f(root1, leaf1))
#         print(f(root2, leaf2))
#         return f(root1, leaf1) == f(root2, leaf2)
#
# if __name__ == '__main__':
#     s = Solution()
#     root1 = create_tree_from_list([3,5,1,6,2,9,8,None,None,7,4])
#     root2 = create_tree_from_list([3,5,1,6,7,4,2,None,None,None,None,None,None,9,8])
#     print(s.leafSimilar(root1, root2))



# class Solution:
#     def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
#         ans = 0
#         cnt = defaultdict(int)
#         cnt[0] = 1
#
#         def dfs(node: Optional[TreeNode], s: int) -> None:
#             if node is None:
#                 return
#             nonlocal ans
#             s += node.val
#             ans += cnt[s - targetSum]
#             cnt[s] += 1
#             dfs(node.left, s)
#             dfs(node.right, s)
#             cnt[s] -= 1  # 恢复现场
#
#         dfs(root, 0)
#         return ans
#
# root = create_tree_from_list([10,5,-3,3,2,None,11,3,-2,None,1])
# s = Solution()
# s.pathSum(root,8)



# num_pos = {'a': 3, 'b': 2, 'c': 2}
# for complement in num_pos:
#     print(complement)
#     print(num_pos[complement])
# TypeError: unhashable type: 'list'

# # 创建一个字符串列表
# str_list = ['H', 'e', 'l', 'l', 'o']
#
# # 使用空字符串作为分隔符将列表元素连接成一个字符串
# result = "c".join(str_list)
#
# print(result)  # 输出: Hello



# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def levelOrder(self, root: Optional[TreeNode]) -> list[list[int]]:
#         if root is None:
#             return []
#         ans = []
#         # 初始化当前, [3,9,20,null,null,15,7]型了
#         cur = [root]
#         print(cur)
#         while cur:
#             #初始化下一层
#             nxt = []
#             #初始化下一层的值
#             vals = []
#             for node in cur:
#                 #把当前的值加到数组
#                 vals.append(node.val)
#                 if node.left:nxt.append(node.left)
#                 if node.right:nxt.append(node.right)
#
#             #换到下一层
#             cur = nxt
#             ans.append(vals)
#         return ans
#
# root = TreeNode(3, TreeNode(9,None,None), TreeNode(20, TreeNode(15), TreeNode(7)))
# s = Solution()
# print(s.levelOrder(root))

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def maxLevelSum(self, root: Optional[TreeNode]) -> int:
#         if root is None:
#             return 0
#         cur = [root]
#         ans = 0
#         max_sum = float('-inf')
#         level = 0
#         while cur:
#             nxt = []
#             vals = []
#             for node in cur:
#                 vals.append(node.val)
#                 if node.left: nxt.append(node.left)
#                 if node.right: nxt.append(node.right)
#             cur = nxt
#             level += 1
#             if sum(vals) > max_sum:
#                 max_sum = sum(vals)
#                 ans = level
#         return ans
# # root = TreeNode(1, TreeNode(7,TreeNode(7),TreeNode(-8)), TreeNode(0, None, None))
# # root = [989,null,10250,98693,-89388,null,null,null,-32127]
# # root = [-100,-200,-300,-20,-5,-10,null]
# root = TreeNode(-100, TreeNode(-200, TreeNode(-20), TreeNode(-5)), TreeNode(-300, TreeNode(-10)))
# # root = TreeNode(989, None, TreeNode(10250, TreeNode(98693), TreeNode(-89388,None,TreeNode(-32127))))
#
# s = Solution()
# print(s.maxLevelSum(root))

# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
# class Solution:
#     def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
#         if root is None:
#             return None
#         if root.val>key:
#             #self.deleteNode(root.left,key) #这样写不行
#             root.left = self.deleteNode(root.left,key)
#         elif root.val<key:
#             #self.deleteNode(root.right,key) #这样写不行
#             root.right = self.deleteNode(root.right,key)
#         #找到了
#         else:
#             #叶子节点
#             if root.left is None and root.right is None:
#                 root = None
#                 return root
#             #key只有左子树
#             elif root.left is not None and root.right is None:
#                 return root.left
#             #key只有右子树
#             elif root.right is not None and root.left is  None:
#                 return root.right
#             #左右子树都有,将 key 位置的值替换成右子树的最小值
#             else:
#                 node = root.right
#                 while node.left:
#                     node=node.left
#                 root.val = node.val
#                 #删除右子树最小值
#                 root.right = self.deleteNode(root.right,node.val)
#         return root
#
#     def dfs(self, node):
#         if node is not None:
#             print(node.val, end=" ")
#             self.dfs(node.left)
#             self.dfs(node.right)
# root = TreeNode(5, TreeNode(3, TreeNode(2), TreeNode(4)), TreeNode(6, None, TreeNode(7)))
# s = Solution()
# ans = s.deleteNode(root,3)
# s.dfs(ans)

# try :
#     result = 10/0
# except ZeroDivisionError as e:
#     print("Error:", e)

# chessboard = ['.' * 4 for _ in range(4)]
# print(chessboard)
from collections import Counter
from collections import Counter
# class Solution:
#     def backtracking(self,candidates,target,startindex,path,summ,used,result):
#         #终止条件
#         if summ>target:return
#         if summ == target:
#             result.append(path[:])
#             return
#         for i in range(startindex,len(candidates)):
#             #对树层去重
#             if i > startindex and candidates[i]==candidates[i-1] and used[i-1] ==0: #i>startindex
#                 continue
#             path.append(candidates[i])
#             summ+=candidates[i]
#             used[i]==1
#             self.backtracking(candidates,target,i+1,path,summ,used,result)
#             path.pop()
#             summ -=candidates[i]
#             used[i]==0
#     def combinationSum2(self, candidates: list[int], target: int) -> list[list[int]]:
#         path = []
#         result = []
#         used = [0]*len(candidates)
#         # sort
#         candidates.sort()
#         self.backtracking(candidates,target,0,path,0,used,result)
#         return result
# s = Solution()
# candidates = [10,1,2,7,6,1,5]
# target = 8
# print(s.combinationSum2(candidates,target))

# class Solution:
#     def threeSum(self, nums: list[int]) -> list[list[int]]:
#         result = []
#
#         #nums = nums.sort() nums返回none
#         nums.sort()
#         for i in range(len(nums)):
#             # 如果第一个元素已经大于0，不需要进一步检查
#             if nums[i] > 0:
#                 return result
#             while i >= 1 and i < len(nums) and nums[i] == nums[i - 1]: i += 1
#             left = i + 1
#             right = len(nums) - 1
#             while left<right:
#                 if nums[i]+nums[left]+nums[right] == 0:
#                     result.append([nums[i],nums[left],nums[right]])
#                     right -= 1
#                     left += 1
#                 if nums[i]+nums[left]+nums[right] < 0:
#                     left += 1
#                 if nums[i]+nums[left]+nums[right] > 0:
#                     right -= 1
#                 # 去重
#
#                 while left<right and nums[left] == nums[left-1]: left+=1
#                 while left<right and right<=len(nums)-2 and nums[right] == nums[right+1]: right-=1
#         return result
# s = Solution()
# nums = [-2,0,1,1,2]
# print(s.threeSum(nums))

# class Solution:
#     def backtracking(self,nums,path,used,result):
#         if len(path) == len(nums):
#             result.append(path[:])
#             return
#         for i in range(len(nums)):
#             if used[i] == 1:
#                 continue
#             path.append(nums[i])
#             used[i] = 1
#             self.backtracking(nums,path,used,result)
#             path.pop()
#             used.pop()
#     def permute(self, nums: list[int]) -> list[list[int]]:
#         result = []
#         path = []
#         used = [0]*len(nums)
#         self.backtracking(nums,path,used,result)
#         return result
# s = Solution()
# nums = [1,2,3]
# print(s.permute(nums))



# class Solution:
#     def backtracking(self,nums,path,used,result):
#         if len(path) == len(nums):
#             result.append(path[:])
#             return
#         for i in range(len(nums)):
#             if used[i] == 1:
#                 continue
#             path.append(nums[i])
#             used[i] = 1
#             self.backtracking(nums,path,used,result)
#             path.pop()
#             used[i] = 0
#     def permute(self, nums: list[int]) -> list[list[int]]:
#         nums.sort()
#         result = []
#         path = []
#         used = [0]*len(nums)
#         self.backtracking(nums,path,used,result)
#         return result
#
# s = Solution()
# nums = [1,1,2]
# print(s.permute(nums))

# class Solution:
#     def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
#         row = len(matrix)
#         col = len(matrix[0])
#         cnt = min(row, col)
#         if cnt % 2 == 0:
#             loop = cnt // 2
#         else:
#             loop = cnt // 2 + 1
#
#         ans = []
#         for offset in range(loop):
#             for j in range(0 + offset, col - 1 - offset):
#                 ans.append(matrix[0 + offset][j])
#             for i in range(0 + offset, row - 1 - offset):
#                 ans.append(matrix[i][col - 1 - offset])
#             for j in range(col - 1 - offset, 0 + offset, -1):
#                 if row - 1 - offset != offset:
#                     ans.append(matrix[row - 1 - offset][j])
#                 else:
#                     ans.append(matrix[row - 1 - offset][col - 1 - offset])
#                     break
#             for i in range(row - 1 - offset, 0 + offset, -1):
#                 if col - 1 - offset != offset:
#                     ans.append(matrix[i][0 + offset])
#                 else:
#                     ans.append(matrix[row - 1 - offset][0 + offset])
#                     break
#         if row == col and row % 2 == 1:
#             ans.append(matrix[row // 2][col // 2])
#
#         return ans

# class Solution:
#     def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
#         row = len(matrix)
#         col = len(matrix[0])
#         cnt = min(row, col)
#         loop = cnt // 2
#
#         ans = []
#         # loop控制offset
#         for offset in range(loop):
#             for j in range(0 + offset, col - 1 - offset):
#                 ans.append(matrix[0 + offset][j])
#             for i in range(0 + offset, row - 1 - offset):
#                 ans.append(matrix[i][col - 1 - offset])
#             for j in range(col - 1 - offset, 0 + offset, -1):
#                 ans.append(matrix[row - 1 - offset][j])
#             for i in range(row - 1 - offset, 0 + offset, -1):
#                 ans.append(matrix[i][0 + offset])
#
#         # 参考螺旋矩阵2，对于奇数边长的矩阵，需要把心加上。
#         if row>col and row % 2 == 1 and col % 2 ==1 or row % 2 ==1 and col%2==0:
#             for i in range(loop, row - (loop)):
#                 ans.append(matrix[i][col - (loop) - 1])
#         if col>row and col % 2 ==1 and row % 2 == 1 or col % 2 ==1 and row%2==0:
#             for j in range(loop, col - (loop)):
#                 ans.append(matrix[row - (loop) - 1][j])
#         if col == row and row % 2 == 1:
#             ans.append(matrix[cnt // 2][cnt // 2])
#
#         return ans
#
# s = Solution()
# matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# print(s.spiralOrder(matrix))

# def test_CompletePack(n, v, wi, vi):
#     dp = [0] * (v + 1)
#     for i in range(n):
#         for j in range(wi[i], v + 1):
#             dp[j] = max(dp[j], dp[j - wi[i]] + vi[i])
#     return dp[v]
#
#
# if __name__ == "__main__":
#     n,v = 0, 0
#     n, v = map(int, input().split())
#
#     # 初始化 wi 和 vi 列表来存储每种研究材料的重量和价值
#     wi = []
#     vi = []
#
#     # 读取接下来的 n 行输入，每行包含 wi 和 vi
#     for _ in range(n):
#         w, vv = map(int, input().split())
#         wi.append(w)
#         vi.append(vv)
#
#     result = test_CompletePack(n, v, wi, vi)
#     print(result)

# class Solution:
#     def combinationSum4(self, nums: list[int], target: int) -> int:
#         # dp[j]为组合数量 j为目前的和
#         dp=[0]*(target+1)
#         #
#         dp[0]=1
#         #有顺序，就要先遍历背包
#         for j in range(target+1):
#             for i in range(len(nums)):
#                 #记得加这句话
#                 if j-nums[i]>=0:
#                     dp[j]+=dp[j-nums[i]]
#                     print('i:',i,'j:',j,dp)
#         return dp[target]
#
# s = Solution()
# target = 5
# nums = [1, 2, 5]
# s.combinationSum4(nums, target)

# def climb(target, step):
#     # dp[j]为方法，j为当前爬的台阶数目
#
#     dp = [0] * (target + 1)
#     dp[0] = 1
#     for j in range(target + 1):
#         for i in range(1, step + 1):
#             dp[j] += dp[j - i]
#     return dp[target]
#
#
# n, m = map(int, input().split())
# res = climb(n, m)
# print(res)

# class Solution:
#     def wordBreak(self, s: str, wordDict: list[str]) -> bool:
#         #dp[j],dp[j]为拼接的字符串,j为目前拼接到的字母的下角标
#         dp=[0]*(len(s))
#
#         for j in range(len(s)):
#             for i in wordDict:
#                 dp[j]+=dp[j-len(i)]
#                 print(dp)
#         if dp[len(s)-1]==s:
#             return True
#         else:
#             return False
#
# S = Solution()
# s = "leetcode"
# wordDict = ["leet", "code"]
# print(S.wordBreak(s, wordDict))
class Solution:
    def robit(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        pre = 0
        cur = 0
        for i in range(0, len(nums)):
            tmp = cur
            cur = max(cur, pre + nums[i])
            pre = tmp
        return cur


    def rob(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        if len(nums) == 2:
            return max(nums[0], nums[1])
        res1 = self.robit(nums[0:len(nums) - 1])
        res2 = self.robit(nums[1:len(nums)])
        res = max(res1, res2)
        return res
s = Solution()
nums = [1,2,3,1]
print(s.rob(nums))
