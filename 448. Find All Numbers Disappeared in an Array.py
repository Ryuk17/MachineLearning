
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        nums.insert(0, 0)
        res = []
        for i in range(1, len(nums)):
            while nums[i] != i and nums[i] != nums[nums[i]]:
                tmp = nums[nums[i]]
                nums[nums[i]] = nums[i]
                nums[i] = tmp

        for i in range(1, len(nums)):
            if nums[i] != i:
                res.append(i)

        return res
