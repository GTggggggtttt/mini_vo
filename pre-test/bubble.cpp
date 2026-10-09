/* 不断比较相邻两个元素。

如果前面的比后面的大：
    交换

每完成一轮：
    当前最大的元素会被“冒泡”到数组末尾。 */

#include <iostream>
#include <vector>
#include <utility>

using namespace std;

void bubbleSort(std::vector<int> &nums)
{
    const int n = static_cast<int>(nums.size());

    /*
     * 外层循环控制“排序多少轮”。
     *
     * n 个元素最多只需要 n - 1 轮。
     *
     * 例如：
     *
     * 5 个元素
     *
     * 第 1 轮：
     * 最大元素移动到最后
     *
     * 第 2 轮：
     * 第二大元素移动到倒数第二
     *
     * ...
     */

    for (int i = 0; i < n - 1; ++i)
    {
        /*
         * 每一轮都比较相邻元素：
         *
         * nums[j]
         * nums[j + 1]
         *
         * 为什么是 n - 1 - i？
         *
         * 因为每完成一轮，
         * 数组末尾就会多一个已经排好序的元素，
         * 后面已经不需要继续比较。
         */
        for (int j = 0; j < n - 1; ++j)
        {
            if (nums[j] > nums[j + 1])
            {
                std::swap(nums[j], nums[j + 1]);
            }
        }
    }
}

void selectionSort(std::vector<int> &nums)
{
    const int n = static_cast<int>(nums.size());
    /*
     * i 表示：
     *
     * 当前准备确定哪个位置的元素。
     *
     * i 左边：
     * 已经排序完成。
     *
     * i 以及右边：
     * 还没有排序。
     */
    for (int i = 0; i < n - 1; ++i)
    { /*
       * 先假设当前 nums[i]
       * 就是剩余部分中最小的元素。
       *
       * minIndex 保存最小元素的“下标”，
       * 而不是直接保存元素值。
       */
        int minIndex = i;
        /*
         * 从 i + 1 开始搜索。
         *
         * 因为 nums[i] 已经作为当前最小值候选，
         * 不需要自己和自己比较。
         */
        for (int j = i + 1; j < n; ++j)
        {
            if (nums[j] < nums[minIndex])
            {
                minIndex = j;
            }
        }

        // 搜索结束以后
        std::swap(nums[i], nums[minIndex]);
    }
}

void insertionSort(vector<int> &nums)
{
    const int n = static_cast<int>(nums.size());
    /*
     * nums[0] 单独一个元素时，
     * 一定是有序的。
     *
     * 所以从 nums[1] 开始处理。
     */
    for (int i = 1; i < n; ++i)
    {
        /*
         * key 是当前准备插入到
         * 前面有序区域中的元素。
         *
         * 必须先保存下来。
         *
         * 因为后面移动元素时，
         * nums[i] 可能会被覆盖。
         */
        const int key = nums[i];
        /*
         * j 从当前元素左边开始。
         */
        int j = i - 1;
        /*
         * 条件：
         *
         * j >= 0
         *
         * 表示还没有移动到数组最左边。
         *
         * nums[j] > key
         *
         * 表示当前元素比 key 大，
         * 那么它应该向右移动。
         */
        while (j >= 0 && nums[j] > key)
        {
            /*
             * 将较大的元素向右移动一个位置。
             *
             * 注意：
             * 这里不是交换。
             *
             * 是直接覆盖。
             */
            nums[j + 1] = nums[j];
            --j;
        }
        /*
         * while 结束以后：
         *
         * j + 1 就是 key 应该插入的位置。
         */
        nums[j + 1] = key;
    }
}

int partition(std::vector<int> &nums, int low, int high)
{
    /*
     * 选择最后一个元素作为 pivot。
     */
    const int pivot = nums[high];
    /*
     * i 表示：
     *
     * “小于等于 pivot 区域”的最后一个位置。
     *
     * 一开始还没有任何元素属于这个区域，
     * 所以：
     *
     * i = low - 1
     */
    int i = low - 1;
    for (int j = low; j < high; ++j)
    {
        if (nums[j] <= pivot)
        {
            ++i;
            std::swap(nums[i], nums[j]);
        }
    }
    std::swap(nums[i + 1], nums[high]);

    return i + 1;
}

int main()
{
    std::vector<int> nums{64, 34, 25, 12, 22, 11, 90};

    // bubbleSort(nums);

    // selectionSort(nums);

    insertionSort(nums);

    for (const int value : nums)
    {
        std::cout << value << ' ';
    }
    std::cout << '\n';
    return 0;
}