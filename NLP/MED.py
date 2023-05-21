"""
@ 动态规划求最小编辑距离
"""


def computeMED(str1, str2, insertCost, deleteCost, replaceCost):
    """计算str1转化为str2所需要的最小编辑代价
       insertCost、deleteCost与replaceCost需为非负有限值
    """
    m = len(str1)+1  # str1的长度加1
    n = len(str2)+1  # str2的长度加1
    # 初始化最小编辑距离矩阵d，shape为m*n
    # d[i][j]表示str1的前i-1项转化为str2的前j-1项所需要的最小代价(从第0项开始)
    # i=0时，表示str1为空字符串，j=0时，表示str2为空字符串
    d = [[0]*n for t in range(m)]
    p = [[(-1, -1)]*n for t in range(m)]  # 回退路径
    for i in range(1, n):
        d[0][i] = i*insertCost  # str1为空字符串时的最小编辑距离
        p[0][i] = (-1, i-1)  # 对应的回退路径
    for j in range(1, m):
        d[j][0] = j*deleteCost  # str2为空字符串时的最小编辑距离
        p[j][0] = (j-1, -1)  # 对应的回退路径
    for i in range(1, m):
        for j in range(1, n):
            # 按照动态规划自底向上的思想，要使得str1的前i-1项转化到与str2的前j-1项相等，有如下考虑方式：
            # 第一种考虑方式是已经使str1的前i-2项转化到与str2的前j-2项相等，通过替换操作使转化后的str1末位项与str2的第j-1项相等，则二者相等
            # 第二种考虑方式是已经使str1的前i-1项转化到与str2的前j-2项相等，通过向转化后的str1末端插入str2的第j-1项使二者相等
            # 第三种考虑方式是已经使str1的前i-2项转化到与str2的前j-1项相等，通过删除转化后的str1的末位项使二者相等
            """------------------------------------------------------------------------------------------------------"""
            # 第一种考虑方式，通过消耗d[i-1][j-1]的代价，已经使str1的前i-2项转化到与str2的前j-2项相等
            # 考虑str1的第i-1项与str2的第j-1项，如果相等，则d[i][j] = d[i-1][j-1]
            # 如果不相等，执行替换操作，则d[i][j] = d[i-1][j-1] + 替换操作需要的代价
            if str1[i-1] == str2[j-1]:
                cond1 = d[i-1][j-1]
            else:
                cond1 = d[i-1][j-1] + replaceCost
            """------------------------------------------------------------------------------------------------------"""
            # 第二种考虑方式，通过消耗d[i][j-1]的代价，已经使str1的前i-1项转化到与str2的前j-2项相等，再执行插入操作即可(尾插)
            # 则有d[i][j] = d[i][j-1] + 插入操作需要的代价
            cond2 = d[i][j-1]+insertCost
            # 第三种考虑方式，通过消耗d[i-1][j]的代价，已经使str1的前i-2项转化到与str2的前j-1项相等，再执行删除操作即可(去尾)
            # 则有d[i][j] = d[i-1][j] + 删除操作需要的代价
            cond3 = d[i-1][j]+deleteCost
            # 三种考虑方式中选择代价最小的，即为str1的前i-1项与str2的前j-1项的最小编辑距离
            d[i][j] = min(cond1, cond2, cond3)
            # 根据操作的不同为回退矩阵赋值
            if d[i][j] == cond1:  p[i][j] = (i-1, j-1)
            elif d[i][j] == cond2:  p[i][j] = (i, j-1)
            elif d[i][j] == cond3:  p[i][j] = (i-1, j)
    """--------------------------------------------------------------------------------------------------------------"""
    find_path = p[m-1][n-1]  # 回退，寻找上一步的路径
    operations = [str2]  # 记录每次操作完成后的str1(逆序)，最后的操作完成后变为str2
    while (find_path[0] != -1 or find_path[1] != -1):
        x, y = find_path
        if x != -1 and y != -1:
            operations.append(str2[:y]+str1[x:])
            find_path = p[x][y]
        elif x == -1:
            operations.append(str2[:y])
            find_path = p[0][y]
        else:
            operations.append(str1[x:])
            find_path = p[x][0]
    operations = operations[:-1]  # 除去空字符串
    operations.append(str1)  # 添加初始字符串str1
    operations = operations[::-1]  # 逆序改为倒序
    operations_ = []
    [operations_.append(opts) for opts in operations if opts not in operations_]  # 删除重复的字符串

    return d[m-1][n-1], operations_


if __name__ == '__main__':
    str1 = "你不是大学生"
    str2 = "あなたわ大学生でわありません"
    # str1 = input()
    # str2 = input()
    insertCost = 1
    deleteCost = 1
    replaceCost = 1
    minEditDistance, operations = computeMED(str1, str2, insertCost, deleteCost, replaceCost)
    print("最小编辑距离为:", minEditDistance)
    print("编辑路径为:",operations)
