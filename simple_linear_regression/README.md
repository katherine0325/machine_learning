0. 前提介绍
为什么需要统计量？
    统计量：描述数据特征
    0.1 集中趋势衡量
    0.1.1 均值（平均数，平均值）mean
        {6, 2, 9, 1, 2}
        (6+2+9+1+2) / 5 = 4
    0.1.2 中位数 median
        将数据中的各个数值按照大小排序，位于中间位置的变量
        1, 2, 2, 6, 9
        找出位于中间的2
        当n为奇数时，直接取处于位置中间的数
        当n为偶数时，取中间两个量的平均值
    0.1.3 众数：数据中出现次数最多的数
    0.2 离散程度衡量
        0.2.1.1 方差（variance）
        如何求方差呢，首先给出一组数，每个数减去平均数（4），平方后相加。除以个数减1，得出来的数就是方差
            公式：![方差公式](/assets/imgs/variance.png)
            案例：
            {6, 2, 9, 1, 2}
            (1) (6-4)^2+(2-4)^2+(9-4)^2+(1-4)^2+(2-4)^2 = 4+4+9+25+4 = 46
            (2) n - 1 = 5 - 1 = 4
            (3) 46/4=11.5

        0.2.1.2 标准差(standard deviation)
            方差开2次方
            公式：![标准差公式](/assets/imgs/standard_deviation.png)
            继续以上案例，它的标准差是
            s = sqrt(11.5) = 3.39


1. 介绍：回归(regression)Y变量为连续数值型(continuous numerical variable)
    如：房价，人数，降雨量
    分类(classification): Y变量为类别型(categorical variable)
    如：颜色类别，电脑类别，有无信誉

2. 简单线性回归(simple linear regression)
    2.1 很多做决定的过程，通常是两个或多个变量之间的关系
    2.2 回归分析(regression analysis)用来建立方程模拟两个或多个变量之间的关系
    2.3 被预测的变量叫做：因变量(dependent variable), y, 输出(output)
    2.4 被用来进行预测的变量叫做：自变量(independent variable), x, 输入(input)

3. 简单线性回归介绍
    3.1 简单线性回归包含一个自变量和一个因变量
    3.2 以上两个变量的关系用一条直线来模拟
    3.3 如果包含两个以上的自变量，则称为多元回归分析(multiple regression)

4. 简单线性回归模型
    4.1 被用来描述因变量（y）和自变量（x）以及偏差（error）之间的关系的方程叫做回归模型
    4.2 简单线性回归的模型是
        y = b0 + b1x + e