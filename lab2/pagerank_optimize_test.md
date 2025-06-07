开始测试所有数据集（优化版）...

========== 测试Slashdot数据集（优化版） ==========
开始加载数据: Slashdot0902.txt
数据加载完成，共 82168 个节点，耗时 2.66 秒

测试方法: power_iteration, 收敛阈值: 1e-06
方法: power_iteration, 阈值: 1e-06, 迭代次数: 63, 计算时间: 0.09秒

测试方法: power_iteration_parallel, 收敛阈值: 1e-06
方法: power_iteration_parallel, 阈值: 1e-06, 迭代次数: 63, 计算时间: 0.49秒

测试方法: matrix_power, 收敛阈值: 1e-06
方法: matrix_power, 阈值: 1e-06, 迭代次数: 63, 计算时间: 0.09秒

Slashdot数据集 - 不同方法的性能比较（阈值=1e-6）:
方法                    迭代次数        计算时间(秒)
power_iteration         63              0.09
power_iteration_parallel                63              0.49
matrix_power            63              0.09

Slashdot数据集 - PageRank值最高的10个网页:
#1: 网页ID=2494, PageRank值=0.00213059
#2: 网页ID=398, PageRank值=0.00197209
#3: 网页ID=381, PageRank值=0.00181529
#4: 网页ID=4805, PageRank值=0.00146000
#5: 网页ID=37, PageRank值=0.00143137
#6: 网页ID=226, PageRank值=0.00121352
#7: 网页ID=5706, PageRank值=0.00107819
#8: 网页ID=4826, PageRank值=0.00088203
#9: 网页ID=219, PageRank值=0.00079160
#10: 网页ID=5057, PageRank值=0.00071598

测试计算方法: power_iteration, 收敛阈值: 0.001
方法: power_iteration, 阈值: 0.001, 迭代次数: 17, 计算时间: 0.13秒

测试计算方法: power_iteration, 收敛阈值: 0.0001
方法: power_iteration, 阈值: 0.0001, 迭代次数: 32, 计算时间: 0.25秒

测试计算方法: power_iteration, 收敛阈值: 1e-05
方法: power_iteration, 阈值: 1e-05, 迭代次数: 48, 计算时间: -1.92秒

测试计算方法: power_iteration, 收敛阈值: 1e-06
方法: power_iteration, 阈值: 1e-06, 迭代次数: 63, 计算时间: 0.55秒

测试计算方法: power_iteration, 收敛阈值: 1e-07
方法: power_iteration, 阈值: 1e-07, 迭代次数: 78, 计算时间: 0.72秒

Slashdot数据集 - 不同收敛阈值下的性能 (方法=power_iteration):
阈值    迭代次数        计算时间(秒)
1e-03   17      0.13
1e-04   32      0.25
1e-05   48      -1.92
1e-06   63      0.55
1e-07   78      0.72

========== 测试Google数据集（优化版） ==========
开始加载数据: web-Google.txt
数据加载完成，共 875713 个节点，耗时 19.37 秒

测试方法: power_iteration, 收敛阈值: 1e-06
方法: power_iteration, 阈值: 1e-06, 迭代次数: 96, 计算时间: 3.38秒

测试方法: power_iteration_parallel, 收敛阈值: 1e-06
方法: power_iteration_parallel, 阈值: 1e-06, 迭代次数: 96, 计算时间: 4.00秒

测试方法: matrix_power, 收敛阈值: 1e-06
方法: matrix_power, 阈值: 1e-06, 迭代次数: 96, 计算时间: 3.04秒

Google数据集 - 不同方法的性能比较（阈值=1e-6）:
方法                    迭代次数        计算时间(秒)
power_iteration         96              3.38
power_iteration_parallel                96              4.00
matrix_power            96              3.04

Google数据集 - PageRank值最高的10个网页:
#1: 网页ID=41909, PageRank值=0.00101621
#2: 网页ID=597621, PageRank值=0.00096984
#3: 网页ID=537039, PageRank值=0.00093411
#4: 网页ID=163075, PageRank值=0.00091012
#5: 网页ID=384666, PageRank值=0.00085877
#6: 网页ID=504140, PageRank值=0.00080940
#7: 网页ID=32163, PageRank值=0.00074789
#8: 网页ID=486980, PageRank值=0.00074301
#9: 网页ID=765334, PageRank值=0.00073720
#10: 网页ID=558791, PageRank值=0.00072782

测试计算方法: power_iteration, 收敛阈值: 0.001
方法: power_iteration, 阈值: 0.001, 迭代次数: 33, 计算时间: -0.55秒

测试计算方法: power_iteration, 收敛阈值: 0.0001
方法: power_iteration, 阈值: 0.0001, 迭代次数: 54, 计算时间: 1.67秒

测试计算方法: power_iteration, 收敛阈值: 1e-05
方法: power_iteration, 阈值: 1e-05, 迭代次数: 75, 计算时间: 3.02秒

测试计算方法: power_iteration, 收敛阈值: 1e-06
方法: power_iteration, 阈值: 1e-06, 迭代次数: 96, 计算时间: 4.14秒

测试计算方法: power_iteration, 收敛阈值: 1e-07
方法: power_iteration, 阈值: 1e-07, 迭代次数: 118, 计算时间: 4.75秒

Google数据集 - 不同收敛阈值下的性能 (方法=power_iteration):
阈值    迭代次数        计算时间(秒)
1e-03   33      -0.55
1e-04   54      1.67
1e-05   75      3.02
1e-06   96      4.14
1e-07   118     4.75

所有测试完成，总耗时: 46.82秒