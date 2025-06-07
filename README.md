# 数据科学与工程算法课程

本仓库包含了数据科学与工程算法课程的完整学习资料，包括理论教材、习题解答和三个核心实验项目。

## 📚 课程概述

数据科学与工程算法是一门专注于大规模数据处理和分析的核心课程，涵盖了现代数据科学中的关键算法和技术。课程内容包括概率不等式、哈希算法、数据素描、采样理论、随机游走、特征值分析、矩阵分解、整数规划等重要主题。

## 📖 理论教材

### 核心理论文档
- `1_inequality.pdf` - 概率不等式理论
- `2_hashing.pdf` & `2_lsh_theory.pdf` - 哈希算法与局部敏感哈希
- `3_sketch.pdf` - 数据素描技术
- `4_sampling.pdf` - 采样理论与方法
- `5_random_walk.pdf` - 随机游走算法
- `7_eigenvalue_cn.pdf` - 特征值分析
- `8_SVD_PCA_cn.pdf` - 奇异值分解与主成分分析
- `9_decomposition_cn.pdf` - 矩阵分解技术
- `10_IP_cn.pdf` & `10_LP.pdf` - 整数规划与线性规划
- `11_covering_cn.pdf` - 覆盖算法
- `12_community_cn.pdf` - 社区发现算法

### 练习资料
- `期中练习题.pdf` - 期中考试练习题
- `习题答案/` - 各章节习题详细解答

## 🔬 实验项目

### Lab 1: 基于时间窗口的电影评分数据流查询系统
**项目描述**: 设计并实现了一个高效的数据流查询系统，能够处理大规模电影评分数据。

**核心技术**:
- 布隆过滤器 (Bloom Filter)
- 时间窗口数据结构
- 概率数据结构
- 数据流处理

**主要功能**:
- 成员查询：检验特定电影是否被评分
- 频度查询：统计电影评分次数
- Top-k查询：查找评分最多的k部电影

**性能优化**:
- 内存使用减少30-60%
- 查询响应时间控制在毫秒级
- 概率版本进一步减少40%内存占用

**文件结构**:
```
lab1/
├── 基于时间窗口的电影评分数据流查询系统设计与实现.md  # 详细实验报告
├── movie_rating_queries.py                           # 基础实现
├── optimized_movie_queries.py                        # 优化版本
├── probabilistic_movie_queries.py                    # 概率数据结构版本
├── benchmark_queries.py                              # 性能测试
└── compare_implementations.py                        # 实现对比
```

### Lab 2: PageRank算法实现与优化
**项目描述**: 实现并优化了PageRank算法，显著提升了在大规模网络数据上的计算效率。

**核心技术**:
- 稀疏矩阵表示
- 向量化计算
- 并行处理
- 矩阵运算优化

**性能提升**:
- Google数据集上实现超过200倍加速
- 保持计算结果准确性
- 支持不同收敛阈值配置

**测试数据集**:
- Slashdot社交网络
- Google网页链接网络

**文件结构**:
```
lab2/
├── Lab2实验报告_PageRank算法实现与优化.md    # 详细实验报告
├── pagerank.py                            # 基础PageRank实现
├── pagerank_optimize.py                   # 优化版本
├── test_pagerank.py                       # 基础版本测试
├── test_pagerank_optimize.py              # 优化版本测试
├── pagerank_test.md                       # 测试结果文档
└── pagerank_optimize_test.md              # 优化版本测试结果
```

### Lab 3: 基于SVD的电影推荐系统
**项目描述**: 基于矩阵分解技术实现了高效的电影推荐系统，解决了大规模数据处理的内存问题。

**核心技术**:
- 奇异值分解 (SVD)
- 稀疏矩阵表示
- 数据采样技术
- 批处理优化

**系统特性**:
- 内存使用从70GB降至4GB以下（减少95%+）
- RMSE评估值约为1.42
- 支持冷启动用户推荐
- 高质量的电影相似性判断

**主要功能**:
- Top-N个性化推荐
- 电影相似性分析
- 推荐系统性能评估
- 隐因子数量影响分析

**文件结构**:
```
lab3/
├── Lab3实验报告_基于SVD的电影推荐系统设计与优化.md   # 详细实验报告
├── movie_recommender.py                            # 主要推荐系统实现
├── demo_recommender.py                             # 演示版本
├── test_recommender.py                             # 测试脚本
├── test_result.md                                  # 测试结果
└── rmse_vs_k_factors.png                          # 隐因子数量vs RMSE图表
```

## 🗃️ 数据集

所有实验项目均基于 **MovieLens-25M** 数据集：
- 25,000,095条用户评分记录
- 62,423部电影
- 162,541名用户
- 时间跨度：1995-2019年

## 🛠️ 技术栈

**编程语言**: Python 3.x

**核心库**:
- NumPy - 数值计算
- SciPy - 科学计算
- Pandas - 数据处理
- scikit-learn - 机器学习
- Matplotlib - 数据可视化

**算法技术**:
- 概率数据结构
- 稀疏矩阵优化
- 并行计算
- 矩阵分解
- 数据流处理

## 🚀 快速开始

### 环境配置
```bash
# 创建虚拟环境
python -m venv venv
.\venv\Scripts\activate

# 安装依赖
pip install numpy scipy pandas scikit-learn matplotlib
```

### 运行实验

**Lab 1 - 数据流查询系统**:
```bash
cd lab1
python movie_rating_queries.py          # 基础版本
python optimized_movie_queries.py       # 优化版本
python benchmark_queries.py             # 性能测试
```

**Lab 2 - PageRank算法**:
```bash
cd lab2
python test_pagerank.py                 # 基础版本测试
python test_pagerank_optimize.py        # 优化版本测试
```

**Lab 3 - 推荐系统**:
```bash
cd lab3
python demo_recommender.py              # 演示版本
python test_recommender.py              # 完整测试
```

## 📊 实验成果

### 性能对比

| 项目 | 优化前 | 优化后 | 提升幅度 |
|------|--------|--------|----------|
| Lab1 内存使用 | 100% | 40-70% | 减少30-60% |
| Lab2 计算速度 | 1x | 200x+ | 200倍加速 |
| Lab3 内存使用 | 70GB+ | <4GB | 减少95%+ |

### 算法准确性

- **Lab1**: 查询精度损失 <5%
- **Lab2**: 保持完全准确性
- **Lab3**: RMSE ≈ 1.42

## 📝 实验报告

每个实验都包含详细的实验报告，涵盖：
- 问题定义与算法设计
- 实现细节与优化策略
- 实验结果与性能分析
- 结论与未来工作方向

## 🤝 贡献

欢迎提交问题报告和改进建议！

## 📄 许可证

本项目仅供学习和研究使用。

---

**课程**: 数据科学与工程算法  
**时间**: 2024-2025学年  
**最后更新**: 2025年6月7日
