import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from movie_rating_queries import MovieRatingQueries
from optimized_movie_queries import OptimizedMovieQueries
from probabilistic_movie_queries import ProbabilisticMovieQueries
from pympler import asizeof

def generate_test_data(num_movies=1000, num_ratings=10000):
    """生成测试数据"""
    movies = list(range(1, num_movies + 1))
    ratings = []
    for _ in range(num_ratings):
        movie_id = random.choice(movies)
        rating = random.randint(1, 5)
        timestamp = random.randint(0, 1000000)
        ratings.append((movie_id, rating, timestamp))
    return ratings

def benchmark_query(query_func, test_data, num_queries=1000):
    """基准测试单个查询函数"""
    times = []
    for _ in range(num_queries):
        movie_id = random.randint(1, 1000)
        start_time = time.time()
        query_func(movie_id)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times), np.std(times)

def benchmark_topk(query_func, test_data, num_queries=100):
    """基准测试Top-k查询"""
    times = []
    for _ in range(num_queries):
        k = random.randint(1, 100)
        start_time = time.time()
        query_func(k)
        end_time = time.time()
        times.append(end_time - start_time)
    return np.mean(times), np.std(times)

def compare_implementations(num_movies=1000, num_ratings=10000, num_queries=1000, enable_optimization=False):
    """比较不同实现的性能"""
    print(f"\n开始性能对比测试...")
    print(f"参数配置:")
    print(f"- 电影数量: {num_movies}")
    print(f"- 评分数量: {num_ratings}")
    print(f"- 查询次数: {num_queries}")
    print(f"- 内存优化: {'启用' if enable_optimization else '禁用'}")
    
    # 生成测试数据
    test_data = generate_test_data(num_movies, num_ratings)
    
    # 初始化实现
    implementations = {
        "基础实现": MovieRatingQueries(),
        "优化实现": OptimizedMovieQueries(),
        "概率实现": ProbabilisticMovieQueries()
    }
    
    # 记录结果
    results = {
        "实现": [],
        "内存使用(MB)": [],
        "成员查询(ms)": [],
        "频率查询(ms)": [],
        "Top-k查询(ms)": []
    }
    
    # 测试每个实现
    for name, impl in implementations.items():
        print(f"\n测试 {name}...")
        
        # 添加数据
        start_time = time.time()
        for movie_id, rating, timestamp in test_data:
            impl.add_rating(movie_id, rating, timestamp)
        add_time = time.time() - start_time
        print(f"数据添加时间: {add_time:.2f}秒")
        
        # 计算内存使用
        memory_usage = asizeof.asizeof(impl) / (1024 * 1024)
        print(f"内存使用: {memory_usage:.2f}MB")
        
        # 如果启用优化，执行内存优化
        if enable_optimization and hasattr(impl, 'optimize_memory'):
            print("执行内存优化...")
            start_time = time.time()
            impl.optimize_memory()
            optimize_time = time.time() - start_time
            print(f"优化时间: {optimize_time:.2f}秒")
            print(f"优化后内存使用: {asizeof.asizeof(impl) / (1024 * 1024):.2f}MB")
        
        # 基准测试
        print("执行基准测试...")
        member_mean, member_std = benchmark_query(impl.member_query, test_data, num_queries)
        freq_mean, freq_std = benchmark_query(impl.frequency_query, test_data, num_queries)
        topk_mean, topk_std = benchmark_topk(impl.top_k_query, test_data, num_queries // 10)
        
        # 记录结果
        results["实现"].append(name)
        results["内存使用(MB)"].append(f"{memory_usage:.2f}")
        results["成员查询(ms)"].append(f"{member_mean*1000:.3f}±{member_std*1000:.3f}")
        results["频率查询(ms)"].append(f"{freq_mean*1000:.3f}±{freq_std*1000:.3f}")
        results["Top-k查询(ms)"].append(f"{topk_mean*1000:.3f}±{topk_std*1000:.3f}")
        
        print(f"测试完成")
    
    # 打印结果表格
    print("\n性能对比结果:")
    print("| 实现 | 内存使用(MB) | 成员查询(ms) | 频率查询(ms) | Top-k查询(ms) |")
    print("|------|--------------|--------------|--------------|---------------|")
    for i in range(len(results["实现"])):
        print(f"| {results['实现'][i]} | {results['内存使用(MB)'][i]} | {results['成员查询(ms)'][i]} | {results['频率查询(ms)'][i]} | {results['Top-k查询(ms)'][i]} |")

def plot_comparison(results, include_optimization=True):
    """绘制比较结果图表"""
    # 创建图表
    if include_optimization:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
    fig.suptitle('电影评分查询系统实现比较', fontsize=16)
    
    # 加载时间
    axes[0, 0].bar(['精确版本', '优化版本'], 
                 [results['加载时间(秒)']['精确版本'], results['加载时间(秒)']['优化版本']])
    axes[0, 0].set_ylabel('时间 (秒)')
    axes[0, 0].set_title('数据加载时间')
    
    # 内存占用
    if include_optimization:
        # 内存使用对比（优化前后）
        memory_data = [
            results['内存占用(MB)']['精确版本_优化前'], 
            results['内存占用(MB)']['精确版本_优化后'],
            results['内存占用(MB)']['优化版本_优化前'], 
            results['内存占用(MB)']['优化版本_优化后']
        ]
        x = np.arange(2)
        width = 0.35
        
        axes[0, 1].bar(x - width/2, [memory_data[0], memory_data[2]], width, label='优化前')
        axes[0, 1].bar(x + width/2, [memory_data[1], memory_data[3]], width, label='优化后')
        
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(['精确版本', '优化版本'])
        axes[0, 1].set_ylabel('内存 (MB)')
        axes[0, 1].set_title('内存占用 (优化前后)')
        axes[0, 1].legend()
        
        # 添加节省百分比标签
        saving_exact = (memory_data[0] - memory_data[1]) / memory_data[0] * 100
        saving_opt = (memory_data[2] - memory_data[3]) / memory_data[2] * 100
        
        axes[0, 1].text(x[0] - width/2, memory_data[0], f'{memory_data[0]:.2f}MB', 
                       ha='center', va='bottom', fontsize=8)
        axes[0, 1].text(x[0] + width/2, memory_data[1], f'{memory_data[1]:.2f}MB\n(-{saving_exact:.1f}%)', 
                       ha='center', va='bottom', fontsize=8, color='green')
        
        axes[0, 1].text(x[1] - width/2, memory_data[2], f'{memory_data[2]:.2f}MB', 
                       ha='center', va='bottom', fontsize=8)
        axes[0, 1].text(x[1] + width/2, memory_data[3], f'{memory_data[3]:.2f}MB\n(-{saving_opt:.1f}%)', 
                       ha='center', va='bottom', fontsize=8, color='green')
    else:
        # 原始内存比较
        axes[0, 1].bar(['精确版本', '优化版本'], 
                     [results['内存占用(MB)']['精确版本_优化前'], results['内存占用(MB)']['优化版本_优化前']])
        axes[0, 1].set_ylabel('内存 (MB)')
        axes[0, 1].set_title('内存占用')
    
    # 查询时间
    query_metrics = ['成员查询时间(毫秒)', '频度查询时间(毫秒)', 'Top-k查询时间(毫秒)']
    
    x = np.arange(len(query_metrics))
    width = 0.35
    
    exact_times = [results[metric]['精确版本'] for metric in query_metrics]
    optimized_times = [results[metric]['优化版本'] for metric in query_metrics]
    
    axes[1, 0].bar(x - width/2, exact_times, width, label='精确版本')
    axes[1, 0].bar(x + width/2, optimized_times, width, label='优化版本')
    
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([m.split('(')[0] for m in query_metrics])
    axes[1, 0].set_ylabel('时间 (毫秒)')
    axes[1, 0].set_title('查询响应时间')
    axes[1, 0].legend()
    
    # 添加数值标签
    for i, v in enumerate(exact_times):
        axes[1, 0].text(i - width/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    for i, v in enumerate(optimized_times):
        axes[1, 0].text(i + width/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    
    # 精度与准确性
    axes[1, 1].axis('off')
    
    if include_optimization:
        accuracy_text = (
            "准确性分析:\n\n"
            f"1. 频度查询平均误差: {results['频度查询平均误差']:.2%}\n"
            f"2. Top-k查询平均相似度: {results['Top-k查询平均相似度']:.2%}\n\n"
            "性能总结:\n"
            f"- 内存优化前差异: {(results['内存占用(MB)']['精确版本_优化前'] - results['内存占用(MB)']['优化版本_优化前']) / results['内存占用(MB)']['精确版本_优化前']:.2%}\n"
            f"- 内存优化后差异: {(results['内存占用(MB)']['精确版本_优化后'] - results['内存占用(MB)']['优化版本_优化后']) / results['内存占用(MB)']['精确版本_优化后']:.2%}\n"
            f"- 精确版本内存优化: {results['内存优化']['精确版本_节省比例']}\n"
            f"- 优化版本内存优化: {results['内存优化']['优化版本_节省比例']}\n"
            f"- 加载时间差异: {(results['加载时间(秒)']['优化版本'] / results['加载时间(秒)']['精确版本'] - 1):.2%}\n"
            "- 查询性能变化:\n"
            f"  • 成员查询: {(results['成员查询时间(毫秒)']['优化版本'] / results['成员查询时间(毫秒)']['精确版本'] - 1):.2%}\n"
            f"  • 频度查询: {(results['频度查询时间(毫秒)']['优化版本'] / results['频度查询时间(毫秒)']['精确版本'] - 1):.2%}\n"
            f"  • Top-k查询: {(results['Top-k查询时间(毫秒)']['优化版本'] / results['Top-k查询时间(毫秒)']['精确版本'] - 1):.2%}\n\n"
            "结论:\n"
            "- 使用Pympler进行内存统计提供了更准确的内存使用数据\n"
            "- 内存优化策略在两种实现上均取得显著效果\n"
            "- 优化版本在保持准确性的前提下进一步降低了内存占用\n"
            "- 稀疏数据结构和数据压缩对大规模数据集尤为有效\n"
        )
    else:
        accuracy_text = (
            "准确性分析:\n\n"
            f"1. 频度查询平均误差: {results['频度查询平均误差']:.2%}\n"
            f"2. Top-k查询平均相似度: {results['Top-k查询平均相似度']:.2%}\n\n"
            "性能总结:\n"
            f"- 内存占用减少: {(results['内存占用(MB)']['精确版本_优化前'] - results['内存占用(MB)']['优化版本_优化前']) / results['内存占用(MB)']['精确版本_优化前']:.2%}\n"
            f"- 加载时间变化: {(results['加载时间(秒)']['优化版本'] / results['加载时间(秒)']['精确版本'] - 1):.2%}\n"
            "- 查询性能变化:\n"
            f"  • 成员查询: {(results['成员查询时间(毫秒)']['优化版本'] / results['成员查询时间(毫秒)']['精确版本'] - 1):.2%}\n"
            f"  • 频度查询: {(results['频度查询时间(毫秒)']['优化版本'] / results['频度查询时间(毫秒)']['精确版本'] - 1):.2%}\n"
            f"  • Top-k查询: {(results['Top-k查询时间(毫秒)']['优化版本'] / results['Top-k查询时间(毫秒)']['精确版本'] - 1):.2%}\n\n"
            "结论:\n"
            "- 优化版本显著减少了内存占用\n"
            "- 概率数据结构在保持较高准确性的同时提供了良好的性能\n"
            "- 两种实现各有优势，根据内存限制和精度要求选择合适的方案\n"
        )
    
    axes[1, 1].text(0, 0.5, accuracy_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if include_optimization:
        plt.savefig('implementation_comparison_with_optimization.png')
    else:
        plt.savefig('implementation_comparison.png')
    
    plt.close()

if __name__ == "__main__":
    ratings_file = os.path.join("ml-25m", "ratings.csv")
    
    if os.path.exists(ratings_file):
        print(f"找到评分文件: {ratings_file}")
        
        # 比较两种实现 (包含内存优化)
        results_with_opt = compare_implementations(ratings_file, enable_optimization=True)
        
        # 绘制比较结果
        plot_comparison(results_with_opt, include_optimization=True)
        print("\n比较完成，结果已保存为 implementation_comparison_with_optimization.png")
    else:
        print(f"错误: 找不到评分文件 {ratings_file}")
        print("请确保MovieLens 25M数据集已解压到正确位置")

    # 运行基准测试
    compare_implementations(num_movies=1000, num_ratings=10000, num_queries=1000, enable_optimization=True) 