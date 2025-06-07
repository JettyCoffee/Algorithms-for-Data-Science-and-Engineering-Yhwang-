import os
import time
import random
import matplotlib.pyplot as plt
import numpy as np
from movie_rating_queries import MovieRatingQueries

def run_benchmarks(ratings_file, time_window_sizes=[3600*24*30, 3600*24*90, 3600*24*180]):
    """
    运行不同时间窗口大小的基准测试
    
    参数:
        ratings_file: 评分文件路径
        time_window_sizes: 要测试的时间窗口大小列表
    """
    results = {
        "窗口大小": [],
        "加载时间(秒)": [],
        "内存占用(MB)": [],
        "成员查询时间(秒)": [],
        "频度查询时间(秒)": [],
        "Top-k查询时间(秒)": []
    }
    
    for window_size in time_window_sizes:
        print(f"\n===== 测试时间窗口大小: {window_size/3600/24:.1f}天 =====")
        
        # 初始化查询系统并记录加载时间
        start_time = time.time()
        query_system = MovieRatingQueries(ratings_file, time_window_size=window_size)
        load_time = time.time() - start_time
        
        # 获取内存使用情况
        memory_usage = query_system.memory_usage()
        
        # 准备测试数据
        movie_ids = list(query_system.movie_ids)
        test_movie_ids = random.sample(movie_ids, min(1000, len(movie_ids)))
        timestamps = np.linspace(
            query_system.min_timestamp, 
            query_system.max_timestamp, 
            20
        ).astype(int)
        
        # 测试成员查询
        start_time = time.time()
        for movie_id in test_movie_ids[:100]:  # 使用前100个电影ID
            for timestamp in timestamps[:5]:  # 使用前5个时间戳
                query_system.membership_query(movie_id, timestamp)
        membership_time = (time.time() - start_time) / (100 * 5)
        
        # 测试频度查询
        start_time = time.time()
        for movie_id in test_movie_ids[:100]:
            for timestamp in timestamps[:5]:
                query_system.frequency_query(movie_id, timestamp)
        frequency_time = (time.time() - start_time) / (100 * 5)
        
        # 测试Top-k查询
        start_time = time.time()
        for k in [10, 50, 100]:
            for timestamp in timestamps[:5]:
                query_system.topk_query(k, timestamp)
        topk_time = (time.time() - start_time) / (3 * 5)
        
        # 记录结果
        results["窗口大小"].append(f"{window_size/3600/24:.1f}天")
        results["加载时间(秒)"].append(load_time)
        results["内存占用(MB)"].append(memory_usage["总占用 (MB)"])
        results["成员查询时间(秒)"].append(membership_time)
        results["频度查询时间(秒)"].append(frequency_time)
        results["Top-k查询时间(秒)"].append(topk_time)
        
        print(f"加载时间: {load_time:.2f}秒")
        print(f"内存占用: {memory_usage['总占用 (MB)']:.2f} MB")
        print(f"平均成员查询时间: {membership_time*1000:.4f} 毫秒")
        print(f"平均频度查询时间: {frequency_time*1000:.4f} 毫秒")
        print(f"平均Top-k查询时间: {topk_time*1000:.4f} 毫秒")
    
    return results

def plot_results(results):
    """绘制基准测试结果"""
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('电影评分查询系统性能分析', fontsize=16)
    
    # 绘制加载时间
    axes[0, 0].bar(results["窗口大小"], results["加载时间(秒)"])
    axes[0, 0].set_ylabel('时间 (秒)')
    axes[0, 0].set_title('数据加载时间')
    
    # 绘制内存占用
    axes[0, 1].bar(results["窗口大小"], results["内存占用(MB)"])
    axes[0, 1].set_ylabel('内存 (MB)')
    axes[0, 1].set_title('内存占用')
    
    # 绘制查询时间
    query_times = [
        results["成员查询时间(秒)"],
        results["频度查询时间(秒)"],
        results["Top-k查询时间(秒)"]
    ]
    labels = ['成员查询', '频度查询', 'Top-k查询']
    
    x = np.arange(len(results["窗口大小"]))
    width = 0.25
    
    for i, (times, label) in enumerate(zip(query_times, labels)):
        axes[1, 0].bar(x + i*width, [t*1000 for t in times], width, label=label)
    
    axes[1, 0].set_xticks(x + width)
    axes[1, 0].set_xticklabels(results["窗口大小"])
    axes[1, 0].set_ylabel('时间 (毫秒)')
    axes[1, 0].set_title('查询响应时间')
    axes[1, 0].legend()
    
    # 总结
    axes[1, 1].axis('off')
    summary_text = (
        "系统性能总结:\n\n"
        f"1. 平均成员查询时间: {np.mean(results['成员查询时间(秒)'])*1000:.4f} 毫秒\n"
        f"2. 平均频度查询时间: {np.mean(results['频度查询时间(秒)'])*1000:.4f} 毫秒\n"
        f"3. 平均Top-k查询时间: {np.mean(results['Top-k查询时间(秒)'])*1000:.4f} 毫秒\n"
        f"4. 平均内存占用: {np.mean(results['内存占用(MB)']):.2f} MB\n\n"
        "结论:\n"
        "- 时间窗口越大，内存占用越少，但加载时间可能增加\n"
        "- 成员查询和频度查询性能表现优秀\n"
        "- Top-k查询相对耗时较长，但仍在毫秒级别\n"
    )
    axes[1, 1].text(0, 0.5, summary_text, fontsize=10, verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('benchmark_results.png')
    plt.close()

if __name__ == "__main__":
    import os
    
    ratings_file = os.path.join("ml-25m", "ratings.csv")
    
    if os.path.exists(ratings_file):
        print(f"找到评分文件: {ratings_file}")
        
        # 使用多个时间窗口大小运行测试
        results = run_benchmarks(
            ratings_file, 
            time_window_sizes=[
                3600*24*30,    # 30天
                3600*24*90,    # 90天
                3600*24*180    # 180天
            ]
        )
        
        # 绘制结果
        plot_results(results)
        print("\n基准测试完成，结果已保存为 benchmark_results.png")
    else:
        print(f"错误: 找不到评分文件 {ratings_file}")
        print("请确保MovieLens 25M数据集已解压到正确位置") 