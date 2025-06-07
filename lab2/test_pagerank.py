import sys
import time
from pagerank import PageRank

def test_slashdot():
    """
    在Slashdot数据集上测试PageRank算法
    """
    print("\n========== 测试Slashdot数据集 ==========")
    
    # 创建PageRank对象
    pr = PageRank(damping_factor=0.85)
    
    # 加载数据
    pr.load_graph("Slashdot0902.txt")
    
    # 计算PageRank
    pr.calculate_pagerank(max_iterations=100, convergence_threshold=1e-6)
    
    # 获取前10个PageRank值最高的网页
    top_10 = pr.get_top_k_pages(k=10)
    
    print("\nSlashdot数据集 - PageRank值最高的10个网页:")
    for i, (node_id, pr_value) in enumerate(top_10):
        print(f"#{i+1}: 网页ID={node_id}, PageRank值={pr_value:.8f}")
    
    # 测试不同收敛阈值
    thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    results = pr.benchmark_convergence(thresholds)
    
    print("\nSlashdot数据集 - 不同收敛阈值下的性能:")
    print("阈值\t迭代次数\t计算时间(秒)")
    for threshold, iterations, calc_time in results:
        print(f"{threshold:.0e}\t{iterations}\t{calc_time:.2f}")
    
    return results

def test_google():
    """
    在Google数据集上测试PageRank算法
    """
    print("\n========== 测试Google数据集 ==========")
    
    # 创建PageRank对象
    pr = PageRank(damping_factor=0.85)
    
    # 加载数据
    pr.load_graph("web-Google.txt")
    
    # 计算PageRank
    pr.calculate_pagerank(max_iterations=100, convergence_threshold=1e-6)
    
    # 获取前10个PageRank值最高的网页
    top_10 = pr.get_top_k_pages(k=10)
    
    print("\nGoogle数据集 - PageRank值最高的10个网页:")
    for i, (node_id, pr_value) in enumerate(top_10):
        print(f"#{i+1}: 网页ID={node_id}, PageRank值={pr_value:.8f}")
    
    # 测试不同收敛阈值
    thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    results = pr.benchmark_convergence(thresholds)
    
    print("\nGoogle数据集 - 不同收敛阈值下的性能:")
    print("阈值\t迭代次数\t计算时间(秒)")
    for threshold, iterations, calc_time in results:
        print(f"{threshold:.0e}\t{iterations}\t{calc_time:.2f}")
    
    return results

if __name__ == "__main__":
    # 测试所有数据集或根据命令行参数选择
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "slashdot":
            test_slashdot()
        elif sys.argv[1].lower() == "google":
            test_google()
        else:
            print(f"未知的数据集: {sys.argv[1]}")
            print("可用选项: slashdot, google")
    else:
        # 默认测试所有数据集
        print("开始测试所有数据集...")
        
        start_time = time.time()
        
        slashdot_results = test_slashdot()
        google_results = test_google()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n所有测试完成，总耗时: {total_time:.2f}秒") 