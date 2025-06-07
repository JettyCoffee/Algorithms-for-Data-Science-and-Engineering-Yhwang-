import sys
import time
from pagerank_optimize import OptimizedPageRank

def test_slashdot(methods=None):
    """
    在Slashdot数据集上测试优化版PageRank算法
    
    参数:
        methods: 要测试的方法列表，如果为None则测试所有方法
    """
    print("\n========== 测试Slashdot数据集（优化版） ==========")
    
    # 创建PageRank对象
    pr = OptimizedPageRank(damping_factor=0.85)
    
    # 加载数据
    pr.load_graph("Slashdot0902.txt")
    
    if methods is None:
        # 比较不同方法的性能
        thresholds = [1e-6]
        results = pr.compare_methods(thresholds=thresholds, max_iterations=100)
        
        # 输出结果汇总
        print("\nSlashdot数据集 - 不同方法的性能比较（阈值=1e-6）:")
        print("方法\t\t\t迭代次数\t计算时间(秒)")
        for method, result_list in results.items():
            for threshold, iterations, calc_time in result_list:
                print(f"{method}\t\t{iterations}\t\t{calc_time:.2f}")
    else:
        # 使用指定方法计算PageRank
        method = methods[0]
        use_parallel = "_parallel" in method
        method = method.replace("_parallel", "")
        
        # 计算PageRank
        pr.calculate_pagerank(method=method, max_iterations=100, convergence_threshold=1e-6, use_parallel=use_parallel)
    
    # 获取前10个PageRank值最高的网页
    top_10 = pr.get_top_k_pages(k=10)
    
    print("\nSlashdot数据集 - PageRank值最高的10个网页:")
    for i, (node_id, pr_value) in enumerate(top_10):
        print(f"#{i+1}: 网页ID={node_id}, PageRank值={pr_value:.8f}")
    
    # 测试不同收敛阈值（使用最佳方法）
    thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    best_method = "power_iteration"  # 或根据比较结果选择最佳方法
    results = pr.benchmark_convergence(method=best_method, thresholds=thresholds)
    
    print(f"\nSlashdot数据集 - 不同收敛阈值下的性能 (方法={best_method}):")
    print("阈值\t迭代次数\t计算时间(秒)")
    for threshold, iterations, calc_time in results:
        print(f"{threshold:.0e}\t{iterations}\t{calc_time:.2f}")
    
    return results

def test_google(methods=None):
    """
    在Google数据集上测试优化版PageRank算法
    
    参数:
        methods: 要测试的方法列表，如果为None则测试所有方法
    """
    print("\n========== 测试Google数据集（优化版） ==========")
    
    # 创建PageRank对象
    pr = OptimizedPageRank(damping_factor=0.85)
    
    # 加载数据
    pr.load_graph("web-Google.txt")
    
    if methods is None:
        # 比较不同方法的性能
        thresholds = [1e-6]
        results = pr.compare_methods(thresholds=thresholds, max_iterations=100)
        
        # 输出结果汇总
        print("\nGoogle数据集 - 不同方法的性能比较（阈值=1e-6）:")
        print("方法\t\t\t迭代次数\t计算时间(秒)")
        for method, result_list in results.items():
            for threshold, iterations, calc_time in result_list:
                print(f"{method}\t\t{iterations}\t\t{calc_time:.2f}")
    else:
        # 使用指定方法计算PageRank
        method = methods[0]
        use_parallel = "_parallel" in method
        method = method.replace("_parallel", "")
        
        # 计算PageRank
        pr.calculate_pagerank(method=method, max_iterations=100, convergence_threshold=1e-6, use_parallel=use_parallel)
    
    # 获取前10个PageRank值最高的网页
    top_10 = pr.get_top_k_pages(k=10)
    
    print("\nGoogle数据集 - PageRank值最高的10个网页:")
    for i, (node_id, pr_value) in enumerate(top_10):
        print(f"#{i+1}: 网页ID={node_id}, PageRank值={pr_value:.8f}")
    
    # 测试不同收敛阈值（使用最佳方法）
    thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    best_method = "power_iteration"  # 或根据比较结果选择最佳方法
    results = pr.benchmark_convergence(method=best_method, thresholds=thresholds)
    
    print(f"\nGoogle数据集 - 不同收敛阈值下的性能 (方法={best_method}):")
    print("阈值\t迭代次数\t计算时间(秒)")
    for threshold, iterations, calc_time in results:
        print(f"{threshold:.0e}\t{iterations}\t{calc_time:.2f}")
    
    return results

def test_compare_with_original():
    """比较优化版与原始版的性能差异"""
    try:
        from pagerank import PageRank
        
        print("\n========== 比较优化版与原始版的性能差异 ==========")
        
        datasets = ["Slashdot0902.txt", "web-Google.txt"]
        threshold = 1e-6
        
        for dataset in datasets:
            print(f"\n数据集: {dataset}")
            
            # 测试原始版
            print("测试原始版PageRank...")
            pr_orig = PageRank(damping_factor=0.85)
            pr_orig.load_graph(dataset)
            
            start_time = time.time()
            iterations_orig, _ = pr_orig.calculate_pagerank(max_iterations=100, convergence_threshold=threshold, verbose=False)
            orig_time = time.time() - start_time
            
            # 测试优化版（使用不同方法）
            pr_opt = OptimizedPageRank(damping_factor=0.85)
            pr_opt.load_graph(dataset)
            
            methods = [
                ("power_iteration", False),
                ("power_iteration", True),
                ("matrix_power", False)
            ]
            
            for method, use_parallel in methods:
                method_name = f"{method}" + ("_parallel" if use_parallel else "")
                print(f"测试优化版PageRank ({method_name})...")
                
                start_time = time.time()
                iterations_opt, _ = pr_opt.calculate_pagerank(
                    method=method, 
                    max_iterations=100, 
                    convergence_threshold=threshold, 
                    verbose=False,
                    use_parallel=use_parallel
                )
                opt_time = time.time() - start_time
                
                # 计算加速比
                speedup = orig_time / opt_time if opt_time > 0 else float('inf')
                
                print(f"方法: {method_name}, 迭代次数: {iterations_opt}, 计算时间: {opt_time:.2f}秒")
                print(f"原始版: 迭代次数: {iterations_orig}, 计算时间: {orig_time:.2f}秒")
                print(f"加速比: {speedup:.2f}倍")
                
                # 比较结果是否一致
                top_orig = pr_orig.get_top_k_pages(k=5)
                top_opt = pr_opt.get_top_k_pages(k=5)
                
                print("原始版和优化版的Top-5结果比较:")
                for i in range(5):
                    orig_id, orig_val = top_orig[i]
                    opt_id, opt_val = top_opt[i]
                    print(f"#{i+1}: 原始版(ID={orig_id}, PR={orig_val:.8f}) vs 优化版(ID={opt_id}, PR={opt_val:.8f})")
        
    except ImportError:
        print("无法导入原始版PageRank进行比较")

if __name__ == "__main__":
    # 测试所有数据集或根据命令行参数选择
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "slashdot":
            if len(sys.argv) > 2:
                test_slashdot(methods=[sys.argv[2]])
            else:
                test_slashdot()
        elif sys.argv[1].lower() == "google":
            if len(sys.argv) > 2:
                test_google(methods=[sys.argv[2]])
            else:
                test_google()
        elif sys.argv[1].lower() == "compare":
            test_compare_with_original()
        else:
            print(f"未知的数据集或命令: {sys.argv[1]}")
            print("可用选项: slashdot, google, compare")
            print("可以指定方法: python test_pagerank_optimize.py slashdot power_iteration_parallel")
    else:
        # 默认测试所有数据集
        print("开始测试所有数据集（优化版）...")
        
        start_time = time.time()
        
        slashdot_results = test_slashdot()
        google_results = test_google()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n所有测试完成，总耗时: {total_time:.2f}秒")
        
        # 与原始版比较
        print("\n是否进行与原始版的比较测试？(y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            test_compare_with_original() 