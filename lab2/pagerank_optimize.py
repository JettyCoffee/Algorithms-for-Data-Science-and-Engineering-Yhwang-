import numpy as np
import time
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
from scipy.sparse import csr_matrix, lil_matrix

# 将process_chunk定义为全局函数，使其可以被pickle
def process_chunk(matrix, vector, start, end):
    """计算部分矩阵和向量的乘积"""
    return matrix[start:end, :].dot(vector)

class OptimizedPageRank:
    def __init__(self, damping_factor=0.85):
        """
        初始化优化版PageRank算法
        
        参数:
            damping_factor: 阻尼系数，通常设为0.85
        """
        self.damping_factor = damping_factor
        self.graph = None
        self.node_map = None
        self.reverse_node_map = None
        self.pagerank_values = None
        self.num_nodes = 0
        self.transition_matrix = None
        
    def load_graph(self, file_path):
        """
        从文件加载图数据并构建转移矩阵
        
        参数:
            file_path: 图数据文件路径
        """
        # 用于存储图的结构，使用邻接表表示
        graph = defaultdict(list)
        # 记录所有出现的节点ID
        all_nodes = set()
        
        print(f"开始加载数据: {file_path}")
        start_time = time.time()
        
        with open(file_path, 'r') as f:
            for line in f:
                # 跳过注释行
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split()
                if len(parts) >= 2:
                    from_node = int(parts[0])
                    to_node = int(parts[1])
                    
                    # 添加到图中
                    graph[from_node].append(to_node)
                    
                    # 记录节点
                    all_nodes.add(from_node)
                    all_nodes.add(to_node)
        
        # 创建节点ID到索引的映射
        node_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
        reverse_node_map = {idx: node for node, idx in node_map.items()}
        
        # 存储图数据和映射
        self.graph = graph
        self.node_map = node_map
        self.reverse_node_map = reverse_node_map
        self.num_nodes = len(node_map)
        
        # 构建转移矩阵（使用稀疏矩阵）
        self._build_transition_matrix()
        
        end_time = time.time()
        print(f"数据加载完成，共 {self.num_nodes} 个节点，耗时 {end_time - start_time:.2f} 秒")
        
        return self.num_nodes
    
    def _build_transition_matrix(self):
        """构建转移概率矩阵"""
        n = self.num_nodes
        # 使用稀疏矩阵以节省内存
        M = lil_matrix((n, n), dtype=np.float64)
        
        # 记录悬挂节点（无出链的节点）
        dangling_nodes = []
        
        # 构建初始转移矩阵
        for node, neighbors in self.graph.items():
            if node in self.node_map:
                i = self.node_map[node]
                out_degree = len(neighbors)
                
                if out_degree > 0:
                    # 计算转移概率 1/出度
                    prob = 1.0 / out_degree
                    for neighbor in neighbors:
                        if neighbor in self.node_map:
                            j = self.node_map[neighbor]
                            M[j, i] = prob  # 列表示从i到j的概率
                else:
                    # 记录悬挂节点
                    dangling_nodes.append(i)
        
        # 转换为CSR格式，更适合矩阵运算
        self.transition_matrix = M.tocsr()
        self.dangling_nodes = dangling_nodes
        
    def matrix_vector_multiply_parallel(self, matrix, vector, num_processes=None):
        """并行计算矩阵向量乘法"""
        if num_processes is None:
            num_processes = multiprocessing.cpu_count()
        
        n = matrix.shape[0]
        chunk_size = max(1, n // num_processes)
        chunks = [(i, min(i + chunk_size, n)) for i in range(0, n, chunk_size)]
        
        result = np.zeros(n)
        
        # 使用ThreadPoolExecutor代替ProcessPoolExecutor
        # 线程池不需要pickle函数，且对于这种IO密集型任务更合适
        with ThreadPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_chunk, matrix, vector, start, end) for start, end in chunks]
            
            for i, future in enumerate(futures):
                start, end = chunks[i]
                result[start:end] = future.result()
        
        return result
    
    def power_iteration(self, max_iterations=100, convergence_threshold=1e-6, verbose=True, use_parallel=True):
        """使用幂迭代法计算PageRank，可选并行计算"""
        if self.transition_matrix is None:
            raise ValueError("请先加载图数据并构建转移矩阵")
        
        n = self.num_nodes
        
        # 初始化PageRank向量，均匀分布
        v = np.ones(n) / n
        
        # 开始迭代计算
        iterations = 0
        converged = False
        
        if verbose:
            print(f"开始PageRank迭代计算，收敛阈值: {convergence_threshold}")
        
        start_time = time.time()
        
        # 创建单位向量用于处理随机跳转
        e = np.ones(n)
        
        # 创建悬挂节点处理向量
        dangling_weights = np.ones(n) / n
        
        # 记录悬挂节点
        dangling_vector = np.zeros(n)
        for i in self.dangling_nodes:
            dangling_vector[i] = 1
        
        while iterations < max_iterations and not converged:
            # 保存上一次迭代的结果，用于计算差值
            prev_v = v.copy()
            
            # 计算来自悬挂节点的贡献（均匀分布到所有节点）
            dangling_contrib = self.damping_factor * np.sum(v * dangling_vector) * dangling_weights
            
            # 使用转移矩阵计算新的PageRank值（可选并行）
            if use_parallel and n > 10000:  # 只有大矩阵才使用并行计算
                new_v = self.matrix_vector_multiply_parallel(self.transition_matrix, v)
                new_v = self.damping_factor * new_v
            else:
                new_v = self.damping_factor * self.transition_matrix.dot(v)
            
            # 添加随机跳转和悬挂节点贡献
            new_v += (1 - self.damping_factor) * e / n + dangling_contrib
            
            # 规范化向量（确保总和为1）
            new_v = new_v / np.sum(new_v)
            
            # 计算差值并更新
            diff = np.sum(np.abs(new_v - prev_v))
            v = new_v
            iterations += 1
            
            if verbose and iterations % 10 == 0:
                print(f"迭代 {iterations}, 差异: {diff:.8f}")
                
            if diff < convergence_threshold:
                converged = True
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        if verbose:
            if converged:
                print(f"PageRank计算收敛，共迭代 {iterations} 次，耗时 {calculation_time:.2f} 秒")
            else:
                print(f"达到最大迭代次数 {max_iterations}，未收敛，差异: {diff:.8f}，耗时 {calculation_time:.2f} 秒")
        
        # 存储计算结果
        self.pagerank_values = v
        
        return iterations, calculation_time
    
    def matrix_power_method(self, max_iterations=100, convergence_threshold=1e-6, verbose=True):
        """使用矩阵快速幂方法计算PageRank"""
        if self.transition_matrix is None:
            raise ValueError("请先加载图数据并构建转移矩阵")
        
        n = self.num_nodes
        
        # 初始化PageRank向量，均匀分布
        v = np.ones(n) / n
        
        # 开始迭代计算
        iterations = 0
        converged = False
        
        if verbose:
            print(f"开始使用矩阵快速幂方法计算PageRank，收敛阈值: {convergence_threshold}")
        
        start_time = time.time()
        
        # 修正转移矩阵，处理悬挂节点和随机跳转
        e = np.ones((n, 1)) / n  # 全1列向量，用于随机跳转
        
        # 对于悬挂节点，设置为均匀分布
        a = np.zeros((1, n))
        for i in self.dangling_nodes:
            a[0, i] = 1
        
        # Google矩阵 = β*M + β*(a^T)(e/n) + (1-β)*(e/n)
        # 由于稀疏矩阵操作，我们将分步计算
        
        # 计算 β*M 部分
        M = self.transition_matrix.copy()
        M = M.multiply(self.damping_factor)
        
        while iterations < max_iterations and not converged:
            # 保存上一次迭代的结果，用于计算差值
            prev_v = v.copy()
            
            # 计算 M*v
            Mv = M.dot(v)
            
            # 添加悬挂节点的贡献
            dangling_contrib = self.damping_factor * np.sum(v[self.dangling_nodes]) / n
            
            # 添加随机跳转贡献
            teleport_contrib = (1 - self.damping_factor) / n
            
            # 组合所有贡献
            new_v = Mv + dangling_contrib * np.ones(n) + teleport_contrib * np.ones(n)
            
            # 规范化向量（确保总和为1）
            new_v = new_v / np.sum(new_v)
            
            # 计算差值并更新
            diff = np.sum(np.abs(new_v - prev_v))
            v = new_v
            iterations += 1
            
            if verbose and iterations % 10 == 0:
                print(f"迭代 {iterations}, 差异: {diff:.8f}")
                
            if diff < convergence_threshold:
                converged = True
        
        end_time = time.time()
        calculation_time = end_time - start_time
        
        if verbose:
            if converged:
                print(f"PageRank计算收敛，共迭代 {iterations} 次，耗时 {calculation_time:.2f} 秒")
            else:
                print(f"达到最大迭代次数 {max_iterations}，未收敛，差异: {diff:.8f}，耗时 {calculation_time:.2f} 秒")
        
        # 存储计算结果
        self.pagerank_values = v
        
        return iterations, calculation_time
    
    def calculate_pagerank(self, method="power_iteration", max_iterations=100, convergence_threshold=1e-6, verbose=True, use_parallel=True):
        """
        计算PageRank值，支持多种计算方法
        
        参数:
            method: 计算方法，"power_iteration"或"matrix_power"
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            verbose: 是否显示详细信息
            use_parallel: 是否使用并行计算
            
        返回:
            迭代次数, 计算时间
        """
        if method == "power_iteration":
            return self.power_iteration(max_iterations, convergence_threshold, verbose, use_parallel)
        elif method == "matrix_power":
            return self.matrix_power_method(max_iterations, convergence_threshold, verbose)
        else:
            raise ValueError(f"未知的计算方法: {method}，可用选项: power_iteration, matrix_power")
    
    def get_top_k_pages(self, k=10):
        """
        获取PageRank值最高的k个网页
        
        参数:
            k: 返回的网页数量
            
        返回:
            包含(节点ID, PageRank值)的列表
        """
        if self.pagerank_values is None:
            raise ValueError("请先计算PageRank值")
        
        # 获取所有(索引, PageRank值)对
        pagerank_with_index = [(i, pr) for i, pr in enumerate(self.pagerank_values)]
        
        # 按PageRank值排序
        pagerank_with_index.sort(key=lambda x: x[1], reverse=True)
        
        # 取前k个并转换回原始节点ID
        top_k = []
        for i in range(min(k, self.num_nodes)):
            idx, pr = pagerank_with_index[i]
            node_id = self.reverse_node_map[idx]
            top_k.append((node_id, pr))
        
        return top_k
    
    def benchmark_convergence(self, method="power_iteration", thresholds=None, max_iterations=1000, use_parallel=True):
        """
        测试不同收敛阈值下的性能
        
        参数:
            method: 计算方法
            thresholds: 收敛阈值列表
            max_iterations: 最大迭代次数
            use_parallel: 是否使用并行计算
            
        返回:
            包含(阈值, 迭代次数, 计算时间)的列表
        """
        if thresholds is None:
            thresholds = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
            
        results = []
        
        for threshold in thresholds:
            print(f"\n测试计算方法: {method}, 收敛阈值: {threshold}")
            iterations, calculation_time = self.calculate_pagerank(
                method=method,
                max_iterations=max_iterations,
                convergence_threshold=threshold,
                verbose=False,
                use_parallel=use_parallel
            )
            
            results.append((threshold, iterations, calculation_time))
            print(f"方法: {method}, 阈值: {threshold}, 迭代次数: {iterations}, 计算时间: {calculation_time:.2f}秒")
        
        return results

    def compare_methods(self, thresholds=None, max_iterations=100):
        """
        比较不同方法的性能
        
        参数:
            thresholds: 收敛阈值列表
            max_iterations: 最大迭代次数
        """
        if thresholds is None:
            thresholds = [1e-6]
            
        methods = [
            ("power_iteration", False),
            ("power_iteration", True),
            ("matrix_power", False)
        ]
        
        results = {}
        
        for method, use_parallel in methods:
            method_name = f"{method}" + ("_parallel" if use_parallel else "")
            results[method_name] = []
            
            for threshold in thresholds:
                print(f"\n测试方法: {method_name}, 收敛阈值: {threshold}")
                
                # 重置PageRank值
                self.pagerank_values = None
                
                # 计算PageRank
                iterations, calculation_time = self.calculate_pagerank(
                    method=method,
                    max_iterations=max_iterations,
                    convergence_threshold=threshold,
                    verbose=False,
                    use_parallel=use_parallel
                )
                
                results[method_name].append((threshold, iterations, calculation_time))
                print(f"方法: {method_name}, 阈值: {threshold}, 迭代次数: {iterations}, 计算时间: {calculation_time:.2f}秒")
        
        return results 