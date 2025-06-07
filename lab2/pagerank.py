import numpy as np
import time
from collections import defaultdict, Counter

class PageRank:
    def __init__(self, damping_factor=0.85):
        """
        初始化PageRank算法
        
        参数:
            damping_factor: 阻尼系数，通常设为0.85
        """
        self.damping_factor = damping_factor
        self.graph = None
        self.node_map = None
        self.reverse_node_map = None
        self.pagerank_values = None
        self.num_nodes = 0
        
    def load_graph(self, file_path):
        """
        从文件加载图数据
        
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
        
        # 创建节点ID到索引的映射（为了使用数组表示PageRank值）
        node_map = {node: idx for idx, node in enumerate(sorted(all_nodes))}
        reverse_node_map = {idx: node for node, idx in node_map.items()}
        
        # 存储图数据和映射
        self.graph = graph
        self.node_map = node_map
        self.reverse_node_map = reverse_node_map
        self.num_nodes = len(node_map)
        
        end_time = time.time()
        print(f"数据加载完成，共 {self.num_nodes} 个节点，耗时 {end_time - start_time:.2f} 秒")
        
        return self.num_nodes
    
    def calculate_pagerank(self, max_iterations=100, convergence_threshold=1e-6, verbose=True):
        """
        计算PageRank值
        
        参数:
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            verbose: 是否显示详细信息
            
        返回:
            迭代次数, 计算时间
        """
        if self.graph is None:
            raise ValueError("请先加载图数据")
        
        start_time = time.time()
        
        # 初始化PageRank值为均匀分布
        pagerank = np.ones(self.num_nodes) / self.num_nodes
        
        # 计算每个节点的出链数量
        out_degree = Counter()
        for node, neighbors in self.graph.items():
            out_degree[node] = len(neighbors)
        
        # 创建随机游走矩阵的稀疏表示
        iterations = 0
        converged = False
        
        if verbose:
            print(f"开始PageRank迭代计算，收敛阈值: {convergence_threshold}")
        
        while iterations < max_iterations and not converged:
            new_pagerank = np.zeros(self.num_nodes)
            
            # 分配来自随机游走的贡献
            for node, neighbors in self.graph.items():
                if node in self.node_map and out_degree[node] > 0:
                    node_idx = self.node_map[node]
                    contribution = self.damping_factor * pagerank[node_idx] / out_degree[node]
                    
                    for neighbor in neighbors:
                        if neighbor in self.node_map:
                            neighbor_idx = self.node_map[neighbor]
                            new_pagerank[neighbor_idx] += contribution
            
            # 添加随机跳转贡献
            teleport_contribution = (1 - self.damping_factor) / self.num_nodes
            new_pagerank += teleport_contribution
            
            # 处理悬挂节点（没有出链的节点）
            for node in self.node_map:
                if node not in self.graph or out_degree[node] == 0:
                    node_idx = self.node_map[node]
                    # 悬挂节点将其PageRank值均匀分配给所有其他节点
                    new_pagerank += self.damping_factor * pagerank[node_idx] / self.num_nodes
            
            # 检查收敛性
            diff = np.sum(np.abs(new_pagerank - pagerank))
            pagerank = new_pagerank
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
        self.pagerank_values = pagerank
        
        return iterations, calculation_time
    
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
    
    def benchmark_convergence(self, thresholds, max_iterations=1000):
        """
        测试不同收敛阈值下的性能
        
        参数:
            thresholds: 收敛阈值列表
            max_iterations: 最大迭代次数
            
        返回:
            包含(阈值, 迭代次数, 计算时间)的列表
        """
        results = []
        
        for threshold in thresholds:
            print(f"\n测试收敛阈值: {threshold}")
            iterations, calculation_time = self.calculate_pagerank(
                max_iterations=max_iterations,
                convergence_threshold=threshold,
                verbose=False
            )
            
            results.append((threshold, iterations, calculation_time))
            print(f"阈值: {threshold}, 迭代次数: {iterations}, 计算时间: {calculation_time:.2f}秒")
        
        return results 