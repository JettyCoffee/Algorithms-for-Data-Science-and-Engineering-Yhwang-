import csv
import heapq
import time
import numpy as np
from collections import defaultdict
from bitarray import bitarray
import mmh3  # MurmurHash3 哈希函数库
from pympler import asizeof

class CountMinSketch:
    """Count-Min Sketch 数据结构，用于频率估计"""
    
    def __init__(self, width=1000, depth=5):
        """
        初始化Count-Min Sketch
        
        参数:
            width: 哈希表宽度
            depth: 哈希表深度（哈希函数个数）
        """
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=np.int32)
        
    def update(self, key, count=1):
        """增加计数"""
        for i in range(self.depth):
            j = mmh3.hash(str(key), i) % self.width
            self.table[i, j] += count
    
    def estimate(self, key):
        """估计频率"""
        result = float('inf')
        for i in range(self.depth):
            j = mmh3.hash(str(key), i) % self.width
            result = min(result, self.table[i, j])
        return result
    
    def get_size_in_bytes(self):
        """获取占用空间大小（字节）"""
        return self.table.size * self.table.itemsize
    
    def optimize(self, threshold=0.01):
        """
        优化Count-Min Sketch内存使用
        
        参数:
            threshold: 清除小于最大值threshold倍的计数
        """
        # 找出每行的最大值
        max_values = np.max(self.table, axis=1)
        
        # 对于每一行，将小于max_value * threshold的值设为0
        for i in range(self.depth):
            self.table[i, self.table[i, :] < max_values[i] * threshold] = 0
        
        # 将稀疏矩阵转换为更节省空间的表示 (如果numpy支持的话)
        if hasattr(np, 'sparse'):
            try:
                from scipy import sparse
                # 将numpy数组转换为稀疏矩阵 (如果稀疏程度足够高)
                if np.count_nonzero(self.table) / self.table.size < 0.3:
                    self.table = sparse.csr_matrix(self.table)
                    self.is_sparse = True
                    return True
            except ImportError:
                pass
        
        return False


class OptimizedMovieQueries:
    """
    优化的电影评分查询系统，使用概率数据结构减少内存使用
    实现三种查询功能:
    1. 成员查询: 检查电影在指定时间戳前是否被评分
    2. 频度查询: 查询电影在指定时间戳前的评分次数
    3. Top-k查询: 查询指定时间戳前评分次数最多的前k个电影
    """
    
    def __init__(self, ratings_file, time_window_size=3600*24*30, sketch_width=10000):
        """
        初始化查询系统
        
        参数:
            ratings_file: ratings.csv文件路径
            time_window_size: 时间窗口大小，默认30天(秒)
            sketch_width: Count-Min Sketch宽度
        """
        self.ratings_file = ratings_file
        self.time_window_size = time_window_size
        self.sketch_width = sketch_width
        
        # 读取最小和最大时间戳
        self.min_timestamp, self.max_timestamp, self.movie_ids = self._scan_timestamps_and_movies()
        
        # 初始化数据结构
        self._init_data_structures()
        
        # 加载数据
        self._load_data()
    
    def _scan_timestamps_and_movies(self):
        """扫描文件获取时间戳范围和所有电影ID"""
        min_ts = float('inf')
        max_ts = 0
        movie_ids = set()
        
        with open(self.ratings_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            for row in reader:
                movie_id = int(row[1])
                timestamp = int(row[3])
                min_ts = min(min_ts, timestamp)
                max_ts = max(max_ts, timestamp)
                movie_ids.add(movie_id)
        
        return min_ts, max_ts, movie_ids
    
    def _init_data_structures(self):
        """初始化数据结构"""
        # 计算时间窗口数量
        self.num_windows = (self.max_timestamp - self.min_timestamp) // self.time_window_size + 1
        
        # 为每个时间窗口创建布隆过滤器 (成员查询)
        self.bloom_filters = []
        # 假定每个窗口最多包含100万条记录，假阳性率为0.01
        bloom_size = self._calculate_bloom_filter_size(1000000, 0.01)
        bloom_hashes = self._calculate_bloom_filter_hashes(bloom_size, 1000000)
        
        for _ in range(self.num_windows):
            bf = bitarray(bloom_size)
            bf.setall(0)
            self.bloom_filters.append(bf)
        
        self.bloom_hash_funcs = self._create_hash_functions(bloom_hashes)
        
        # 使用Count-Min Sketch进行频度估计 (频度查询)
        self.count_min_sketches = [CountMinSketch(width=self.sketch_width, depth=5) 
                                  for _ in range(self.num_windows)]
        
        # 保存每个窗口中评分频率最高的电影 (用于Top-k查询)
        self.top_movies_per_window = [defaultdict(int) for _ in range(self.num_windows)]
        
        # 累积Top-k结果
        self.cumulative_top_movies = [defaultdict(int) for _ in range(self.num_windows)]
        
        # 确定高频电影阈值 (保存top 5%的电影精确计数)
        self.top_movie_threshold = max(100, len(self.movie_ids) // 20)
    
    def _calculate_bloom_filter_size(self, n, p):
        """计算布隆过滤器大小"""
        m = -1 * (n * np.log(p)) / (np.log(2) ** 2)
        return int(m)
    
    def _calculate_bloom_filter_hashes(self, m, n):
        """计算布隆过滤器哈希函数个数"""
        k = (m / n) * np.log(2)
        return max(1, int(k))
    
    def _create_hash_functions(self, num_functions):
        """创建哈希函数"""
        def hash_func(i):
            def hash_i(x):
                return mmh3.hash(str(x), i) % len(self.bloom_filters[0])
            return hash_i
        
        return [hash_func(i) for i in range(num_functions)]
    
    def _get_window_index(self, timestamp):
        """获取时间戳对应的窗口索引"""
        if timestamp < self.min_timestamp:
            return -1
        return min(self.num_windows - 1, (timestamp - self.min_timestamp) // self.time_window_size)
    
    def _add_to_bloom_filter(self, window_idx, movie_id):
        """将电影ID添加到对应窗口的布隆过滤器"""
        for hash_func in self.bloom_hash_funcs:
            bit_pos = hash_func(movie_id)
            self.bloom_filters[window_idx][bit_pos] = 1
    
    def _is_in_bloom_filter(self, window_idx, movie_id):
        """检查电影ID是否在对应窗口的布隆过滤器中"""
        for hash_func in self.bloom_hash_funcs:
            bit_pos = hash_func(movie_id)
            if not self.bloom_filters[window_idx][bit_pos]:
                return False
        return True
    
    def _load_data(self):
        """加载评分数据到数据结构"""
        print("开始加载数据...")
        start_time = time.time()
        
        # 第一遍：统计每个窗口中每部电影的评分次数
        window_movie_counts = [defaultdict(int) for _ in range(self.num_windows)]
        
        with open(self.ratings_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            
            for row in reader:
                movie_id = int(row[1])
                timestamp = int(row[3])
                
                window_idx = self._get_window_index(timestamp)
                if window_idx >= 0:
                    window_movie_counts[window_idx][movie_id] += 1
        
        # 对于每个窗口，确定需要保存精确计数的高频电影
        for window_idx, counts in enumerate(window_movie_counts):
            # 选择评分次数最多的电影保存精确计数
            top_movies = heapq.nlargest(self.top_movie_threshold, 
                                        counts.items(), key=lambda x: x[1])
            self.top_movies_per_window[window_idx] = dict(top_movies)
        
        # 第二遍：加载到数据结构
        with open(self.ratings_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            
            for row in reader:
                movie_id = int(row[1])
                timestamp = int(row[3])
                
                window_idx = self._get_window_index(timestamp)
                if window_idx >= 0:
                    # 添加到布隆过滤器（成员查询）
                    self._add_to_bloom_filter(window_idx, movie_id)
                    
                    # 更新Count-Min Sketch（频度查询）
                    self.count_min_sketches[window_idx].update(movie_id)
        
        # 构建累积Top-k结果
        for i in range(self.num_windows):
            if i == 0:
                self.cumulative_top_movies[i] = dict(self.top_movies_per_window[i])
            else:
                # 合并前一个窗口的结果和当前窗口的高频电影
                self.cumulative_top_movies[i] = defaultdict(int, self.cumulative_top_movies[i-1])
                for movie_id, count in self.top_movies_per_window[i].items():
                    self.cumulative_top_movies[i][movie_id] += count
        
        print(f"数据加载完成，耗时 {time.time() - start_time:.2f} 秒")
    
    def membership_query(self, movie_id, timestamp):
        """
        成员查询：检查电影在指定时间戳前是否被评分
        
        参数:
            movie_id: 电影ID
            timestamp: 查询时间戳
            
        返回:
            bool: 是否被评分过
        """
        if timestamp < self.min_timestamp or movie_id not in self.movie_ids:
            return False
        
        window_idx = self._get_window_index(timestamp)
        
        # 检查每个窗口的布隆过滤器
        for i in range(window_idx + 1):
            if self._is_in_bloom_filter(i, movie_id):
                return True
                
        return False
    
    def frequency_query(self, movie_id, timestamp):
        """
        频度查询：查询电影在指定时间戳前的评分次数
        
        参数:
            movie_id: 电影ID
            timestamp: 查询时间戳
            
        返回:
            int: 评分次数（可能是估计值）
        """
        if timestamp < self.min_timestamp or movie_id not in self.movie_ids:
            return 0
        
        window_idx = self._get_window_index(timestamp)
        
        # 累积所有窗口中的评分次数
        total_count = 0
        for i in range(window_idx + 1):
            # 检查是否有精确计数
            if movie_id in self.top_movies_per_window[i]:
                total_count += self.top_movies_per_window[i][movie_id]
            else:
                # 使用Count-Min Sketch估计频率
                total_count += self.count_min_sketches[i].estimate(movie_id)
        
        return total_count
    
    def topk_query(self, k, timestamp):
        """
        Top-k查询：查询指定时间戳前评分次数最多的前k个电影
        
        参数:
            k: 返回电影数量
            timestamp: 查询时间戳
            
        返回:
            list: 包含(movie_id, count)元组的列表
        """
        if timestamp < self.min_timestamp:
            return []
        
        window_idx = self._get_window_index(timestamp)
        
        # 获取累积的高频电影计数
        counts = self.cumulative_top_movies[window_idx]
        
        # 如果需要更多的前k个电影，则需要计算其他电影的评分次数
        if len(counts) < k:
            # 为所有电影计算频率
            all_counts = defaultdict(int)
            for movie_id in self.movie_ids:
                all_counts[movie_id] = self.frequency_query(movie_id, timestamp)
            return heapq.nlargest(k, all_counts.items(), key=lambda x: x[1])
        
        # 否则，直接从高频电影列表中获取前k个
        return heapq.nlargest(k, counts.items(), key=lambda x: x[1])

    def memory_usage(self):
        """使用Pympler估计数据结构的内存占用"""
        # 使用Pympler的asizeof进行更准确的内存占用统计
        bloom_size = asizeof.asizeof(self.bloom_filters)
        sketch_size = asizeof.asizeof(self.count_min_sketches)
        top_movies_size = asizeof.asizeof(self.top_movies_per_window)
        cumulative_size = asizeof.asizeof(self.cumulative_top_movies)
        
        # 统计其他重要数据结构的内存占用
        hash_funcs_size = asizeof.asizeof(self.bloom_hash_funcs)
        movie_ids_size = asizeof.asizeof(self.movie_ids)
        
        total_size = bloom_size + sketch_size + top_movies_size + cumulative_size + hash_funcs_size + movie_ids_size
        
        return {
            "布隆过滤器 (bytes)": bloom_size,
            "Count-Min Sketch (bytes)": sketch_size,
            "高频电影计数器 (bytes)": top_movies_size,
            "累积Top-k (bytes)": cumulative_size,
            "哈希函数 (bytes)": hash_funcs_size,
            "电影ID集合 (bytes)": movie_ids_size,
            "总占用 (MB)": total_size / (1024 * 1024)
        }
    
    def optimize_memory(self):
        """尝试优化内存使用"""
        # 计算当前内存占用
        before = self.memory_usage()
        
        # 1. 优化Count-Min Sketch - 稀疏化
        sketches_optimized = 0
        for cms in self.count_min_sketches:
            if cms.optimize(threshold=0.005):
                sketches_optimized += 1
                
        # 2. 对于稀疏窗口的高频电影，转为普通字典
        for i, counter in enumerate(self.top_movies_per_window):
            if len(counter) < len(self.movie_ids) * 0.05:
                self.top_movies_per_window[i] = dict(counter)
                
        # 3. 优化布隆过滤器
        # 这里可以实现更复杂的布隆过滤器压缩策略
        
        # 4. 合并相似度高的时间窗口
        if self.num_windows > 10:
            # 实际应用中可以基于窗口内容相似度合并
            pass
        
        # 计算优化后的内存占用
        after = self.memory_usage()
        
        savings = (before["总占用 (MB)"] - after["总占用 (MB)"]) / before["总占用 (MB)"] * 100
        
        return {
            "优化前 (MB)": before["总占用 (MB)"],
            "优化后 (MB)": after["总占用 (MB)"],
            "优化的Sketch数量": sketches_optimized,
            "节省比例": f"{savings:.2f}%"
        }


# 示例使用方法
if __name__ == "__main__":
    import os
    
    # 假设ml-25m数据集在当前目录下的ml-25m文件夹中
    ratings_file = os.path.join("ml-25m", "ratings.csv")
    
    # 初始化查询系统
    query_system = OptimizedMovieQueries(ratings_file, time_window_size=3600*24*90)  # 90天窗口
    
    # 性能测试
    print("\n==== 性能测试 ====")
    
    # 测试成员查询
    start_time = time.time()
    movie_id = 100  # 测试电影ID
    timestamp = int(time.time())  # 当前时间
    result = query_system.membership_query(movie_id, timestamp)
    print(f"成员查询结果: 电影{movie_id}在时间戳{timestamp}前{'被评分过' if result else '未被评分'}")
    print(f"成员查询耗时: {time.time() - start_time:.6f}秒")
    
    # 测试频度查询
    start_time = time.time()
    result = query_system.frequency_query(movie_id, timestamp)
    print(f"频度查询结果: 电影{movie_id}在时间戳{timestamp}前被评分{result}次 (可能是估计值)")
    print(f"频度查询耗时: {time.time() - start_time:.6f}秒")
    
    # 测试Top-k查询
    start_time = time.time()
    k = 10
    results = query_system.topk_query(k, timestamp)
    print(f"Top-{k}查询结果:")
    for movie_id, count in results:
        print(f"  电影ID: {movie_id}, 评分次数: {count}")
    print(f"Top-k查询耗时: {time.time() - start_time:.6f}秒")
    
    # 内存占用
    print("\n==== 内存占用 ====")
    memory_usage = query_system.memory_usage()
    for key, value in memory_usage.items():
        print(f"{key}: {value}")
    
    # 尝试优化内存
    print("\n==== 内存优化 ====")
    optimization_result = query_system.optimize_memory()
    for key, value in optimization_result.items():
        print(f"{key}: {value}") 