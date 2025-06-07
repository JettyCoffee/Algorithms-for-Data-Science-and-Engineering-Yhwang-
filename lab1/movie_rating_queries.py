import csv
import heapq
import time
from collections import defaultdict
import numpy as np
from bitarray import bitarray
from pympler import asizeof

class MovieRatingQueries:
    """
    电影评分查询系统：实现三种查询功能的高效数据结构
    1. 成员查询：检查电影在指定时间戳前是否被评分
    2. 频度查询：查询电影在指定时间戳前的评分次数
    3. Top-k查询：查询指定时间戳前评分次数最多的k部电影
    """
    
    def __init__(self, ratings_file, time_window_size=3600*24*30):
        """
        初始化查询结构
        
        参数:
            ratings_file: ratings.csv文件路径
            time_window_size: 时间窗口大小，默认为30天(秒)
        """
        self.ratings_file = ratings_file
        self.time_window_size = time_window_size
        
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
        
        # 电影评分计数器 (频度查询)
        self.movie_count_per_window = [defaultdict(int) for _ in range(self.num_windows)]
        
        # 累积计数 (用于Top-k查询)
        self.cumulative_counts = [defaultdict(int) for _ in range(self.num_windows)]
    
    def _calculate_bloom_filter_size(self, n, p):
        """计算布隆过滤器大小"""
        m = -1 * (n * np.log(p)) / (np.log(2) ** 2)
        return int(m)
    
    def _calculate_bloom_filter_hashes(self, m, n):
        """计算布隆过滤器哈希函数个数"""
        k = (m / n) * np.log(2)
        return int(k)
    
    def _create_hash_functions(self, num_functions):
        """创建哈希函数"""
        def hash_func(i):
            def hash_i(x):
                return hash(str(x) + str(i)) % len(self.bloom_filters[0])
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
        
        with open(self.ratings_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过标题行
            
            for row in reader:
                movie_id = int(row[1])
                timestamp = int(row[3])
                
                window_idx = self._get_window_index(timestamp)
                if window_idx >= 0:
                    # 添加到布隆过滤器
                    self._add_to_bloom_filter(window_idx, movie_id)
                    
                    # 更新计数器
                    self.movie_count_per_window[window_idx][movie_id] += 1
            
        # 构建累积计数
        for i in range(self.num_windows):
            if i == 0:
                self.cumulative_counts[i] = dict(self.movie_count_per_window[i])
            else:
                self.cumulative_counts[i] = defaultdict(int)
                for movie_id in self.movie_ids:
                    self.cumulative_counts[i][movie_id] = (
                        self.cumulative_counts[i-1].get(movie_id, 0) + 
                        self.movie_count_per_window[i].get(movie_id, 0)
                    )
        
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
            int: 评分次数
        """
        if timestamp < self.min_timestamp or movie_id not in self.movie_ids:
            return 0
        
        window_idx = self._get_window_index(timestamp)
        
        # 使用累积计数直接获取结果
        return self.cumulative_counts[window_idx].get(movie_id, 0)
    
    def topk_query(self, k, timestamp):
        """
        Top-k查询：查询指定时间戳前评分次数最多的k部电影
        
        参数:
            k: 返回电影数量
            timestamp: 查询时间戳
            
        返回:
            list: 包含(movie_id, count)元组的列表
        """
        if timestamp < self.min_timestamp:
            return []
        
        window_idx = self._get_window_index(timestamp)
        
        # 使用堆来找出前k个电影
        counts = self.cumulative_counts[window_idx]
        return heapq.nlargest(k, counts.items(), key=lambda x: x[1])

    def memory_usage(self):
        """使用Pympler估计数据结构的内存占用"""
        # 使用Pympler的asizeof进行更准确的内存占用统计
        bloom_size = asizeof.asizeof(self.bloom_filters)
        counter_size = asizeof.asizeof(self.movie_count_per_window)
        cumulative_size = asizeof.asizeof(self.cumulative_counts)
        
        # 统计其他重要数据结构的内存占用
        hash_funcs_size = asizeof.asizeof(self.bloom_hash_funcs)
        movie_ids_size = asizeof.asizeof(self.movie_ids)
        
        total_size = bloom_size + counter_size + cumulative_size + hash_funcs_size + movie_ids_size
        
        return {
            "布隆过滤器 (bytes)": bloom_size,
            "计数器 (bytes)": counter_size,
            "累积计数 (bytes)": cumulative_size,
            "哈希函数 (bytes)": hash_funcs_size,
            "电影ID集合 (bytes)": movie_ids_size,
            "总占用 (MB)": total_size / (1024 * 1024)
        }
        
    def optimize_memory(self):
        """尝试优化内存使用"""
        # 计算当前内存占用
        before = self.memory_usage()
        
        # 1. 清理不必要的中间结果
        if hasattr(self, '_temp_data'):
            del self._temp_data
            
        # 2. 压缩布隆过滤器 - 如果有多个连续的0或1，可以考虑使用游程编码
        
        # 3. 对于稀疏窗口，考虑使用稀疏表示
        for i, counter in enumerate(self.movie_count_per_window):
            # 如果窗口中的电影数量很少（低于总电影数的10%），转为普通字典
            if len(counter) < len(self.movie_ids) * 0.1:
                self.movie_count_per_window[i] = dict(counter)
        
        # 计算优化后的内存占用
        after = self.memory_usage()
        
        savings = (before["总占用 (MB)"] - after["总占用 (MB)"]) / before["总占用 (MB)"] * 100
        
        return {
            "优化前 (MB)": before["总占用 (MB)"],
            "优化后 (MB)": after["总占用 (MB)"],
            "节省比例": f"{savings:.2f}%"
        }


# 示例使用方法
if __name__ == "__main__":
    import os
    
    # 假设ml-25m数据集在当前目录下的ml-25m文件夹中
    ratings_file = os.path.join("ml-25m", "ratings.csv")
    
    # 初始化查询系统（使用较大的时间窗口以减少内存占用）
    query_system = MovieRatingQueries(ratings_file, time_window_size=3600*24*90)  # 90天窗口
    
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
    print(f"频度查询结果: 电影{movie_id}在时间戳{timestamp}前被评分{result}次")
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