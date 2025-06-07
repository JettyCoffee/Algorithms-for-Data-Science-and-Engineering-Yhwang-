#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于矩阵因式分解的电影推荐系统
使用SVD方法为用户生成个性化的电影推荐列表
"""

import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
import warnings

__all__ = ['MovieRecommender']


class MovieRecommender:
    """电影推荐系统类"""
    
    def __init__(self, data_path, k_factors=50):
        """
        初始化推荐系统
        
        参数:
            data_path: MovieLens数据集的路径
            k_factors: 矩阵分解的隐因子数量
        """
        self.data_path = data_path
        self.k_factors = k_factors
        self.ratings_df = None
        self.movies_df = None
        self.user_ratings_matrix = None
        self.U = None
        self.sigma = None
        self.Vt = None
        self.predicted_ratings = None
        
    def load_data(self):
        """加载MovieLens数据集"""
        # 加载电影数据
        movies_path = os.path.join(self.data_path, 'movies.csv')
        self.movies_df = pd.read_csv(movies_path)
        
        # 加载评分数据
        ratings_path = os.path.join(self.data_path, 'ratings.csv')
        self.ratings_df = pd.read_csv(ratings_path)
        
        print(f"加载了 {len(self.movies_df)} 部电影和 {len(self.ratings_df)} 条评分记录")
        return self
    
    def prepare_data(self, sample_users=None, sample_movies=None):
        """
        准备数据：构建用户-电影评分矩阵
        
        参数:
            sample_users: 要使用的用户数量，None表示使用全部用户
            sample_movies: 要使用的电影数量，None表示使用全部电影
        """
        if self.ratings_df is None:
            self.load_data()
            
        # 可以选择性地减少数据量
        ratings = self.ratings_df
        
        if sample_users is not None:
            # 获取评分数量最多的N个用户
            top_users = ratings['userId'].value_counts().head(sample_users).index
            ratings = ratings[ratings['userId'].isin(top_users)]
            print(f"采样了 {len(top_users)} 个用户进行分析")
            
        if sample_movies is not None:
            # 获取评分数量最多的N部电影
            top_movies = ratings['movieId'].value_counts().head(sample_movies).index
            ratings = ratings[ratings['movieId'].isin(top_movies)]
            print(f"采样了 {len(top_movies)} 部电影进行分析")
        
        # 创建用户和电影的映射字典
        unique_users = ratings['userId'].unique()
        unique_movies = ratings['movieId'].unique()
        
        self.user_map = {user: i for i, user in enumerate(unique_users)}
        self.movie_map = {movie: i for i, movie in enumerate(unique_movies)}
        
        # 反向映射用于推荐结果
        self.reverse_user_map = {i: user for user, i in self.user_map.items()}
        self.reverse_movie_map = {i: movie for movie, i in self.movie_map.items()}
        
        # 保存用户和电影列表
        self.user_ids = list(unique_users)
        self.movie_ids = list(unique_movies)
        
        # 构建稀疏矩阵
        user_indices = [self.user_map[user] for user in ratings['userId']]
        movie_indices = [self.movie_map[movie] for movie in ratings['movieId']]
        
        # 使用float32降低内存使用
        rating_values = ratings['rating'].astype(np.float32).values
        
        # 创建CSR稀疏矩阵
        n_users = len(self.user_map)
        n_movies = len(self.movie_map)
        self.user_ratings_sparse = csr_matrix((rating_values, (user_indices, movie_indices)), 
                                            shape=(n_users, n_movies))
                                            
        print(f"创建了形状为 {self.user_ratings_sparse.shape} 的稀疏用户-电影评分矩阵")
        print(f"矩阵密度: {self.user_ratings_sparse.nnz / (n_users * n_movies):.6f}")
        
        return self
    
    def train_model(self):
        """使用SVD训练推荐模型"""
        if not hasattr(self, 'user_ratings_sparse'):
            self.prepare_data(sample_users=5000, sample_movies=5000)
        
        # 计算每个用户的平均评分（处理稀疏矩阵）
        user_ratings_array = self.user_ratings_sparse.toarray()
        
        # 计算用户的平均评分（只考虑非零评分）
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # 使用掩码处理零值，只计算实际评分的平均值
            ratings_mask = user_ratings_array > 0
            masked_ratings = np.ma.masked_array(user_ratings_array, mask=~ratings_mask)
            user_ratings_mean = np.ma.mean(masked_ratings, axis=1).filled(0)
            
        # 对原始评分数据进行中心化（只对非零元素进行中心化）
        # 创建新的评分矩阵用于中心化
        ratings_centered = user_ratings_array.copy()
        for i in range(ratings_centered.shape[0]):
            # 只对用户实际评分的电影进行中心化处理
            ratings_centered[i, ratings_mask[i]] -= user_ratings_mean[i]
        
        # 使用SVD进行矩阵分解，k值不能超过矩阵的最小维度
        k = min(self.k_factors, min(ratings_centered.shape) - 1)
        print(f"使用 {k} 个隐因子进行SVD分解")
        
        try:
            self.U, self.sigma, self.Vt = svds(ratings_centered.astype(np.float32), k=k)
            
            # 将奇异值转换为对角矩阵
            sigma_diag_matrix = np.diag(self.sigma)
            
            # 在预测时可以按需为特定用户计算预测评分，而不是一次性计算整个矩阵
            # 我们不再一次性计算整个预测矩阵，以节省内存
            print(f"完成SVD模型训练，使用 {k} 个隐因子")
            
            # 保存用户平均评分，以便在推荐时使用
            self.user_ratings_mean = user_ratings_mean
            
        except Exception as e:
            print(f"SVD训练失败: {e}")
            # 降低k值，再次尝试
            reduced_k = max(k // 2, 1)
            print(f"尝试使用降低的隐因子数量: {reduced_k}")
            self.k_factors = reduced_k
            return self.train_model()  # 递归调用，降低维度重试
            
        return self
    
    def recommend_movies(self, user_id, top_n=10):
        """
        为指定用户推荐电影
        
        参数:
            user_id: 用户ID
            top_n: 推荐电影的数量
        
        返回:
            包含推荐电影信息的DataFrame
        """
        if not hasattr(self, 'U') or self.U is None:
            self.train_model()
        
        # 检查用户是否存在
        if user_id not in self.user_ids:
            print(f"用户ID {user_id} 不存在，无法提供推荐")
            return pd.DataFrame()
        
        # 获取用户映射后的索引
        user_idx = self.user_map[user_id]
        
        # 获取用户评分数据（稀疏矩阵的一行）
        user_ratings = self.user_ratings_sparse[user_idx].toarray().flatten()
        
        # 为该用户计算预测评分（而不是加载整个预测矩阵）
        user_mean_rating = self.user_ratings_mean[user_idx]
        
        # 根据SVD模型计算预测评分
        user_prediction = user_mean_rating + np.dot(
            np.dot(self.U[user_idx, :], np.diag(self.sigma)), 
            self.Vt
        )
        
        # 找出用户尚未评分的电影
        unrated_movies_mask = user_ratings == 0
        
        # 获取预测评分最高的N部尚未评分的电影
        # 只考虑那些用户未评分过的电影
        prediction_for_unrated = user_prediction * unrated_movies_mask
        
        # 获取预测评分最高的电影索引
        recommended_idxs = np.argsort(prediction_for_unrated)[::-1][:top_n]
        
        # 将内部索引转换回原始电影ID
        recommended_movie_ids = [self.reverse_movie_map[idx] for idx in recommended_idxs]
        
        # 获取推荐电影的详细信息
        recommended_movies = self.movies_df[self.movies_df['movieId'].isin(recommended_movie_ids)].copy()
        
        # 添加预测评分列
        predicted_ratings = [user_prediction[self.movie_map[movie_id]] 
                           for movie_id in recommended_movies['movieId']]
        recommended_movies['predicted_rating'] = predicted_ratings
        
        # 按预测评分排序
        recommended_movies = recommended_movies.sort_values(by='predicted_rating', ascending=False)
        
        return recommended_movies

    def get_similar_movies(self, movie_id, top_n=10):
        """
        找出与指定电影最相似的电影
        
        参数:
            movie_id: 电影ID
            top_n: 返回相似电影的数量
        
        返回:
            包含相似电影信息的DataFrame
        """
        if not hasattr(self, 'Vt') or self.Vt is None:
            self.train_model()
            
        # 检查电影是否存在
        if movie_id not in self.movie_ids:
            print(f"电影ID {movie_id} 不存在，无法找到相似电影")
            return pd.DataFrame()
            
        # 获取电影映射后的索引
        if movie_id not in self.movie_map:
            print(f"电影ID {movie_id} 不在训练样本中，无法找到相似电影")
            return pd.DataFrame()
            
        movie_idx = self.movie_map[movie_id]
        
        # 获取该电影在隐因子空间中的表示
        movie_vector = self.Vt[:, movie_idx]
        
        # 计算与所有电影的余弦相似度 (使用批处理以减少内存使用)
        batch_size = 1000  # 每批处理的电影数量
        n_movies = self.Vt.shape[1]
        similarities = np.zeros(n_movies)
        
        movie_vector_normalized = movie_vector / np.linalg.norm(movie_vector)
        
        # 分批计算相似度
        for start_idx in range(0, n_movies, batch_size):
            end_idx = min(start_idx + batch_size, n_movies)
            
            # 获取当前批次的电影矩阵
            movie_batch = self.Vt[:, start_idx:end_idx].T
            
            # 归一化当前批次
            movie_batch_normalized = normalize(movie_batch, axis=1)
            
            # 计算相似度
            batch_similarities = movie_batch_normalized @ movie_vector_normalized
            
            # 存储批次结果
            similarities[start_idx:end_idx] = batch_similarities
        
        # 排除电影本身
        similarities[movie_idx] = -1
        
        # 获取相似度最高的N部电影的索引
        similar_movie_idxs = np.argsort(similarities)[::-1][:top_n]
        
        # 将内部索引转换回原始电影ID
        similar_movie_ids = [self.reverse_movie_map[idx] for idx in similar_movie_idxs]
        
        # 获取相似电影的详细信息
        similar_movies = self.movies_df[self.movies_df['movieId'].isin(similar_movie_ids)].copy()
        
        # 添加相似度列
        similarity_scores = [similarities[self.movie_map[movie_id]] for movie_id in similar_movies['movieId']]
        similar_movies['similarity'] = similarity_scores
        
        # 按相似度排序
        similar_movies = similar_movies.sort_values(by='similarity', ascending=False)
        
        return similar_movies

    def evaluate_model(self, test_fraction=0.2, sample_size=10000, random_state=42):
        """
        评估模型性能，使用RMSE（均方根误差）作为评估指标
        
        参数:
            test_fraction: 测试集比例
            sample_size: 用于评估的样本数量，以避免内存溢出
            random_state: 随机种子
        
        返回:
            RMSE值
        """
        if self.ratings_df is None:
            self.load_data()
        
        # 为了避免内存问题，可以对评分数据进行抽样
        if sample_size and len(self.ratings_df) > sample_size:
            ratings_sample = self.ratings_df.sample(n=sample_size, random_state=random_state)
            print(f"从 {len(self.ratings_df)} 条评分记录中抽样 {sample_size} 条用于评估")
        else:
            ratings_sample = self.ratings_df
            
        # 随机分割训练集和测试集
        from sklearn.model_selection import train_test_split
        train_data, test_data = train_test_split(
            ratings_sample, 
            test_size=test_fraction, 
            random_state=random_state
        )
        
        # 创建用户和电影的映射字典
        unique_users = train_data['userId'].unique()
        unique_movies = train_data['movieId'].unique()
        
        user_map = {user: i for i, user in enumerate(unique_users)}
        movie_map = {movie: i for i, movie in enumerate(unique_movies)}
        
        # 构建训练数据的稀疏矩阵
        user_indices = [user_map[user] for user in train_data['userId']]
        movie_indices = [movie_map[movie] for movie in train_data['movieId']]
        rating_values = train_data['rating'].astype(np.float32).values
        
        n_users = len(user_map)
        n_movies = len(movie_map)
        train_matrix = csr_matrix((rating_values, (user_indices, movie_indices)), 
                                shape=(n_users, n_movies))
        
        # 计算每个用户的平均评分
        train_array = train_matrix.toarray()
        
        # 使用掩码处理零值，只计算实际评分的平均值
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            ratings_mask = train_array > 0
            masked_ratings = np.ma.masked_array(train_array, mask=~ratings_mask)
            user_ratings_mean = np.ma.mean(masked_ratings, axis=1).filled(0)
            
        # 对原始评分数据进行中心化
        ratings_centered = train_array.copy()
        for i in range(ratings_centered.shape[0]):
            # 只对用户实际评分的电影进行中心化处理
            ratings_centered[i, ratings_mask[i]] -= user_ratings_mean[i]
        
        # 使用SVD进行矩阵分解（降低k值以适应内存）
        k = min(self.k_factors, min(ratings_centered.shape) - 1)
        print(f"使用 {k} 个隐因子进行SVD分解（评估模式）")
        
        try:
            U, sigma, Vt = svds(ratings_centered.astype(np.float32), k=k)
            
            # 计算测试集中的预测评分
            test_rmse = 0
            test_count = len(test_data)
            
            if test_count > 0:
                squared_error_sum = 0
                valid_predictions = 0
                
                # 分批处理测试数据，避免内存问题
                batch_size = 1000
                
                for batch_start in range(0, len(test_data), batch_size):
                    batch_end = min(batch_start + batch_size, len(test_data))
                    test_batch = test_data.iloc[batch_start:batch_end]
                    
                    for _, row in test_batch.iterrows():
                        user_id = row['userId']
                        movie_id = row['movieId']
                        actual_rating = row['rating']
                        
                        # 检查用户和电影是否在训练数据中
                        if user_id in user_map and movie_id in movie_map:
                            user_idx = user_map[user_id]
                            movie_idx = movie_map[movie_id]
                            
                            # 计算预测评分
                            user_vec = U[user_idx, :]
                            movie_vec = Vt[:, movie_idx]
                            
                            # 预测评分 = 用户平均评分 + 用户向量 * Sigma * 电影向量
                            predicted_rating = user_ratings_mean[user_idx] + np.dot(np.dot(user_vec, np.diag(sigma)), movie_vec)
                            
                            # 累加平方误差
                            squared_error_sum += (predicted_rating - actual_rating) ** 2
                            valid_predictions += 1
                
                if valid_predictions > 0:
                    test_rmse = np.sqrt(squared_error_sum / valid_predictions)
                    
            print(f"模型评估 - RMSE: {test_rmse:.4f} (基于 {valid_predictions}/{test_count} 条有效测试数据)")
            return test_rmse
            
        except Exception as e:
            print(f"评估模型时SVD分解失败: {e}")
            # 可以尝试进一步减小样本大小或降低k值再次评估
            if sample_size > 1000:
                reduced_sample = sample_size // 2
                print(f"尝试使用更小的样本量: {reduced_sample}")
                return self.evaluate_model(test_fraction, reduced_sample, random_state)
            else:
                print("无法评估模型，样本量已经非常小")
                return None


if __name__ == "__main__":
    # 设置数据集路径
    data_path = "./ml-25m"
    
    # 初始化并训练推荐系统
    print("初始化电影推荐系统...")
    recommender = MovieRecommender(data_path, k_factors=20)
    
    # 加载数据
    print("\n加载数据...")
    recommender.load_data()
    
    # 准备数据，限制用户和电影数量以避免内存问题
    print("\n准备数据...")
    # 只使用评分次数最多的5000个用户和5000部电影
    recommender.prepare_data(sample_users=5000, sample_movies=2000)
    
    # 训练模型
    print("\n训练模型...")
    recommender.train_model()
    
    # 为用户1推荐10部电影
    user_id = 1
    print(f"\n为用户 {user_id} 生成推荐...")
    recommendations = recommender.recommend_movies(user_id, top_n=10)
    
    print(f"\n为用户 {user_id} 推荐的电影:")
    if not recommendations.empty:
        print(recommendations[['movieId', 'title', 'genres', 'predicted_rating']])
    else:
        print(f"无法为用户 {user_id} 生成推荐。")
        
    # 尝试另一个用户
    user_id = 2
    print(f"\n为用户 {user_id} 生成推荐...")
    recommendations = recommender.recommend_movies(user_id, top_n=10)
    
    print(f"\n为用户 {user_id} 推荐的电影:")
    if not recommendations.empty:
        print(recommendations[['movieId', 'title', 'genres', 'predicted_rating']])
    else:
        print(f"无法为用户 {user_id} 生成推荐。")
        
    # 获取一部流行电影的相似电影
    movie_title = "Toy Story"
    matching_movies = recommender.movies_df[recommender.movies_df['title'].str.contains(movie_title)].head(1)
    
    if not matching_movies.empty:
        movie_id = matching_movies.iloc[0]['movieId']
        movie_title = matching_movies.iloc[0]['title']
        
        print(f"\n找到电影: {movie_title} (ID: {movie_id})")
        print(f"\n寻找与电影 '{movie_title}' 相似的电影...")
        similar_movies = recommender.get_similar_movies(movie_id, top_n=10)
        
        if not similar_movies.empty:
            print(f"\n与电影 '{movie_title}' 相似的电影:")
            print(similar_movies[['title', 'genres', 'similarity']])
        else:
            print(f"无法为电影 '{movie_title}' 找到相似电影。")
