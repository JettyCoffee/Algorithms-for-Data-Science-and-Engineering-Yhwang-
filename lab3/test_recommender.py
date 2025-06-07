#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电影推荐系统测试脚本
用于测试和演示基于SVD的电影推荐系统
"""

import os
import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from movie_recommender import MovieRecommender

def test_basic_functionality(data_path, sample_users=5000, sample_movies=2000):
    """测试推荐系统的基本功能"""
    print("\n===== 测试基本功能 =====")
    
    # 初始化推荐系统
    recommender = MovieRecommender(data_path, k_factors=20)
    
    # 加载数据
    start_time = time.time()
    recommender.load_data()
    print(f"数据加载耗时: {time.time() - start_time:.2f} 秒")
    
    # 准备数据
    start_time = time.time()
    recommender.prepare_data(sample_users=sample_users, sample_movies=sample_movies)
    print(f"数据准备耗时: {time.time() - start_time:.2f} 秒")
    
    # 训练模型
    start_time = time.time()
    recommender.train_model()
    print(f"模型训练耗时: {time.time() - start_time:.2f} 秒")
    
    return recommender

def test_movie_recommendations(recommender, user_ids, top_n=10):
    """测试电影推荐功能"""
    print("\n===== 测试电影推荐功能 =====")
    
    for user_id in user_ids:
        # 检查用户是否在训练数据中
        if user_id not in recommender.user_ids:
            print(f"用户 {user_id} 不在训练数据集中，跳过推荐")
            continue
            
        # 为用户生成推荐
        start_time = time.time()
        recommendations = recommender.recommend_movies(user_id, top_n=top_n)
        print(f"为用户 {user_id} 生成推荐耗时: {time.time() - start_time:.2f} 秒")
        
        # 显示推荐结果
        if not recommendations.empty:
            print(f"\n为用户 {user_id} 推荐的 {top_n} 部电影:")
            for idx, row in recommendations.iterrows():
                print(f"{row['title']} (预测评分: {row['predicted_rating']:.2f})")
        else:
            print(f"无法为用户 {user_id} 生成推荐")
        print("="*50)

def test_similar_movies(recommender, movie_titles, top_n=10):
    """测试相似电影推荐功能"""
    print("\n===== 测试相似电影推荐功能 =====")
    
    # 查找电影ID
    movies_df = recommender.movies_df
    
    for title in movie_titles:
        # 查找与标题匹配的电影
        matched_movies = movies_df[movies_df['title'].str.contains(title, case=False, regex=False)]
        
        if matched_movies.empty:
            print(f"找不到标题包含 '{title}' 的电影")
            continue
            
        # 使用第一个匹配结果
        movie_id = matched_movies.iloc[0]['movieId']
        movie_title = matched_movies.iloc[0]['title']
        
        # 检查电影是否在训练数据中
        if not hasattr(recommender, 'movie_map') or movie_id not in recommender.movie_map:
            print(f"电影 '{movie_title}' (ID: {movie_id}) 不在训练数据集中，跳过相似电影查找")
            continue
            
        # 查找相似电影
        start_time = time.time()
        similar_movies = recommender.get_similar_movies(movie_id, top_n=top_n)
        print(f"为电影 '{movie_title}' 查找相似电影耗时: {time.time() - start_time:.2f} 秒")
        
        # 显示相似电影
        if not similar_movies.empty:
            print(f"\n与电影 '{movie_title}' 相似的 {top_n} 部电影:")
            for idx, row in similar_movies.iterrows():
                print(f"{row['title']} (相似度: {row['similarity']:.2f})")
        else:
            print(f"无法为电影 '{movie_title}' 找到相似电影")
        print("="*50)

def test_model_evaluation(data_path, k_values, sample_size=5000):
    """测试不同隐因子数量对模型性能的影响"""
    print("\n===== 测试模型评估 =====")
    
    # 存储不同k值的RMSE结果
    rmse_results = []
    
    for k in k_values:
        print(f"\n测试k = {k}的模型性能")
        recommender = MovieRecommender(data_path, k_factors=k)
        recommender.load_data()
        
        # 评估模型
        start_time = time.time()
        rmse = recommender.evaluate_model(test_fraction=0.2, sample_size=sample_size, random_state=42)
        print(f"模型评估耗时: {time.time() - start_time:.2f} 秒")
        
        if rmse is not None:
            rmse_results.append((k, rmse))
    
    # 绘制k值与RMSE的关系图
    if rmse_results:
        k_values, rmse_values = zip(*rmse_results)
        
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, rmse_values, 'o-')
        plt.xlabel('隐因子数量 (k)')
        plt.ylabel('RMSE')
        plt.title('隐因子数量对推荐系统性能的影响')
        plt.grid(True)
        plt.savefig('rmse_vs_k_factors.png')
        plt.close()
        
        print(f"\n已保存RMSE vs k因子关系图到 'rmse_vs_k_factors.png'")
    
    return rmse_results

def test_cold_start(recommender):
    """测试冷启动问题（评分较少的用户）"""
    print("\n===== 测试冷启动问题 =====")
    
    # 统计每个用户的评分数量
    ratings_df = recommender.ratings_df
    
    # 只考虑在训练数据中的用户
    if hasattr(recommender, 'user_map'):
        valid_users = recommender.user_map.keys()
        ratings_subset = ratings_df[ratings_df['userId'].isin(valid_users)]
        user_rating_counts = ratings_subset['userId'].value_counts()
    else:
        user_rating_counts = ratings_df['userId'].value_counts()
    
    # 找出评分数量最少的5个用户（至少有5个评分）
    cold_users = user_rating_counts[user_rating_counts >= 5].nsmallest(5).index.tolist()
    
    if cold_users:
        print(f"评分数量最少的5个用户: {cold_users}")
        
        # 为这些用户生成推荐
        for user_id in cold_users:
            rating_count = user_rating_counts[user_id]
            print(f"\n用户 {user_id} (评分数量: {rating_count}):")
            
            # 检查用户是否在训练数据中
            if hasattr(recommender, 'user_map') and user_id not in recommender.user_map:
                print(f"用户 {user_id} 不在训练数据中，跳过")
                continue
                
            recommendations = recommender.recommend_movies(user_id, top_n=5)
            if not recommendations.empty:
                for idx, row in recommendations.iterrows():
                    print(f"{row['title']} (预测评分: {row['predicted_rating']:.2f})")
            else:
                print(f"无法为用户 {user_id} 生成推荐")
    else:
        print("未找到满足条件的冷启动用户")

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试基于SVD的电影推荐系统')
    parser.add_argument('--data_path', type=str, default='./ml-25m',
                        help='MovieLens数据集路径')
    parser.add_argument('--test_users', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='要测试的用户ID列表')
    parser.add_argument('--test_movies', type=str, nargs='+', default=['Toy Story', 'Matrix', 'Star Wars', 'Inception'],
                        help='要测试相似性的电影标题列表')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 15],
                        help='要测试的隐因子数量列表')
    parser.add_argument('--top_n', type=int, default=10,
                        help='推荐结果数量')
    parser.add_argument('--sample_users', type=int, default=5000,
                        help='用于训练的用户采样数量')
    parser.add_argument('--sample_movies', type=int, default=2000,
                        help='用于训练的电影采样数量')
    parser.add_argument('--evaluation_sample', type=int, default=5000,
                        help='用于评估的样本数量')
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        # 测试基本功能
        recommender = test_basic_functionality(
            args.data_path, 
            sample_users=args.sample_users, 
            sample_movies=args.sample_movies
        )
        
        # 测试电影推荐
        test_movie_recommendations(recommender, args.test_users, top_n=args.top_n)
        
        # 测试相似电影推荐
        test_similar_movies(recommender, args.test_movies, top_n=args.top_n)
        
        # 测试冷启动问题
        test_cold_start(recommender)
        
        # 测试不同隐因子数量对模型性能的影响
        test_model_evaluation(
            args.data_path, 
            args.k_values, 
            sample_size=args.evaluation_sample
        )
    except MemoryError:
        print("\n遇到内存错误！请尝试减少采样数量或降低隐因子数量。")
    except Exception as e:
        print(f"\n执行测试时发生错误: {str(e)}")
