#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
电影推荐系统演示脚本
展示基于SVD的电影推荐系统的主要功能
"""

import os
import argparse
import pandas as pd
from movie_recommender import MovieRecommender


def show_welcome():
    """显示欢迎信息"""
    print("\n" + "="*80)
    print("🎬  基于SVD的电影推荐系统演示  🎬")
    print("="*80)
    print("本系统使用矩阵分解技术为用户提供个性化电影推荐")
    print("数据来源: MovieLens 25M Dataset")
    print("="*80 + "\n")


def initialize_system(data_path, sample_users, sample_movies, k_factors):
    """初始化推荐系统"""
    print("正在初始化推荐系统...")
    recommender = MovieRecommender(data_path, k_factors=k_factors)
    
    print(f"加载数据中...")
    recommender.load_data()
    
    print(f"准备用户-电影评分矩阵 (采样 {sample_users} 用户, {sample_movies} 电影)...")
    recommender.prepare_data(sample_users=sample_users, sample_movies=sample_movies)
    
    print(f"训练SVD模型 (使用 {k_factors} 个隐因子)...")
    recommender.train_model()
    
    print("\n✅ 系统初始化完成!")
    return recommender


def user_recommendation_demo(recommender):
    """演示用户推荐功能"""
    print("\n" + "="*80)
    print("👤  用户电影推荐演示  👤")
    print("="*80)
    
    # 查找评分最多的几个用户
    if hasattr(recommender, 'user_map') and recommender.user_map:
        user_ids = list(recommender.user_map.keys())
        if user_ids:
            top_users = user_ids[:5]  # 取前5个活跃用户
            print(f"演示用户ID: {top_users}")
            
            for user_id in top_users[:2]:  # 只展示前2个用户的推荐
                print(f"\n为用户 {user_id} 生成推荐中...")
                recommendations = recommender.recommend_movies(user_id, top_n=5)
                
                if not recommendations.empty:
                    print(f"\n🎯 用户 {user_id} 的个性化电影推荐:")
                    for idx, row in recommendations.iterrows():
                        print(f"📽️  {row['title']} ({row['genres']}) - 预测评分: {row['predicted_rating']:.2f}")
                else:
                    print(f"⚠️ 无法为用户 {user_id} 生成推荐。")
        else:
            print("⚠️ 系统中没有足够的用户数据进行演示。")
    else:
        print("⚠️ 推荐系统未正确初始化用户映射。")


def movie_similarity_demo(recommender):
    """演示电影相似性推荐功能"""
    print("\n" + "="*80)
    print("🎞️  电影相似性推荐演示  🎞️")
    print("="*80)
    
    # 演示电影列表
    demo_movies = [
        "Toy Story",        # 动画片
        "The Godfather",    # 犯罪/剧情片
        "The Matrix",       # 科幻片
        "Titanic",          # 爱情片
        "Pulp Fiction"      # 昆汀作品
    ]
    
    for title in demo_movies:
        # 查找匹配的电影
        matched_movies = recommender.movies_df[recommender.movies_df['title'].str.contains(title, case=False)]
        
        if matched_movies.empty:
            print(f"\n⚠️ 找不到标题包含 '{title}' 的电影。")
            continue
            
        # 使用第一个匹配结果
        movie = matched_movies.iloc[0]
        movie_id = movie['movieId']
        movie_title = movie['title']
        movie_genres = movie['genres']
        
        print(f"\n📽️ 电影: {movie_title} ({movie_genres})")
        
        # 检查电影是否在训练数据中
        if not hasattr(recommender, 'movie_map') or movie_id not in recommender.movie_map:
            print(f"⚠️ 此电影不在训练数据集中，无法提供相似电影。")
            continue
            
        print("寻找相似电影中...")
        similar_movies = recommender.get_similar_movies(movie_id, top_n=5)
        
        if not similar_movies.empty:
            print(f"\n🎯 与 '{movie_title}' 相似的电影:")
            for idx, row in similar_movies.iterrows():
                print(f"🎬 {row['title']} ({row['genres']}) - 相似度: {row['similarity']:.2f}")
        else:
            print(f"⚠️ 无法为电影 '{movie_title}' 找到相似电影。")


def search_movie_demo(recommender):
    """演示电影搜索功能"""
    print("\n" + "="*80)
    print("🔍  电影搜索与推荐演示  🔍")
    print("="*80)
    
    while True:
        query = input("\n输入电影标题关键词 (输入q退出): ")
        if query.lower() == 'q':
            break
            
        # 搜索匹配的电影
        matched_movies = recommender.movies_df[recommender.movies_df['title'].str.contains(query, case=False)]
        
        if matched_movies.empty:
            print(f"⚠️ 找不到标题包含 '{query}' 的电影。")
            continue
            
        # 显示搜索结果
        print(f"\n找到 {len(matched_movies)} 部匹配电影:")
        for i, (idx, movie) in enumerate(matched_movies.head(5).iterrows()):
            print(f"{i+1}. {movie['title']} ({movie['genres']}) - ID: {movie['movieId']}")
            
        # 询问用户选择
        choice = input("\n选择电影编号查看相似推荐 (输入q返回): ")
        if choice.lower() == 'q':
            continue
            
        try:
            choice_idx = int(choice) - 1
            if choice_idx < 0 or choice_idx >= len(matched_movies.head(5)):
                print("⚠️ 无效的选择。")
                continue
                
            selected_movie = matched_movies.iloc[choice_idx]
            movie_id = selected_movie['movieId']
            
            # 检查电影是否在训练数据中
            if not hasattr(recommender, 'movie_map') or movie_id not in recommender.movie_map:
                print(f"⚠️ 电影 '{selected_movie['title']}' 不在训练数据中，无法提供相似电影。")
                continue
                
            # 获取相似电影
            similar_movies = recommender.get_similar_movies(movie_id, top_n=10)
            
            if not similar_movies.empty:
                print(f"\n🎯 与 '{selected_movie['title']}' 相似的电影:")
                for idx, row in similar_movies.iterrows():
                    print(f"🎬 {row['title']} ({row['genres']}) - 相似度: {row['similarity']:.2f}")
            else:
                print(f"⚠️ 无法为电影 '{selected_movie['title']}' 找到相似电影。")
                
        except ValueError:
            print("⚠️ 请输入有效的数字。")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='电影推荐系统演示')
    parser.add_argument('--data_path', type=str, default='./ml-25m',
                        help='MovieLens数据集路径')
    parser.add_argument('--sample_users', type=int, default=5000,
                        help='用于训练的用户采样数量')
    parser.add_argument('--sample_movies', type=int, default=2000,
                        help='用于训练的电影采样数量')
    parser.add_argument('--k_factors', type=int, default=20,
                        help='SVD隐因子数量')
    parser.add_argument('--interactive', action='store_true',
                        help='启用交互式电影搜索')
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_arguments()
    
    # 显示欢迎信息
    show_welcome()
    
    try:
        # 初始化推荐系统
        recommender = initialize_system(
            args.data_path, 
            args.sample_users, 
            args.sample_movies, 
            args.k_factors
        )
        
        # 演示用户推荐功能
        user_recommendation_demo(recommender)
        
        # 演示电影相似性推荐功能
        movie_similarity_demo(recommender)
        
        # 是否启用交互式电影搜索
        if args.interactive:
            search_movie_demo(recommender)
            
        print("\n" + "="*80)
        print("🎉  演示完成! 感谢使用电影推荐系统  🎉")
        print("="*80)
        
    except MemoryError:
        print("\n⚠️ 内存错误! 请尝试减少采样数量或降低隐因子数量。")
    except Exception as e:
        print(f"\n⚠️ 发生错误: {str(e)}")
