正在初始化推荐系统...
加载数据中...
加载了 62423 部电影和 25000095 条评分记录
准备用户-电影评分矩阵 (采样 5000 用户, 2000 电影)...
采样了 5000 个用户进行分析
采样了 2000 部电影进行分析
创建了形状为 (5000, 2000) 的稀疏用户-电影评分矩阵
矩阵密度: 0.362497
训练SVD模型 (使用 20 个隐因子)...
使用 20 个隐因子进行SVD分解
完成SVD模型训练，使用 20 个隐因子

✅ 系统初始化完成!

================================================================================
👤  用户电影推荐演示  👤
================================================================================
演示用户ID: [12, 72, 187, 321, 426]

为用户 12 生成推荐中...

🎯 用户 12 的个性化电影推荐:
📽️  Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981) (Action|Adventure) - 预测评分: 4.14
📽️  Eternal Sunshine of the Spotless Mind (2004) (Drama|Romance|Sci-Fi) - 预测评分: 4.12
📽️  One Flew Over the Cuckoo's Nest (1975) (Drama) - 预测评分: 4.08
📽️  Willy Wonka & the Chocolate Factory (1971) (Children|Comedy|Fantasy|Musical) - 预测评分: 4.08
📽️  Office Space (1999) (Comedy|Crime) - 预测评分: 4.02

为用户 72 生成推荐中...

🎯 用户 72 的个性化电影推荐:
📽️  To Kill a Mockingbird (1962) (Drama) - 预测评分: 4.31
📽️  Toy Story (1995) (Adventure|Animation|Children|Comedy|Fantasy) - 预测评分: 4.27
📽️  Shakespeare in Love (1998) (Comedy|Drama|Romance) - 预测评分: 4.24
📽️  Life Is Beautiful (La Vita è bella) (1997) (Comedy|Drama|Romance|War) - 预测评分: 4.19
📽️  Sting, The (1973) (Comedy|Crime) - 预测评分: 4.16

================================================================================
🎞️  电影相似性推荐演示  🎞️
================================================================================

📽️ 电影: Toy Story (1995) (Adventure|Animation|Children|Comedy|Fantasy)
寻找相似电影中...

🎯 与 'Toy Story (1995)' 相似的电影:
🎬 Toy Story 2 (1999) (Adventure|Animation|Children|Comedy|Fantasy) - 相似度: 0.94
🎬 Aladdin (1992) (Adventure|Animation|Children|Comedy|Musical) - 相似度: 0.85
🎬 Monsters, Inc. (2001) (Adventure|Animation|Children|Comedy|Fantasy) - 相似度: 0.84
🎬 Toy Story 3 (2010) (Adventure|Animation|Children|Comedy|Fantasy|IMAX) - 相似度: 0.84
🎬 Finding Nemo (2003) (Adventure|Animation|Children|Comedy) - 相似度: 0.82

📽️ 电影: Battle of the Godfathers (1973) (Action|Drama)
⚠️ 此电影不在训练数据集中，无法提供相似电影。

📽️ 电影: Return to Source: The Philosophy of The Matrix (2004) (Documentary)
⚠️ 此电影不在训练数据集中，无法提供相似电影。

📽️ 电影: Titanic (1997) (Drama|Romance)
寻找相似电影中...

🎯 与 'Titanic (1997)' 相似的电影:
🎬 E.T. the Extra-Terrestrial (1982) (Children|Drama|Sci-Fi) - 相似度: 0.71
🎬 Ghost (1990) (Comedy|Drama|Fantasy|Romance|Thriller) - 相似度: 0.70
🎬 Jurassic Park (1993) (Action|Adventure|Sci-Fi|Thriller) - 相似度: 0.67
🎬 Fatal Attraction (1987) (Drama|Thriller) - 相似度: 0.63
🎬 Pretty Woman (1990) (Comedy|Romance) - 相似度: 0.59

📽️ 电影: Pulp Fiction (1994) (Comedy|Crime|Drama|Thriller)
寻找相似电影中...

🎯 与 'Pulp Fiction (1994)' 相似的电影:
🎬 Reservoir Dogs (1992) (Crime|Mystery|Thriller) - 相似度: 0.91
🎬 Fight Club (1999) (Action|Crime|Drama|Thriller) - 相似度: 0.78
🎬 Kill Bill: Vol. 1 (2003) (Action|Crime|Thriller) - 相似度: 0.74
🎬 Goodfellas (1990) (Crime|Drama) - 相似度: 0.73
🎬 Godfather, The (1972) (Crime|Drama) - 相似度: 0.72

================================================================================
🎉  演示完成! 感谢使用电影推荐系统  🎉
================================================================================
root@Latte-de-Furina:~/saunfa# cd /root/saunfa && python demo_recommender.py --sample_users 3000 --sample_movies 3000 --interactive

================================================================================
🎬  基于SVD的电影推荐系统演示  🎬
================================================================================
本系统使用矩阵分解技术为用户提供个性化电影推荐
数据来源: MovieLens 25M Dataset
================================================================================

正在初始化推荐系统...
加载数据中...
加载了 62423 部电影和 25000095 条评分记录
准备用户-电影评分矩阵 (采样 3000 用户, 3000 电影)...
采样了 3000 个用户进行分析
采样了 3000 部电影进行分析
创建了形状为 (3000, 3000) 的稀疏用户-电影评分矩阵
矩阵密度: 0.334132
训练SVD模型 (使用 20 个隐因子)...
使用 20 个隐因子进行SVD分解
完成SVD模型训练，使用 20 个隐因子

✅ 系统初始化完成!

================================================================================
👤  用户电影推荐演示  👤
================================================================================
演示用户ID: [187, 426, 541, 548, 626]

为用户 187 生成推荐中...

🎯 用户 187 的个性化电影推荐:
📽️  Guardians of the Galaxy (2014) (Action|Adventure|Sci-Fi) - 预测评分: 4.52
📽️  Deadpool (2016) (Action|Adventure|Comedy|Sci-Fi) - 预测评分: 4.51
📽️  Snatch (2000) (Comedy|Crime|Thriller) - 预测评分: 4.47
📽️  Usual Suspects, The (1995) (Crime|Mystery|Thriller) - 预测评分: 4.36
📽️  Star Trek Into Darkness (2013) (Action|Adventure|Sci-Fi|IMAX) - 预测评分: 4.35

为用户 426 生成推荐中...

🎯 用户 426 的个性化电影推荐:
📽️  Cool Hand Luke (1967) (Drama) - 预测评分: 3.85
📽️  Bridge on the River Kwai, The (1957) (Adventure|Drama|War) - 预测评分: 3.78
📽️  Spartacus (1960) (Action|Drama|Romance|War) - 预测评分: 3.46
📽️  Free Willy 2: The Adventure Home (1995) (Adventure|Children|Drama) - 预测评分: 3.44
📽️  Mortal Kombat: Annihilation (1997) (Action|Adventure|Fantasy) - 预测评分: 3.43

================================================================================
🎞️  电影相似性推荐演示  🎞️
================================================================================

📽️ 电影: Toy Story (1995) (Adventure|Animation|Children|Comedy|Fantasy)
寻找相似电影中...

🎯 与 'Toy Story (1995)' 相似的电影:
🎬 Toy Story 2 (1999) (Adventure|Animation|Children|Comedy|Fantasy) - 相似度: 0.94
🎬 Finding Nemo (2003) (Adventure|Animation|Children|Comedy) - 相似度: 0.84
🎬 Monsters, Inc. (2001) (Adventure|Animation|Children|Comedy|Fantasy) - 相似度: 0.84
🎬 Aladdin (1992) (Adventure|Animation|Children|Comedy|Musical) - 相似度: 0.83
🎬 Beauty and the Beast (1991) (Animation|Children|Fantasy|Musical|Romance|IMAX) - 相似度: 0.77

📽️ 电影: Battle of the Godfathers (1973) (Action|Drama)
⚠️ 此电影不在训练数据集中，无法提供相似电影。

📽️ 电影: Return to Source: The Philosophy of The Matrix (2004) (Documentary)
⚠️ 此电影不在训练数据集中，无法提供相似电影。

📽️ 电影: Titanic (1997) (Drama|Romance)
寻找相似电影中...

🎯 与 'Titanic (1997)' 相似的电影:
🎬 E.T. the Extra-Terrestrial (1982) (Children|Drama|Sci-Fi) - 相似度: 0.75
🎬 Jurassic Park (1993) (Action|Adventure|Sci-Fi|Thriller) - 相似度: 0.72
🎬 Ghost (1990) (Comedy|Drama|Fantasy|Romance|Thriller) - 相似度: 0.67
🎬 Speed (1994) (Action|Romance|Thriller) - 相似度: 0.61
🎬 Dances with Wolves (1990) (Adventure|Drama|Western) - 相似度: 0.61

📽️ 电影: Pulp Fiction (1994) (Comedy|Crime|Drama|Thriller)
寻找相似电影中...

🎯 与 'Pulp Fiction (1994)' 相似的电影:
🎬 Reservoir Dogs (1992) (Crime|Mystery|Thriller) - 相似度: 0.91
🎬 Goodfellas (1990) (Crime|Drama) - 相似度: 0.78
🎬 Fight Club (1999) (Action|Crime|Drama|Thriller) - 相似度: 0.77
🎬 Kill Bill: Vol. 1 (2003) (Action|Crime|Thriller) - 相似度: 0.73
🎬 Full Metal Jacket (1987) (Drama|War) - 相似度: 0.73

================================================================================
🔍  电影搜索与推荐演示  🔍
================================================================================

输入电影标题关键词 (输入q退出): Titanic

找到 15 部匹配电影:
1. Titanic (1997) (Drama|Romance) - ID: 1721
2. Chambermaid on the Titanic, The (Femme de chambre du Titanic, La) (1998) (Romance) - ID: 2157
3. Raise the Titanic (1980) (Drama|Thriller) - ID: 3403
4. Titanic (1953) (Action|Drama) - ID: 3404
5. Titanica (1992) (Documentary|IMAX) - ID: 4864

选择电影编号查看相似推荐 (输入q返回): 1

🎯 与 'Titanic (1997)' 相似的电影:
🎬 E.T. the Extra-Terrestrial (1982) (Children|Drama|Sci-Fi) - 相似度: 0.75
🎬 Jurassic Park (1993) (Action|Adventure|Sci-Fi|Thriller) - 相似度: 0.72
🎬 Ghost (1990) (Comedy|Drama|Fantasy|Romance|Thriller) - 相似度: 0.67
🎬 Speed (1994) (Action|Romance|Thriller) - 相似度: 0.61
🎬 Dances with Wolves (1990) (Adventure|Drama|Western) - 相似度: 0.61
🎬 Fatal Attraction (1987) (Drama|Thriller) - 相似度: 0.58
🎬 Blair Witch Project, The (1999) (Drama|Horror|Thriller) - 相似度: 0.58
🎬 Pretty Woman (1990) (Comedy|Romance) - 相似度: 0.58
🎬 A.I. Artificial Intelligence (2001) (Adventure|Drama|Sci-Fi) - 相似度: 0.58
🎬 Sleepless in Seattle (1993) (Comedy|Drama|Romance) - 相似度: 0.57

输入电影标题关键词 (输入q退出): E.T. the Extra-Terrestrial

找到 1 部匹配电影:
1. E.T. the Extra-Terrestrial (1982) (Children|Drama|Sci-Fi) - ID: 1097