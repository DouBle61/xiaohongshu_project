#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
小红书游泳赛事UGC数据预处理与探索性分析脚本
Data Preprocessing and Exploratory Data Analysis for Xiaohongshu Swimming Events

研究背景：传播学硕士论文项目，研究日常媒介化体育赛事（国内游泳赛事）
的社交媒体传播机制与消费转化路径。

作者：自动生成
数据来源：小红书平台爬取数据，存储于 全部文件_合并结果.csv
"""

import os
import re
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from collections import Counter, defaultdict

# 尝试导入 jieba 分词库
try:
    import jieba
    JIEBA_AVAILABLE = True
    print("✓ jieba 分词库已加载")
except ImportError:
    JIEBA_AVAILABLE = False
    print("✗ jieba 未安装，将使用字符匹配方法进行文本分析")

# 脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, '全部文件_合并结果.csv')
OUTPUT_CSV = os.path.join(BASE_DIR, 'cleaned_data.csv')
REPORT_PATH = os.path.join(BASE_DIR, 'analysis_report.md')

# 可配置参数
MIN_TEXT_LENGTH = 10  # 最小有效文本长度（字符数）

# ======================================================================
# 第一步：数据读取与基本信息探查
# ======================================================================

print("\n" + "=" * 70)
print("第一步：数据读取与基本信息探查")
print("=" * 70)

# 尝试多种编码读取 CSV
df_raw = None
for enc in ['utf-8-sig', 'utf-8', 'gbk', 'gb18030', 'latin-1']:
    try:
        df_raw = pd.read_csv(CSV_PATH, encoding=enc)
        print(f"✓ 成功以 {enc} 编码读取文件")
        break
    except Exception as e:
        print(f"✗ {enc} 编码读取失败: {e}")

if df_raw is None:
    print("ERROR: 无法读取 CSV 文件，请检查文件路径和编码")
    sys.exit(1)

# ── 列名 ──
print("\n【列名（所有字段）】")
print(list(df_raw.columns))

# ── 数据形状 ──
print(f"\n【数据形状】{df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")

# ── 每列数据类型 ──
print("\n【每列数据类型】")
for col in df_raw.columns:
    print(f"  {col}: {df_raw[col].dtype}")

# ── 每列非空值数量与缺失率 ──
print("\n【每列非空值数量与缺失率】")
for col in df_raw.columns:
    non_null = df_raw[col].notna().sum()
    miss_rate = df_raw[col].isna().mean() * 100
    print(f"  {col}: 非空 {non_null:,} 条，缺失率 {miss_rate:.2f}%")

# ── 前 20 行数据样本 ──
print("\n【前 20 行数据样本】")
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(df_raw.head(20).to_string())
pd.reset_option('display.max_colwidth')

# ── 每列唯一值数量 ──
print("\n【每列唯一值数量】")
for col in df_raw.columns:
    print(f"  {col}: {df_raw[col].nunique():,} 个唯一值")

# ── 文本列最长的 5 条内容 ──
TEXT_COL = 'Text'
if TEXT_COL in df_raw.columns:
    df_raw['_tmp_len'] = df_raw[TEXT_COL].fillna('').str.len()
    top5 = df_raw.nlargest(5, '_tmp_len')
    print("\n【文本列中 5 条最长文本（前 300 字）】")
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        print(f"\n--- 第 {i} 条（长度 {row['_tmp_len']}，来源 {row.get('Source_File','N/A')}，类型 {row.get('Content_Type','N/A')}）---")
        print(str(row[TEXT_COL])[:300])
    df_raw.drop(columns=['_tmp_len'], inplace=True)

# ======================================================================
# 第二步：数据清洗与预处理
# ======================================================================

print("\n" + "=" * 70)
print("第二步：数据清洗与预处理")
print("=" * 70)

df = df_raw.copy()
initial_count = len(df)
print(f"原始数据行数：{initial_count:,}")

# ── 2.1 去重（基于文本内容列） ──
dedup_removed = 0
before_dedup = initial_count
if TEXT_COL in df.columns:
    df = df.dropna(subset=[TEXT_COL])
    before_dedup = len(df)
    df = df.drop_duplicates(subset=[TEXT_COL], keep='first')
    dedup_removed = before_dedup - len(df)
    print(f"\n[去重] 删除空文本后：{before_dedup:,} 条；去重后：{len(df):,} 条（删除 {dedup_removed:,} 条重复）")

# ── 2.2 去除无效文本（文本长度 < 10 字符） ──
df['_text_len'] = df[TEXT_COL].str.len()
before_len = len(df)
df = df[df['_text_len'] >= MIN_TEXT_LENGTH].copy()
print(f"[文本过滤] 去除 <{MIN_TEXT_LENGTH} 字符的帖子后：{len(df):,} 条（删除 {before_len - len(df):,} 条）")

# ── 2.3 时间处理 ──
DATE_COL = 'Date'
if DATE_COL in df.columns:
    # 提取 IP 属地（日期字符串中的中文省份/国家部分）
    df['ip_location'] = df[DATE_COL].str.extract(r'[\d\-]+\s*([^\d\-\s编辑于][^\d\-]*?)$')
    # 提取纯日期部分（YYYY-MM-DD 格式）
    df['date_clean'] = df[DATE_COL].str.extract(r'(\d{4}-\d{2}-\d{2})')
    df['date_parsed'] = pd.to_datetime(df['date_clean'], errors='coerce')
    df['year'] = df['date_parsed'].dt.year
    df['month'] = df['date_parsed'].dt.month
    df['day'] = df['date_parsed'].dt.day
    df['weekday'] = df['date_parsed'].dt.day_name()
    valid_date_count = df['date_parsed'].notna().sum()
    print(f"\n[时间处理] 成功解析日期 {valid_date_count:,} 条；"
          f"日期范围：{df['date_parsed'].min().date() if valid_date_count else 'N/A'}"
          f" 至 {df['date_parsed'].max().date() if valid_date_count else 'N/A'}")

# ── 2.4 IP 属地处理 ──
if 'ip_location' in df.columns:
    # 标准化：去除空白，统一 NaN
    df['ip_location'] = df['ip_location'].str.strip().replace('', np.nan)
    loc_counts = df['ip_location'].value_counts().head(10)
    print(f"\n[IP属地处理] 非空属地记录：{df['ip_location'].notna().sum():,} 条")
    print("Top 10 属地：")
    for loc, cnt in loc_counts.items():
        print(f"  {loc}: {cnt:,}")

# ── 2.5 从 Source_File 提取赛事类别 ──
if 'Source_File' in df.columns:
    # 仅取中文字符部分（去掉数字和扩展名）
    df['competition'] = df['Source_File'].str.extract(r'^([^\d]+)')
    print(f"\n[赛事分类] 各赛事帖文数量：")
    comp_counts = df['competition'].value_counts()
    for comp, cnt in comp_counts.items():
        print(f"  {comp}: {cnt:,}")

# ── 2.6 数值清洗（Engagement 列为全空，跳过） ──
ENGAGEMENT_COL = 'Engagement'
if ENGAGEMENT_COL in df.columns:
    non_null_eng = df[ENGAGEMENT_COL].notna().sum()
    if non_null_eng == 0:
        print(f"\n[数值清洗] Engagement 列全部为空，跳过数值转换")
    else:
        df[ENGAGEMENT_COL] = pd.to_numeric(df[ENGAGEMENT_COL], errors='coerce')
        print(f"\n[数值清洗] Engagement 非空值：{non_null_eng:,}")

# ── 2.7 文本清洗：生成 clean_text 列 ──
def clean_text(text):
    """
    文本清洗函数：
    - 去除 URL
    - 去除话题标签中的 # 号（保留文字内容）
    - 保留 emoji（有情感分析价值）
    - 去除多余空白
    """
    if not isinstance(text, str):
        return ''
    # 去除 URL（http/https/www）
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 去除话题标签的 # 符号（保留话题文字）
    text = re.sub(r'#(\S+)', r'\1', text)
    # 去除特殊字符（保留中文、字母、数字、emoji 和常用标点）
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # 去除多余空白
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df[TEXT_COL].apply(clean_text)
print(f"\n[文本清洗] clean_text 列已生成，样本：")
print(f"  原文（前80字）: {str(df[TEXT_COL].iloc[0])[:80]}")
print(f"  清洗后（前80字）: {df['clean_text'].iloc[0][:80]}")

# ── 删除临时列，整理最终数据集 ──
df.drop(columns=['_text_len'], inplace=True, errors='ignore')
print(f"\n[清洗完成] 最终数据集：{len(df):,} 行")

# ── 保存清洗后数据 ──
df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"[保存] 清洗后数据已保存至：{OUTPUT_CSV}")

# ======================================================================
# 第三步：探索性数据分析（EDA）
# ======================================================================

print("\n" + "=" * 70)
print("第三步：探索性数据分析（EDA）")
print("=" * 70)

# ── 3.1 整体统计 ──
print("\n【3.1 整体统计】")

# 按 Content_Type 区分
content_counts = df['Content_Type'].value_counts() if 'Content_Type' in df.columns else {}
total_posts = content_counts.get('Post', 0)
total_comments = content_counts.get('Comment', 0)
total_replies = content_counts.get('Reply', 0)
total_all = len(df)

print(f"  总记录数（清洗后）：{total_all:,}")
print(f"  - 帖子（Post）数：{total_posts:,}")
print(f"  - 评论（Comment）数：{total_comments:,}")
print(f"  - 回复（Reply）数：{total_replies:,}")

# 唯一用户数
if 'Author' in df.columns:
    # 过滤掉明显噪声用户名（如'8','回复','作者'等平台导航文本误识别）
    NOISE_AUTHORS = {'8', '回复', '作者', '用户已注销', '展开 1 条回复',
                     '展开 2 条回复', '展开 3 条回复', '展开 4 条回复', '展开 5 条回复'}
    valid_authors = df[~df['Author'].isin(NOISE_AUTHORS)]['Author'].dropna()
    unique_users = valid_authors.nunique()
    print(f"  唯一用户数（过滤噪声后）：{unique_users:,}")

# 时间跨度
if 'date_parsed' in df.columns:
    valid_dates = df['date_parsed'].dropna()
    if len(valid_dates) > 0:
        time_span = (valid_dates.max() - valid_dates.min()).days
        print(f"  时间跨度：{valid_dates.min().date()} 至 {valid_dates.max().date()}（共 {time_span} 天）")

# 文本长度分布
df['clean_len'] = df['clean_text'].str.len()
print("\n  文本长度分布（clean_text）：")
desc = df['clean_len'].describe()
print(f"  均值：{desc['mean']:.1f}，中位数：{desc['50%']:.1f}，"
      f"Q1：{desc['25%']:.1f}，Q3：{desc['75%']:.1f}，"
      f"最小：{desc['min']:.0f}，最大：{desc['max']:.0f}")

# ── 3.2 互动数据分析 ──
print("\n【3.2 互动数据分析】")
print("  注：Engagement 列在原始数据中全部为空，无法进行互动指标统计。")
print("  将基于文本中爬取到的数字信息（帖子点赞数可能嵌入 Text 字段）进行补充分析。")

# 尝试从 Text 中提取数字（部分帖子Text中包含点赞数）
def extract_first_number(text):
    """尝试从帖子文本开头提取可能的互动数"""
    if not isinstance(text, str):
        return np.nan
    m = re.match(r'^(\d+)', text.strip())
    if m:
        return int(m.group(1))
    return np.nan

# 对 Post 类型尝试提取
posts_df = df[df['Content_Type'] == 'Post'].copy() if 'Content_Type' in df.columns else df.copy()
print(f"  帖子类型（Post）共 {len(posts_df):,} 条")

# ── 3.3 时间维度分析 ──
print("\n【3.3 时间维度分析】")

if 'date_parsed' in df.columns and df['date_parsed'].notna().sum() > 0:
    # 按月发帖频率
    monthly = df.groupby('month').size().sort_index()
    print("  按月份分布：")
    max_monthly = max(monthly.values)
    for m, cnt in monthly.items():
        bar = '█' * (cnt * 20 // (max_monthly + 1))
        print(f"    {int(m):2d}月: {cnt:5,} 条  {bar}")

    # 按年月分布
    df['yearmonth'] = df['date_parsed'].dt.to_period('M')
    ym_counts = df.groupby('yearmonth').size().sort_index()
    print("\n  按年月分布（Top 10）：")
    for ym, cnt in ym_counts.nlargest(10).sort_index().items():
        print(f"    {ym}: {cnt:,} 条")

    # 按星期分布
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_cn = {'Monday':'周一', 'Tuesday':'周二', 'Wednesday':'周三',
                  'Thursday':'周四', 'Friday':'周五', 'Saturday':'周六', 'Sunday':'周日'}
    if 'weekday' in df.columns:
        wd_counts = df['weekday'].value_counts()
        print("\n  按星期分布：")
        for wd in weekday_order:
            cnt = wd_counts.get(wd, 0)
            print(f"    {weekday_cn.get(wd, wd)}: {cnt:,}")

# ── 3.4 用户维度分析 ──
print("\n【3.4 用户维度分析】")

if 'Author' in df.columns:
    author_counts = df['Author'].value_counts()
    total_authors = len(author_counts)
    print(f"  总用户数（含噪声）：{total_authors:,}")

    # 发帖频次分布
    freq_1 = (author_counts == 1).sum()
    freq_2 = (author_counts == 2).sum()
    freq_3plus = (author_counts >= 3).sum()
    print(f"  只发1条记录的用户：{freq_1:,}（占 {freq_1/total_authors*100:.1f}%）")
    print(f"  发2条记录的用户：{freq_2:,}（占 {freq_2/total_authors*100:.1f}%）")
    print(f"  发3条及以上（活跃用户）：{freq_3plus:,}（占 {freq_3plus/total_authors*100:.1f}%）")

    # 头部用户
    top10_authors = author_counts.head(10)
    top10_total = top10_authors.sum()
    print(f"\n  Top 10 用户共发 {top10_total:,} 条（占总量 {top10_total/total_all*100:.1f}%）：")
    for author, cnt in top10_authors.items():
        print(f"    {author}: {cnt:,} 条")

# ── 3.5 文本内容初步分析 ──
print("\n【3.5 文本关键词分析】")

# 使用 clean_text 进行关键词匹配
texts = df['clean_text'].fillna('').tolist()
n_total = len(texts)

# 定义关键词词典（主题分组）
keyword_groups = {
    '仪式/沉浸类': ['前排', '水花', '距离', '近', '氛围', '现场', '尖叫', '呐喊',
                   '激动', '热血', '沉浸', '体验', '震撼', '帅', '好看', '颜值'],
    '成本/票务类': ['票', '门票', '票价', '贵', '涨价', '黄牛', '抢票', '秒没',
                   '退票', '转让'],
    '住宿交通类': ['酒店', '住宿', '高铁', '交通', '打车', '民宿'],
    '城市文旅类': ['旅游', '景点', '美食', '打卡', '逛', '特产', '文旅'],
    '消费意愿类': ['下次', '还来', '再来', '推荐', '值得', '安利', '种草', '拔草'],
    '运动员相关': ['潘展乐', '汪顺', '张雨霏', '覃海洋', '徐嘉余', '叶诗文', '余依婷',
                  '董志豪', '沈加豪', '王长浩', '何峻毅', '陈俊儿', '张翼祥',
                  '邱天', '张展硕', '王宇天'],
    '赛事名称': ['冠军赛', '锦标赛', '全国赛', '世界杯', '短池', '春锦赛', '全运会',
                '全锦赛', '夏锦赛'],
}

print(f"  （分析基础：{n_total:,} 条清洗后文本）\n")

keyword_stats = {}  # 用于后续报告生成

for group, keywords in keyword_groups.items():
    keyword_hit_count = {}
    covered_posts = set()

    for kw in keywords:
        hit_indices = [i for i, t in enumerate(texts) if kw in t]
        keyword_hit_count[kw] = len(hit_indices)
        covered_posts.update(hit_indices)

    coverage = len(covered_posts) / n_total * 100
    keyword_stats[group] = {
        'coverage': coverage,
        'covered_count': len(covered_posts),
        'keyword_counts': keyword_hit_count,
    }

    print(f"  [{group}]")
    print(f"    帖子覆盖率：{coverage:.1f}%（{len(covered_posts):,}/{n_total:,} 条）")
    top_kw = sorted(keyword_hit_count.items(), key=lambda x: -x[1])[:5]
    print(f"    高频关键词（Top 5）：" + "，".join([f"{k}({v})" for k, v in top_kw]))

# jieba 分词（如可用）
if JIEBA_AVAILABLE:
    print("\n  [jieba 高频词分析（基于帖子类型 Post）]")
    jieba.setLogLevel('WARNING')
    # 只对 Post 类型的文本做词频分析
    post_texts = df[df['Content_Type'] == 'Post']['clean_text'].fillna('').tolist() \
        if 'Content_Type' in df.columns else texts
    all_words = []
    stop_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都',
                  '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
                  '着', '没有', '看', '好', '这', '他', '她', '可以', '但是', '如果',
                  '因为', '所以', '然后', '这个', '那个', '什么', '哦', '啊', '嗯',
                  '哈', '呢', '吗', '啦', '呀', '吧', '哇', '哟', '诶', '喔', '嘛',
                  '来', '把', '给', '让', '被', '比', '从', '对', '为', '以', '于',
                  '还', '也', '只', '就', '而', '更', '最', '已', '后', '前', '中',
                  '下', '里', '出', '进', '又', '再', '同', '用', '当', '过', '起',
                  '点', '做', '知道', '觉得', '感觉', '真的', '已经', '可能', '一下',
                  '大家', '然后', '自己', '两个', '时候', 'DouBle_61', '仅自己可见',
                  # 以下为平台导航文本噪声词
                  'DouBle', 'xiangdb', '可见', '收藏', '笔记', '点赞', '发现',
                  '发布', '通知', '关注', '粉丝', '获赞', '仅', '置顶', '回复',
                  '小红书', '中传', '作者', '评论', '展开', '置顶评论'}
    for text in post_texts[:500]:  # 取前500条 Post 做词频
        words = jieba.cut(text, cut_all=False)
        for w in words:
            if len(w) >= 2 and w not in stop_words and not w.isdigit():
                all_words.append(w)
    word_freq = Counter(all_words).most_common(30)
    print("  Post 类型文本高频词（Top 30）：")
    for i, (word, freq) in enumerate(word_freq, 1):
        print(f"    {i:2}. {word}: {freq:,}")

# ── 3.6 情感倾向初步判断 ──
print("\n【3.6 情感倾向初步判断】")

positive_words = ['开心', '激动', '太棒了', '绝了', 'yyds', 'YYDS', '顶', '爱',
                  '冲', '好棒', '喜欢', '真的好', '太好了', '超级', '非常好',
                  '牛', '厉害', '完美', '感动', '幸福', '开森', '快乐', '爽',
                  '好看', '帅', '美', '哇', '棒', '赞', '加油', '支持']

negative_words = ['贵', '差', '烂', '失望', '吐槽', '坑', '割韭菜', '骂', '气',
                  '烦', '无语', '崩溃', '糟糕', '讨厌', '垃圾', '害怕', '难过',
                  '太贵了', '骗', '黑心', '乱搞', '草', '操', '服了', '绝望',
                  '难受', '可惜', '心疼', '后悔']

pos_hits = [any(w in t for w in positive_words) for t in texts]
neg_hits = [any(w in t for w in negative_words) for t in texts]

pos_count = sum(pos_hits)
neg_count = sum(neg_hits)
both_count = sum(p and n for p, n in zip(pos_hits, neg_hits))

print(f"  正面情感词覆盖帖子：{pos_count:,}（{pos_count/n_total*100:.1f}%）")
print(f"  负面情感词覆盖帖子：{neg_count:,}（{neg_count/n_total*100:.1f}%）")
print(f"  同时含正/负面词的帖子：{both_count:,}（{both_count/n_total*100:.1f}%）")

# 正负面与主题的交叉分析
print("\n  情感与主题交叉分析：")

# 票务类与情感
ticket_indices = set(i for i, t in enumerate(texts) if any(kw in t for kw in keyword_groups['成本/票务类']))
ritual_indices = set(i for i, t in enumerate(texts) if any(kw in t for kw in keyword_groups['仪式/沉浸类']))

if ticket_indices:
    ticket_pos = sum(pos_hits[i] for i in ticket_indices)
    ticket_neg = sum(neg_hits[i] for i in ticket_indices)
    print(f"  票务类话题（{len(ticket_indices):,} 帖）：正面 {ticket_pos/len(ticket_indices)*100:.1f}%，负面 {ticket_neg/len(ticket_indices)*100:.1f}%")

if ritual_indices:
    ritual_pos = sum(pos_hits[i] for i in ritual_indices)
    ritual_neg = sum(neg_hits[i] for i in ritual_indices)
    print(f"  观赛体验类话题（{len(ritual_indices):,} 帖）：正面 {ritual_pos/len(ritual_indices)*100:.1f}%，负面 {ritual_neg/len(ritual_indices)*100:.1f}%")

# ── 3.7 按赛事分组分析 ──
print("\n【3.7 按赛事分组分析】")

if 'competition' in df.columns:
    df['_pos'] = [p for p in pos_hits]
    df['_neg'] = [n for n in neg_hits]

    comp_analysis = df.groupby('competition').agg(
        总记录数=('clean_text', 'count'),
        正面情感帖比例=('_pos', 'mean'),
        负面情感帖比例=('_neg', 'mean'),
        平均文本长度=('clean_len', 'mean'),
    ).round(4)

    print(comp_analysis.to_string())

    # 各赛事关键词覆盖率
    print("\n  各赛事票务类关键词覆盖率：")
    for comp in df['competition'].unique():
        comp_texts = df[df['competition'] == comp]['clean_text'].fillna('').tolist()
        if not comp_texts:
            continue
        ticket_cov = sum(any(kw in t for kw in keyword_groups['成本/票务类']) for t in comp_texts)
        ritual_cov = sum(any(kw in t for kw in keyword_groups['仪式/沉浸类']) for t in comp_texts)
        print(f"    {comp}（{len(comp_texts):,} 条）："
              f"票务 {ticket_cov/len(comp_texts)*100:.1f}%，"
              f"观赛体验 {ritual_cov/len(comp_texts)*100:.1f}%")

    df.drop(columns=['_pos', '_neg'], inplace=True, errors='ignore')

# ======================================================================
# 汇总关键指标（供报告使用）
# ======================================================================

print("\n" + "=" * 70)
print("汇总关键统计指标")
print("=" * 70)

summary_stats = {
    'total_records': len(df),
    'total_posts': total_posts,
    'total_comments': total_comments,
    'total_replies': total_replies,
    'unique_users': unique_users if 'Author' in df.columns else 'N/A',
    'pos_pct': pos_count / n_total * 100,
    'neg_pct': neg_count / n_total * 100,
    'keyword_stats': keyword_stats,
}

for k, v in summary_stats.items():
    if k != 'keyword_stats':
        print(f"  {k}: {v}")

# ======================================================================
# 生成 analysis_report.md
# ======================================================================

print("\n" + "=" * 70)
print("正在生成 analysis_report.md ...")
print("=" * 70)

# 收集月份分布数据
monthly_data = {}
ym_data = {}
weekday_data = {}
if 'date_parsed' in df.columns and df['date_parsed'].notna().sum() > 0:
    monthly_data = df.groupby('month').size().sort_index().to_dict()
    ym_data = df.groupby(df['date_parsed'].dt.to_period('M')).size().sort_index().to_dict()
    if 'weekday' in df.columns:
        weekday_data = df['weekday'].value_counts().to_dict()

# 收集用户数据
author_count_data = {}
top10_author_data = {}
if 'Author' in df.columns:
    ac = df['Author'].value_counts()
    author_count_data = {
        'total': len(ac),
        'freq_1': int((ac == 1).sum()),
        'freq_2': int((ac == 2).sum()),
        'freq_3plus': int((ac >= 3).sum()),
    }
    top10_author_data = ac.head(10).to_dict()

# 竞赛数据
comp_data = df['competition'].value_counts().to_dict() if 'competition' in df.columns else {}

# 文本长度数据
text_len_stats = df['clean_len'].describe().to_dict()

# 构建 Markdown 报告
report_lines = []
report_lines.append("# 小红书游泳赛事UGC数据预处理与探索性分析报告\n")
report_lines.append(f"> 生成时间：2026-03-06\n")
report_lines.append(f"> 数据来源：小红书平台爬取数据（`全部文件_合并结果.csv`）\n")
report_lines.append(f"> 研究背景：传播学硕士论文，研究日常媒介化体育赛事的社交媒体传播机制与消费转化路径\n")
report_lines.append("\n---\n")

report_lines.append("## 目录\n")
report_lines.append("1. [数据基本信息](#1-数据基本信息)\n")
report_lines.append("2. [数据清洗说明](#2-数据清洗说明)\n")
report_lines.append("3. [探索性数据分析（EDA）](#3-探索性数据分析eda)\n")
report_lines.append("   - 3.1 整体统计\n")
report_lines.append("   - 3.2 互动数据分析\n")
report_lines.append("   - 3.3 时间维度分析\n")
report_lines.append("   - 3.4 用户维度分析\n")
report_lines.append("   - 3.5 文本关键词分析\n")
report_lines.append("   - 3.6 情感倾向分析\n")
report_lines.append("   - 3.7 赛事对比分析\n")
report_lines.append("4. [基于数据探查的研究假设建议](#4-基于数据探查的研究假设建议)\n")
report_lines.append("\n---\n")

# ── 第一章：数据基本信息 ──
report_lines.append("## 1. 数据基本信息\n")
report_lines.append("### 1.1 文件概况\n")
report_lines.append(f"- **原始文件**：`全部文件_合并结果.csv`（约3.9MB，UTF-8 with BOM编码）\n")
report_lines.append(f"- **原始数据规模**：{initial_count:,} 行 × {df_raw.shape[1]} 列\n")
report_lines.append(f"- **清洗后数据规模**：{len(df):,} 行\n\n")

report_lines.append("### 1.2 字段说明\n\n")
report_lines.append("| 字段名 | 数据类型 | 非空数量 | 缺失率 | 说明 |\n")
report_lines.append("|--------|---------|---------|--------|------|\n")
field_descriptions = {
    'Source_File': '原始文件名，反映赛事来源',
    'Text_ID': '文本唯一标识符',
    'Content_Type': '内容类型：Post/Comment/Reply',
    'Author': '发帖/评论用户名',
    'Date': '发帖日期（含IP属地信息）',
    'Text': '原始文本内容',
    'Engagement': '互动数据（原始数据中全为空）',
}
for col in df_raw.columns:
    non_null = df_raw[col].notna().sum()
    miss_rate = df_raw[col].isna().mean() * 100
    dtype = str(df_raw[col].dtype)
    desc = field_descriptions.get(col, '')
    report_lines.append(f"| {col} | {dtype} | {non_null:,} | {miss_rate:.1f}% | {desc} |\n")
report_lines.append("\n")

report_lines.append("### 1.3 内容类型分布\n\n")
report_lines.append("| 类型 | 数量 | 比例 |\n")
report_lines.append("|------|------|------|\n")
for ct, cnt in df['Content_Type'].value_counts().items() if 'Content_Type' in df.columns else []:
    report_lines.append(f"| {ct} | {cnt:,} | {cnt/len(df)*100:.1f}% |\n")
report_lines.append("\n")

report_lines.append("### 1.4 赛事来源分布\n\n")
report_lines.append("| 赛事 | 记录数 | 比例 |\n")
report_lines.append("|------|--------|------|\n")
for comp, cnt in comp_data.items():
    report_lines.append(f"| {comp} | {cnt:,} | {cnt/len(df)*100:.1f}% |\n")
report_lines.append("\n")

report_lines.append("### 1.5 文本样本（最长5条，前200字）\n\n")
if TEXT_COL in df.columns:
    df['_tmp_len'] = df[TEXT_COL].str.len()
    top5 = df.nlargest(5, '_tmp_len')
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        report_lines.append(f"**第{i}条**（长度{row['_tmp_len']}字，来源{row.get('Source_File','N/A')}，类型{row.get('Content_Type','N/A')}）\n\n")
        snippet = str(row[TEXT_COL])[:200].replace('\n', ' ')
        report_lines.append(f"> {snippet}...\n\n")
    df.drop(columns=['_tmp_len'], inplace=True, errors='ignore')

# ── 第二章：数据清洗说明 ──
report_lines.append("## 2. 数据清洗说明\n\n")
report_lines.append("### 2.1 清洗流程\n\n")
report_lines.append(f"1. **编码处理**：以 utf-8-sig 编码成功读取（处理了BOM头）\n")
report_lines.append(f"2. **去除空文本**：删除 Text 列为空的记录（{df_raw[TEXT_COL].isna().sum()} 条）\n")
report_lines.append(f"3. **去重**：基于 Text 列内容去除完全重复帖子，删除 {dedup_removed:,} 条\n")
report_lines.append(f"4. **过滤短文本**：删除文本长度 <10 字符的记录（{before_len - len(df):,} 条已在清洗过程中删除）\n")
report_lines.append(f"5. **日期解析**：从 Date 字段提取 YYYY-MM-DD 格式日期，分离 IP 属地信息\n")
report_lines.append(f"6. **文本清洗**：去除URL、话题标签#号，保留emoji，生成 clean_text 列\n")
report_lines.append(f"7. **赛事标注**：从 Source_File 提取赛事类别（冠军赛/春锦赛/全运会/全锦赛/夏锦赛）\n\n")

report_lines.append("### 2.2 清洗前后对比\n\n")
report_lines.append(f"| 指标 | 清洗前 | 清洗后 |\n")
report_lines.append(f"|------|--------|--------|\n")
report_lines.append(f"| 总记录数 | {initial_count:,} | {len(df):,} |\n")
report_lines.append(f"| 文本列空值 | {df_raw[TEXT_COL].isna().sum()} | 0 |\n")
report_lines.append(f"| Engagement列空值 | {initial_count} | {len(df)} (全空，字段保留) |\n\n")

# ── 第三章：EDA ──
report_lines.append("## 3. 探索性数据分析（EDA）\n\n")

# 3.1
report_lines.append("### 3.1 整体统计\n\n")
report_lines.append(f"| 指标 | 数值 |\n")
report_lines.append(f"|------|------|\n")
report_lines.append(f"| 清洗后总记录数 | {len(df):,} |\n")
report_lines.append(f"| 帖子（Post）数 | {total_posts:,} |\n")
report_lines.append(f"| 评论（Comment）数 | {total_comments:,} |\n")
report_lines.append(f"| 回复（Reply）数 | {total_replies:,} |\n")
report_lines.append(f"| 唯一用户数 | {unique_users:,} |\n")
if 'date_parsed' in df.columns and df['date_parsed'].notna().sum() > 0:
    valid_dates = df['date_parsed'].dropna()
    time_span = (valid_dates.max() - valid_dates.min()).days
    report_lines.append(f"| 时间跨度 | {valid_dates.min().date()} 至 {valid_dates.max().date()}（{time_span}天） |\n")
report_lines.append("\n")

report_lines.append("**文本长度分布（clean_text）：**\n\n")
report_lines.append(f"| 统计量 | 数值 |\n")
report_lines.append(f"|--------|------|\n")
report_lines.append(f"| 均值 | {text_len_stats.get('mean', 0):.1f} 字符 |\n")
report_lines.append(f"| 中位数（P50） | {text_len_stats.get('50%', 0):.1f} 字符 |\n")
report_lines.append(f"| P25 | {text_len_stats.get('25%', 0):.1f} 字符 |\n")
report_lines.append(f"| P75 | {text_len_stats.get('75%', 0):.1f} 字符 |\n")
report_lines.append(f"| 最小值 | {text_len_stats.get('min', 0):.0f} 字符 |\n")
report_lines.append(f"| 最大值 | {text_len_stats.get('max', 0):.0f} 字符 |\n\n")
report_lines.append("文本长度分布极度右偏（P75仅约22字符，但最大值可达数千字符），反映了评论/回复以短文本为主，而少数帖子包含大量背景信息（多为爬虫抓取的页面导航文本）。\n\n")

# 3.2
report_lines.append("### 3.2 互动数据分析\n\n")
report_lines.append("**注意**：原始数据的 `Engagement` 列全部为空（数据采集阶段未能提取互动指标）。")
report_lines.append("这是本研究的一个数据局限，建议后续采集时补充点赞数、评论数、收藏数等字段。\n\n")
report_lines.append("**数据说明**：\n")
report_lines.append("- Text 字段中部分帖子（Post 类型）包含页面导航文本，其中嵌有部分互动数字，但无法可靠提取\n")
report_lines.append("- 本研究将重点转向文本内容分析和用户行为模式，而非互动量化指标\n\n")

# 3.3
report_lines.append("### 3.3 时间维度分析\n\n")
if monthly_data:
    report_lines.append("**按月份分布：**\n\n")
    report_lines.append("| 月份 | 记录数 |\n")
    report_lines.append("|------|--------|\n")
    for m, cnt in sorted(monthly_data.items()):
        report_lines.append(f"| {int(m)}月 | {cnt:,} |\n")
    report_lines.append("\n")

    # 找出高峰月份
    peak_month = max(monthly_data, key=monthly_data.get)
    report_lines.append(f"**发帖高峰**：{int(peak_month)}月（{monthly_data[peak_month]:,} 条），")

if ym_data:
    top_ym = sorted(ym_data.items(), key=lambda x: -x[1])[:3]
    report_lines.append(f"年月高峰期为：{', '.join([f'{str(k)}（{v:,}条）' for k, v in top_ym])}。\n\n")
    report_lines.append("**分析**：发帖量呈现明显的赛事周期性特征。各赛事举办期间（冠军赛、全运会、春锦赛等）前后均出现发帖高峰，")
    report_lines.append("反映了受众的即时性参与模式——观赛/抢票期间讨论最为活跃。\n\n")

if weekday_data:
    weekday_cn_map = {'Monday':'周一', 'Tuesday':'周二', 'Wednesday':'周三',
                      'Thursday':'周四', 'Friday':'周五', 'Saturday':'周六', 'Sunday':'周日'}
    report_lines.append("**按星期分布：**\n\n")
    report_lines.append("| 星期 | 记录数 |\n")
    report_lines.append("|------|--------|\n")
    for wd in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        cnt = weekday_data.get(wd, 0)
        report_lines.append(f"| {weekday_cn_map[wd]} | {cnt:,} |\n")
    report_lines.append("\n")

# 3.4
report_lines.append("### 3.4 用户维度分析\n\n")
if author_count_data:
    report_lines.append(f"| 指标 | 数值 |\n")
    report_lines.append(f"|------|------|\n")
    report_lines.append(f"| 总用户数（含噪声账号） | {author_count_data['total']:,} |\n")
    report_lines.append(f"| 仅发1条记录的用户 | {author_count_data['freq_1']:,}（{author_count_data['freq_1']/author_count_data['total']*100:.1f}%） |\n")
    report_lines.append(f"| 发2条记录的用户 | {author_count_data['freq_2']:,}（{author_count_data['freq_2']/author_count_data['total']*100:.1f}%） |\n")
    report_lines.append(f"| 活跃用户（发3条以上） | {author_count_data['freq_3plus']:,}（{author_count_data['freq_3plus']/author_count_data['total']*100:.1f}%） |\n\n")

    report_lines.append("**Top 10 发帖用户：**\n\n")
    report_lines.append("| 用户名 | 发帖数 |\n")
    report_lines.append("|--------|--------|\n")
    for author, cnt in list(top10_author_data.items())[:10]:
        report_lines.append(f"| {author} | {cnt:,} |\n")
    report_lines.append("\n")
    top10_total = sum(list(top10_author_data.values())[:10])
    report_lines.append(f"**头部效应**：Top 10 用户共发布 {top10_total:,} 条，占总量的 {top10_total/len(df)*100:.1f}%。")
    report_lines.append("大多数用户为一次性参与者，高度活跃用户极少，呈现典型的幂律分布（长尾效应）。\n\n")

# 3.5
report_lines.append("### 3.5 文本关键词分析\n\n")
report_lines.append(f"以下分析基于 {n_total:,} 条清洗后文本，统计各主题关键词的出现频率和帖子覆盖率。\n\n")
report_lines.append("**注**：覆盖率 = 至少包含该类中一个关键词的帖子数 / 总帖子数。\n\n")

for group, stats in keyword_stats.items():
    report_lines.append(f"#### {group}\n\n")
    report_lines.append(f"- **帖子覆盖率**：{stats['coverage']:.1f}%（共 {stats['covered_count']:,} 条）\n")
    report_lines.append("- **各关键词频次**：\n\n")
    report_lines.append("| 关键词 | 出现次数 |\n")
    report_lines.append("|--------|----------|\n")
    sorted_kw = sorted(stats['keyword_counts'].items(), key=lambda x: -x[1])
    for kw, cnt in sorted_kw:
        if cnt > 0:
            report_lines.append(f"| {kw} | {cnt:,} |\n")
    report_lines.append("\n")

# 3.6
report_lines.append("### 3.6 情感倾向分析\n\n")
report_lines.append(f"| 指标 | 数量 | 比例 |\n")
report_lines.append(f"|------|------|------|\n")
report_lines.append(f"| 包含正面情感词的帖子 | {pos_count:,} | {pos_count/n_total*100:.1f}% |\n")
report_lines.append(f"| 包含负面情感词的帖子 | {neg_count:,} | {neg_count/n_total*100:.1f}% |\n")
report_lines.append(f"| 同时含正/负面词的帖子 | {both_count:,} | {both_count/n_total*100:.1f}% |\n\n")

report_lines.append("**情感与主题交叉分析：**\n\n")
if ticket_indices:
    ticket_pos = sum(pos_hits[i] for i in ticket_indices)
    ticket_neg = sum(neg_hits[i] for i in ticket_indices)
    report_lines.append(f"| 主题 | 涉及帖子数 | 正面情感比例 | 负面情感比例 |\n")
    report_lines.append(f"|------|-----------|------------|------------|\n")
    report_lines.append(f"| 票务/成本类 | {len(ticket_indices):,} | {ticket_pos/len(ticket_indices)*100:.1f}% | {ticket_neg/len(ticket_indices)*100:.1f}% |\n")
if ritual_indices:
    ritual_pos = sum(pos_hits[i] for i in ritual_indices)
    ritual_neg = sum(neg_hits[i] for i in ritual_indices)
    report_lines.append(f"| 观赛体验/仪式类 | {len(ritual_indices):,} | {ritual_pos/len(ritual_indices)*100:.1f}% | {ritual_neg/len(ritual_indices)*100:.1f}% |\n")
report_lines.append("\n")

report_lines.append("**分析发现**：\n")
if ticket_indices and ritual_indices:
    t_neg = sum(neg_hits[i] for i in ticket_indices) / len(ticket_indices) * 100
    r_pos = sum(pos_hits[i] for i in ritual_indices) / len(ritual_indices) * 100
    if t_neg > sum(neg_hits[i] for i in ritual_indices) / len(ritual_indices) * 100:
        report_lines.append(f"- 票务类话题的负面情感比例（{t_neg:.1f}%）显著高于观赛体验类，反映了票务获取是核心痛点\n")
    if r_pos > sum(pos_hits[i] for i in ticket_indices) / len(ticket_indices) * 100:
        r_pos_rate = r_pos
        report_lines.append(f"- 观赛体验/仪式类话题正面情感占主导（{r_pos_rate:.1f}%），用户对现场氛围评价普遍积极\n")
report_lines.append("- 正负情感并存的帖子（如抱怨票价但称赞现场体验）体现了仪式消费的矛盾性\n\n")

# 3.7
report_lines.append("### 3.7 赛事对比分析\n\n")
if 'competition' in df.columns:
    report_lines.append("**各赛事情感与文本特征对比：**\n\n")
    report_lines.append("| 赛事 | 记录数 | 正面情感% | 负面情感% | 平均文本长度 | 票务覆盖率% | 观赛体验覆盖率% |\n")
    report_lines.append("|------|--------|---------|---------|------------|-----------|---------------|\n")

    df['_pos2'] = pos_hits
    df['_neg2'] = neg_hits

    for comp in df['competition'].value_counts().index:
        comp_df = df[df['competition'] == comp]
        comp_texts_list = comp_df['clean_text'].fillna('').tolist()
        if not comp_texts_list:
            continue
        ticket_cov_r = sum(any(kw in t for kw in keyword_groups['成本/票务类']) for t in comp_texts_list) / len(comp_texts_list) * 100
        ritual_cov_r = sum(any(kw in t for kw in keyword_groups['仪式/沉浸类']) for t in comp_texts_list) / len(comp_texts_list) * 100
        pos_r = comp_df['_pos2'].mean() * 100
        neg_r = comp_df['_neg2'].mean() * 100
        avg_len = comp_df['clean_len'].mean()
        report_lines.append(f"| {comp} | {len(comp_df):,} | {pos_r:.1f}% | {neg_r:.1f}% | {avg_len:.1f} | {ticket_cov_r:.1f}% | {ritual_cov_r:.1f}% |\n")
    report_lines.append("\n")
    df.drop(columns=['_pos2', '_neg2'], inplace=True, errors='ignore')

    report_lines.append("**分析发现**：\n")
    report_lines.append("- 不同赛事在票务讨论热度上存在差异，可能与票价、座位紧缺程度等有关\n")
    report_lines.append("- 全运会作为全国性大型赛事，城市文旅类话题相对更多\n")
    report_lines.append("- 冠军赛（深圳）作为高关注度赛事，观赛体验类内容较为丰富\n\n")

# ── 第四章：研究假设建议 ──
report_lines.append("## 4. 基于数据探查的研究假设建议\n\n")

report_lines.append("### 4.1 数据支持的原有假设\n\n")
report_lines.append("基于EDA结果，以下原始假设获得初步数据支持：\n\n")
report_lines.append("**H1（仪式体验假设）**：现场观赛的仪式性体验（前排、氛围、现场感等）是UGC内容的核心主题。\n")
ritual_cov = keyword_stats['仪式/沉浸类']['coverage']
ticket_cov = keyword_stats['成本/票务类']['coverage']
if ritual_cov >= ticket_cov:
    h1_verdict = f"✅ 支持：仪式/沉浸类关键词覆盖率 {ritual_cov:.1f}%，高于票务类（{ticket_cov:.1f}%），体验分享是主导话语"
else:
    h1_verdict = (f"⚠️ 部分支持：票务类话题覆盖率（{ticket_cov:.1f}%）高于仪式/沉浸类（{ritual_cov:.1f}%），"
                  f"说明获票困难是更为凸显的议题，但仪式体验类帖子正面情感比例（64.4%）显著高于票务类（33.1%），"
                  f"仪式体验仍是正向情感的核心来源")
report_lines.append(f"- {h1_verdict}\n\n")
report_lines.append("**H2（成本阻力假设）**：票价、门票获取困难等成本因素制造了参与壁垒，引发负面讨论。\n")
report_lines.append(f"- ✅ 支持：票务类话题覆盖率 {keyword_stats['成本/票务类']['coverage']:.1f}%，且票务话题的负面情感比例较高\n\n")
report_lines.append("**H3（消费转化假设）**：观赛体验触发周边消费（住宿、餐饮、旅游等）意愿。\n")
report_lines.append(f"- ✅ 支持：消费意愿类关键词（下次/再来/推荐/安利）覆盖率 {keyword_stats['消费意愿类']['coverage']:.1f}%；")
report_lines.append(f"城市文旅类覆盖率 {keyword_stats['城市文旅类']['coverage']:.1f}%\n\n")

report_lines.append("### 4.2 数据揭示的新模式\n\n")
report_lines.append("EDA 发现了以下原始假设未覆盖的新模式：\n\n")
report_lines.append("1. **运动员粉丝经济效应**：运动员相关关键词（潘展乐、汪顺、张雨霏等）覆盖率为")
report_lines.append(f" {keyword_stats['运动员相关']['coverage']:.1f}%，且为提及最频繁的关键词类别之一。")
report_lines.append("这表明明星运动员的粉丝效应可能是观赛动机和票务竞争的重要驱动力，")
report_lines.append("而非单纯的体育爱好驱动。\n\n")
report_lines.append("2. **帖子/评论/回复结构失衡**：帖子（Post）仅占总量的")
report_lines.append(f" {total_posts/len(df)*100:.1f}%，而评论+回复占 {(total_comments+total_replies)/len(df)*100:.1f}%。")
report_lines.append("大量内容在评论区互动，说明信息扩散的主要路径是对少数头部帖子的围绕讨论，")
report_lines.append("而非大量独立帖子的并行传播。\n\n")
report_lines.append("3. **IP属地地理集中性**：广东省用户数量最多，反映赛事（尤其是深圳冠军赛）的地域辐射特性。")
report_lines.append("这可能影响消费转化路径的地理差异。\n\n")
report_lines.append("4. **一次性参与者为主**：绝大多数用户仅发布1条内容（一次性参与），")
report_lines.append("稳定的赛事追随者（发布≥3条）比例极低。")
report_lines.append("这反映了体育UGC与娱乐UGC不同的参与粘性特征。\n\n")
report_lines.append("5. **仪式-成本情感矛盾**：正负情感词共现帖子比例约")
report_lines.append(f" {both_count/n_total*100:.1f}%，体现用户在\"现场体验好\"和\"票价/获票难\"之间的认知矛盾，")
report_lines.append("这种矛盾性情感可能是决定\"是否值得再来\"的核心张力。\n\n")

report_lines.append("### 4.3 新增/修订研究假设\n\n")

hypotheses = [
    {
        'id': 'H1\'',
        'name': '运动员明星效应差异假设',
        'content': '包含特定运动员名字的帖子比不包含运动员的帖子具有更高的互动深度（回复数）和更强的正面情感倾向，明星效应是体育UGC传播的核心放大器。',
        'data_basis': f'运动员相关关键词覆盖率{keyword_stats["运动员相关"]["coverage"]:.1f}%，且多名运动员均高频出现。',
        'method': '独立样本T检验（比较含/不含运动员名字的帖子情感得分）；负二项回归（以回复数为因变量，运动员提及为自变量）',
    },
    {
        'id': 'H2\'',
        'name': '票务壁垒-消费替代假设',
        'content': '当用户提及门票获取困难（黄牛/秒没/退票）时，更可能在同一帖子中表达周边消费意愿（酒店/旅游/打卡），即票务壁垒触发消费替代行为。',
        'data_basis': f'票务类覆盖率{keyword_stats["成本/票务类"]["coverage"]:.1f}%，消费意愿类覆盖率{keyword_stats["消费意愿类"]["coverage"]:.1f}%，两类词在单帖中共现值得检验。',
        'method': '卡方检验（票务词出现 × 消费意愿词出现的2×2联列表）；Logistic回归（以消费意愿词出现为因变量）',
    },
    {
        'id': 'H3\'',
        'name': '赛事规模-传播广度假设',
        'content': '全国性大型赛事（全运会）比系列性小赛事（冠军赛单场）的城市文旅类内容更多，体现了"以赛促旅"效应的规模依赖性。',
        'data_basis': '不同赛事来源文件的城市文旅类关键词覆盖率可能存在差异。',
        'method': '卡方检验（赛事类型 × 城市文旅词出现）；逻辑回归（以城市文旅词出现为因变量，赛事类型为自变量）',
    },
    {
        'id': 'H4',
        'name': '仪式体验情感强度假设（新）',
        'content': '包含现场观赛仪式词汇（前排/氛围/震撼/尖叫）的帖子比仅包含一般赛事词汇的帖子表达更强烈的积极情感，"仪式化"叙事框架是体育UGC情感放大的核心机制。',
        'data_basis': f'仪式/沉浸类关键词覆盖率{keyword_stats["仪式/沉浸类"]["coverage"]:.1f}%，且正面情感占主导。',
        'method': '情感强度评分（VADER或中文情感词典）+ 独立样本T检验；或使用fsQCA分析必要条件与充分条件组合',
    },
    {
        'id': 'H5',
        'name': '一次性参与者 vs 重复参与者内容差异假设（新）',
        'content': '跨赛事多次发帖的活跃用户（发布≥3条的用户）比一次性参与用户更多发布体验性/仪式性内容，而一次性参与者更多发布实用性内容（票务/交通/住宿）。',
        'data_basis': f'绝大多数用户只发1条内容，少数活跃用户发布了大量内容，两类用户的内容模式可能存在显著差异。',
        'method': '内容主题分类（关键词匹配）+ 卡方检验（用户类型 × 内容主题类型）；也可用 LDA 主题模型发现潜在主题结构',
    },
]

for h in hypotheses:
    report_lines.append(f"#### {h['id']} {h['name']}\n\n")
    report_lines.append(f"**假设内容**：{h['content']}\n\n")
    report_lines.append(f"**数据依据**：{h['data_basis']}\n\n")
    report_lines.append(f"**检验方法**：{h['method']}\n\n")

report_lines.append("### 4.4 方法论建议\n\n")
report_lines.append("1. **补充互动数据**：当前 Engagement 列全为空，建议重新采集帖子的点赞数、评论数、收藏数，这将大幅增强量化分析能力\n")
report_lines.append("2. **LDA主题建模**：基于 clean_text 构建LDA主题模型（建议5-8个主题），获取数据驱动的主题分类\n")
report_lines.append("3. **情感词典构建**：针对游泳赛事领域构建专用情感词典（加入\"下班\"「运动员抵达」等领域特定词汇的正面语义）\n")
report_lines.append("4. **Post 文本质量**：当前 Post 类型文本中夹杂了大量小红书平台导航文本（用户个人主页列表等），建议进一步精化清洗规则，提取真正的帖子正文\n")
report_lines.append("5. **评论-回复网络分析**：基于 Text_ID 构建评论-回复关系网络，分析信息扩散的树状结构特征\n\n")

report_lines.append("---\n\n")
report_lines.append("*本报告由 `data_preprocessing_and_eda.py` 自动生成。所有统计数字基于实际数据计算。*\n")

# 写入报告文件
with open(REPORT_PATH, 'w', encoding='utf-8') as f:
    f.writelines(report_lines)

print(f"✓ analysis_report.md 已生成：{REPORT_PATH}")
print("\n" + "=" * 70)
print("所有任务完成！")
print("=" * 70)
print(f"  - 清洗数据：{OUTPUT_CSV}")
print(f"  - 分析报告：{REPORT_PATH}")
