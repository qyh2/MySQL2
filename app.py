import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from scipy.cluster.hierarchy import dendrogram, linkage
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm, glm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import warnings
import json
import os
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

# ==========================================
# ---------- 学术图表格式全局配置 ----------
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

academic_template = {
    "layout": {
        "font": {"family": "Times New Roman, SimSun", "size": 14, "color": "#000"},
        "title": {"font": {"size": 16, "family": "Times New Roman, SimSun"}, "x": 0.5, "xanchor": "center"},
        "xaxis": {"showline": True, "linewidth": 1.5, "linecolor": "black", "mirror": True, "ticks": "outside", "showgrid": False},
        "yaxis": {"showline": True, "linewidth": 1.5, "linecolor": "black", "mirror": True, "ticks": "outside", "showgrid": False},
        "plot_bgcolor": "white", "paper_bgcolor": "white"
    }
}
pio.templates["academic"] = academic_template
pio.templates.default = "academic"

# ==========================================
# 1. 页面设置与 CSS
# ==========================================
st.set_page_config(page_title="长期定位试验数据平台", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* 隐藏侧边栏 radio 按钮的圆点 */
    [data-testid="stSidebar"] .stRadio [data-baseweb="radio"] {
        display: none !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        padding-left: 0 !important;
        gap: 0 !important;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
        gap: 0.5rem;
    }
    [data-testid="stSidebar"] {
        background-color: #e8f5e9;
        padding-top: 2rem;
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: 700;
        color: #1e3a2f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stRadio label {
        font-size: 16px;
        font-weight: 500;
        padding: 8px 12px;
        border-radius: 8px;
        transition: background 0.2s;
        cursor: pointer;
    }
    .stRadio label:hover {
        background-color: #c8e6c9;
    }
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #1e3a2f;
        background-color: #e8f5e9;
        padding: 0.8rem 2rem;
        margin-top: -0.5rem;
        margin-bottom: 1rem;
        border-radius: 0 0 12px 12px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: left;
    }
    .page-title {
        background-color: #e8f5e9;
        padding: 0.6rem 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        font-size: 1.8rem;
        font-weight: 600;
        color: #1e3a2f;
    }
    .intro-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        height: 100%;
    }
    .intro-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
        border-left: 4px solid #66bb6a;
        padding-left: 0.8rem;
    }
    .intro-text {
        font-size: 1rem;
        line-height: 1.6;
        text-align: justify;
        color: #2c3e50;
    }
    .unit-list {
        list-style: none;
        padding-left: 0;
    }
    .unit-list li {
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    .person-list {
        margin-top: 1rem;
        padding-left: 1.2rem;
    }
    .person-list li {
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 数据库连接与辅助函数
# ==========================================
DB_CONFIG = {
    "host": "localhost",
    "port": 3306,
    "user": "root",
    "password": "Qyh327122-",
    "database": "agridata_chinese"
}

def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

@st.cache_data(ttl=60)
def load_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM `长期定位试验总表`", conn)
    conn.close()
    return df

try:
    df = load_data()
    db_status = "数据库已连接"
except Exception as e:
    df = pd.DataFrame()
    db_status = f"数据库连接失败: {e}"

numeric_cols = [c for c in df.columns if '(' in c or '指数' in c or 'pH' in c] if not df.empty else []

def get_tukey_letters(tukey_result, groups):
    summary = pd.DataFrame(data=tukey_result._results_table.data[1:], columns=tukey_result._results_table.data[0])
    letters = {g: '' for g in groups}
    current_letter = 'a'
    for i in range(len(groups)):
        if letters[groups[i]] == '':
            letters[groups[i]] += current_letter
            for j in range(i+1, len(groups)):
                match = summary[((summary['group1'] == groups[i]) & (summary['group2'] == groups[j])) | 
                                ((summary['group1'] == groups[j]) & (summary['group2'] == groups[i]))]
                if not match.empty and not match['reject'].values[0]:
                    letters[groups[j]] += current_letter
            current_letter = chr(ord(current_letter) + 1)
    for g in letters:
        letters[g] = ''.join(sorted(set(letters[g])))
    return letters

def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def init_omics_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS `omics_data` (
            `id` INT AUTO_INCREMENT PRIMARY KEY,
            `sample_name` VARCHAR(100) NOT NULL,
            `data_type` VARCHAR(50) NOT NULL,
            `file_path` VARCHAR(500) NOT NULL,
            `upload_date` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            `description` TEXT,
            `metadata` JSON
        ) ENGINE=InnoDB
    """)
    conn.commit()
    cursor.close()
    conn.close()
init_omics_table()

# ==========================================
# 3. 静态 GIS 基础数据字典
# ==========================================
SITE_GEO = pd.DataFrame([
    {"试验点": "北京房山", "代码": "BJFS", "lat": 39.7, "lon": 116.0, "alt": 50, "soil": "褐土", "climate": "暖温带季风", "sys": "轮作"},
    {"试验点": "北京海淀", "代码": "BJHD", "lat": 40.0, "lon": 116.3, "alt": 60, "soil": "褐土", "climate": "暖温带季风", "sys": "连作"},
    {"试验点": "天津武清", "代码": "TJWQ", "lat": 39.4, "lon": 117.0, "alt": 5,  "soil": "潮土", "climate": "暖温带季风", "sys": "轮作"},
    {"试验点": "河北赵县", "代码": "HBZX", "lat": 37.8, "lon": 114.8, "alt": 40, "soil": "潮土", "climate": "暖温带季风", "sys": "轮作"},
    {"试验点": "山东潍坊", "代码": "SDWF", "lat": 36.7, "lon": 119.1, "alt": 20, "soil": "棕壤", "climate": "暖温带季风", "sys": "轮作"},
    {"试验点": "沈阳海城", "代码": "SYHC", "lat": 40.9, "lon": 122.7, "alt": 30, "soil": "棕壤", "climate": "中温带季风", "sys": "轮作"}
])

# ==========================================
# 4. 登录验证
# ==========================================
def login_form():
    with st.form("login"):
        st.write("## 登录")
        username = st.text_input("用户名")
        password = st.text_input("密码", type="password")
        submitted = st.form_submit_button("登录")
        if submitted:
            user_pass = st.secrets.get("users", {})
            if username in user_pass and user_pass[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                roles = st.secrets.get("roles", {})
                st.session_state.role = roles.get(username, "viewer")
                st.success("登录成功！")
                st.rerun()
            else:
                st.error("用户名或密码错误")
    return st.session_state.get("authenticated", False)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_form()
    st.stop()

# ==========================================
# 5. 侧边栏导航
# ==========================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">数据中台</div>', unsafe_allow_html=True)
    page = st.radio(
        "",
        ["项目介绍", "全景数据看板", "高阶统计与预测", "数据录入中心", "质量管理与布局", "实时监测", "系统设置"],
        label_visibility="collapsed"
    )

def page_title(title):
    st.markdown(f'<div class="page-title">{title}</div>', unsafe_allow_html=True)

# ==========================================
# 页面 0：项目介绍
# ==========================================
if page == "项目介绍":
    st.markdown('<div class="main-title">长期定位试验数据平台</div>', unsafe_allow_html=True)
    page_title("项目简介")
    st.markdown("""
    <div style="display: flex; gap: 2rem; flex-wrap: wrap;">
        <div style="flex: 2; min-width: 300px;">
            <div class="intro-card">
                <div class="intro-title">项目简介</div>
                <div class="intro-text">
                    本数据库依托北京市农林科学院植物营养与资源环境研究所主持的农业农村部环渤海设施农业土壤连作障碍治理修复技术模式联网研究项目构建。针对环渤海地区占全国设施蔬菜面积30%以上、以日光温室和塑料大棚为主体的重要产区中普遍存在的土壤次生盐渍化、酸化、板结、根结线虫严重及连作障碍等突出问题，联合沈阳农业大学、河北省农林科学院、天津师范大学、潍坊科技学院等科研合作单位，构建了“识别障碍—明确目标—分类改良”的系统性技术路径，形成轻度障碍采用土壤调理加合理施肥、中度障碍采用耕翻加调理改良加养分管理、重度障碍采用人工基质栽培加滴灌的分级治理技术体系，并在山东、河北、天津、辽宁等省市布设多个长期定位试验点，通过定期采集0~60cm不同土层样品、监测作物全生育期长势与产量、分析土壤理化性质与生物多样性等指标，系统积累了涵盖土壤健康、水肥利用效率及蔬菜品质的多维度长期定位观测数据。相关技术成果在设施蔬菜主产区累计推广20余万亩，番茄单产提高10%以上。本数据库旨在为设施土壤障碍演变规律研究、治理技术效果评估、区域适应性技术模式优化及政府科学决策提供坚实的数据支撑。
                </div>
            </div>
        </div>
        <div style="flex: 1; min-width: 280px;">
            <div class="intro-card">
                <div class="intro-title">参与单位与核心人员</div>
                <div><strong>项目主持单位</strong><br>北京市农林科学院植物营养与资源环境研究所</div>
                <div style="margin-top: 1rem;"><strong>科研合作单位</strong>
                    <ul class="unit-list">
                        <li>沈阳农业大学</li>
                        <li>河北省农林科学院</li>
                        <li>天津师范大学</li>
                        <li>潍坊科技学院</li>
                    </ul>
                </div>
                <div><strong>核心研究团队</strong>
                    <ul class="person-list">
                        <li>魏丹 研究员（项目负责人，北京市农林科学院）</li>
                        <li>邹洪涛 博士（沈阳农业大学）</li>
                        <li>王丽英 研究员（河北省农林科学院）</li>
                        <li>张国刚 博士（天津师范大学）</li>
                        <li>张敬敏 博士（潍坊科技学院）</li>
                        <li>李艳、金梁、丁建莉 等（北京市农林科学院）</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 页面 1：全景数据看板
# ==========================================
elif page == "全景数据看板":
    page_title("全景数据看板与空间分析")
    if df.empty:
        st.warning("暂无数据。")
    else:
        with st.expander("全局数据筛选器", expanded=False):
            sc1, sc2, sc3 = st.columns(3)
            sel_year = sc1.multiselect("年份", df['测定年份'].unique(), default=df['测定年份'].unique())
            sel_site = sc2.multiselect("试验点", df['试验点'].unique(), default=df['试验点'].unique())
            sel_crop = sc3.multiselect("作物", df['当季种植作物'].unique(), default=df['当季种植作物'].unique())
            filtered_df = df[(df['测定年份'].isin(sel_year)) & (df['试验点'].isin(sel_site)) & (df['当季种植作物'].isin(sel_crop))]

        tab1, tab2, tab_gis = st.tabs(["数据探索图表", "指标相关性热力图", "空间分布与 GIS 插值"])
        
        with tab1:
            ctrl_col, plot_col = st.columns([1, 4])
            with ctrl_col:
                y_axis = st.selectbox("纵坐标 (Y轴)", numeric_cols, index=0)
                x_axis = st.selectbox("横坐标 (X轴)", ['处理编号', '测定年份', '试验点'])
                chart_type = st.radio("图表类型", ["箱线图", "柱状图", "散点图", "均值±标准差"])
            with plot_col:
                if chart_type == "箱线图":
                    fig = px.box(filtered_df, x=x_axis, y=y_axis, color='当季种植作物', color_discrete_sequence=px.colors.qualitative.Pastel)
                elif chart_type == "柱状图":
                    agg_df = filtered_df.groupby([x_axis, '当季种植作物'])[y_axis].mean().reset_index()
                    fig = px.bar(agg_df, x=x_axis, y=y_axis, color='当季种植作物', barmode='group')
                elif chart_type == "散点图":
                    fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color='当季种植作物', hover_data=['试验点', '处理编号'])
                else:
                    agg_stats = filtered_df.groupby([x_axis, '当季种植作物'])[y_axis].agg(['mean', 'std']).reset_index()
                    fig = go.Figure()
                    for crop in agg_stats['当季种植作物'].unique():
                        sub = agg_stats[agg_stats['当季种植作物'] == crop]
                        fig.add_trace(go.Scatter(
                            x=sub[x_axis], y=sub['mean'],
                            error_y=dict(type='data', array=sub['std'], visible=True, color='black'),
                            mode='markers+lines', name=crop
                        ))
                    fig.update_layout(yaxis_title=y_axis, xaxis_title=x_axis)
                fig.update_layout(title=f"<b>{y_axis} 随 {x_axis} 的变化</b>", paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            selected_corr = st.multiselect("纳入分析的指标", numeric_cols, default=numeric_cols[:8] if len(numeric_cols)>8 else numeric_cols)
            if len(selected_corr) > 1:
                corr_matrix = filtered_df[selected_corr].corr()
                p_matrix = np.ones((len(selected_corr), len(selected_corr)))
                for i, col1 in enumerate(selected_corr):
                    for j, col2 in enumerate(selected_corr):
                        if i != j:
                            _, p_val = stats.pearsonr(filtered_df[col1].dropna(), filtered_df[col2].dropna())
                            p_matrix[i, j] = p_val
                fig_corr = px.imshow(corr_matrix, text_auto=False, aspect="auto", color_continuous_scale="RdBu_r",
                                      title="指标相关性热力图（颜色表示相关系数，方块上标注显著性）")
                for i in range(len(selected_corr)):
                    for j in range(len(selected_corr)):
                        if i == j:
                            text = "1.00"
                        else:
                            r = corr_matrix.iloc[i, j]
                            p = p_matrix[i, j]
                            stars = p_to_stars(p)
                            text = f"{r:.2f}<br>{stars}"
                        fig_corr.add_annotation(
                            x=j, y=i, text=text,
                            showarrow=False,
                            font=dict(size=10),
                            xref="x", yref="y"
                        )
                fig_corr.update_layout(xaxis=dict(tickvals=list(range(len(selected_corr))), ticktext=selected_corr),
                                       yaxis=dict(tickvals=list(range(len(selected_corr))), ticktext=selected_corr),
                                       width=800, height=800)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("请选择至少2个指标。")

        with tab_gis:
            st.markdown("##### 试验点地理分布与属性空间插值")
            gis_ind = st.selectbox("选择要在地图上插值分析的指标 (基于均值)", numeric_cols, index=0)
            agg_site = filtered_df.groupby('试验点')[gis_ind].mean().reset_index()
            map_df = pd.merge(SITE_GEO, agg_site, on="试验点", how="inner").dropna()
            if len(map_df) >= 4:
                mc1, mc2 = st.columns([1, 1])
                with mc1:
                    fig_map = px.scatter_mapbox(
                        map_df, lat="lat", lon="lon", hover_name="试验点", hover_data=["soil", "climate", gis_ind],
                        color=gis_ind, size=gis_ind, color_continuous_scale="Viridis", size_max=20, zoom=5,
                        title=f"<b>{gis_ind} 在各试验点的分布</b>"
                    )
                    fig_map.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":40,"l":0,"b":0})
                    st.plotly_chart(fig_map, use_container_width=True)
                with mc2:
                    grid_x, grid_y = np.mgrid[114:123:100j, 36:42:100j]
                    grid_z = griddata((map_df['lon'], map_df['lat']), map_df[gis_ind], (grid_x, grid_y), method='cubic')
                    fig_contour = go.Figure(data=go.Contour(
                        z=grid_z.T, x=np.linspace(114, 123, 100), y=np.linspace(36, 42, 100),
                        colorscale='Viridis', contours_coloring='heatmap', connectgaps=True, opacity=0.8
                    ))
                    fig_contour.add_trace(go.Scatter(x=map_df['lon'], y=map_df['lat'], mode='markers+text',
                                                     text=map_df['试验点'], textposition="top right",
                                                     marker=dict(size=10, color='red', line=dict(width=2, color='white'))))
                    fig_contour.update_layout(title=f"<b>{gis_ind} 空间变异等值线图 (插值模拟)</b>", xaxis_title="经度", yaxis_title="纬度")
                    st.plotly_chart(fig_contour, use_container_width=True)
            else:
                st.warning("筛选条件下的试验点少于 4 个，无法进行可靠的空间插值。")

# ==========================================
# 页面 2：高阶统计与预测
# ==========================================
elif page == "高阶统计与预测":
    page_title("高阶统计、多维分析与机器学习预测")
    if df.empty:
        st.warning("暂无数据。")
    else:
        with st.expander("设定分析数据范围", expanded=False):
            fc1, fc2, fc3 = st.columns(3)
            stat_year = fc1.multiselect("限定年份", df['测定年份'].unique(), default=df['测定年份'].unique())
            stat_site = fc2.multiselect("限定试验点", df['试验点'].unique(), default=df['试验点'].unique())
            stat_crop = fc3.multiselect("限定作物", df['当季种植作物'].unique(), default=df['当季种植作物'].unique())
            stat_df = df[(df['测定年份'].isin(stat_year)) & (df['试验点'].isin(stat_site)) & (df['当季种植作物'].isin(stat_crop))]

        tab_anova, tab_pca, tab_rda, tab_cluster, tab_ml = st.tabs([
            "方差分析", "主成分分析", "冗余分析", "聚类分析", "随机森林预测"
        ])
        
        with tab_anova:
            anova_type = st.radio("方差分析类型：", ["单因素方差分析 (One-way)", "双因素方差分析 (Two-way)"], horizontal=True)
            if anova_type == "单因素方差分析 (One-way)":
                st.info("将比较不同【处理编号】对所选指标的显著性差异，并自动标记 LSD/Tukey 字母 (a, b, c)。")
                a_indicator = st.selectbox("因变量 (Y轴分析指标)", numeric_cols, key='a_ind1')
                if st.button("运行单因素 ANOVA", type="primary"):
                    anova_clean_df = stat_df.dropna(subset=[a_indicator])
                    if anova_clean_df['处理编号'].nunique() < 2:
                        st.error("当前筛选的数据池中，【处理编号】不足2个，无法执行方差分析。")
                    else:
                        try:
                            groups = [group[a_indicator].values for name, group in anova_clean_df.groupby('处理编号')]
                            f_stat, p_val = stats.f_oneway(*groups)
                            tukey = pairwise_tukeyhsd(endog=anova_clean_df[a_indicator], groups=anova_clean_df['处理编号'], alpha=0.05)
                            means_df = anova_clean_df.groupby('处理编号')[a_indicator].agg(['mean', 'std']).reset_index()
                            means_df = means_df.sort_values(by='mean', ascending=False)
                            letters_dict = get_tukey_letters(tukey, means_df['处理编号'].tolist())
                            means_df['letter'] = means_df['处理编号'].map(letters_dict)
                            
                            st.subheader("各处理平均值、标准偏差与显著性标记")
                            st.dataframe(means_df.rename(columns={'mean': '平均值', 'std': '标准偏差', 'letter': '显著性标记'}).style.format({
                                '平均值': '{:.2f}', '标准偏差': '{:.2f}'
                            }), use_container_width=True)
                            st.success(f"**ANOVA结果**：F = {f_stat:.2f}, P = {p_val:.4f}  |  {'具有极显著差异' if p_val<0.01 else '具有显著差异' if p_val<0.05 else '未达显著水平'}")
                            fig_bar = go.Figure()
                            fig_bar.add_trace(go.Bar(
                                x=means_df['处理编号'], y=means_df['mean'],
                                error_y=dict(type='data', array=means_df['std'], visible=True, color='black', thickness=1.5),
                                marker_color='#bdc3c7', marker_line_color='black', marker_line_width=1.5
                            ))
                            max_y = means_df['mean'].max() + means_df['std'].max()
                            for _, row in means_df.iterrows():
                                fig_bar.add_annotation(
                                    x=row['处理编号'], y=row['mean'] + row['std'] + (max_y*0.05),
                                    text=f"<b>{row['letter']}</b>", showarrow=False,
                                    font=dict(family="Times New Roman", size=14, color="black")
                                )
                            fig_bar.update_layout(
                                title=f"<b>不同处理对 {a_indicator} 的影响</b>",
                                xaxis_title="<b>处理编号 (Treatment)</b>", yaxis_title=f"<b>{a_indicator}</b>",
                                width=800, height=500
                            )
                            st.plotly_chart(fig_bar, use_container_width=False)
                        except Exception as e:
                            st.error(f"分析异常：{e}")
            else:
                c1, c2, c3 = st.columns(3)
                y_col = c1.selectbox("因变量 (Y)", numeric_cols, key='a_ind2')
                f1_col = c2.selectbox("因子 1", ['处理编号', '测定年份', '试验点', '当季种植作物'], index=0)
                f2_col = c3.selectbox("因子 2", ['处理编号', '测定年份', '试验点', '当季种植作物'], index=1)
                if st.button("运行双因素 ANOVA", type="primary"):
                    if f1_col == f2_col:
                        st.error("两个因子不能相同。")
                    else:
                        temp_df = pd.DataFrame({'Y': stat_df[y_col], 'F1': stat_df[f1_col].astype(str), 'F2': stat_df[f2_col].astype(str)}).dropna()
                        try:
                            model = ols('Y ~ C(F1) + C(F2) + C(F1):C(F2)', data=temp_df).fit()
                            anova_table = sm.stats.anova_lm(model, typ=2)
                            anova_table.index = [f1_col, f2_col, f"交互作用 ({f1_col} × {f2_col})", '残差 (Residual)']
                            anova_table.columns = ['平方和 (SS)', '自由度 (df)', 'F 值 (F)', 'P 值 (PR(>F))']
                            st.dataframe(anova_table.style.format("{:.4f}"))
                        except Exception as e:
                            st.error(f"分析失败，可能是数据缺失或单组无变异：{e}")

        with tab_pca:
            pca_cols = st.multiselect("参与主成分分析的指标 (至少3个)", numeric_cols, default=numeric_cols[:6] if len(numeric_cols)>6 else numeric_cols)
            pca_group = st.selectbox("主成分分析图着色依据：", ['处理编号', '测定年份', '试验点', '当季种植作物'])
            if st.button("运行主成分分析", type="primary"):
                if len(pca_cols) < 3:
                    st.warning("至少选择 3 个指标。")
                else:
                    pca_df = stat_df.dropna(subset=pca_cols + [pca_group])
                    if pca_df.empty:
                        st.error("所选变量含空值，清洗后无数据。")
                    else:
                        X_std = StandardScaler().fit_transform(pca_df[pca_cols])
                        pca = PCA()
                        scores = pca.fit_transform(X_std)
                        loadings = pca.components_.T
                        exp_var = pca.explained_variance_ratio_
                        cum_var = np.cumsum(exp_var)
                        fig_scree = go.Figure()
                        fig_scree.add_trace(go.Bar(x=np.arange(1, len(exp_var)+1), y=exp_var, name="解释方差比例"))
                        fig_scree.add_trace(go.Scatter(x=np.arange(1, len(exp_var)+1), y=cum_var, name="累积比例", mode='lines+markers', yaxis="y2"))
                        fig_scree.update_layout(title="主成分碎石图", xaxis_title="主成分序号", yaxis_title="解释方差比例", yaxis2=dict(title="累积比例", overlaying='y', side='right'), template="academic")
                        st.plotly_chart(fig_scree, use_container_width=True)
                        fig_load = go.Figure()
                        fig_load.add_trace(go.Scatter(x=loadings[:,0], y=loadings[:,1], mode='markers+text', text=pca_cols, textposition="top center", marker=dict(size=8, color='blue')))
                        fig_load.add_shape(type="line", x0=-1, y0=0, x1=1, y1=0, line=dict(dash="dash", color="gray"))
                        fig_load.add_shape(type="line", x0=0, y0=-1, x1=0, y1=1, line=dict(dash="dash", color="gray"))
                        fig_load.update_layout(title="主成分载荷图", xaxis_title=f"PC1 ({exp_var[0]*100:.2f}%)", yaxis_title=f"PC2 ({exp_var[1]*100:.2f}%)", template="academic")
                        st.plotly_chart(fig_load, use_container_width=True)
                        fig_pca = px.scatter(pca_df, x=scores[:,0], y=scores[:,1], color=pca_group, title="主成分得分图", labels={'x': f"PC1 ({exp_var[0]*100:.2f}%)", 'y': f"PC2 ({exp_var[1]*100:.2f}%)"})
                        fig_pca.update_traces(marker=dict(size=10, line=dict(width=1, color='black')))
                        fig_pca.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_pca.add_vline(x=0, line_dash="dash", line_color="gray")
                        st.plotly_chart(fig_pca, use_container_width=True)
                        if scores.shape[1] >= 3:
                            fig_3d = px.scatter_3d(pca_df, x=scores[:,0], y=scores[:,1], z=scores[:,2], color=pca_group, title="主成分三维得分图", labels={'x': f"PC1 ({exp_var[0]*100:.2f}%)", 'y': f"PC2 ({exp_var[1]*100:.2f}%)", 'z': f"PC3 ({exp_var[2]*100:.2f}%)"})
                            st.plotly_chart(fig_3d, use_container_width=True)

        with tab_rda:
            st.info("分析环境理化因子 (X) 对响应变量 (Y) 群落变异的解释程度。")
            rc1, rc2 = st.columns(2)
            rda_x_cols = rc1.multiselect("环境变量 (X) (如 pH, 有机质)", numeric_cols, default=[c for c in numeric_cols if '有机质' in c or '全氮' in c])
            rda_y_cols = rc2.multiselect("响应变量 (Y) (如 产量, 光合)", numeric_cols, default=[c for c in numeric_cols if '产量' in c or '单果重' in c])
            rda_group = st.selectbox("样点分组依据：", ['处理编号', '试验点'])
            if st.button("运行 RDA 冗余分析", type="primary"):
                if len(rda_x_cols) == 0 or len(rda_y_cols) == 0:
                    st.warning("X 和 Y 均需至少 1 个指标。")
                else:
                    rda_df = stat_df.dropna(subset=rda_x_cols + rda_y_cols + [rda_group])
                    X_scaled = StandardScaler().fit_transform(rda_df[rda_x_cols])
                    Y_scaled = StandardScaler().fit_transform(rda_df[rda_y_cols])
                    B, _, _, _ = np.linalg.lstsq(X_scaled, Y_scaled, rcond=None)
                    Y_fitted = X_scaled @ B
                    pca_rda = PCA(n_components=2)
                    site_scores = pca_rda.fit_transform(Y_fitted)
                    exp_var = pca_rda.explained_variance_ratio_ * 100
                    env_scores = np.corrcoef(X_scaled.T, site_scores.T)[:len(rda_x_cols), len(rda_x_cols):]
                    fig_rda = go.Figure()
                    for grp in rda_df[rda_group].unique():
                        mask = rda_df[rda_group] == grp
                        fig_rda.add_trace(go.Scatter(x=site_scores[mask,0], y=site_scores[mask,1], mode='markers', name=str(grp), marker=dict(size=10, line=dict(width=1, color='black'))))
                    multiplier = np.max(np.abs(site_scores)) * 0.8
                    for i, col in enumerate(rda_x_cols):
                        fig_rda.add_annotation(x=env_scores[i,0]*multiplier, y=env_scores[i,1]*multiplier, text=col, ax=0, ay=0, showarrow=True, arrowhead=3, arrowwidth=2, arrowcolor="blue", font=dict(color="blue", size=12))
                    fig_rda.add_hline(y=0, line_dash="dot", line_color="gray")
                    fig_rda.add_vline(x=0, line_dash="dot", line_color="gray")
                    fig_rda.update_layout(title="RDA 三序图", xaxis_title=f"RDA1 ({exp_var[0]:.2f}%)", yaxis_title=f"RDA2 ({exp_var[1]:.2f}%)", width=800, height=600)
                    st.plotly_chart(fig_rda, use_container_width=False)
                    corr_env_resp = np.corrcoef(X_scaled.T, Y_scaled.T)[:len(rda_x_cols), len(rda_x_cols):]
                    fig_heat = px.imshow(corr_env_resp, x=rda_y_cols, y=rda_x_cols, text_auto=".2f", color_continuous_scale="RdBu_r", title="环境因子与响应变量相关性热力图")
                    st.plotly_chart(fig_heat, use_container_width=True)

        with tab_cluster:
            clust_cols = st.multiselect("参与聚类的特征指标", numeric_cols, default=numeric_cols[:5])
            if st.button("绘制处理间层次聚类树状图", type="primary"):
                if len(clust_cols) < 2:
                    st.warning("至少选择 2 个指标")
                else:
                    clust_df = stat_df.groupby('处理编号')[clust_cols].mean().dropna()
                    if len(clust_df) > 1:
                        Z = linkage(StandardScaler().fit_transform(clust_df), method='ward')
                        fig_clust, ax = plt.subplots(figsize=(8, 5))
                        dendrogram(Z, labels=clust_df.index.tolist(), leaf_rotation=0, color_threshold=0.7*max(Z[:,2]), ax=ax)
                        ax.set_title("不同处理的层次聚类分析 (Ward)", fontsize=16, fontweight='bold')
                        ax.set_ylabel("聚类距离 (Distance)", fontsize=14, fontweight='bold')
                        ax.spines['top'].set_visible(True)
                        ax.spines['right'].set_visible(True)
                        ax.spines['bottom'].set_linewidth(1.5)
                        ax.spines['left'].set_linewidth(1.5)
                        st.pyplot(fig_clust)
                    else:
                        st.error("数据不足。")

        with tab_ml:
            st.markdown("##### 基于随机森林的未来趋势预测")
            st.write("根据历史测定数据（年份、处理、作物等特征），预测未来 3 年的指标走势及不确定性区间，并输出特征重要性。")
            mc1, mc2, mc3 = st.columns(3)
            ml_target = mc1.selectbox("预测目标 (Y)", numeric_cols, index=0)
            ml_site = mc2.selectbox("目标试验点", stat_df['试验点'].unique())
            ml_crop = mc3.selectbox("目标作物", stat_df['当季种植作物'].unique())
            if st.button("训练模型并预测未来三年", type="primary"):
                train_data = df[(df['试验点'] == ml_site) & (df['当季种植作物'] == ml_crop)].dropna(subset=[ml_target])
                if len(train_data) < 20:
                    st.error("该条件下的历史数据样本量过少 (<20)，无法进行可靠的模型训练。")
                else:
                    X = train_data[['测定年份', '处理编号']]
                    y = train_data[ml_target].values
                    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    X_encoded = np.hstack([X[['测定年份']].values, encoder.fit_transform(X[['处理编号']])])
                    feature_names = ['年份'] + [f'处理_{c}' for c in encoder.categories_[0]]
                    rf = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf.fit(X_encoded, y)
                    st.success(f"模型训练完毕！(基于 {len(train_data)} 个样本)")
                    importance = rf.feature_importances_
                    imp_df = pd.DataFrame({'特征': feature_names, '重要性': importance}).sort_values('重要性', ascending=False)
                    fig_imp = px.bar(imp_df, x='特征', y='重要性', title="随机森林特征重要性", labels={'重要性': '贡献度'}, template="academic")
                    st.plotly_chart(fig_imp, use_container_width=True)
                    last_year = train_data['测定年份'].max()
                    future_years = [last_year + 1, last_year + 2, last_year + 3]
                    treatments = train_data['处理编号'].unique()
                    future_records = []
                    for y_f in future_years:
                        for t in treatments:
                            future_records.append([y_f, t])
                    X_fut = pd.DataFrame(future_records, columns=['测定年份', '处理编号'])
                    X_fut_enc = np.hstack([X_fut[['测定年份']].values, encoder.transform(X_fut[['处理编号']])])
                    preds = []
                    for estimator in rf.estimators_:
                        preds.append(estimator.predict(X_fut_enc))
                    preds = np.array(preds)
                    X_fut['预测均值'] = preds.mean(axis=0)
                    X_fut['置信下限'] = X_fut['预测均值'] - preds.std(axis=0) * 1.96
                    X_fut['置信上限'] = X_fut['预测均值'] + preds.std(axis=0) * 1.96
                    hist_agg = train_data.groupby(['测定年份', '处理编号'])[ml_target].mean().reset_index()
                    fig_pred = go.Figure()
                    colors = px.colors.qualitative.D3
                    for i, trt in enumerate(treatments):
                        c = colors[i % len(colors)]
                        hist_t = hist_agg[hist_agg['处理编号'] == trt]
                        fig_pred.add_trace(go.Scatter(x=hist_t['测定年份'], y=hist_t[ml_target], mode='lines+markers', name=f"历史-{trt}", line=dict(color=c, width=2)))
                        fut_t = X_fut[X_fut['处理编号'] == trt]
                        connect_x = [hist_t['测定年份'].iloc[-1]] + fut_t['测定年份'].tolist()
                        connect_y = [hist_t[ml_target].iloc[-1]] + fut_t['预测均值'].tolist()
                        fig_pred.add_trace(go.Scatter(x=connect_x, y=connect_y, mode='lines', line=dict(color=c, dash='dash'), name=f"预测-{trt}"))
                        if i == 0:
                            fig_pred.add_trace(go.Scatter(
                                x=fut_t['测定年份'].tolist() + fut_t['测定年份'].tolist()[::-1],
                                y=fut_t['置信上限'].tolist() + fut_t['置信下限'].tolist()[::-1],
                                fill='toself', fillcolor=c, opacity=0.2, line=dict(color='rgba(255,255,255,0)'),
                                hoverinfo="skip", showlegend=True, name=f"{trt} 95%置信区间"
                            ))
                    fig_pred.update_layout(title=f"{ml_site} {ml_crop} {ml_target} 的历史演变与未来三年预测", xaxis_title="年份", yaxis_title=ml_target)
                    st.plotly_chart(fig_pred, use_container_width=True)

# ==========================================
# 页面 3：数据录入中心
# ==========================================
elif page == "数据录入中心":
    page_title("数据录入中心")
    st.markdown("支持单条实时录入、实验室批量上传，以及组学数据文件上传。")
    entry_tab1, entry_tab2, entry_tab3 = st.tabs(["单条录入 (田间/温室)", "批量上传 (实验室Excel)", "组学数据上传"])
    
    with entry_tab1:
        st.info("提示：未测定的指标留空，系统自动以 NULL 存入数据库。自定义指标将以 JSON 格式存储。")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("添加作物指标", key="add_crop_outside"):
                if 'crop_indicators' not in st.session_state:
                    st.session_state.crop_indicators = []
                st.session_state.crop_indicators.append({"name": "", "value": None})
                st.rerun()
            if st.button("清空所有作物指标", key="clear_crop_outside"):
                st.session_state.crop_indicators = []
                st.rerun()
        with col_btn2:
            if st.button("添加土壤指标", key="add_soil_outside"):
                if 'soil_indicators' not in st.session_state:
                    st.session_state.soil_indicators = []
                st.session_state.soil_indicators.append({"name": "", "value": None})
                st.rerun()
            if st.button("清空所有土壤指标", key="clear_soil_outside"):
                st.session_state.soil_indicators = []
                st.rerun()
        
        with st.form("single_entry_form", clear_on_submit=True):
            st.markdown("#### 1. 确认时空与处理信息")
            c1, c2, c3, c4 = st.columns(4)
            year = c1.number_input("测定年份", value=2024, step=1, format="%d")
            season = c2.selectbox("茬口", ["春茬", "秋冬茬"])
            site = c3.selectbox("试验点", ["北京房山", "北京海淀", "天津武清", "河北赵县", "山东潍坊", "沈阳海城"])
            plant_sys = c4.selectbox("种植制度", ["番茄-黄瓜轮作", "番茄连作"])
            c5, c6, c7 = st.columns(3)
            trt = c5.selectbox("处理编号", [f"T{i}" for i in range(1, 8)])
            rep = c6.radio("重复(区组)", [1, 2, 3], horizontal=True)
            crop = c7.selectbox("当季种植作物", ["番茄", "黄瓜"])
            
            st.markdown("#### 2. 固定指标")
            with st.expander("作物形态与产量指标", expanded=True):
                m1, m2, m3 = st.columns(3)
                yield_v = m1.number_input("总产量(kg/hm2)", value=None)
                fruit_v = m2.number_input("单果重(g)", value=None)
                height_v = m3.number_input("株高(cm)", value=None)
                stem_v = m1.number_input("茎粗(mm)", value=None)
                spad_v = m2.number_input("叶绿素(SPAD)", value=None)
                photo_v = m3.number_input("净光合速率(μmol/m2/s)", value=None)
            with st.expander("作物品质指标"):
                q1, q2 = st.columns(2)
                solids_v = q1.number_input("可溶性固形物(%)", value=None)
                sugar_v = q2.number_input("可溶性糖(%)", value=None)
                acid_v = q1.number_input("可滴定酸(%)", value=None)
                protein_v = q2.number_input("可溶性蛋白(mg/g)", value=None)
            with st.expander("土壤基础理化"):
                s1, s2 = st.columns(2)
                ph_v = s1.number_input("pH", value=None)
                ec_v = s2.number_input("电导率(μS/cm)", value=None)
            with st.expander("土壤养分含量"):
                n1, n2, n3, n4 = st.columns(4)
                som_v = n1.number_input("有机质(g/kg)", value=None)
                tn_v = n2.number_input("全氮(g/kg)", value=None)
                ap_v = n3.number_input("速效磷(mg/kg)", value=None)
                ak_v = n4.number_input("速效钾(mg/kg)", value=None)
            with st.expander("土壤碳氮组分"):
                c1, c2, c3 = st.columns(3)
                doc_v = c1.number_input("溶解性有机碳DOC(mg/kg)", value=None)
                don_v = c2.number_input("溶解性有机氮DON(mg/kg)", value=None)
                mbc_v = c3.number_input("微生物量碳MBC(mg/kg)", value=None)
                mbn_v = c1.number_input("微生物量氮MBN(mg/kg)", value=None)
                poc_v = c2.number_input("颗粒态有机碳POC(g/kg)", value=None)
            with st.expander("土壤微生物与酶活性"):
                e1, e2, e3 = st.columns(3)
                urease_v = e1.number_input("脲酶活性(U/g)", value=None)
                catalase_v = e2.number_input("过氧化氢酶(U/g)", value=None)
                sucrase_v = e3.number_input("蔗糖酶(U/g)", value=None)
                shannon_v = e1.number_input("细菌Shannon指数", value=None, format="%.4f")
                chao1_v = e2.number_input("细菌Chao1指数", value=None)

            st.markdown("#### 3. 自定义作物指标（可添加多个）")
            if 'crop_indicators' not in st.session_state:
                st.session_state.crop_indicators = []
            for i, ind in enumerate(st.session_state.crop_indicators):
                col1, col2 = st.columns([2, 1])
                with col1:
                    new_name = st.text_input(f"指标名称 {i+1}", value=ind.get("name", ""), key=f"crop_name_{i}")
                with col2:
                    new_val = st.number_input(f"值 {i+1}", value=ind.get("value", 0.0), format="%.4f", key=f"crop_val_{i}")
                st.session_state.crop_indicators[i] = {"name": new_name, "value": new_val}
            
            st.markdown("#### 4. 自定义土壤指标（可添加多个）")
            if 'soil_indicators' not in st.session_state:
                st.session_state.soil_indicators = []
            for i, ind in enumerate(st.session_state.soil_indicators):
                col1, col2 = st.columns([2, 1])
                with col1:
                    new_name = st.text_input(f"指标名称 {i+1}", value=ind.get("name", ""), key=f"soil_name_{i}")
                with col2:
                    new_val = st.number_input(f"值 {i+1}", value=ind.get("value", 0.0), format="%.4f", key=f"soil_val_{i}")
                st.session_state.soil_indicators[i] = {"name": new_name, "value": new_val}
            
            submit_btn = st.form_submit_button("确认无误，提交入库", type="primary")
            if submit_btn:
                new_record = {
                    "测定年份": year, "茬口": season, "试验点": site, "种植制度": plant_sys,
                    "处理编号": trt, "重复": rep, "当季种植作物": crop, 
                    "总产量(kg/hm2)": yield_v, "单果重(g)": fruit_v, "株高(cm)": height_v, 
                    "茎粗(mm)": stem_v, "叶绿素(SPAD)": spad_v, "净光合速率(μmol/m2/s)": photo_v, 
                    "可溶性固形物(%)": solids_v, "可溶性糖(%)": sugar_v, "可滴定酸(%)": acid_v, "可溶性蛋白(mg/g)": protein_v, 
                    "pH": ph_v, "电导率(μS/cm)": ec_v,
                    "有机质(g/kg)": som_v, "全氮(g/kg)": tn_v, "速效磷(mg/kg)": ap_v, "速效钾(mg/kg)": ak_v,
                    "溶解性有机碳DOC(mg/kg)": doc_v, "溶解性有机氮DON(mg/kg)": don_v, 
                    "微生物量碳MBC(mg/kg)": mbc_v, "微生物量氮MBN(mg/kg)": mbn_v, "颗粒态有机碳POC(g/kg)": poc_v,
                    "脲酶活性(U/g)": urease_v, "过氧化氢酶(U/g)": catalase_v, "蔗糖酶(U/g)": sucrase_v,
                    "细菌Shannon指数": shannon_v, "细菌Chao1指数": chao1_v
                }
                crop_indicators_json = {ind["name"]: ind["value"] for ind in st.session_state.crop_indicators if ind["name"]}
                soil_indicators_json = {ind["name"]: ind["value"] for ind in st.session_state.soil_indicators if ind["name"]}
                new_record["其他作物指标"] = json.dumps(crop_indicators_json) if crop_indicators_json else None
                new_record["其他土壤指标"] = json.dumps(soil_indicators_json) if soil_indicators_json else None
                insert_record = {k: v for k, v in new_record.items() if v is not None}
                try:
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute("DESCRIBE `长期定位试验总表`")
                    columns_info = cursor.fetchall()
                    table_columns = [col[0] for col in columns_info if col[0] != '数据ID']
                    insert_columns = [col for col in table_columns if col in insert_record]
                    values = [insert_record[col] for col in insert_columns]
                    placeholders = ', '.join(['%s'] * len(insert_columns))
                    insert_sql = f"INSERT INTO `长期定位试验总表` ({', '.join([f'`{col}`' for col in insert_columns])}) VALUES ({placeholders})"
                    cursor.execute(insert_sql, values)
                    conn.commit()
                    cursor.close()
                    conn.close()
                    st.toast(f"{site} {trt}-R{rep} 录入成功！")
                    st.cache_data.clear()
                except Exception as e:
                    st.error(f"录入失败: {e}")

    with entry_tab2:
        st.info("操作指南：在本地 Excel 中排好版，选中数据 Ctrl+C，点击下方表格的左上角空白单元格 Ctrl+V 粘贴。")
        if not df.empty:
            cols = df.columns.tolist()
            if '数据ID' in cols:
                cols.remove('数据ID')
            template_df = pd.DataFrame(columns=cols)
        else:
            template_df = pd.DataFrame(columns=["测定年份", "试验点", "处理编号", "重复", "总产量(kg/hm2)"])
        with st.container():
            edited_df = st.data_editor(template_df, num_rows="dynamic", use_container_width=True, height=400)
        if st.button("批量保存至云端", type="primary"):
            final_df = edited_df.dropna(how='all')
            if final_df.empty:
                st.warning("表格中没有有效数据，请粘贴后再保存。")
            else:
                try:
                    final_df = final_df.replace({np.nan: None})
                    conn = get_connection()
                    cursor = conn.cursor()
                    cursor.execute("DESCRIBE `长期定位试验总表`")
                    table_columns = [col[0] for col in cursor.fetchall() if col[0] != '数据ID']
                    insert_columns = [col for col in final_df.columns if col in table_columns]
                    if not insert_columns:
                        st.error("数据框列名与数据库表列名不匹配，无法插入。")
                    else:
                        insert_sql = f"INSERT INTO `长期定位试验总表` ({', '.join([f'`{col}`' for col in insert_columns])}) VALUES ({', '.join(['%s']*len(insert_columns))})"
                        data_to_insert = [tuple(row[col] for col in insert_columns) for _, row in final_df.iterrows()]
                        cursor.executemany(insert_sql, data_to_insert)
                        conn.commit()
                        cursor.close()
                        conn.close()
                        st.toast("入库完毕！")
                        st.success(f"成功录入 {len(final_df)} 条化验数据，已同步至云端。")
                        st.cache_data.clear()
                except Exception as e:
                    st.error(f"写入冲突或字段类型错误: {e}")

    with entry_tab3:
        st.markdown("##### 组学数据上传")
        st.info("支持上传16S扩增子测序、宏基因组、代谢组学等数据文件，并记录样本元数据。")
        with st.form("omics_form"):
            sample_name = st.text_input("样本名称 (如：BJFS_CK_1_2024)")
            data_type = st.selectbox("数据类型", ["16S rRNA", "宏基因组", "代谢组", "转录组", "其他"])
            description = st.text_area("描述信息 (可选)")
            st.markdown("##### 元数据（可选，如处理、重复等信息）")
            metadata = {}
            cols = st.columns(3)
            with cols[0]:
                meta_key1 = st.text_input("元数据键1", placeholder="处理编号")
                meta_val1 = st.text_input("值1", placeholder="T1")
            with cols[1]:
                meta_key2 = st.text_input("元数据键2", placeholder="重复")
                meta_val2 = st.text_input("值2", placeholder="1")
            with cols[2]:
                meta_key3 = st.text_input("元数据键3", placeholder="其他")
                meta_val3 = st.text_input("值3")
            if meta_key1:
                metadata[meta_key1] = meta_val1
            if meta_key2:
                metadata[meta_key2] = meta_val2
            if meta_key3:
                metadata[meta_key3] = meta_val3
            uploaded_file = st.file_uploader("选择文件", type=["fastq", "fq", "csv", "txt", "xlsx", "tsv"])
            submitted = st.form_submit_button("上传数据")
            if submitted:
                if not sample_name or not uploaded_file:
                    st.error("请填写样本名称并选择文件。")
                else:
                    upload_dir = "uploads/omics"
                    os.makedirs(upload_dir, exist_ok=True)
                    file_ext = uploaded_file.name.split('.')[-1]
                    file_name = f"{sample_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{file_ext}"
                    file_path = os.path.join(upload_dir, file_name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    try:
                        conn = get_connection()
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT INTO `omics_data` (`sample_name`, `data_type`, `file_path`, `description`, `metadata`)
                            VALUES (%s, %s, %s, %s, %s)
                        """, (sample_name, data_type, file_path, description, json.dumps(metadata)))
                        conn.commit()
                        cursor.close()
                        conn.close()
                        st.success(f"文件已上传成功！保存路径：{file_path}")
                    except Exception as e:
                        st.error(f"数据库记录失败: {e}")
        st.markdown("### 已上传的组学数据")
        try:
            conn = get_connection()
            omics_df = pd.read_sql("SELECT * FROM `omics_data` ORDER BY `upload_date` DESC", conn)
            conn.close()
            if not omics_df.empty:
                st.dataframe(omics_df[['id', 'sample_name', 'data_type', 'upload_date', 'description']], use_container_width=True)
                for idx, row in omics_df.iterrows():
                    if os.path.exists(row['file_path']):
                        with open(row['file_path'], "rb") as f:
                            st.download_button(f"下载 {row['sample_name']} 数据", f, file_name=os.path.basename(row['file_path']), key=row['id'])
            else:
                st.info("暂无上传数据。")
        except Exception as e:
            st.error(f"加载组学数据失败: {e}")

# ==========================================
# 页面 4：质量管理与布局
# ==========================================
elif page == "质量管理与布局":
    page_title("底层数据运维、质量预警与田间布局")
    tab_quality, tab_layout, tab_raw = st.tabs(["数据质量预警诊断", "田间网格布局图", "底层数据(删改)"])

    with tab_quality:
        st.markdown("##### 自动异常检测算法 (Isolation Forest) & 缺失值诊断")
        st.write("系统会自动扫描数据库全量数据，揪出可能是手工录入错误的离群点。")
        if st.button("开始全库数据体检", type="primary"):
            if df.empty:
                st.error("无数据")
            else:
                missing = df.isnull().sum()
                missing = missing[missing > 0]
                if not missing.empty:
                    st.warning(f"发现缺失值字段：\n {missing.to_dict()}")
                if numeric_cols:
                    clean_df = df.dropna(subset=numeric_cols)
                    if clean_df.empty:
                        st.warning("用于异常检测的数据不足，请检查数据完整性。")
                    else:
                        clf = IsolationForest(contamination=0.02, random_state=42)
                        preds = clf.fit_predict(clean_df[numeric_cols])
                        outliers = clean_df[preds == -1]
                        if not outliers.empty:
                            st.error(f"警报：系统检测出 {len(outliers)} 条高度疑似录入错误的异常记录！建议核实或在底表中删除。")
                            st.dataframe(outliers[['数据ID', '测定年份', '试验点', '处理编号'] + numeric_cols[:3]], use_container_width=True)
                            st.info("已自动记录日志。如需配置异常数据通过邮件推送给实验室主任，请在后台解开 SMTP 代码注释。")
                        else:
                            st.success("数据质量极佳，未发现明显数值异常。")

    with tab_layout:
        st.markdown("##### 标准化田间小区布局可视化 (模拟随机区组设计 RCBD)")
        st.write("直观查看田间位置效应。可根据处理编号，或特定指标的热力值来给小区染色。")
        lc1, lc2 = st.columns([1, 3])
        with lc1:
            l_site = st.selectbox("选择试验点查看布局", df['试验点'].unique())
            l_year = st.selectbox("选择考察年份", df['测定年份'].unique())
            l_crop = st.selectbox("选择考察作物", df['当季种植作物'].unique())
            l_color = st.selectbox("网格染色依据", ['(按处理分组着色)'] + numeric_cols)
        with lc2:
            layout_df = df[(df['试验点']==l_site) & (df['测定年份']==l_year) & (df['当季种植作物']==l_crop)]
            if layout_df.empty:
                st.warning("该时空条件下无试验数据。")
            else:
                grid_z = []
                grid_text = []
                treats = [f"T{i}" for i in range(1, 8)]
                for rep in [1, 2, 3]:
                    row_z = []
                    row_text = []
                    for trt in treats:
                        cell = layout_df[(layout_df['重复']==rep) & (layout_df['处理编号']==trt)]
                        if not cell.empty:
                            if l_color == '(按处理分组着色)':
                                val = int(trt.replace('T', ''))
                            else:
                                val = float(cell[l_color].values[0])
                            row_z.append(val)
                            row_text.append(f"小区:{trt}-R{rep}<br>值:{val:.2f}" if l_color != '(按处理分组着色)' else f"{trt}-R{rep}")
                        else:
                            row_z.append(np.nan)
                            row_text.append("缺区")
                    grid_z.append(row_z)
                    grid_text.append(row_text)
                fig_grid = go.Figure(data=go.Heatmap(
                    z=grid_z, text=grid_text, texttemplate="%{text}", hoverinfo="text",
                    colorscale="YlGnBu",
                    showscale=l_color != '(按处理分组着色)'
                ))
                fig_grid.update_layout(
                    title=f"<b>{l_site}田间试验布局及位置热力图</b>",
                    xaxis=dict(title="田间列排布 (模拟处理 1-7)", tickvals=list(range(7)), ticktext=treats),
                    yaxis=dict(title="区组 (重复)", tickvals=[0,1,2], ticktext=["Rep 1", "Rep 2", "Rep 3"], autorange="reversed"),
                    width=800, height=400, plot_bgcolor='white'
                )
                st.plotly_chart(fig_grid, use_container_width=True)

    with tab_raw:
        st.markdown("##### 底层数据管理与删除")
        st.dataframe(df, use_container_width=True, height=300)
        del_id = st.number_input("输入要删除的异常数据 ID", min_value=0, step=1, value=0)
        if st.button("永久删除该记录"):
            if del_id > 0:
                conn = get_connection()
                cur = conn.cursor()
                cur.execute("DELETE FROM `长期定位试验总表` WHERE `数据ID` = %s", (del_id,))
                conn.commit()
                if cur.rowcount > 0:
                    st.success("删除成功！")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning("ID不存在")
                conn.close()

# ==========================================
# 页面 5：实时监测
# ==========================================
elif page == "实时监测":
    page_title("实时监测 (IoT 数据采集)")
    st.info("本页面模拟自动传感器数据采集。实际应用中可替换为真实设备数据接口。")
    now = datetime.now()
    times = [now - timedelta(hours=i) for i in range(24, 0, -1)]
    temp_data = 20 + 5 * np.sin(np.linspace(0, 2*np.pi, 24)) + np.random.randn(24) * 1
    hum_data = 60 + 10 * np.cos(np.linspace(0, 2*np.pi, 24)) + np.random.randn(24) * 2
    ec_data = 0.5 + 0.2 * np.sin(np.linspace(0, 2*np.pi, 24)) + np.random.randn(24) * 0.05
    df_sensor = pd.DataFrame({
        "时间": times,
        "土壤温度 (℃)": temp_data,
        "土壤湿度 (%)": hum_data,
        "电导率 (mS/cm)": ec_data
    })
    st.subheader("传感器实时曲线")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_sensor["时间"], y=df_sensor["土壤温度 (℃)"], mode='lines+markers', name="温度"))
    fig.add_trace(go.Scatter(x=df_sensor["时间"], y=df_sensor["土壤湿度 (%)"], mode='lines+markers', name="湿度", yaxis="y2"))
    fig.add_trace(go.Scatter(x=df_sensor["时间"], y=df_sensor["电导率 (mS/cm)"], mode='lines+markers', name="电导率", yaxis="y3"))
    fig.update_layout(
        title="实时监测数据",
        xaxis_title="时间",
        yaxis=dict(title="温度 (℃)", side="left"),
        yaxis2=dict(title="湿度 (%)", overlaying="y", side="right"),
        yaxis3=dict(title="电导率 (mS/cm)", overlaying="y", side="right", anchor="free", position=0.95)
    )
    st.plotly_chart(fig, use_container_width=True)
    if df_sensor["土壤温度 (℃)"].iloc[-1] > 28:
        st.warning("⚠️ 土壤温度偏高，请注意！")
    if df_sensor["土壤湿度 (%)"].iloc[-1] < 50:
        st.warning("⚠️ 土壤湿度偏低，请及时灌溉！")
    st.subheader("最新读数")
    st.dataframe(df_sensor.tail(5).sort_values("时间", ascending=False))
    if st.button("保存当前数据到数据库（模拟）"):
        st.success("数据已模拟写入传感器历史表（实际需要创建对应表）。")

# ==========================================
# 页面 6：系统设置
# ==========================================
elif page == "系统设置":
    page_title("系统设置")
    st.info("系统配置与账户管理")
    st.subheader("账户管理")
    st.write(f"当前登录用户：{st.session_state.username}")
    if st.button("退出登录"):
        st.session_state.clear()
        st.rerun()
    st.subheader("数据库连接测试")
    if st.button("测试数据库连接"):
        try:
            conn = get_connection()
            conn.close()
            st.success("数据库连接正常")
        except Exception as e:
            st.error(f"连接失败: {e}")
    st.subheader("数据备份")
    if st.button("导出数据为 CSV"):
        if not df.empty:
            csv = df.to_csv(index=False)
            st.download_button("下载 CSV", csv, "longterm_data.csv", "text/csv")
        else:
            st.warning("无数据可导出")
    st.subheader("关于")
    st.markdown("长期定位试验数据平台 V3.0")
    st.markdown("© 2026 北京市农林科学院植物营养与资源环境研究所")
