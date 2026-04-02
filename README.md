markdown
# 长期定位试验数据平台

## 项目简介

本平台依托北京市农林科学院植物营养与资源环境研究所主持的农业农村部环渤海设施农业土壤连作障碍治理修复技术模式联网研究项目构建。针对环渤海地区设施农业土壤退化问题，联合多家科研单位，构建了系统性技术路径，积累了多维度长期定位观测数据。平台提供数据可视化、统计分析、数据录入、质量预警、组学数据管理等功能，旨在为土壤障碍演变规律研究、治理技术评估及科学决策提供数据支撑。

## 主要功能

- **数据可视化**：支持指标筛选、箱线图、柱状图、散点图、均值±标准差图、相关性热力图（含显著性标记）
- **空间分析**：试验点地理分布与属性空间插值（GIS）
- **高阶统计**：方差分析（ANOVA）、主成分分析（PCA）、冗余分析（RDA）、聚类分析、随机森林预测
- **数据录入**：单条录入（含自定义指标）、批量上传（Excel）、组学数据上传（16S/宏基因组/代谢组等）
- **质量管理**：异常检测（Isolation Forest）、缺失值诊断、田间布局网格图、底层数据删除
- **实时监测**：模拟IoT传感器数据采集与报警
- **系统设置**：账户管理、数据库连接测试、数据导出

## 技术栈

- 前端/后端：Streamlit
- 数据库：MySQL
- 数据分析：Pandas, NumPy, SciPy, Scikit-learn, Statsmodels
- 可视化：Plotly, Matplotlib
- 部署：Streamlit Cloud / 自建服务器

## 安装与运行

### 1. 克隆仓库

```bash
git clone https://github.com/qyh2/MySQL.git
cd MySQL
2. 安装依赖
bash
pip install -r requirements.txt
3. 配置数据库
创建 MySQL 数据库 agridata_chinese。

修改 app.py 中的数据库连接配置，或使用 .streamlit/secrets.toml。

4. 配置用户登录
在 .streamlit/secrets.toml 中设置用户名和密码。

5. 运行应用
bash
streamlit run app.py
访问 http://localhost:8501

部署到 Streamlit Cloud
将代码推送到 GitHub 仓库。

在 Streamlit Cloud 上，点击 New app，选择该仓库。

在 Advanced settings 中，将 .streamlit/secrets.toml 的内容粘贴到 Secrets 文本框。

点击 Deploy。

数据表结构
主要表：长期定位试验总表（观测数据）、omics_data（组学数据）。具体字段请参考 app.py 中的 CREATE TABLE 语句。

注意事项
首次运行会自动创建 omics_data 表，但 长期定位试验总表 需手动创建或通过已有数据导入。

组学文件上传后存储在 uploads/omics 目录，部署到云平台后建议改用云存储。

实时监测为模拟数据，实际接入IoT设备需修改数据采集接口。

贡献者
项目负责人：魏丹（北京市农林科学院）

开发团队：李艳、金梁、丁建莉 等
