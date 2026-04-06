import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Configuración de página
st.set_page_config(
    page_title="Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .kpi-card:hover {
        transform: translateY(-5px);
    }
    .kpi-value {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .kpi-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .kpi-change {
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        font-size: 0.9rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Cargar datos del CSV"""
    try:
        # Intentar cargar desde el archivo local primero
        if os.path.exists('sales_data_sample.csv'):
            df = pd.read_csv('sales_data_sample.csv', encoding='latin-1')
        else:
            # Fallback: crear datos de ejemplo si no existe el archivo
            st.warning("Archivo CSV no encontrado. Usando datos de ejemplo.")
            return create_sample_data()
        
        # Limpiar y procesar datos
        df.columns = df.columns.str.strip()
        
        # Convertir fechas
        df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], format='mixed', dayfirst=True)
        df['YEAR'] = df['ORDERDATE'].dt.year
        df['MONTH'] = df['ORDERDATE'].dt.month
        df['MONTH_NAME'] = df['ORDERDATE'].dt.month_name()
        df['QTR'] = 'Q' + df['QTR_ID'].astype(str)
        
        # Calcular métricas adicionales
        df['TOTAL_ORDER_VALUE'] = df['QUANTITYORDERED'] * df['PRICEEACH']
        df['DISCOUNT'] = 1 - (df['PRICEEACH'] / df['MSRP'])
        df['DISCOUNT'] = df['DISCOUNT'].apply(lambda x: max(0, x))
        
        return df
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return create_sample_data()

def create_sample_data():
    """Crear datos de ejemplo si no hay CSV"""
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range(start='2003-01-01', end='2005-12-31', periods=n)
    
    return pd.DataFrame({
        'ORDERNUMBER': range(10100, 10100 + n),
        'QUANTITYORDERED': np.random.randint(10, 100, n),
        'PRICEEACH': np.random.uniform(30, 120, n),
        'SALES': np.random.uniform(1000, 10000, n),
        'ORDERDATE': dates,
        'STATUS': np.random.choice(['Shipped', 'Resolved', 'Cancelled', 'On Hold'], n, p=[0.7, 0.1, 0.1, 0.1]),
        'QTR_ID': np.random.choice([1, 2, 3, 4], n),
        'MONTH_ID': np.random.randint(1, 13, n),
        'YEAR_ID': np.random.choice([2003, 2004, 2005], n),
        'PRODUCTLINE': np.random.choice(['Classic Cars', 'Motorcycles', 'Trucks and Buses', 
                                        'Vintage Cars', 'Planes', 'Ships', 'Trains'], n),
        'MSRP': np.random.uniform(50, 200, n),
        'CITY': np.random.choice(['NYC', 'Paris', 'London', 'Tokyo', 'Sydney'], n),
        'COUNTRY': np.random.choice(['USA', 'France', 'UK', 'Japan', 'Australia'], n),
        'TERRITORY': np.random.choice(['NA', 'EMEA', 'APAC', 'Japan'], n),
        'DEALSIZE': np.random.choice(['Small', 'Medium', 'Large'], n, p=[0.4, 0.4, 0.2]),
        'YEAR': dates.year,
        'MONTH': dates.month,
        'MONTH_NAME': dates.month_name(),
        'QTR': 'Q' + np.random.choice([1, 2, 3, 4], n).astype(str),
        'TOTAL_ORDER_VALUE': np.random.uniform(1000, 10000, n),
        'DISCOUNT': np.random.uniform(0, 0.3, n)
    })

def create_kpi_cards(df, df_filtered):
    """Crear tarjetas KPI"""
    st.markdown('<div class="section-title">📈 Indicadores Clave de Rendimiento (KPIs)</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Total Sales
    total_sales = df_filtered['SALES'].sum()
    total_sales_prev = df[(df['ORDERDATE'] < df_filtered['ORDERDATE'].min()) & 
                         (df['ORDERDATE'] >= df_filtered['ORDERDATE'].min() - pd.Timedelta(days=365))]['SALES'].sum()
    sales_change = ((total_sales - total_sales_prev) / total_sales_prev * 100) if total_sales_prev > 0 else 0
    
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">💰 Ventas Totales</div>
            <div class="kpi-value">${total_sales:,.0f}</div>
            <div class="kpi-change">{'📈' if sales_change > 0 else '📉'} {abs(sales_change):.1f}% vs período anterior</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Total Orders
    total_orders = df_filtered['ORDERNUMBER'].nunique()
    total_orders_prev = df[(df['ORDERDATE'] < df_filtered['ORDERDATE'].min()) & 
                          (df['ORDERDATE'] >= df_filtered['ORDERDATE'].min() - pd.Timedelta(days=365))]['ORDERNUMBER'].nunique()
    orders_change = ((total_orders - total_orders_prev) / total_orders_prev * 100) if total_orders_prev > 0 else 0
    
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📦 Pedidos Totales</div>
            <div class="kpi-value">{total_orders:,}</div>
            <div class="kpi-change">{'📈' if orders_change > 0 else '📉'} {abs(orders_change):.1f}% vs período anterior</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Average Order Value
    avg_order = df_filtered['SALES'].mean()
    
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🎯 Valor Promedio</div>
            <div class="kpi-value">${avg_order:,.2f}</div>
            <div class="kpi-change">Por pedido</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Top Product Line
    top_product = df_filtered.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
    top_product_sales = df_filtered.groupby('PRODUCTLINE')['SALES'].sum().max()
    
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🏆 Top Producto</div>
            <div class="kpi-value" style="font-size: 1.8rem;">{top_product}</div>
            <div class="kpi-change">${top_product_sales:,.0f} en ventas</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Success Rate
    success_rate = (df_filtered[df_filtered['STATUS'] == 'Shipped'].shape[0] / df_filtered.shape[0]) * 100
    
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">✅ Tasa de Éxito</div>
            <div class="kpi-value">{success_rate:.1f}%</div>
            <div class="kpi-change">Pedidos entregados</div>
        </div>
        """, unsafe_allow_html=True)

def create_filters(df):
    """Crear filtros en la sidebar"""
    st.sidebar.markdown("## 🔍 Filtros")
    
    # Filtro por año
    years = sorted(df['YEAR_ID'].dropna().unique().tolist()) if 'YEAR_ID' in df.columns else sorted(df['YEAR'].dropna().unique().tolist())
    year_filter = st.sidebar.multiselect("Año", years, default=years)
    
    # Filtro por país
    countries = sorted(df['COUNTRY'].dropna().unique().tolist())
    country_filter = st.sidebar.multiselect("País", countries, default=countries)
    
    # Filtro por línea de producto
    product_lines = sorted(df['PRODUCTLINE'].dropna().unique().tolist())
    product_filter = st.sidebar.multiselect("Línea de Producto", product_lines, default=product_lines)
    
    # Filtro por estado
    statuses = sorted(df['STATUS'].dropna().unique().tolist())
    status_filter = st.sidebar.multiselect("Estado del Pedido", statuses, default=statuses)
    
    # Filtro por tamaño de trato
    deal_sizes = sorted(df['DEALSIZE'].dropna().unique().tolist())
    deal_filter = st.sidebar.multiselect("Tamaño del Trato", deal_sizes, default=deal_sizes)
    
    # Filtro por territorio
    territories = sorted(df['TERRITORY'].dropna().unique().tolist())
    territory_filter = st.sidebar.multiselect("Territorio", territories, default=territories)
    
    # Rango de ventas
    min_sales, max_sales = int(df['SALES'].min()), int(df['SALES'].max())
    sales_range = st.sidebar.slider("Rango de Ventas ($)", min_sales, max_sales, (min_sales, max_sales))
    
    return year_filter, country_filter, product_filter, status_filter, deal_filter, territory_filter, sales_range

def filter_data(df, year_filter, country_filter, product_filter, status_filter, deal_filter, territory_filter, sales_range):
    """Aplicar filtros al dataframe"""
    if 'YEAR_ID' in df.columns:
        df_filtered = df[df['YEAR_ID'].isin(year_filter)]
    else:
        df_filtered = df[df['YEAR'].isin(year_filter)]
    
    df_filtered = df_filtered[
        (df_filtered['COUNTRY'].isin(country_filter)) &
        (df_filtered['PRODUCTLINE'].isin(product_filter)) &
        (df_filtered['STATUS'].isin(status_filter)) &
        (df_filtered['DEALSIZE'].isin(deal_filter)) &
        (df_filtered['TERRITORY'].isin(territory_filter)) &
        (df_filtered['SALES'].between(sales_range[0], sales_range[1]))
    ]
    
    return df_filtered

def create_univariate_analysis(df):
    """Análisis descriptivo univariado"""
    st.markdown('<div class="section-title">📊 Análisis Univariado</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Distribución de Ventas", "Variables Categóricas", "Estadísticas Descriptivas"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma de ventas
            fig = px.histogram(df, x='SALES', nbins=50, 
                              title='Distribución de Ventas',
                              color_discrete_sequence=['#667eea'])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Histograma de cantidad
            fig = px.histogram(df, x='QUANTITYORDERED', nbins=50,
                              title='Distribución de Cantidad Pedida',
                              color_discrete_sequence=['#764ba2'])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Boxplot de ventas por tamaño de trato
            fig = px.box(df, x='DEALSIZE', y='SALES', 
                        title='Ventas por Tamaño de Trato',
                        color='DEALSIZE',
                        color_discrete_map={'Small': '#FF6B6B', 'Medium': '#4ECDC4', 'Large': '#45B7D1'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Distribución de precios
            fig = px.histogram(df, x='PRICEEACH', nbins=50,
                              title='Distribución de Precios por Unidad',
                              color_discrete_sequence=['#96CEB4'])
            fig.update_layout(bargap=0.1)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de barras de países
            country_counts = df['COUNTRY'].value_counts().head(15)
            fig = px.bar(x=country_counts.index, y=country_counts.values,
                        title='Top 15 Países por Número de Pedidos',
                        labels={'x': 'País', 'y': 'Número de Pedidos'},
                        color=country_counts.values,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Gráfico de líneas de producto
            product_counts = df['PRODUCTLINE'].value_counts()
            fig = px.pie(values=product_counts.values, names=product_counts.index,
                        title='Distribución por Línea de Producto',
                        hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Estado de pedidos
            status_counts = df['STATUS'].value_counts()
            fig = px.bar(x=status_counts.index, y=status_counts.values,
                        title='Distribución de Estados de Pedido',
                        color=status_counts.values,
                        color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Territorios
            territory_counts = df['TERRITORY'].value_counts()
            fig = px.pie(values=territory_counts.values, names=territory_counts.index,
                        title='Distribución por Territorio',
                        hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### Estadísticas Descriptivas Numéricas")
        numeric_cols = ['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        if numeric_cols:
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            st.markdown("### Correlación entre Variables Numéricas")
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           color_continuous_scale='RdBu_r',
                           title='Matriz de Correlación')
            st.plotly_chart(fig, use_container_width=True)

def create_bivariate_analysis(df):
    """Análisis descriptivo bivariado"""
    st.markdown('<div class="section-title">🔗 Análisis Bivariado</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Ventas vs Tiempo", "Ventas vs Categorías", "Scatter Plots", "Análisis por País"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Ventas por año
            sales_by_year = df.groupby('YEAR')['SALES'].sum().reset_index()
            fig = px.bar(sales_by_year, x='YEAR', y='SALES',
                        title='Ventas Totales por Año',
                        color='SALES',
                        color_continuous_scale='YlOrRd')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Tendencia mensual
            sales_by_month = df.groupby(['YEAR', 'MONTH'])['SALES'].sum().reset_index()
            sales_by_month['Date'] = pd.to_datetime(sales_by_month['YEAR'].astype(str) + '-' + sales_by_month['MONTH'].astype(str) + '-01')
            sales_by_month = sales_by_month.sort_values('Date')
            
            fig = px.line(sales_by_month, x='Date', y='SALES',
                         title='Tendencia de Ventas Mensuales',
                         markers=True)
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Ventas por trimestre
            sales_by_qtr = df.groupby('QTR')['SALES'].sum().reset_index()
            fig = px.bar(sales_by_qtr, x='QTR', y='SALES',
                        title='Ventas por Trimestre',
                        color='SALES',
                        color_continuous_scale='Greens')
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Heatmap de ventas por mes y año
            if 'YEAR' in df.columns and 'MONTH' in df.columns:
                pivot_table = df.pivot_table(values='SALES', index='YEAR', columns='MONTH', aggfunc='sum')
                fig = px.imshow(pivot_table, text_auto='.0f', aspect="auto",
                               color_continuous_scale='YlOrRd',
                               title='Heatmap de Ventas por Mes y Año')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Ventas por línea de producto
            sales_by_product = df.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=True)
            fig = px.bar(x=sales_by_product.values, y=sales_by_product.index,
                        orientation='h',
                        title='Ventas por Línea de Producto',
                        color=sales_by_product.values,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Ventas por tamaño de trato
            sales_by_deal = df.groupby('DEALSIZE')['SALES'].agg(['sum', 'count', 'mean']).reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=sales_by_deal['DEALSIZE'], y=sales_by_deal['sum'],
                                name='Ventas Totales', marker_color='#667eea'), secondary_y=False)
            fig.add_trace(go.Scatter(x=sales_by_deal['DEALSIZE'], y=sales_by_deal['mean'],
                                    mode='lines+markers', name='Ventas Promedio',
                                    line=dict(color='#FF6B6B', width=3)), secondary_y=True)
            fig.update_layout(title='Ventas y Promedio por Tamaño de Trato')
            st.plotly_chart(fig, use_container_width=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            # Ventas por territorio
            sales_by_territory = df.groupby('TERRITORY')['SALES'].sum().sort_values(ascending=False)
            fig = px.pie(values=sales_by_territory.values, names=sales_by_territory.index,
                        title='Distribución de Ventas por Territorio',
                        hole=0.4)
            st.plotly_chart(fig, use_container_width=True)
        
        with col4:
            # Ventas por estado
            sales_by_status = df.groupby('STATUS')['SALES'].sum()
            fig = px.bar(x=sales_by_status.index, y=sales_by_status.values,
                        title='Ventas por Estado del Pedido',
                        color=sales_by_status.values,
                        color_continuous_scale='Tealgrn')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter: Cantidad vs Ventas
            fig = px.scatter(df, x='QUANTITYORDERED', y='SALES',
                           title='Cantidad Pedida vs Ventas',
                           color='DEALSIZE',
                           size='PRICEEACH',
                           hover_data=['PRODUCTLINE', 'COUNTRY'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Scatter: Precio vs Ventas
            fig = px.scatter(df, x='PRICEEACH', y='SALES',
                           title='Precio por Unidad vs Ventas',
                           color='PRODUCTLINE',
                           size='QUANTITYORDERED',
                           hover_data=['STATUS'])
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Top países por ventas
        top_countries = df.groupby('COUNTRY').agg({
            'SALES': 'sum',
            'ORDERNUMBER': 'nunique',
            'QUANTITYORDERED': 'sum'
        }).sort_values('SALES', ascending=False).head(10).reset_index()
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Ventas Totales', 'Número de Pedidos'))
        fig.add_trace(go.Bar(x=top_countries['COUNTRY'], y=top_countries['SALES'],
                            name='Ventas', marker_color='#667eea'), row=1, col=1)
        fig.add_trace(go.Bar(x=top_countries['COUNTRY'], y=top_countries['ORDERNUMBER'],
                            name='Pedidos', marker_color='#764ba2'), row=1, col=2)
        fig.update_layout(title='Top 10 Países por Ventas y Pedidos', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

def create_predictive_analysis(df):
    """Análisis predictivo"""
    st.markdown('<div class="section-title">🔮 Análisis Predictivo</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Predicción de Ventas", "Modelo de Clasificación", "Análisis de Tendencias"])
    
    with tab1:
        st.markdown("### Predicción de Ventas por Tiempo")
        
        # Preparar datos para series temporales
        df_time = df.groupby(df['ORDERDATE'].dt.to_period('M'))['SALES'].sum().reset_index()
        df_time['ORDERDATE'] = df_time['ORDERDATE'].dt.to_timestamp()
        df_time = df_time.sort_values('ORDERDATE')
        
        # Crear variables para el modelo
        df_time['Month_Num'] = (df_time['ORDERDATE'] - df_time['ORDERDATE'].min()).dt.days / 30
        df_time['Month_Num'] = df_time['Month_Num'].round()
        
        X = df_time[['Month_Num']].values
        y = df_time['SALES'].values
        
        # Entrenar modelo
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"${rmse:,.2f}")
        col2.metric("R² Score", f"{r2:.3f}")
        col3.metric("Coeficiente", f"{model.coef_[0]:,.2f}")
        
        # Gráfico de predicción
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_time['ORDERDATE'], y=df_time['SALES'],
                                mode='lines+markers', name='Ventas Reales',
                                line=dict(color='#667eea', width=3)))
        
        # Línea de tendencia
        x_range = np.linspace(df_time['Month_Num'].min(), df_time['Month_Num'].max(), 100)
        y_pred_line = model.predict(x_range.reshape(-1, 1))
        dates_range = pd.date_range(df_time['ORDERDATE'].min(), df_time['ORDERDATE'].max(), periods=100)
        
        fig.add_trace(go.Scatter(x=dates_range, y=y_pred_line,
                                mode='lines', name='Tendencia Predictiva',
                                line=dict(color='#FF6B6B', width=3, dash='dash')))
        
        # Predicción futura
        future_months = np.arange(df_time['Month_Num'].max() + 1, df_time['Month_Num'].max() + 7).reshape(-1, 1)
        future_sales = model.predict(future_months)
        future_dates = pd.date_range(df_time['ORDERDATE'].max() + pd.DateOffset(months=1), periods=6, freq='M')
        
        fig.add_trace(go.Scatter(x=future_dates, y=future_sales,
                                mode='lines+markers', name='Predicción Futura (6 meses)',
                                line=dict(color='#4ECDC4', width=3, dash='dot'),
                                marker=dict(size=10)))
        
        fig.update_layout(title='Predicción de Ventas con Regresión Lineal',
                         xaxis_title='Fecha', yaxis_title='Ventas ($)',
                         hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"💡 **Insight**: El modelo predice un incremento/decremento de ${model.coef_[0]:,.2f} en ventas por mes.")
    
    with tab2:
        st.markdown("### Predicción de Tamaño de Trato")
        
        # Preparar datos para clasificación
        df_model = df[['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'DEALSIZE']].dropna()
        
        if 'DISCOUNT' in df.columns:
            df_model['DISCOUNT'] = df['DISCOUNT']
        
        # Encode target
        le = LabelEncoder()
        df_model['DEALSIZE_ENCODED'] = le.fit_transform(df_model['DEALSIZE'])
        
        X = df_model[['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MSRP']].values
        y = df_model['DEALSIZE_ENCODED'].values
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Modelo de regresión logística (clasificación)
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        
        st.metric("Precisión del Modelo", f"{accuracy:.2%}")
        
        # Importancia de características
        feature_importance = pd.DataFrame({
            'Característica': ['Ventas', 'Cantidad', 'Precio', 'MSRP'],
            'Coeficiente': np.abs(clf.coef_[0])
        }).sort_values('Coeficiente', ascending=True)
        
        fig = px.bar(feature_importance, x='Coeficiente', y='Característica',
                    orientation='h',
                    title='Importancia de Características en la Predicción',
                    color='Coeficiente',
                    color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
        # Demo interactiva
        st.markdown("### 🎮 Demo Interactiva: Predice el Tamaño del Trato")
        
        col1, col2 = st.columns(2)
        with col1:
            demo_sales = st.number_input("Ventas ($)", min_value=0, max_value=20000, value=5000)
            demo_quantity = st.number_input("Cantidad", min_value=1, max_value=100, value=30)
        
        with col2:
            demo_price = st.number_input("Precio por Unidad ($)", min_value=0, max_value=200, value=80)
            demo_msrp = st.number_input("MSRP ($)", min_value=0, max_value=300, value=100)
        
        if st.button("Predecir Tamaño de Trato"):
            demo_input = np.array([[demo_sales, demo_quantity, demo_price, demo_msrp]])
            prediction = clf.predict(demo_input)
            predicted_label = le.inverse_transform(prediction)[0]
            
            st.success(f"🎯 **Predicción**: El tamaño del trato es **{predicted_label}**")
    
    with tab3:
        st.markdown("### 📊 Análisis de Tendencias y Estacionalidad")
        
        # Ventas por mes y línea de producto
        monthly_product = df.groupby([df['ORDERDATE'].dt.to_period('M'), 'PRODUCTLINE'])['SALES'].sum().unstack()
        monthly_product.index = monthly_product.index.to_timestamp()
        monthly_product = monthly_product.fillna(0)
        
        fig = go.Figure()
        for product in monthly_product.columns:
            fig.add_trace(go.Scatter(x=monthly_product.index, y=monthly_product[product],
                                    mode='lines', name=product))
        
        fig.update_layout(title='Evolución de Ventas por Línea de Producto',
                         xaxis_title='Fecha', yaxis_title='Ventas ($)',
                         hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de estacionalidad
        st.markdown("### 🗓️ Patrones Estacionales")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Ventas por día de la semana
            df_copy = df.copy()
            df_copy['DAY_OF_WEEK'] = df_copy['ORDERDATE'].dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_sales = df_copy.groupby('DAY_OF_WEEK')['SALES'].mean().reindex(day_order)
            
            fig = px.bar(x=day_sales.index, y=day_sales.values,
                        title='Ventas Promedio por Día de la Semana',
                        color=day_sales.values,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Ventas por mes (promedio)
            month_sales = df.groupby('MONTH_NAME')['SALES'].mean()
            month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']
            month_sales = month_sales.reindex(month_order)
            
            fig = px.bar(x=month_sales.index, y=month_sales.values,
                        title='Ventas Promedio por Mes',
                        color=month_sales.values,
                        color_continuous_scale='YlOrRd')
            st.plotly_chart(fig, use_container_width=True)

def main():
    """Función principal"""
    # Header
    st.markdown('<h1 class="main-header">📊 Sales Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Análisis descriptivo y predictivo de datos de ventas con inteligencia de negocios</p>', unsafe_allow_html=True)
    
    # Cargar datos
    with st.spinner('Cargando datos...'):
        df = load_data()
    
    if df.empty:
        st.error("No se pudieron cargar los datos.")
        return
    
    # Sidebar con filtros
    year_filter, country_filter, product_filter, status_filter, deal_filter, territory_filter, sales_range = create_filters(df)
    
    # Aplicar filtros
    df_filtered = filter_data(df, year_filter, country_filter, product_filter, status_filter, deal_filter, territory_filter, sales_range)
    
    if df_filtered.empty:
        st.warning("No hay datos que coincidan con los filtros seleccionados.")
        return
    
    # Mostrar KPIs
    create_kpi_cards(df, df_filtered)
    
    # Tabs para diferentes análisis
    tab_uni, tab_bi, tab_pred = st.tabs(["📊 Análisis Univariado", "🔗 Análisis Bivariado", "🔮 Análisis Predictivo"])
    
    with tab_uni:
        create_univariate_analysis(df_filtered)
    
    with tab_bi:
        create_bivariate_analysis(df_filtered)
    
    with tab_pred:
        create_predictive_analysis(df_filtered)
    
    # Tabla de datos
    st.markdown('<div class="section-title">📋 Datos Filtrados</div>', unsafe_allow_html=True)
    st.dataframe(df_filtered.head(100), use_container_width=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>📊 Sales Analytics Dashboard | Desarrollado con Streamlit y Plotly</p>
        <p>© 2024 - Análisis de Ventas</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
