import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io

# Page config
st.set_page_config(page_title="Profitability & Pricing Optimizer", layout="wide", page_icon="💰")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'elasticity_tests' not in st.session_state:
    st.session_state.elasticity_tests = {}

# Helper functions
def generate_sample_data():
    """Generate sample transaction data"""
    np.random.seed(42)
    n_records = 500
    
    products = ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones', 'Webcam', 'USB Cable', 'Desk Lamp']
    categories = ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Audio', 'Video', 'Accessories', 'Furniture']
    
    data = {
        'order_id': [f'ORD{i:05d}' for i in range(1, n_records + 1)],
        'product': np.random.choice(products, n_records),
        'category': [categories[products.index(p)] for p in np.random.choice(products, n_records)],
        'quantity': np.random.randint(1, 5, n_records),
        'unit_price': np.random.uniform(10, 1000, n_records).round(2),
        'slash_price': np.random.uniform(12, 1200, n_records).round(2),  # Original/MSRP price
        'cost_per_unit': np.random.uniform(5, 700, n_records).round(2),
        'discount_pct': np.random.choice([0, 5, 10, 15, 20], n_records, p=[0.4, 0.2, 0.2, 0.15, 0.05]),
        'shipping_cost': np.random.uniform(0, 20, n_records).round(2),
        'date': pd.date_range(start='2024-01-01', periods=n_records, freq='D')[:n_records]
    }
    
    df = pd.DataFrame(data)
    # Ensure slash price is higher than unit price
    df['slash_price'] = df.apply(lambda x: max(x['slash_price'], x['unit_price'] * 1.1), axis=1)
    df['cost_per_unit'] = df.apply(lambda x: min(x['cost_per_unit'], x['unit_price'] * 0.8), axis=1)
    return df

def calculate_profitability(df):
    """Calculate profitability metrics"""
    df = df.copy()
    
    # Calculate revenue and costs
    df['gross_revenue'] = df['quantity'] * df['unit_price']
    df['discount_amount'] = df['gross_revenue'] * (df['discount_pct'] / 100)
    df['net_revenue'] = df['gross_revenue'] - df['discount_amount']
    df['total_cost'] = (df['quantity'] * df['cost_per_unit']) + df['shipping_cost']
    df['profit'] = df['net_revenue'] - df['total_cost']
    df['profit_margin_pct'] = (df['profit'] / df['net_revenue'] * 100).round(2)
    
    # Calculate slash price discount
    df['slash_discount_pct'] = ((df['slash_price'] - df['unit_price']) / df['slash_price'] * 100).round(2)
    df['perceived_savings'] = df['slash_price'] - df['unit_price']
    
    return df

def classify_product_velocity(df):
    """Classify products by sales velocity (fast/mid/slow movers)"""
    product_sales = df.groupby('product').agg({
        'quantity': 'sum',
        'order_id': 'count'
    }).reset_index()
    
    # Calculate velocity score (units * frequency)
    product_sales['velocity_score'] = product_sales['quantity'] * product_sales['order_id']
    
    # Classify into terciles
    terciles = product_sales['velocity_score'].quantile([0.33, 0.67])
    
    def classify(score):
        if score <= terciles.iloc[0]:
            return 'Slow Mover'
        elif score <= terciles.iloc[1]:
            return 'Mid Mover'
        else:
            return 'Fast Mover'
    
    product_sales['velocity_class'] = product_sales['velocity_score'].apply(classify)
    
    return product_sales[['product', 'velocity_class', 'velocity_score']]

def estimate_elasticity_from_data(df, product_name):
    """Estimate price elasticity from historical data"""
    product_df = df[df['product'] == product_name].copy()
    
    if len(product_df) < 10:
        return None, "Insufficient data"
    
    # Group by similar price points
    product_df['price_bin'] = pd.cut(product_df['unit_price'], bins=5)
    
    aggregated = product_df.groupby('price_bin').agg({
        'unit_price': 'mean',
        'quantity': 'sum'
    }).reset_index().dropna()
    
    if len(aggregated) < 3:
        return None, "Insufficient price variation"
    
    # Simple elasticity calculation: % change in quantity / % change in price
    aggregated = aggregated.sort_values('unit_price')
    
    elasticities = []
    for i in range(1, len(aggregated)):
        price_change_pct = (aggregated.iloc[i]['unit_price'] - aggregated.iloc[i-1]['unit_price']) / aggregated.iloc[i-1]['unit_price']
        qty_change_pct = (aggregated.iloc[i]['quantity'] - aggregated.iloc[i-1]['quantity']) / aggregated.iloc[i-1]['quantity']
        
        if price_change_pct != 0:
            elasticity = qty_change_pct / price_change_pct
            elasticities.append(elasticity)
    
    if elasticities:
        avg_elasticity = np.mean(elasticities)
        return avg_elasticity, "Estimated from data"
    
    return None, "Could not calculate"

def simulate_price_change(df, price_change_pct, elasticity=-1.5):
    """Simulate impact of price changes with demand elasticity"""
    df_sim = df.copy()
    
    # Calculate new prices
    df_sim['new_unit_price'] = df_sim['unit_price'] * (1 + price_change_pct / 100)
    
    # Apply demand elasticity
    demand_change_pct = elasticity * price_change_pct
    df_sim['new_quantity'] = (df_sim['quantity'] * (1 + demand_change_pct / 100)).clip(lower=0)
    
    # Calculate new metrics
    df_sim['new_gross_revenue'] = df_sim['new_quantity'] * df_sim['new_unit_price']
    df_sim['new_discount_amount'] = df_sim['new_gross_revenue'] * (df_sim['discount_pct'] / 100)
    df_sim['new_net_revenue'] = df_sim['new_gross_revenue'] - df_sim['new_discount_amount']
    df_sim['new_total_cost'] = (df_sim['new_quantity'] * df_sim['cost_per_unit']) + df_sim['shipping_cost']
    df_sim['new_profit'] = df_sim['new_net_revenue'] - df_sim['new_total_cost']
    df_sim['new_profit_margin_pct'] = (df_sim['new_profit'] / df_sim['new_net_revenue'] * 100).round(2)
    
    # Calculate changes
    df_sim['revenue_change'] = df_sim['new_net_revenue'] - df_sim['net_revenue']
    df_sim['profit_change'] = df_sim['new_profit'] - df_sim['profit']
    df_sim['margin_change'] = df_sim['new_profit_margin_pct'] - df_sim['profit_margin_pct']
    
    return df_sim

def generate_pricing_recommendation(elasticity_abs, current_margin, velocity_class):
    """Generate smart pricing recommendation based on elasticity and context"""
    recommendations = []
    
    # Elasticity-based rules
    if elasticity_abs > 1.5:
        recommendations.append({
            'type': 'warning',
            'title': '⚠️ High Price Sensitivity Detected',
            'message': f'Elasticity: {-elasticity_abs:.2f} - Avoid deep permanent discounts',
            'action': 'Use targeted promotions (limited time, specific segments) instead of across-the-board price cuts'
        })
    else:
        recommendations.append({
            'type': 'info',
            'title': '✅ Low Price Sensitivity',
            'message': f'Elasticity: {-elasticity_abs:.2f} - Price increases are relatively safe',
            'action': 'Consider modest price increases (5-10%) to improve margins'
        })
    
    # Velocity-based recommendations
    if velocity_class == 'Fast Mover':
        if current_margin < 25:
            recommendations.append({
                'type': 'success',
                'title': '🚀 Fast Mover + Low Margin = Opportunity',
                'message': 'High volume can absorb small price increases',
                'action': 'Test 5-10% price increase on this popular item'
            })
    elif velocity_class == 'Slow Mover':
        if current_margin > 40:
            recommendations.append({
                'type': 'info',
                'title': '🐌 Slow Mover + High Margin = Volume Opportunity',
                'message': 'Price may be limiting sales',
                'action': 'Test 10-15% price decrease to stimulate demand'
            })
    
    return recommendations

# Header
st.markdown('<div class="main-header">💰 Profitability & Pricing Optimizer</div>', unsafe_allow_html=True)
st.markdown("**Analyze profitability and run elasticity-based pricing experiments by product velocity**")

# Sidebar
with st.sidebar:
    st.header("📁 Data Upload")
    
    upload_option = st.radio("Choose data source:", ["Upload CSV", "Use Sample Data"])
    
    if upload_option == "Upload CSV":
        uploaded_file = st.file_uploader("Upload your transaction data (CSV)", type=['csv'])
        
        with st.expander("📋 Required Columns"):
            st.markdown("""
            Your CSV should include:
            - `order_id` - Unique order identifier
            - `product` - Product name
            - `category` - Product category
            - `quantity` - Units sold
            - `unit_price` - Actual selling price
            - `slash_price` - Original/MSRP price (for discount display)
            - `cost_per_unit` - Cost per unit
            - `discount_pct` - Discount percentage
            - `shipping_cost` - Shipping cost per order
            - `date` - Transaction date (optional)
            """)
        
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.success(f"✅ Loaded {len(df)} records")
    else:
        if st.button("Generate Sample Data", type="primary"):
            df = generate_sample_data()
            st.session_state.df = df
            st.success(f"✅ Generated {len(df)} sample records")

# Main content
if st.session_state.df is not None:
    df = st.session_state.df
    
    # Calculate profitability
    df_analyzed = calculate_profitability(df)
    
    # Classify product velocity
    velocity_df = classify_product_velocity(df_analyzed)
    df_analyzed = df_analyzed.merge(velocity_df, on='product', how='left')
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🔍 Product Analysis", 
        "🧪 Elasticity Testing",
        "💡 Pricing Simulator", 
        "📈 Recommendations"
    ])
    
    with tab1:
        st.subheader("Overall Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_revenue = df_analyzed['net_revenue'].sum()
            st.metric("Total Revenue", f"${total_revenue:,.2f}")
        
        with col2:
            total_profit = df_analyzed['profit'].sum()
            st.metric("Total Profit", f"${total_profit:,.2f}")
        
        with col3:
            avg_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
            st.metric("Avg Profit Margin", f"{avg_margin:.2f}%")
        
        with col4:
            total_orders = len(df_analyzed)
            st.metric("Total Orders", f"{total_orders:,}")
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Profitability by category
            cat_profit = df_analyzed.groupby('category').agg({
                'net_revenue': 'sum',
                'profit': 'sum'
            }).reset_index()
            cat_profit['margin_pct'] = (cat_profit['profit'] / cat_profit['net_revenue'] * 100).round(2)
            
            fig = px.bar(cat_profit, x='category', y='profit', 
                        title='Profit by Category',
                        labels={'profit': 'Profit ($)', 'category': 'Category'},
                        color='margin_pct',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Product velocity distribution
            velocity_counts = velocity_df['velocity_class'].value_counts().reset_index()
            velocity_counts.columns = ['Velocity', 'Count']
            
            fig = px.pie(velocity_counts, names='Velocity', values='Count',
                        title='Product Velocity Distribution',
                        color='Velocity',
                        color_discrete_map={
                            'Fast Mover': '#2ecc71',
                            'Mid Mover': '#f39c12',
                            'Slow Mover': '#e74c3c'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        # Velocity performance comparison
        st.markdown("---")
        st.subheader("Performance by Product Velocity")
        
        velocity_perf = df_analyzed.groupby('velocity_class').agg({
            'product': 'nunique',
            'net_revenue': 'sum',
            'profit': 'sum',
            'quantity': 'sum'
        }).reset_index()
        velocity_perf.columns = ['Velocity Class', 'Products', 'Revenue', 'Profit', 'Units Sold']
        velocity_perf['Profit Margin %'] = (velocity_perf['Profit'] / velocity_perf['Revenue'] * 100).round(2)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Revenue', x=velocity_perf['Velocity Class'], 
                            y=velocity_perf['Revenue'], marker_color='lightblue'))
        fig.add_trace(go.Bar(name='Profit', x=velocity_perf['Velocity Class'], 
                            y=velocity_perf['Profit'], marker_color='darkblue'))
        fig.update_layout(title='Revenue & Profit by Velocity Class', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Product-Level Analysis")
        
        # Filter by velocity class
        velocity_filter = st.multiselect(
            "Filter by Velocity Class:",
            options=['Fast Mover', 'Mid Mover', 'Slow Mover'],
            default=['Fast Mover', 'Mid Mover', 'Slow Mover']
        )
        
        # Aggregated product metrics
        product_summary = df_analyzed[df_analyzed['velocity_class'].isin(velocity_filter)].groupby(['product', 'velocity_class']).agg({
            'quantity': 'sum',
            'net_revenue': 'sum',
            'total_cost': 'sum',
            'profit': 'sum',
            'order_id': 'count',
            'slash_discount_pct': 'mean',
            'unit_price': 'mean',
            'cost_per_unit': 'mean'
        }).reset_index()
        
        product_summary.columns = ['Product', 'Velocity', 'Units Sold', 'Revenue', 'Total Cost', 
                                   'Profit', 'Orders', 'Avg Slash Discount %', 'Avg Price', 'Avg Cost']
        product_summary['Profit Margin %'] = (product_summary['Profit'] / product_summary['Revenue'] * 100).round(2)
        product_summary['Avg Order Value'] = (product_summary['Revenue'] / product_summary['Orders']).round(2)
        
        # Sort by profit
        product_summary = product_summary.sort_values('Profit', ascending=False)
        
        # Display table with color coding
        def color_profit_margin(val):
            """Color code profit margins"""
            if pd.isna(val):
                return ''
            if val < 0:
                return 'background-color: #ffcccc'
            elif val < 20:
                return 'background-color: #fff4cc'
            elif val < 40:
                return 'background-color: #e6f7ff'
            else:
                return 'background-color: #ccffcc'
        
        st.dataframe(product_summary.style.format({
            'Units Sold': '{:,.0f}',
            'Revenue': '${:,.2f}',
            'Total Cost': '${:,.2f}',
            'Profit': '${:,.2f}',
            'Orders': '{:,.0f}',
            'Avg Slash Discount %': '{:.1f}%',
            'Avg Price': '${:,.2f}',
            'Avg Cost': '${:,.2f}',
            'Profit Margin %': '{:.2f}%',
            'Avg Order Value': '${:,.2f}'
        }).applymap(color_profit_margin, subset=['Profit Margin %']),
        use_container_width=True, height=400)
        
        # Slash pricing analysis by velocity
        st.markdown("---")
        st.subheader("Slash Pricing Effectiveness by Velocity")
        
        slash_analysis = df_analyzed.groupby('velocity_class').agg({
            'slash_discount_pct': 'mean',
            'perceived_savings': 'mean',
            'quantity': 'sum',
            'profit_margin_pct': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(slash_analysis, x='velocity_class', y='slash_discount_pct',
                        title='Avg Slash Discount % by Velocity Class',
                        labels={'slash_discount_pct': 'Avg Discount %', 'velocity_class': 'Velocity Class'},
                        color='velocity_class',
                        color_discrete_map={
                            'Fast Mover': '#2ecc71',
                            'Mid Mover': '#f39c12',
                            'Slow Mover': '#e74c3c'
                        })
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(slash_analysis, x='slash_discount_pct', y='profit_margin_pct',
                           size='quantity', color='velocity_class',
                           title='Discount vs Margin by Velocity',
                           labels={'slash_discount_pct': 'Avg Discount %', 
                                  'profit_margin_pct': 'Avg Profit Margin %'},
                           color_discrete_map={
                               'Fast Mover': '#2ecc71',
                               'Mid Mover': '#f39c12',
                               'Slow Mover': '#e74c3c'
                           })
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🧪 Price Elasticity Testing by Velocity Class")
        
        st.markdown("""
        **Test pricing across all products** grouped by velocity to estimate real demand elasticity. 
        This helps you understand how fast, mid, and slow movers respond differently to price changes.
        """)
        
        # Create unfiltered product summary for elasticity testing
        product_summary_all = df_analyzed.groupby(['product', 'velocity_class']).agg({
            'quantity': 'sum',
            'net_revenue': 'sum',
            'total_cost': 'sum',
            'profit': 'sum',
            'order_id': 'count',
            'slash_discount_pct': 'mean',
            'unit_price': 'mean',
            'cost_per_unit': 'mean'
        }).reset_index()
        
        product_summary_all.columns = ['Product', 'Velocity', 'Units Sold', 'Revenue', 'Total Cost', 
                                   'Profit', 'Orders', 'Avg Slash Discount %', 'Avg Price', 'Avg Cost']
        product_summary_all['Profit Margin %'] = (product_summary_all['Profit'] / product_summary_all['Revenue'] * 100).round(2)
        
        # Group products by velocity
        velocity_groups = velocity_df.groupby('velocity_class')['product'].apply(list).to_dict()
        
        # Tabs for each velocity class
        velocity_tabs = st.tabs(["🚀 Fast Movers", "📊 Mid Movers", "🐌 Slow Movers"])
        
        for idx, (velocity_class, tab) in enumerate(zip(['Fast Mover', 'Mid Mover', 'Slow Mover'], velocity_tabs)):
            with tab:
                if velocity_class not in velocity_groups:
                    st.warning(f"No {velocity_class} products found in dataset")
                    continue
                
                products_in_class = velocity_groups[velocity_class]
                st.info(f"**{len(products_in_class)} products** in {velocity_class} category")
                
                # Summary stats for this velocity class
                class_data = df_analyzed[df_analyzed['velocity_class'] == velocity_class]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Units Sold", f"{class_data['quantity'].sum():,.0f}")
                with col2:
                    st.metric("Avg Margin", f"{class_data['profit_margin_pct'].mean():.1f}%")
                with col3:
                    st.metric("Total Profit", f"${class_data['profit'].sum():,.2f}")
                
                st.markdown("---")
                
                # Test each product in this velocity class
                for product in products_in_class:
                    product_data = product_summary_all[product_summary_all['Product'] == product].iloc[0]
                    
                    with st.expander(f"**{product}**"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Current Avg Price", f"${product_data['Avg Price']:.2f}")
                            st.metric("Current Margin", f"{product_data['Profit Margin %']:.1f}%")
                        
                        with col2:
                            st.metric("Units Sold", f"{product_data['Units Sold']:.0f}")
                            st.metric("Total Profit", f"${product_data['Profit']:.2f}")
                        
                        with col3:
                            # Estimate elasticity from data
                            estimated_elasticity, status = estimate_elasticity_from_data(df_analyzed, product)
                            
                            if estimated_elasticity:
                                st.metric("Estimated Elasticity", f"{estimated_elasticity:.2f}")
                                st.caption(status)
                                
                                # Store in session state
                                st.session_state.elasticity_tests[product] = estimated_elasticity
                                
                                # Recommendation
                                if abs(estimated_elasticity) > 1.5:
                                    st.error("⚠️ **High sensitivity**: Avoid deep discounts")
                                else:
                                    st.success("✅ **Low sensitivity**: Price changes safer")
                            else:
                                st.info("Need more price variation")
                                st.caption(status)
                                # Use default elasticity
                                st.session_state.elasticity_tests[product] = -1.5
                        
                        # Simulate small test
                        st.markdown("##### Quick Test Simulation")
                        test_col1, test_col2 = st.columns(2)
                        
                        with test_col1:
                            test_increase = st.number_input(
                                "Test price increase %",
                                min_value=0,
                                max_value=20,
                                value=5,
                                step=1,
                                key=f"inc_{product}"
                            )
                        
                        with test_col2:
                            test_decrease = st.number_input(
                                "Test price decrease %",
                                min_value=0,
                                max_value=20,
                                value=10,
                                step=1,
                                key=f"dec_{product}"
                            )
                        
                        if st.button(f"Run Test for {product}", key=f"test_{product}"):
                            product_df = df_analyzed[df_analyzed['product'] == product]
                            elasticity = st.session_state.elasticity_tests.get(product, -1.5)
                            
                            # Test increase
                            sim_increase = simulate_price_change(product_df, test_increase, elasticity)
                            profit_change_inc = sim_increase['profit_change'].sum()
                            
                            # Test decrease
                            sim_decrease = simulate_price_change(product_df, -test_decrease, elasticity)
                            profit_change_dec = sim_decrease['profit_change'].sum()
                            
                            st.markdown("**Test Results:**")
                            result_col1, result_col2 = st.columns(2)
                            
                            with result_col1:
                                st.metric(
                                    f"+{test_increase}% Price Increase",
                                    f"${profit_change_inc:,.2f}",
                                    f"{'📈' if profit_change_inc > 0 else '📉'}"
                                )
                            
                            with result_col2:
                                st.metric(
                                    f"-{test_decrease}% Price Decrease",
                                    f"${profit_change_dec:,.2f}",
                                    f"{'📈' if profit_change_dec > 0 else '📉'}"
                                )
                            
                            # Recommendation
                            if profit_change_inc > profit_change_dec and profit_change_inc > 0:
                                st.success(f"✅ **Recommendation**: Increase price by {test_increase}%")
                            elif profit_change_dec > 0:
                                st.success(f"✅ **Recommendation**: Decrease price by {test_decrease}% to drive volume")
                            else:
                                st.warning("⚠️ **Recommendation**: Keep current pricing")
        
        # Summary of elasticity tests
        st.markdown("---")
        st.markdown("#### 📊 Elasticity Testing Summary by Velocity Class")
        
        if st.session_state.elasticity_tests:
            elasticity_summary = []
            for product, elasticity in st.session_state.elasticity_tests.items():
                velocity = velocity_df[velocity_df['product'] == product]['velocity_class'].values[0]
                elasticity_summary.append({
                    'Product': product,
                    'Velocity Class': velocity,
                    'Elasticity': elasticity,
                    'Price Sensitivity': 'High (|ε| > 1.5)' if abs(elasticity) > 1.5 else 'Low (|ε| ≤ 1.5)',
                    'Discount Strategy': 'Targeted promos only' if abs(elasticity) > 1.5 else 'Flexible discounting OK'
                })
            
            elasticity_df = pd.DataFrame(elasticity_summary)
            st.dataframe(elasticity_df, use_container_width=True)
            
            # Summary by velocity class
            col1, col2 = st.columns(2)
            
            with col1:
                # Count by velocity and sensitivity
                sensitivity_summary = elasticity_df.groupby(['Velocity Class', 'Price Sensitivity']).size().reset_index(name='Count')
                fig = px.bar(sensitivity_summary, x='Velocity Class', y='Count', color='Price Sensitivity',
                           title='Price Sensitivity Distribution by Velocity',
                           barmode='group')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average elasticity by velocity
                avg_elasticity = elasticity_df.groupby('Velocity Class')['Elasticity'].mean().reset_index()
                fig = px.bar(avg_elasticity, x='Velocity Class', y='Elasticity',
                           title='Average Elasticity by Velocity Class',
                           color='Elasticity',
                           color_continuous_scale='RdYlGn',
                           color_continuous_midpoint=-1.5)
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **💡 Pricing Rule Applied:**
            - If |elasticity| > 1.5 → Customers are very price sensitive. Avoid deep permanent discounts. Use targeted, time-limited promotions instead.
            - If |elasticity| ≤ 1.5 → Customers are less price sensitive. You have more flexibility with pricing changes.
            """)
    
    with tab4:
        st.subheader("Pricing Scenario Simulator")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Simulation Parameters")
            
            price_change = st.slider(
                "Price Change (%)",
                min_value=-50,
                max_value=50,
                value=10,
                step=5,
                help="Positive = price increase, Negative = price decrease"
            )
            
            # Let user choose elasticity or use estimated
            use_estimated = st.checkbox("Use estimated elasticity from tests", value=True)
            
            if not use_estimated:
                elasticity = st.slider(
                    "Demand Elasticity",
                    min_value=-3.0,
                    max_value=-0.5,
                    value=-1.5,
                    step=0.1,
                    help="How sensitive demand is to price changes"
                )
            
            filter_by = st.selectbox("Simulate for:", ["All Products", "By Velocity Class"] + df_analyzed['product'].unique().tolist())
            
            # Velocity class filter if selected
            velocity_class_filter = None
            if filter_by == "By Velocity Class":
                velocity_class_filter = st.selectbox(
                    "Select Velocity Class:",
                    ["Fast Mover", "Mid Mover", "Slow Mover"]
                )
            
            # Set elasticity
            if use_estimated and filter_by not in ["All Products", "By Velocity Class"] and filter_by in st.session_state.elasticity_tests:
                elasticity = st.session_state.elasticity_tests[filter_by]
                st.info(f"Using estimated elasticity: {elasticity:.2f}")
            elif use_estimated:
                elasticity = -1.5
                st.warning("Using default elasticity: -1.5 (run elasticity tests first)")
            
            if st.button("Run Simulation", type="primary"):
                st.session_state.simulation_done = True
        
        with col2:
            if st.session_state.get('simulation_done', False):
                # Filter data
                if filter_by == "All Products":
                    df_sim = simulate_price_change(df_analyzed, price_change, elasticity)
                elif filter_by == "By Velocity Class":
                    df_sim = simulate_price_change(
                        df_analyzed[df_analyzed['velocity_class'] == velocity_class_filter], 
                        price_change, 
                        elasticity
                    )
                else:
                    df_sim = simulate_price_change(
                        df_analyzed[df_analyzed['product'] == filter_by], 
                        price_change, 
                        elasticity
                    )
                
                st.markdown("#### Simulation Results")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    revenue_change = df_sim['revenue_change'].sum()
                    revenue_pct = (revenue_change / df_sim['net_revenue'].sum() * 100) if df_sim['net_revenue'].sum() > 0 else 0
                    st.metric(
                        "Revenue Impact",
                        f"${revenue_change:,.2f}",
                        f"{revenue_pct:.1f}%"
                    )
                
                with col_b:
                    profit_change = df_sim['profit_change'].sum()
                    profit_pct = (profit_change / df_sim['profit'].sum() * 100) if df_sim['profit'].sum() != 0 else 0
                    st.metric(
                        "Profit Impact",
                        f"${profit_change:,.2f}",
                        f"{profit_pct:.1f}%"
                    )
                
                with col_c:
                    avg_margin_change = df_sim['margin_change'].mean()
                    st.metric(
                        "Avg Margin Change",
                        f"{avg_margin_change:.2f}pp"
                    )
                
                # Elasticity warning
                if abs(elasticity) > 1.5 and price_change < -10:
                    st.error("⚠️ **Warning**: High elasticity + deep discount = risky strategy. Consider targeted promotions instead.")
                elif abs(elasticity) > 1.5 and price_change > 10:
                    st.warning("⚠️ **Caution**: High elasticity means volume will drop significantly with price increases.")
                
                # Comparison chart
                comparison = pd.DataFrame({
                    'Metric': ['Revenue', 'Profit', 'Margin %'],
                    'Current': [
                        df_sim['net_revenue'].sum(),
                        df_sim['profit'].sum(),
                        (df_sim['profit'].sum() / df_sim['net_revenue'].sum() * 100) if df_sim['net_revenue'].sum() > 0 else 0
                    ],
                    'Simulated': [
                        df_sim['new_net_revenue'].sum(),
                        df_sim['new_profit'].sum(),
                        (df_sim['new_profit'].sum() / df_sim['new_net_revenue'].sum() * 100) if df_sim['new_net_revenue'].sum() > 0 else 0
                    ]
                })
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='Current', x=comparison['Metric'], y=comparison['Current'], marker_color='lightblue'))
                fig.add_trace(go.Bar(name='Simulated', x=comparison['Metric'], y=comparison['Simulated'], marker_color='darkblue'))
                fig.update_layout(title='Current vs Simulated Performance', barmode='group')
                st.plotly_chart(fig, use_container_width=True)
                
                # Product-level impact
                if filter_by in ["All Products", "By Velocity Class"]:
                    st.markdown("#### Impact by Product")
                    product_impact = df_sim.groupby('product').agg({
                        'profit_change': 'sum',
                        'margin_change': 'mean',
                        'velocity_class': 'first'
                    }).reset_index().sort_values('profit_change', ascending=False)
                    
                    fig = px.bar(product_impact, x='product', y='profit_change',
                               title='Profit Change by Product',
                               labels={'profit_change': 'Profit Change ($)', 'product': 'Product'},
                               color='velocity_class',
                               color_discrete_map={
                                   'Fast Mover': '#2ecc71',
                                   'Mid Mover': '#f39c12',
                                   'Slow Mover': '#e74c3c'
                               })
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show velocity class breakdown
                    if filter_by == "All Products":
                        st.markdown("#### Impact by Velocity Class")
                        velocity_impact = df_sim.groupby('velocity_class').agg({
                            'profit_change': 'sum',
                            'revenue_change': 'sum'
                        }).reset_index()
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            fig = px.bar(velocity_impact, x='velocity_class', y='profit_change',
                                       title='Profit Change by Velocity Class',
                                       color='velocity_class',
                                       color_discrete_map={
                                           'Fast Mover': '#2ecc71',
                                           'Mid Mover': '#f39c12',
                                           'Slow Mover': '#e74c3c'
                                       })
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col_b:
                            fig = px.bar(velocity_impact, x='velocity_class', y='revenue_change',
                                       title='Revenue Change by Velocity Class',
                                       color='velocity_class',
                                       color_discrete_map={
                                           'Fast Mover': '#2ecc71',
                                           'Mid Mover': '#f39c12',
                                           'Slow Mover': '#e74c3c'
                                       })
                            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("Pricing Recommendations by Velocity Class")
        
        # Get product metrics with velocity
        product_metrics = df_analyzed.groupby(['product', 'velocity_class']).agg({
            'unit_price': 'mean',
            'cost_per_unit': 'mean',
            'profit_margin_pct': 'mean',
            'quantity': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        # Add elasticity data
        product_metrics['elasticity'] = product_metrics['product'].map(
            lambda x: st.session_state.elasticity_tests.get(x, -1.5)
        )
        product_metrics['elasticity_abs'] = product_metrics['elasticity'].abs()
        
        # Generate recommendations by velocity class
        st.markdown("#### 🎯 Smart Pricing Actions")
        
        # Tabs for each velocity class
        rec_tabs = st.tabs(["🚀 Fast Movers", "📊 Mid Movers", "🐌 Slow Movers"])
        
        for velocity_class, tab in zip(['Fast Mover', 'Mid Mover', 'Slow Mover'], rec_tabs):
            with tab:
                velocity_products = product_metrics[product_metrics['velocity_class'] == velocity_class]
                
                if len(velocity_products) == 0:
                    st.warning(f"No {velocity_class} products found")
                    continue
                
                # Summary for this velocity class
                st.markdown(f"### {velocity_class} Strategy Overview")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_elasticity = velocity_products['elasticity'].mean()
                    st.metric("Avg Elasticity", f"{avg_elasticity:.2f}")
                with col2:
                    avg_margin = velocity_products['profit_margin_pct'].mean()
                    st.metric("Avg Margin", f"{avg_margin:.1f}%")
                with col3:
                    total_profit = velocity_products['profit'].sum()
                    st.metric("Total Profit", f"${total_profit:,.2f}")
                
                # High elasticity products
                high_elasticity = velocity_products[velocity_products['elasticity_abs'] > 1.5]
                
                if len(high_elasticity) > 0:
                    st.markdown("---")
                    st.markdown("#### ⚠️ High Price Sensitivity Products (|ε| > 1.5)")
                    st.warning("**Rule**: Avoid deep permanent discounts. Use targeted, time-limited promotions.")
                    
                    for _, row in high_elasticity.iterrows():
                        with st.expander(f"**{row['product']}**"):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Current Price", f"${row['unit_price']:.2f}")
                                st.metric("Current Margin", f"{row['profit_margin_pct']:.1f}%")
                            
                            with col2:
                                st.metric("Elasticity", f"{row['elasticity']:.2f}")
                                st.metric("Units Sold", f"{row['quantity']:.0f}")
                            
                            with col3:
                                st.metric("Total Profit", f"${row['profit']:.2f}")
                            
                            # Generate recommendations
                            recommendations = generate_pricing_recommendation(
                                row['elasticity_abs'],
                                row['profit_margin_pct'],
                                row['velocity_class']
                            )
                            
                            for rec in recommendations:
                                if rec['type'] == 'warning':
                                    st.error(f"**{rec['title']}**\n\n{rec['message']}\n\n✅ **Action**: {rec['action']}")
                                elif rec['type'] == 'info':
                                    st.info(f"**{rec['title']}**\n\n{rec['message']}\n\n✅ **Action**: {rec['action']}")
                                elif rec['type'] == 'success':
                                    st.success(f"**{rec['title']}**\n\n{rec['message']}\n\n✅ **Action**: {rec['action']}")
                            
                            # Specific tactical recommendations
                            st.markdown("**💡 Recommended Tactics:**")
                            st.markdown("""
                            - ✅ Flash sales (24-48 hours only)
                            - ✅ Limited quantity offers
                            - ✅ Bundle with complementary products
                            - ✅ Loyalty rewards (non-price incentives)
                            - ❌ Across-the-board permanent price cuts
                            - ❌ Deep discounts (>20%)
                            """)
                
                # Low elasticity products
                low_elasticity = velocity_products[velocity_products['elasticity_abs'] <= 1.5]
                
                if len(low_elasticity) > 0:
                    st.markdown("---")
                    st.markdown("#### ✅ Low Price Sensitivity Products (|ε| ≤ 1.5)")
                    st.success("**Rule**: More flexible pricing. Price increases are safer, discounts can be broader.")
                    
                    # Price increase opportunities
                    low_margin_low_elast = low_elasticity[low_elasticity['profit_margin_pct'] < 25].sort_values('quantity', ascending=False)
                    
                    if len(low_margin_low_elast) > 0:
                        st.markdown("##### 🔴 Price INCREASE Opportunities")
                        
                        for _, row in low_margin_low_elast.iterrows():
                            with st.expander(f"**{row['product']}** - Low margin + Low sensitivity"):
                                recommended_price = row['unit_price'] * 1.10  # 10% increase
                                new_margin = ((recommended_price - row['cost_per_unit']) / recommended_price * 100)
                                
                                # Simulate the impact
                                product_df = df_analyzed[df_analyzed['product'] == row['product']]
                                sim_result = simulate_price_change(product_df, 10, row['elasticity'])
                                profit_impact = sim_result['profit_change'].sum()
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Current State:**")
                                    st.write(f"Price: ${row['unit_price']:.2f}")
                                    st.write(f"Margin: {row['profit_margin_pct']:.1f}%")
                                    st.write(f"Elasticity: {row['elasticity']:.2f}")
                                    st.write(f"Monthly profit: ${row['profit']:.2f}")
                                
                                with col2:
                                    st.markdown("**Recommended Change:**")
                                    st.write(f"New price: ${recommended_price:.2f} (+10%)")
                                    st.write(f"New margin: {new_margin:.1f}%")
                                    st.write(f"Expected volume impact: {10 * row['elasticity']:.1f}%")
                                    st.write(f"Profit impact: ${profit_impact:,.2f}")
                                
                                if profit_impact > 0:
                                    st.success(f"✅ **Recommendation**: Increase price to ${recommended_price:.2f}. Expected additional profit: ${profit_impact:,.2f}")
                                else:
                                    st.warning("⚠️ Test smaller increments (5%) first")
                    
                    # Price decrease opportunities
                    high_margin_low_elast = low_elasticity[
                        (low_elasticity['profit_margin_pct'] > 40) & 
                        (low_elasticity['quantity'] < low_elasticity['quantity'].median())
                    ]
                    
                    if len(high_margin_low_elast) > 0:
                        st.markdown("##### 🟢 Price DECREASE Opportunities")
                        
                        for _, row in high_margin_low_elast.iterrows():
                            with st.expander(f"**{row['product']}** - High margin + Low volume"):
                                recommended_price = row['unit_price'] * 0.90  # 10% decrease
                                new_margin = ((recommended_price - row['cost_per_unit']) / recommended_price * 100)
                                
                                # Simulate the impact
                                product_df = df_analyzed[df_analyzed['product'] == row['product']]
                                sim_result = simulate_price_change(product_df, -10, row['elasticity'])
                                profit_impact = sim_result['profit_change'].sum()
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Current State:**")
                                    st.write(f"Price: ${row['unit_price']:.2f}")
                                    st.write(f"Margin: {row['profit_margin_pct']:.1f}%")
                                    st.write(f"Units sold: {row['quantity']:.0f}")
                                    st.write(f"Monthly profit: ${row['profit']:.2f}")
                                
                                with col2:
                                    st.markdown("**Recommended Change:**")
                                    st.write(f"New price: ${recommended_price:.2f} (-10%)")
                                    st.write(f"New margin: {new_margin:.1f}%")
                                    st.write(f"Expected volume increase: {-10 * row['elasticity']:.1f}%")
                                    st.write(f"Profit impact: ${profit_impact:,.2f}")
                                
                                if profit_impact > 0:
                                    st.success(f"✅ **Recommendation**: Decrease price to ${recommended_price:.2f}. Expected additional profit: ${profit_impact:,.2f}")
                                else:
                                    st.info("💡 Current pricing may be optimal")
        
        # Loss-making products across all velocity classes
        loss_making = product_metrics[product_metrics['profit'] < 0]
        
        if len(loss_making) > 0:
            st.markdown("---")
            st.markdown("#### 🚨 Loss-Making Products - URGENT ACTION REQUIRED")
            
            for _, row in loss_making.iterrows():
                breakeven_price = row['cost_per_unit'] * 1.2  # 20% margin
                price_increase_needed = ((breakeven_price - row['unit_price']) / row['unit_price'] * 100)
                
                st.error(f"""
                **{row['product']}** ({row['velocity_class']}) is losing money  
                - Current: ${row['unit_price']:.2f} | Cost: ${row['cost_per_unit']:.2f} | Loss: ${row['profit']:.2f}
                - Elasticity: {row['elasticity']:.2f}
                
                **Immediate Actions:**
                1. Increase price by {price_increase_needed:.1f}% to ${breakeven_price:.2f}, OR
                2. Reduce cost per unit to ${row['unit_price'] * 0.8:.2f}, OR
                3. Discontinue product if neither option is viable
                """)
        
        # Export recommendations
        st.markdown("---")
        if st.button("📥 Export Complete Analysis Report"):
            # Create comprehensive report
            buffer = io.StringIO()
            buffer.write("=" * 80 + "\n")
            buffer.write("PROFITABILITY & PRICING ANALYSIS REPORT\n")
            buffer.write("WITH ELASTICITY TESTING BY VELOCITY CLASS\n")
            buffer.write("=" * 80 + "\n\n")
            buffer.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            buffer.write("OVERALL PERFORMANCE\n")
            buffer.write("-" * 80 + "\n")
            buffer.write(f"Total Revenue: ${df_analyzed['net_revenue'].sum():,.2f}\n")
            buffer.write(f"Total Profit: ${df_analyzed['profit'].sum():,.2f}\n")
            buffer.write(f"Overall Margin: {(df_analyzed['profit'].sum() / df_analyzed['net_revenue'].sum() * 100):.2f}%\n\n")
            
            buffer.write("VELOCITY CLASS SUMMARY\n")
            buffer.write("-" * 80 + "\n")
            velocity_summary = df_analyzed.groupby('velocity_class').agg({
                'product': 'nunique',
                'net_revenue': 'sum',
                'profit': 'sum'
            })
            velocity_summary.to_string(buffer)
            buffer.write("\n\n")
            
            buffer.write("ELASTICITY TEST RESULTS\n")
            buffer.write("-" * 80 + "\n")
            for product, elasticity in st.session_state.elasticity_tests.items():
                velocity = velocity_df[velocity_df['product'] == product]['velocity_class'].values[0]
                sensitivity = "HIGH" if abs(elasticity) > 1.5 else "LOW"
                buffer.write(f"{product} ({velocity}): {elasticity:.2f} ({sensitivity} price sensitivity)\n")
            buffer.write("\n")
            
            buffer.write("PRODUCT PERFORMANCE\n")
            buffer.write("-" * 80 + "\n")
            product_summary.to_string(buffer)
            
            # Download
            st.download_button(
                label="Download Report (TXT)",
                data=buffer.getvalue(),
                file_name=f"pricing_analysis_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

else:
    st.info("👈 Please upload your transaction data or generate sample data to begin analysis")
    
    # Display instructions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Features
        - **Velocity Classification**: Automatically categorize products as Fast/Mid/Slow movers
        - **Slash Price Analysis**: Understand discount perception impacts
        - **Elasticity Testing**: Test ALL products grouped by velocity class
        - **Smart Recommendations**: Get elasticity-based pricing guidance per velocity tier
        - **Pricing Simulation**: Model price changes by velocity class or individual products
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Getting Started
        1. Upload your sales CSV with slash_price column
        2. Products automatically categorized by velocity (Fast/Mid/Slow)
        3. Run elasticity tests on all products by velocity class
        4. Get smart recommendations based on |ε| > 1.5 rule
        5. Simulate and export your pricing strategy
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with Streamlit | Enhanced Profitability & Pricing Optimizer v2.0</div>",
    unsafe_allow_html=True
)
