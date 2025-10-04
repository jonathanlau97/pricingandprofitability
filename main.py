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
        'cost_per_unit': np.random.uniform(5, 700, n_records).round(2),
        'discount_pct': np.random.choice([0, 5, 10, 15, 20], n_records, p=[0.4, 0.2, 0.2, 0.15, 0.05]),
        'shipping_cost': np.random.uniform(0, 20, n_records).round(2),
        'date': pd.date_range(start='2024-01-01', periods=n_records, freq='D')[:n_records]
    }
    
    df = pd.DataFrame(data)
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
    
    return df

def simulate_price_change(df, price_change_pct, elasticity=-1.5):
    """Simulate impact of price changes with demand elasticity"""
    df_sim = df.copy()
    
    # Calculate new prices
    df_sim['new_unit_price'] = df_sim['unit_price'] * (1 + price_change_pct / 100)
    
    # Apply demand elasticity (% change in quantity / % change in price)
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

# Header
st.markdown('<div class="main-header">💰 Profitability & Pricing Optimizer</div>', unsafe_allow_html=True)
st.markdown("**Analyze profitability and simulate pricing scenarios with demand elasticity**")

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
            - `unit_price` - Price per unit
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
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Product Analysis", "💡 Pricing Simulator", "📈 Recommendations"])
    
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
            # Top profitable products
            prod_profit = df_analyzed.groupby('product').agg({
                'profit': 'sum',
                'quantity': 'sum'
            }).reset_index().sort_values('profit', ascending=False).head(10)
            
            fig = px.bar(prod_profit, x='profit', y='product', 
                        orientation='h',
                        title='Top 10 Products by Profit',
                        labels={'profit': 'Profit ($)', 'product': 'Product'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Profitability distribution
        st.subheader("Profitability Distribution")
        fig = px.histogram(df_analyzed, x='profit_margin_pct', 
                          nbins=50,
                          title='Distribution of Profit Margins',
                          labels={'profit_margin_pct': 'Profit Margin (%)'})
        fig.add_vline(x=0, line_dash="dash", line_color="red", 
                     annotation_text="Break-even")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Product-Level Analysis")
        
        # Aggregated product metrics
        product_summary = df_analyzed.groupby('product').agg({
            'quantity': 'sum',
            'net_revenue': 'sum',
            'total_cost': 'sum',
            'profit': 'sum',
            'order_id': 'count'
        }).reset_index()
        
        product_summary.columns = ['Product', 'Units Sold', 'Revenue', 'Total Cost', 'Profit', 'Orders']
        product_summary['Profit Margin %'] = (product_summary['Profit'] / product_summary['Revenue'] * 100).round(2)
        product_summary['Avg Order Value'] = (product_summary['Revenue'] / product_summary['Orders']).round(2)
        
        # Sort by profit
        product_summary = product_summary.sort_values('Profit', ascending=False)
        
        # Display table
        st.dataframe(product_summary.style.format({
            'Units Sold': '{:,.0f}',
            'Revenue': '${:,.2f}',
            'Total Cost': '${:,.2f}',
            'Profit': '${:,.2f}',
            'Orders': '{:,.0f}',
            'Profit Margin %': '{:.2f}%',
            'Avg Order Value': '${:,.2f}'
        }).background_gradient(subset=['Profit Margin %'], cmap='RdYlGn', vmin=-20, vmax=50),
        use_container_width=True, height=400)
        
        # Loss-making products
        loss_products = product_summary[product_summary['Profit'] < 0]
        if len(loss_products) > 0:
            st.warning(f"⚠️ {len(loss_products)} product(s) are currently unprofitable:")
            st.dataframe(loss_products[['Product', 'Profit', 'Profit Margin %']], use_container_width=True)
    
    with tab3:
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
            
            elasticity = st.slider(
                "Demand Elasticity",
                min_value=-3.0,
                max_value=-0.5,
                value=-1.5,
                step=0.1,
                help="How sensitive demand is to price changes. -1.5 = 1% price increase leads to 1.5% demand decrease"
            )
            
            filter_by = st.selectbox("Simulate for:", ["All Products"] + df_analyzed['product'].unique().tolist())
            
            if st.button("Run Simulation", type="primary"):
                st.session_state.simulation_done = True
        
        with col2:
            if st.session_state.get('simulation_done', False):
                # Filter data
                if filter_by == "All Products":
                    df_sim = simulate_price_change(df_analyzed, price_change, elasticity)
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
                if filter_by == "All Products":
                    st.markdown("#### Impact by Product")
                    product_impact = df_sim.groupby('product').agg({
                        'profit_change': 'sum',
                        'margin_change': 'mean'
                    }).reset_index().sort_values('profit_change', ascending=False)
                    
                    fig = px.bar(product_impact, x='product', y='profit_change',
                               title='Profit Change by Product',
                               labels={'profit_change': 'Profit Change ($)', 'product': 'Product'},
                               color='margin_change',
                               color_continuous_scale='RdYlGn')
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Pricing Recommendations")
        
        # Identify opportunities
        product_metrics = df_analyzed.groupby('product').agg({
            'unit_price': 'mean',
            'cost_per_unit': 'mean',
            'profit_margin_pct': 'mean',
            'quantity': 'sum',
            'profit': 'sum'
        }).reset_index()
        
        # Low margin, high volume products
        low_margin = product_metrics[product_metrics['profit_margin_pct'] < 20].sort_values('quantity', ascending=False).head(5)
        
        # High margin, low volume products
        high_margin = product_metrics[product_metrics['profit_margin_pct'] > 40].sort_values('quantity', ascending=True).head(5)
        
        # Loss-making products
        loss_making = product_metrics[product_metrics['profit'] < 0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 Price Increase Opportunities")
            st.markdown("**Low margin, high volume products:**")
            if len(low_margin) > 0:
                for _, row in low_margin.iterrows():
                    recommended_price = row['unit_price'] * 1.15  # 15% increase
                    new_margin = ((recommended_price - row['cost_per_unit']) / recommended_price * 100)
                    st.info(f"""
                    **{row['product']}**  
                    Current: ${row['unit_price']:.2f} ({row['profit_margin_pct']:.1f}% margin)  
                    Recommended: ${recommended_price:.2f} (+15%) → {new_margin:.1f}% margin  
                    Volume: {row['quantity']:.0f} units
                    """)
            else:
                st.success("No immediate opportunities found")
        
        with col2:
            st.markdown("#### 🟢 Price Decrease Opportunities")
            st.markdown("**High margin, low volume products:**")
            if len(high_margin) > 0:
                for _, row in high_margin.iterrows():
                    recommended_price = row['unit_price'] * 0.90  # 10% decrease
                    new_margin = ((recommended_price - row['cost_per_unit']) / recommended_price * 100)
                    st.info(f"""
                    **{row['product']}**  
                    Current: ${row['unit_price']:.2f} ({row['profit_margin_pct']:.1f}% margin)  
                    Recommended: ${recommended_price:.2f} (-10%) → {new_margin:.1f}% margin  
                    Volume: {row['quantity']:.0f} units
                    """)
            else:
                st.success("No immediate opportunities found")
        
        if len(loss_making) > 0:
            st.markdown("#### ⚠️ Loss-Making Products - Immediate Action Required")
            for _, row in loss_making.iterrows():
                breakeven_price = row['cost_per_unit'] * 1.2  # 20% margin
                price_increase_needed = ((breakeven_price - row['unit_price']) / row['unit_price'] * 100)
                st.error(f"""
                **{row['product']}** is losing money  
                Current: ${row['unit_price']:.2f} | Cost: ${row['cost_per_unit']:.2f} | Loss: ${row['profit']:.2f}  
                **Action:** Increase price by {price_increase_needed:.1f}% to ${breakeven_price:.2f} or reduce costs
                """)
        
        # Export recommendations
        st.markdown("---")
        if st.button("📥 Export Analysis Report"):
            # Create comprehensive report
            buffer = io.StringIO()
            buffer.write("PROFITABILITY & PRICING ANALYSIS REPORT\n")
            buffer.write("=" * 50 + "\n\n")
            buffer.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            buffer.write(f"Total Revenue: ${df_analyzed['net_revenue'].sum():,.2f}\n")
            buffer.write(f"Total Profit: ${df_analyzed['profit'].sum():,.2f}\n")
            buffer.write(f"Overall Margin: {(df_analyzed['profit'].sum() / df_analyzed['net_revenue'].sum() * 100):.2f}%\n\n")
            
            buffer.write("\nPRODUCT PERFORMANCE:\n")
            buffer.write("-" * 50 + "\n")
            product_summary.to_string(buffer)
            
            # Download
            st.download_button(
                label="Download Report (TXT)",
                data=buffer.getvalue(),
                file_name=f"profitability_report_{datetime.now().strftime('%Y%m%d')}.txt",
                mime="text/plain"
            )

else:
    st.info("👈 Please upload your transaction data or generate sample data to begin analysis")
    
    # Display instructions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📊 Features
        - **Profitability Analysis**: Calculate profit at product, order, and category levels
        - **Margin Insights**: Identify high and low margin products
        - **Pricing Simulation**: Model price changes with demand elasticity
        - **Smart Recommendations**: Get AI-powered pricing suggestions
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Getting Started
        1. Upload your sales/transaction CSV file
        2. Review overall profitability metrics
        3. Analyze product-level performance
        4. Run pricing simulations
        5. Export recommendations
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with Streamlit | Profitability & Pricing Optimizer v1.0</div>",
    unsafe_allow_html=True
)
