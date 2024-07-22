
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_correlations(data, injection_wells, producing_wells):
    correlations = {}
    for prod_well in producing_wells:
        prod_col = f"{prod_well}_WaterCut"
        if prod_col not in data.columns:
            continue
        correlations[prod_well] = {}
        for inj_well in injection_wells:
            inj_col = f"{inj_well}_WI_Rate__b_d"
            if inj_col not in data.columns:
                inj_col = f"{inj_well}_GI_Rate__MMscf_d"
                if inj_col not in data.columns:
                    continue
            corr = data[prod_col].corr(data[inj_col])
            correlations[prod_well][inj_well] = corr
    return correlations

def correlation_analysis():
    st.title('Injection-Production Correlation Analysis')

    # Load data
    @st.cache_data
    def load_data():
        return pd.read_pickle('test_data.pkl')  # Adjust the file path as needed

    data = load_data()

    # Identify wells
    producing_wells = sorted(set([col.split('_')[0] for col in data.columns if '_WaterCut' in col]))
    injection_wells = sorted(set([col.split('_')[0] for col in data.columns if '_WI_Rate__b_d' in col or '_GI_Rate__MMscf_d' in col]))

    # Calculate correlations
    correlations = calculate_correlations(data, injection_wells, producing_wells)

    # User selection
    selected_producing_well = st.selectbox('Select Producing Well for Correlation Analysis', producing_wells)

    if selected_producing_well in correlations:
        st.write(f"Correlation coefficients between {selected_producing_well} Water Cut and Injection Rates:")
        corr_data = pd.DataFrame.from_dict(correlations[selected_producing_well], orient='index', columns=['Correlation'])
        corr_data = corr_data.sort_values('Correlation', ascending=False)
        st.dataframe(corr_data)

        # Visualize correlations
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(corr_data.T, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title(f'Correlations with {selected_producing_well} Water Cut')
        st.pyplot(fig)

        # Visualize top correlations
        top_n = 5  # Show top 5 correlations
        fig, ax = plt.subplots(figsize=(10, 6))
        top_corr = corr_data.head(top_n)
        top_corr['Correlation'].plot(kind='bar', ax=ax)
        ax.set_title(f'Top {top_n} Correlations with {selected_producing_well} Water Cut')
        ax.set_ylabel('Correlation Coefficient')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        # Scatter plot for top correlation
        top_inj_well = top_corr.index[0]
        inj_col = f"{top_inj_well}_WI_Rate__b_d" if f"{top_inj_well}_WI_Rate__b_d" in data.columns else f"{top_inj_well}_GI_Rate__MMscf_d"
        prod_col = f"{selected_producing_well}_WaterCut"

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(data[inj_col], data[prod_col])
        ax.set_xlabel(f'{top_inj_well} Injection Rate')
        ax.set_ylabel(f'{selected_producing_well} Water Cut')
        ax.set_title(f'Scatter Plot: {top_inj_well} Injection Rate vs {selected_producing_well} Water Cut')
        st.pyplot(fig)

    else:
        st.write(f"No correlation data available for {selected_producing_well}")

if __name__ == "__main__":
    correlation_analysis()
