# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the trained model (ensure the model file is in the same directory)
# model = joblib.load('model.pkl')

# # Function to process new peptide sequences
# def process_peptide_sequences(peptides):
#     # Example processing function, replace with actual preprocessing steps
#     compositions = []
#     for peptide in peptides:
#         composition = {aa: peptide.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
#         compositions.append(composition)
#     return pd.DataFrame(compositions)

# # Streamlit app
# st.title("ABPep-C")
# st.write("Classify peptide sequences as active or inactive against biofilm")

# # Input: Peptide sequences
# peptide_input = st.text_area("Enter peptide sequences (one per line)")
# peptides = peptide_input.split('\n')

# if st.button("Classify"):
#     if peptides:
#         # Process the input peptides
#         peptide_df = process_peptide_sequences(peptides)
        
#         # Predict using the trained model
#         predictions = model.predict(peptide_df)
#         results = pd.DataFrame({
#             'Peptide': peptides,
#             'Prediction': predictions
#         })
#         results['Prediction'] = results['Prediction'].map({0: 'Inactive', 1: 'Active'})
        
#         # Display the results
#         st.write("Classification Results")
#         st.write(results)
        
#         # Display interactive graphs
#         st.write("Prediction Distribution")
#         fig, ax = plt.subplots()
#         sns.countplot(x='Prediction', data=results, ax=ax)
#         st.pyplot(fig)
        
#         st.write("Amino Acid Composition of Peptides")
#         amino_acid_counts = peptide_df.sum().reset_index()
#         amino_acid_counts.columns = ['Amino Acid', 'Count']
#         fig, ax = plt.subplots()
#         sns.barplot(x='Amino Acid', y='Count', data=amino_acid_counts, ax=ax)
#         st.pyplot(fig)
#     else:
#         st.write("Please enter peptide sequences.")

# # Save this script as app.py and run it using: streamlit run app.
#######################################################################################################################################
# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the trained model (ensure the model file is in the same directory)
# model = joblib.load('model.pkl')

# # Function to process new peptide sequences
# def process_peptide_sequences(peptides):
#     # Example processing function, replace with actual preprocessing steps
#     compositions = []
#     for peptide in peptides:
#         composition = {aa: peptide.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
#         compositions.append(composition)
#     return pd.DataFrame(compositions)

# # Custom CSS for font size and color
# st.markdown("""
#     <style>
#     .title {
#         font-size: 48px !important;
#         color: #4CAF50;
#     }
#     .subheader {
#         font-size: 24px !important;
#         color: #FF5722;
#     }
#     .text {
#         font-size: 18px !important;
#     }
#     </style>
#     """, unsafe_allow_html=True)

# # Streamlit app
# st.markdown('<h1 class="title">Ab-PepC</h1>', unsafe_allow_html=True)
# st.markdown('<h2 class="subheader">Classify peptide sequences as active or inactive against biofilm</h2>', unsafe_allow_html=True)

# # Input: Peptide sequences
# peptide_input = st.text_area("Enter peptide sequences (one per line)")
# peptides = peptide_input.split('\n')

# if st.button("Classify"):
#     if peptides:
#         # Process the input peptides
#         peptide_df = process_peptide_sequences(peptides)
        
#         # Predict using the trained model
#         predictions = model.predict(peptide_df)
#         results = pd.DataFrame({
#             'Peptide': peptides,
#             'Prediction': predictions
#         })
#         results['Prediction'] = results['Prediction'].map({0: 'Inactive', 1: 'Active'})
        
#         # Display the results
#         st.markdown('<h3 class="subheader">Classification Results</h3>', unsafe_allow_html=True)
#         st.dataframe(results)
        
#         # Display interactive graphs
#         st.markdown('<h3 class="subheader">Prediction Distribution</h3>', unsafe_allow_html=True)
#         fig, ax = plt.subplots()
#         sns.countplot(x='Prediction', data=results, ax=ax)
#         ax.set_xlabel('Prediction', fontsize=18)
#         ax.set_ylabel('Count', fontsize=18)
#         st.pyplot(fig)
        
#         st.markdown('<h3 class="subheader">Amino Acid Composition of Peptides</h3>', unsafe_allow_html=True)
#         amino_acid_counts = peptide_df.sum().reset_index()
#         amino_acid_counts.columns = ['Amino Acid', 'Count']
#         fig, ax = plt.subplots()
#         sns.barplot(x='Amino Acid', y='Count', data=amino_acid_counts, ax=ax)
#         ax.set_xlabel('Amino Acid', fontsize=18)
#         ax.set_ylabel('Count', fontsize=18)
#         st.pyplot(fig)
#     else:
#         st.write("Please enter peptide sequences.")
#######################################################################################################################################
import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model (ensure the model file is in the same directory)
model = joblib.load('model.pkl')

# Function to process new peptide sequences
def process_peptide_sequences(peptides):
    # Example processing function, replace with actual preprocessing steps
    compositions = []
    for peptide in peptides:
        composition = {aa: peptide.count(aa) for aa in 'ACDEFGHIKLMNPQRSTVWY'}
        compositions.append(composition)
    return pd.DataFrame(compositions)

# Custom CSS for font size and color
st.markdown("""
    <style>
    .title {
        font-size: 48px !important;
        color: #4CAF50;
    }
    .subheader {
        font-size: 24px !important;
        color: #FF5722;
    }
    .text {
        font-size: 18px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Streamlit app
col1, col2 = st.columns([1, 4])  # Adjust the width ratio as needed
col1.image('Ab-PepC_logo.png', width=150)  # Add your logo file path here
with col2:
    st.markdown('<h1 class="title">ABPep-C</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="subheader">Classify peptide sequences as active or inactive against biofilm</h2>', unsafe_allow_html=True)

# Input: Peptide sequences
peptide_input = st.text_area("Enter peptide sequences (one per line)")
peptides = peptide_input.split('\n')

if st.button("Classify"):
    if peptides:
        # Process the input peptides
        peptide_df = process_peptide_sequences(peptides)
        
        # Predict using the trained model
        predictions = model.predict(peptide_df)
        results = pd.DataFrame({
            'Peptide': peptides,
            'Prediction': predictions
        })
        results['Prediction'] = results['Prediction'].map({0: 'Inactive', 1: 'Active'})
        
        # Display the results
        st.markdown('<h3 class="subheader">Classification Results</h3>', unsafe_allow_html=True)
        st.dataframe(results)
        
        # Display interactive graphs
        st.markdown('<h3 class="subheader">Prediction Distribution</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.countplot(x='Prediction', data=results, ax=ax)
        ax.set_xlabel('Prediction', fontsize=18)
        ax.set_ylabel('Count', fontsize=18)
        st.pyplot(fig)
        
        st.markdown('<h3 class="subheader">Amino Acid Composition of Peptides</h3>', unsafe_allow_html=True)
        amino_acid_counts = peptide_df.sum().reset_index()
        amino_acid_counts.columns = ['Amino Acid', 'Count']
        fig, ax = plt.subplots()
        sns.barplot(x='Amino Acid', y='Count', data=amino_acid_counts, ax=ax)
        ax.set_xlabel('Amino Acid', fontsize=18)
        ax.set_ylabel('Count', fontsize=18)
        st.pyplot(fig)
    else:
        st.write("Please enter peptide sequences.")

