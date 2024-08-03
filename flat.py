import streamlit as st
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import pandas as pd


#page head
st.set_page_config(page_title="ML", page_icon=":rocket:", layout="wide", initial_sidebar_state="expanded")

st.markdown("<h1 style='text-align: center; color: blue;'>SINGAPORE FLAT RESALE PRICE PREDICTION</h1>", unsafe_allow_html=True)
# ------------------------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["***HOME***&nbsp;&nbsp;&nbsp;&nbsp;",'***ABOUT-MODEL***&nbsp;&nbsp;&nbsp;&nbsp;','***PREDICTION***&nbsp;&nbsp;&nbsp;&nbsp;',"***ACCURACY***&nbsp;&nbsp;&nbsp;&nbsp;", "***DATA***&nbsp;&nbsp;&nbsp;&nbsp;","***CONTACT US***&nbsp;&nbsp;&nbsp;&nbsp;"])


# ------------------------------------------------------------------------------------------

css = '''
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size:2rem;
    }
</style>
'''
st.markdown(css, unsafe_allow_html=True)
# -------------------------------------------tab1-----------------------------------------------

with tab1 : 

  st.markdown("""
    Empower your decisions in the competitive resale flat market with our machine learning-driven price prediction tool. 
    Accurately estimate resale values based on factors like location, flat type, and floor area. 
    Our user-friendly Streamlit application, backed by advanced regression models, makes predictions accessible to all. 
    Hosted on the Render platform, our tool ensures seamless internet access for users. Explore the future with continuous model retraining and real-time updates.
    """)  
  
def load_data_and_model():
    with open("S_Model_R.pkl", "rb") as file:
        prediction_model = pickle.load(file)

    data = pd.read_csv("Final_df.csv")

    return prediction_model, data

# Load data and model
Prediction, data = load_data_and_model()

# Sample a subset of the data for faster visualization
sampled_data = data.sample(frac=0.1, random_state=42)

# Assuming 'no_of_rooms' is one of your features
X = sampled_data[["year", "town", "floor_area_sqm", "lease_commence_year", "no_of_rooms"]]
Y = sampled_data[["resale_price"]]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

Prediction.fit(X_train, y_train)
Ans = Prediction.predict(X_test)
Final = r2_score(y_test, Ans)

feature_importances = pd.Series(Prediction.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# -------------------------------------------tab2-----------------------------------------------

# path2 =  '1.jpg'
# L = Image.open(path2)

with tab2 :
    # st.image(L, width=1300)
    st.markdown("[CLICK HERE TO KNOW PROBLEM STATEMENT OF THE PROJECT](https://docs.google.com/document/d/1mPb68zw8G-iFNcFr4hSAp7yIXc3-0JVlFVKBPf-0Hxo/edit)")
    
    if st.button('CLICK HERE TO KNOW ABOUT FEATURE IMPORTANCE'):
        M = pd.DataFrame(feature_importances, columns=['COLUMN_IMPORTANCE'])
        M.index.name = 'COLUMNS_NAME'
        st.write(M)

# -------------------------------------------tab3-----------------------------------------------
# X = data[["year", "town", "floor_area_sqm", "lease_commence_year", "no_of_rooms"]]

def town_mapping(town_number):
    town_dict = {
        0: 'ANG MO KIO',
        1: 'BEDOK',
        2: 'BISHAN',
        3: 'BUKIT BATOK',
        4: 'BUKIT MERAH',
        5: 'BUKIT PANJANG',
        6: 'BUKIT TIMAH',
        7: 'CENTRAL AREA',
        8: 'CHOA CHU KANG',
        9: 'CLEMENTI',
        10: 'GEYLANG',
        11: 'HOUGANG',
        12: 'JURONG EAST',
        13: 'JURONG WEST',
        14: 'KALLANG/WHAMPOA',
        15: 'MARINE PARADE',
        16: 'PASIR RIS',
        17: 'PUNGGOL',
        18: 'QUEENSTOWN',
        19: 'SEMBAWANG',
        20: 'SENGKANG',
        21: 'SERANGOON',
        22: 'TAMPINES',
        23: 'TOA PAYOH',
        24: 'WOODLANDS',
        25: 'YISHUN',
        26: 'SEMBAWANG'
    }
    return town_dict.get(town_number, -1)

data_encoded = pd.get_dummies(data, columns=['town'])

with tab3:
    st.title("Singapore Resale Flat Prices Prediction")

    unique_years = sorted(data['year'].unique())
    unique_town_numbers = list(range(27))  # 0 to 26
    unique_floor_area_options = sorted(data['floor_area_sqm'].unique())
    unique_lease_commence_options = sorted(data['lease_commence_year'].unique())

    A = st.selectbox('ENTER YEAR', unique_years)
    B = st.selectbox('ENTER TOWN', unique_town_numbers, format_func=lambda x: town_mapping(x))
    D = st.selectbox('ENTER FLOOR_AREA_OPTIONS', unique_floor_area_options)
    E = st.selectbox('ENTER LEASE_COMMENCE_OPTIONS', unique_lease_commence_options)

    if A is not None and B != -1 and D is not None and E is not None:
        # Use one-hot encoding for town
        town_encoded = [0] * 27
        town_encoded[B] = 1
        # Predict using the model
        W = Prediction.predict([[A] + town_encoded + [D, E]])
        for i, v in enumerate(W):
            st.markdown(f'<h1 style="text-align: center; color:red;">PREDICTED RESALE PRICE: {v:.5f}</h1>', unsafe_allow_html=True)



#-------------------------------------------tab4-----------------------------------------------


with tab4 :
  # path3 = '1.jpg'
  # O = Image.open(path3)
  # width, height = O.size
  # st.image(O, width=1300)
  st.subheader("For any regressor or classification model, what is accuracy, and why is it so important?")
  st.write("Accuracy in the context of regression or classification models is a performance metric that measures the proportion of correct predictions made by the model out of the total instances. It is a fundamental evaluation criterion, reflecting the model's ability to provide accurate and reliable results.Accuracy is crucial because it offers a clear and easily interpretable indication of how well the model performs. In classification tasks, it shows the percentage of correctly classified instances among the total predictions. For regression tasks, accuracy is often measured using metrics like R-squared or Mean Squared Error, providing insight into the precision of the model's predictions.High accuracy signifies that the model is effectively capturing the underlying patterns and relationships in the data, making it a valuable tool for decision-making and prediction. It instills confidence in the model's predictive capabilities, making it an essential metric for assessing and comparing different models. However, it's important to note that accuracy alone may not be sufficient in all scenarios, and other metrics such as precision, recall, F1 score, or specific regression metrics may be considered depending on the nature of the problem.")
  if st.button("CLICK HERE TO CHECK ACCURACY OF THE MODEL"):
    accuracy_percentage = Final * 100  # Convert accuracy to percentage
    st.markdown(f'<h1 style="text-align: center; color:blue;">Accuracy Score: {accuracy_percentage:.5f}%</h1>', unsafe_allow_html=True)
    st.progress(Final) # Add a progress bar to visualize accuracy

    # Comparison with Baseline
    baseline_accuracy = 0.75  # Replace with an appropriate baseline accuracy
    st.write(f"Model Accuracy vs. Baseline Accuracy: {Final:.5%} vs. {baseline_accuracy:.5%}")


# -------------------------------------------tab5-----------------------------------------------


 
with tab5 :
  st.subheader("What is DATA in machine learning")
  st.write('In the realm of machine learning, "data" is the fundamental building block that fuels the learning process of algorithms. It encompasses various types, each serving a distinct purpose in the model development cycle. The "training data" forms the bedrock of model learning, consisting of input features and corresponding output labels or target values. Through exposure to this dataset, the model discerns patterns and relationships, honing its predictive capabilities. As the model undergoes refinement, a portion of the data is reserved for "validation," aiding in the optimization of hyperparameters. Following training and validation, the model encounters "testing data," a set separate from the training process, crucial for assessing the model\'s performance on new, unseen instances. Data manifests in the form of "input features," representing the variables or attributes used for predictions, and "output labels" or "target values" in supervised learning, signifying the values the model aims to predict. "Unlabeled data" is prevalent in unsupervised learning scenarios, where the model identifies patterns without explicit target values. In mathematical terms, input features are often structured as a "features matrix," denoted as (X), and output labels or target values as a "target vector," represented by (y). The quality, quantity, and relevance of data are pivotal factors influencing a model\'s efficacy. Preprocessing and cleaning steps are undertaken to refine the dataset, ensuring optimal learning conditions. The successful navigation of machine learning tasks requires a nuanced understanding of the data\'s nature, guiding the selection of suitable algorithms and techniques. Ultimately, data in machine learning is the catalyst for model intelligence, empowering algorithms to make informed predictions and decisions based on learned patterns.')
  path8 = "C:/Users/prabh/Downloads/Datascience/Project/Singapore_resale/merged_df.csv"
  with open(path8,"rb") as data:
    Y = data.read()
    st.download_button('DOWNLOAD DATASET BEFORE PREPROCESSING AND FEATURE SELECTION', Y, key='file_download', file_name='dataset.csv')
  
  path9 = "C:/Users/prabh/Downloads/Datascience/Project/Singapore_resale/Colab/Final_data1.csv"
  with open(path9,"rb") as data:
    Z = data.read()
    st.download_button('DOWNLOAD DATASET AFTER PREPROCESSING AND FEATURE SELECTION', Z,file_name='final_data.csv')


# # -------------------------------------------tab7-----------------------------------------------


with tab6 :
  with st.spinner("PLEASE WAIT..."):
    # Display information and links
    st.image(Image.open('s.jpg'), width=650)
    st.markdown("[CLICK HERE TO KNOW LINKEDIN PROFILE](https://www.linkedin.com/in/prabhu-sabharish-671871259/)")
    st.markdown("[CLICK HERE TO KNOW GITHUB PROFILE](https://github.com/Prabhusabharish)")
    
    # Add button to reveal email
    if st.button("CLICK HERE TO KNOW EMAIL"):
        st.write('prabhusabharish78@gmail.com')