import streamlit as st
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
import pickle 

# Title 
col = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']

st.title('California Housing Price Prediction')

st.image('https://mir-s3-cdn-cf.behance.net/project_modules/max_1200/cd1d28100171245.5f034bd42e35f.jpg')

st.header('Model of housing prices to predict median house values in California',divider=True)

# st.subheader('''User Must Enter Given values to predict Price: 
# ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']''')

st.sidebar.title('Select House Features 🏠')

st.sidebar.image('https://cdn.carrot.com/uploads/sites/40700/2020/06/Buy-and-Sell-a-House.jpg')

temp_df = pd.read_csv('california.csv')

random.seed(17)

all_values=[]

for i in temp_df[col]:
    min_value, max_value = temp_df[i].agg(['min','max'])

    var = st.sidebar.slider(f'Select {i} value',int(min_value),int(max_value),
                      random.randint(int(min_value),int(max_value)))

    all_values.append(var)
    
ss = StandardScaler()
ss.fit(temp_df[col])

final_value = ss.transform([all_values])


with open('House_price_pred_ridge_model.pkl','rb') as f:
    chatgpt = pickle.load(f)

price = chatgpt.predict(final_value)[0]

import time

st.write(pd.DataFrame(dict(zip(col,all_values)),index = [1]))

progress_bar = st.progress(0)
placeholder = st.empty()
placeholder.subheader('Predicting price!!')
place = st.empty()
place.image('https://cdnl.iconscout.com/lottie/premium/thumb/search-analytics-3574324-3035912.gif',width = 80)
if price>0:
    
    for i in range(100):
        time.sleep(0.05)
        progress_bar.progress(i + 1)
        
    body = f'Predicted Median House Price: ${round(price,2)} Thousand Dollars'
     # st.subheader(body)
    placeholder.empty()
    place.empty()
    
    st.success(body)

else:
    body = 'Invalid House Features Values'
    st.warning(body)




    
