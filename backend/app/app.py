import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from datetime import date
from datetime import datetime, timedelta                                                                                                                                                            

import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.dates import MonthLocator, DateFormatter, WeekdayLocator
from PIL import Image
from streamlit_lottie import st_lottie
from streamlit_folium import folium_static
import os
import time
import folium
import random
import requests
# import math

from fire_smoke_detection import FireSmokeDetection
from session_state import SessionState


def get_fire_and_smoke_df(fire_detector, data_dir):
  # Fetch images from directoies
  image_list = []
  for subdir, dirs, files in os.walk(data_dir):
    for file in files:

      filepath = subdir + os.sep + file
      
      if filepath.endswith(".jpg") or filepath.endswith("jpeg"):
        image_list.append(filepath)

  random.shuffle(image_list)

  # Calculate predictions and construct the dataframe
  df = pd.DataFrame(columns=['date', 'pred', 'raw_pred'])
  start = datetime.now()
  for i in range(len(image_list)):
    pred, raw_pred = fire_detector.get_pred(image_list[i])
    str_date = (start - timedelta(days=i)).strftime("%m/%d/%Y")
    df.loc[i] = [str(str_date)] + [pred, raw_pred]
  
  return df, image_list

def show_fire_summary_view(df, image_list):
  str_header = "Summary View of Farm for the last {} days".format(len(image_list))
  
  # adding a map
  m = folium.Map(location=[37.49264, -120.14093], zoom_start=16)
  # add marker for Liberty Bell
  tooltip = "Liberty Bell"
  folium.Marker(
          [37.49264, -120.14093], popup="Liberty Bell", tooltip=tooltip
          ).add_to(m)
  
  # call to render Folium map in Streamlit
  folium_static(m)

  st.header(str_header)
  
  st.subheader('Fire Severity Monitoring')

  plot_df = df
  plot_df['fire_severity'] = np.exp(1 - plot_df['raw_pred']) + random.uniform(0.0, 0.1)
  plot_df['smoke_severity'] = np.sqrt(np.exp(1 - plot_df['raw_pred']) + random.uniform(0.0, 0.1))
  
  fire_chart = alt.Chart(plot_df).mark_circle().encode(x='date', 
                  y='fire_severity', size='fire_severity', 
                  color='fire_severity', tooltip=['date', 'fire_severity'])
  st.altair_chart(fire_chart)

  st.subheader('Smoke Severity Monitoring')

  smoke_chart = alt.Chart(plot_df).mark_circle().encode(x='date', 
                  y='smoke_severity', size='smoke_severity', 
                color='smoke_severity', tooltip=['date', 'smoke_severity'])
  st.altair_chart(smoke_chart)
  return


def show_fire_daily_view(df, image_list,  label_dict):
  str_header = "Daily View of Farm for the last {} days".format(len(image_list))
  st.header(str_header)

  img_loc = st.empty()

  # Calculate predictions and constructu display titles
  title_list = []
  for idx, row in df.iterrows():
    title = "Date: " + row['date'] + ", Prediction: " + label_dict[row['pred']]
    title_list.append(title) 


  # Loop through the images. 
  idx = 0
  while True:
    if idx >= df.shape[0]:
      idx = 0

    # Show image on StreamLit
    img_loc.image(image_list[idx], caption=title_list[idx], use_column_width=True)
    time.sleep(3)
    idx = idx + 1
  return


def get_feature(selected_feature):
    if selected_feature == "Home":
        st.balloons()
                
        header = st.beta_container()

        with header:
          
          st.title('Welcome to Smart Farm Survelliance!')
          st.write(" ")
          st.write(" ")
          st.write(" ")
          
          st.write('This app provides **Smart Farm Solutions** like:')
          st.write('* Fire & Smoke Detection')
          st.write('* Object & Vehicle Detection')
          st.write('* Cattle Detection & counting')
          st.write('* Livestock Behaviour Monitoring')
          st.write(""" *** """)
    elif selected_feature == "Fire & Smoke Detection":
          st.title('Fire & Smoke Detection')
          st.write(""" *** """)
    
          
          street = st.sidebar.text_input("Street", "3655 Old Toll Rd")
          city = st.sidebar.text_input("City", "Hornitos")
          province = st.sidebar.text_input("Province", "California")
          country = st.sidebar.text_input("Country", "USA")
          zipcode = st.sidebar.text_input("Zipcode", "95325")
          data_source = st.sidebar.text_input("Data Source", "Satellite Imagery")


          view_type = st.sidebar.selectbox("View Type", ("Summary", "Daily"))
          classifier_name = st.sidebar.selectbox("Select Classifier", ("EfficientNet", "VGG16", "Baseline CNN"))

          
          data_dir = "./app/data"
          
          if classifier_name == "EfficientNet":
            model_dir = "./app/models/VGG16_fine_tuned.h5"
          else:
            model_dir = "./app/models/VGG16_lr-4.h5"

          fire_detector = FireSmokeDetection(data_dir, model_dir)

          # get fire and smoke data frame
          fire_df, image_list = get_fire_and_smoke_df(fire_detector, data_dir)

          if view_type == "Daily":
            label_dict = fire_detector.label_dict()
            show_fire_daily_view(fire_df, image_list, label_dict)
          else:
            show_fire_summary_view(fire_df, image_list)


    elif selected_feature == "Object & Vehicle Detection":
          st.title('Object & Vehicle Detection')
          st.write(""" *** """)
          
          classifier_name = st.sidebar.selectbox("Select Classifier", ("MaskRCNN", "Faster RCNN", "SSD"))
          uploaded_file = st.file_uploader("Choose an image....", type="jpeg")
    
          if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
            st.write("")
            
            if st.button('predict'):
                st.write("Predicting the result....")
                st.write("Objects detected")
    
    elif selected_feature == "Cattle Detection & Counting":
          st.title('Cattle Detection & Counting')
          st.write(""" *** """)
          classifier_name = st.sidebar.selectbox("Select Classifier", ("YOLOv4", "U-Net"))
    else:
          st.title('Livestock Behaviour Monitoring')
          st.write(""" *** """)
          classifier_name = st.sidebar.selectbox("Select Classifier", ("YOLOv4", "LSTM"))
    return


def load_lottieurl(url: str):
  r = requests.get(url)
  if r.status_code != 200:
      return None
  return r.json()

def main():

  lottie_farm = load_lottieurl('https://assets4.lottiefiles.com/datafiles/eZXzHZZ2e9Apt25/data.json')
  st_lottie(lottie_farm, speed=1, width=1000, height=400, key="initial")

  today = date.today()
  st.sidebar.image('./app/data/logo_image.png', use_column_width=True)
  st.sidebar.markdown('## **SanJose State University**')
  st.sidebar.markdown('**Group A - Team 2**')
  st.sidebar.write("**Today's date:**", today)

  selected_feature = st.sidebar.selectbox("What do you want to do", ("Home","Fire & Smoke Detection", "Object & Vehicle Detection", "Cattle Detection & Counting", "Livestock Behaviour Monitoring"))
  get_feature(selected_feature)
  return


main()