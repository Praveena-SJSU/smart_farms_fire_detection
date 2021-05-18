import streamlit as st
import numpy as np
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from matplotlib.dates import MonthLocator, DateFormatter, WeekdayLocator
from PIL import Image
from streamlit_folium import folium_static
import os
import time
import folium
import random

from fire_smoke_detection import FireSmokeDetection
from session_state import SessionState


today = date.today()
    

st.sidebar.markdown('## **SanJose State University**')
st.sidebar.markdown('**Group A - Team 2**')
st.sidebar.write("**Today's date:**", today)

selected_feature = st.sidebar.selectbox("What do you want to do", ("Home","Fire & Smoke Detection", "Object & Vehicle Detection", "Cattle Detection & Counting", "Livestock Behaviour Monitoring"))

def get_feature(selected_feature):
    if selected_feature == "Home":
        
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
    
          classifier_name = st.sidebar.selectbox("Select Classifier", ("VGG16", "EfficientNet", "Baseline CNN"))

          street = st.sidebar.text_input("Street", "555 Brannan Street")
          city = st.sidebar.text_input("City", "San Francisco")
          province = st.sidebar.text_input("Province", "California")
          country = st.sidebar.text_input("Country", "USA")
          
          data_dir = "./app/data"
          model_dir = "./app/models/VGG16_lr-4.h5"
          fire_detector = FireSmokeDetection(data_dir, model_dir)

          img_loc = st.empty()

          if fire_detector is None:
              st.write("Could not create fire_detector")
          else:

              label_dict = fire_detector.label_dict()

              # Fetch images from directoies
              image_list = []
              for subdir, dirs, files in os.walk(data_dir):
                for file in files:

                  filepath = subdir + os.sep + file
                  
                  if filepath.endswith(".jpg") or filepath.endswith("jpeg"):
                    image_list.append(filepath)

              random.shuffle(image_list)

              # Calculate predictions and constructu display titles
              title_list = []
              for i in range(len(image_list)):
                pred = fire_detector.get_pred(image_list[i])
                title = "Image ID: " + str(i) + ", Prediction: " + label_dict[pred]
                title_list.append(title) 



              # Show each image 
              session_state = SessionState.get(page_number = 0)
              last_page = len(image_list)
              prv, _ ,nxt = st.beta_columns([1, 10, 1])
              
              if nxt.button("Next"):
                if session_state.page_number + 1 > last_page:
                  session_state.page_number = 0
                else:
                  session_state.page_number += 1

              if prv.button("Previous"):
                if session_state.page_number - 1 < 0:
                  session_state.page_number = last_page
                else:
                  session_state.page_number -= 1

              idx = session_state.page_number

                # Show image on StreamLit
              img_loc.image(image_list[idx], caption=title_list[idx], use_column_width=True)


              # idx = 0
              # while True:
              #     st.write(idx)
              #     if idx < 0 or idx >= len(image_list):
              #       idx = 0 ## start over

              #     img_path = image_list[idx]

              #     # Show prediction
              #     pred = fire_detector.get_pred(img_path)
              #     title = "Image ID: " + str(idx) + ", Prediction: " + label_dict[pred]

              #       # Show image on StreamLit
              #     img_loc.image(img_path, caption=title, use_column_width=True)

              #     if st.button("next image"):
              #       idx = idx + 1
              #     elif st.button("prev image"):
              #       idx = idx - 1

                  # advance = False
                  # while not advance:
                  #   advance = but_loc.button('next image', time.time())
                  #   if advance:
                  #     idx = idx + 1
                  #     st.write("Hello" + idx)
                  #     break
            


              # img, label, pred = fire_detector.get_all_img_label_preds()
              

              # # Generating collage of plots 
              # # fig = plt.figure(figsize=(10, 10))
              # st.write('Classification by the model')
              # # plt.axis('off')

              # for i, img_i in enumerate(img[:5]):
              #   # ax = fig.add_subplot(4, 5, i+1)
              #   # plt.axis('off')
              #   title = "pred: " + label_dict[pred[i]]
              #   # ax.imshow(img_i)
              #   st.image(img_i, caption=title, use_column_width=True)

          # uploaded_file = st.file_uploader("Choose an image....", type="jpg")
        
          # if uploaded_file is not None:
          #   image = Image.open(uploaded_file)
          #   st.image(image, caption='Uploaded Image', use_column_width=True)
            
          #   st.write("")
                
          #   if st.button('predict'):
          #       model = load_model('./VGG16_lr-4.h5')
          #       st.write("Predicting the result....")
          #       st.write("Fire")

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

get_feature(selected_feature)

