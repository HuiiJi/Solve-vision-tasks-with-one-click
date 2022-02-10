from io import StringIO
from pathlib import Path
import streamlit as st
from detect_FFA import detect
import os
import sys
import cv2
import numpy as np
import argparse
from PIL import Image

if __name__ == '__main__':
    st.image('header2.png')
    st.subheader('1. 使用')
    st.write('·您可以使用左侧来进行文件选择并上传')
    code1 = '''You can use the left side to select and upload files'''
    st.video('introduction.mp4')
    st.code(code1, language='bash')
    st.subheader('2. 功能')
    st.write('·当前可实现图像去雾、图像去噪和图像去雨')
    code2 = 'Vision tasks: dehaze, denoisy, derain'
    st.code(code2, language='bash')
    st.subheader('3. 运行区')

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    opt = parser.parse_args()

    source_button= ("图片上传", "视频上传" )
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source_button)), format_func=lambda x: source_button[x])

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])

        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'data/images/{uploaded_file.name}')
                opt.source = f'data/images/{uploaded_file.name}'
        else:
            is_valid = False

    else:
        uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.video(uploaded_file)
                with open(uploaded_file.name, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                opt.source = f'data/videos/{uploaded_file.name}'
        else:
            is_valid = False

    if is_valid:
        option = st.selectbox("加载文件成功", ['请选择任务', '视觉去雾', '视觉去雨', '视觉去噪'])

        if option == "视觉去雨":
            detect(opt, task = 'derain')
            if source_index == 0:
                with open('runs/detect/clean.jpg', "rb") as file:
                    btn = st.download_button(
                        label="点击下载",
                        data=file,
                        file_name='clean.jpg',
                    )
                with st.spinner(text='Preparing Images'):
                    st.image('runs/detect/clean.jpg')

            else:           
                with open('runs/detect/clean.mp4', "rb") as file:
                    btn = st.download_button(
                        label="点击下载",
                        data=file,
                        file_name='clean.mp4',
                    )
                with st.spinner(text='Preparing Images'):
                    st.video('runs/detect/clean.mp4')


        if option == "视觉去噪":
            detect(opt, task = 'denoisy')
            if source_index == 0:
                with open('runs/detect/clean.jpg', "rb") as file:
                    btn = st.download_button(
                        label="点击下载",
                        data=file,
                        file_name='clean.jpg',
                    )

                with st.spinner(text='Preparing Images'):
                    st.image('runs/detect/clean.jpg')


            else:
                with open('runs/detect/clean.mp4', "rb") as file:
                    btn = st.download_button(
                        label="点击下载",
                        data=file,
                        file_name='clean.mp4',
                    )

                with st.spinner(text='Preparing Video'):
                        st.video('runs/detect/clean.mp4',  format='mp4', start_time=0)


        if option == "视觉去雾":
            detect(opt, task='dehaze')
            if source_index == 0:
                with open('runs/detect/clean.jpg', "rb") as file:
                    btn = st.download_button(
                        label="点击下载",
                        data=file,
                        file_name='clean.jpg',
                    )

                with st.spinner(text='Preparing Images'):
                    st.image('runs/detect/clean.jpg')

            else:

                with st.spinner(text='Preparing Video'):
                        st.video('runs/detect/clean.mp4',  format='mp4', start_time=0)
                with open('runs/detect/clean.mp4', "rb") as file:
                    btn = st.download_button(
                        label="点击下载",
                        data=file,
                        file_name='clean.mp4',
                    )



