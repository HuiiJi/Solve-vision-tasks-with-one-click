from io import StringIO
from pathlib import Path
import streamlit as st
from detect_DMSHN import detect, save_dir
import os
import sys
import argparse
from PIL import Image

def get_subdirs(b='.'):
    '''
        Returns all sub-directories in a specific Path
    '''
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    '''
        Returns the latest folder in a runs\detect
    '''
    return max(get_subdirs(os.path.join('runs', 'detect')), key=os.path.getmtime)


if __name__ == '__main__':
    # 展示一级标题
    st.image('header.jpg')


    st.subheader('1. 使用')
    st.write('·您可以使用左侧栏来进行文件上传')
    code1 = '''You can use the left side to select and upload files'''
    st.code(code1, language='bash')
    st.subheader('2. 功能')
    st.write('·当前可实现图像去雾、图像去噪和图像去雨')
    code2 = 'Vision tasks: dehaze, denoisy, derain'
    st.code(code2, language='bash')
    st.subheader('3. 运行区')



    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str,
                        default='images', help='source')
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--update', action='store_true',
                        help='update all models')
    parser.add_argument('--project', default='runs/detect',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')

    opt = parser.parse_args()


    source = ("图片上传", "视频上传(正在维护)" )
    source_index = st.sidebar.selectbox("选择输入", range(
        len(source)), format_func=lambda x: source[x])
    cloumns1, cloumns2= st.columns(2)

    if source_index == 0:
        uploaded_file = st.sidebar.file_uploader(
            "上传图片", type=['png', 'jpeg', 'jpg'])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text='资源加载中...'):
                st.sidebar.image(uploaded_file)
                picture = Image.open(uploaded_file)
                picture = picture.save(f'images/{uploaded_file.name}')
                opt.source = f'images/{uploaded_file.name}'
        else:
            is_valid = False
    # else:
    #     uploaded_file = st.sidebar.file_uploader("上传视频", type=['mp4'])
    #     if uploaded_file is not None:
    #         is_valid = True
    #         with st.spinner(text='资源加载中...'):
    #             st.sidebar.video(uploaded_file)
    #             with open(os.path.join("data", "videos", uploaded_file.name), "wb") as f:
    #                 f.write(uploaded_file.getbuffer())
    #             opt.source = f'data/videos/{uploaded_file.name}'
    #     else:
    #         is_valid = False
    if is_valid:
        option = st.selectbox("加载文件成功", ['请选择任务', '图像去雾', '图像去雨', '图像去噪'])

        if option == "图像去雨":
            detect(opt, task = 'derain')
            if source_index == 0:
                with open(str(save_dir) + '.jpg', "rb") as file:
                    btn = st.download_button(
                        label="Download image",
                        data=file,
                        file_name=str(save_dir) + '.jpg',
                        mime=str(save_dir) + '.jpg'
                    )
                with st.spinner(text='Preparing Images'):
                    # for img in os.listdir(get_detection_folder()):
                    st.image(str(save_dir) + '.jpg')
                    st.balloons()

        if option == "图像去噪":
            detect(opt, task = 'denoisy')
            if source_index == 0:
                with open(str(save_dir) + '.jpg', "rb") as file:
                    btn = st.download_button(
                        label="Download image",
                        data=file,
                        file_name=str(save_dir) + '.jpg',
                        mime=str(save_dir) + '.jpg'
                    )
                with st.spinner(text='Preparing Images'):
                    # for img in os.listdir(get_detection_folder()):
                    st.image(str(save_dir) + '.jpg')
                    st.balloons()
                
            # else:
            #     with st.spinner(text='Preparing Video'):
            #         for vid in os.listdir(get_detection_folder()):
            #             st.video(str(Path(f'{get_detection_folder()}') / vid))
            #
            #         st.balloons()

        if option == "图像去雾":
            detect(opt, task='dehaze')
            if source_index == 0:
              with open(str(save_dir) + '.jpg', "rb") as file:
                btn = st.download_button(
                    label="Download image",
                    data=file,
                    file_name=str(save_dir) + '.jpg',
                    mime=str(save_dir) + '.jpg'
                      )
                with st.spinner(text='Preparing Images'):
                    # for img in os.listdir(get_detection_folder()):
                    st.image(str(save_dir) + '.jpg')
                    st.balloons()
              
                  
            # else:
            #     with st.spinner(text='Preparing Video'):
            #         for vid in os.listdir(get_detection_folder()):
            #             st.video(str(Path(f'{get_detection_folder()}') / vid))
            #
            #         st.balloons()

