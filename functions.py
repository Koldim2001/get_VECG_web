import os
import math
import warnings

import pyedflib
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.subplots as sp
from scipy import signal
from matplotlib.pyplot import figure
import scipy.signal
import neurokit2 as nk
import plotly.express as px
from shapely.geometry import Polygon
import plotly.graph_objects as go
import torch
from torchvision import transforms
from models_for_inference.model import *
import plotly.io as pio



def visualize_rotate(data):
    # Функция по визуализации вращения по нажатию кнопки
    x_eye, y_eye, z_eye = 1.25, 1.25, 0.8
    frames=[]

    def rotate_z(x, y, z, theta):
        w = x+1j*y
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z

    for t in np.arange(0, 10.26, 0.025):
        xe, ye, ze = rotate_z(x_eye, y_eye, z_eye, -t)
        frames.append(dict(layout=dict(scene=dict(camera=dict(eye=dict(x=xe, 
                                                                       y=ye,
                                                                       z=ze))))))
    fig = go.Figure(data=data,
        layout=go.Layout(
            updatemenus=[dict(type='buttons',
                showactive=False,
                y=1,
                x=0.8,
                xanchor='left',
                yanchor='bottom',
                pad=dict(t=45, r=10),
                buttons=[dict(label='Запуск вращения',
                    method='animate',
                    args=[None, dict(frame=dict(duration=20, redraw=True),
                        transition=dict(duration=0),
                        fromcurrent=True,
                        mode='immediate'
                        )]
                    )
                ])]
        ),
        frames=frames
    )

    return fig


def show_3d(x, y, z):
    # Отображение 3D интерактивного окна
    data=[go.Scatter3d(x=x, y=y, z=z,
                       mode='lines+markers', opacity=1)]
    fig = visualize_rotate(data)
    fig.update_traces(marker=dict(size=3),
                      line=dict(width=5))
    fig.update_layout(title_text="3D представление ВЭКГ")
    fig.update_layout(height=800)
    return fig


def convert_to_posix_path(windows_path):
    # Перевод пути к формату posix:
    posix_path = windows_path.replace('\\', '/')
    return posix_path


def rename_columns(df):
    # Приводит к правильному виду данные в df:
    new_columns = []
    for column in df.columns:
        new_columns.append(column[:-4])
    df.columns = new_columns
    return df


def discrete_signal_resample_for_DL(signal, old_sampling_rate, new_sampling_rate):
    """
    Осуществление ресемплирования перед DL инференсом
    """
    num_points_new = int(len(signal) * new_sampling_rate / old_sampling_rate)

    # Используем scipy.signal.resample для изменения дискретизации
    new_signal = scipy.signal.resample(signal, num_points_new)

    return new_signal


def discrete_signal_resample(signal, time, new_sampling_rate):
    """
    Осуществление ресемплирования
    """
    # Текущая частота дискретизации
    current_sampling_rate = 1 / np.mean(np.diff(time))

    # Количество точек в новой дискретизации
    num_points_new = int(len(signal) * new_sampling_rate / current_sampling_rate)

    # Используем scipy.signal.resample для изменения дискретизации
    new_signal = scipy.signal.resample(signal, num_points_new)
    new_time = np.linspace(time[0], time[-1], num_points_new)

    return new_signal, new_time


def calculate_area(points):
    # Считает площадь замкнутого полигона
    polygon = Polygon(points)
    area_inside_loop = polygon.area
    return area_inside_loop


def find_mean(df_term):
    # Считает средние значения петель
    x_center = df_term.x.mean()
    y_center = df_term.y.mean()
    z_center = df_term.z.mean()
    return [x_center, y_center, z_center]


def find_qrst_angle(mean_qrs, mean_t, name=''):
    """
    Определение угла QRST
    """
    # Преобразуем списки в numpy массивы
    mean_qrs = np.array(mean_qrs)
    mean_t = np.array(mean_t)

    # Находим угол между векторами в радианах
    dot_product = np.dot(mean_qrs, mean_t)
    norm_qrs = np.linalg.norm(mean_qrs)
    norm_t = np.linalg.norm(mean_t)
    angle_radians = np.arccos(dot_product / (norm_qrs * norm_t))

    # Конвертируем угол из радиан в градусы
    angle_degrees = np.degrees(angle_radians)
    #print(f"Угол QRST {name}равен {round(angle_degrees, 2)} градусов")

    return angle_degrees


def make_vecg(df_term):
    # Получает значения ВЭКГ из ЭКГ
    DI = df_term['ECG I']
    DII = df_term['ECG II']
    V1 = df_term['ECG V1']
    V2 = df_term['ECG V2']
    V3 = df_term['ECG V3']
    V4 = df_term['ECG V4']
    V5 = df_term['ECG V5']
    V6 = df_term['ECG V6']

    df_term['x'] = -(-0.172*V1-0.074*V2+0.122*V3+0.231*V4+0.239*V5+0.194*V6+0.156*DI-0.01*DII)
    df_term['y'] = (0.057*V1-0.019*V2-0.106*V3-0.022*V4+0.041*V5+0.048*V6-0.227*DI+0.887*DII)
    df_term['z'] = -(-0.229*V1-0.31*V2-0.246*V3-0.063*V4+0.055*V5+0.108*V6+0.022*DI+0.102*DII)
    return df_term

    
def loop(df_term, name, plotly_figures, show=False):
    # Подсчет и отображение площади петли
    if name == 'T':
        name_loop = 'ST-T'
    else:
        name_loop = name

    if show:
        # Создаем подокно с тремя графиками в ряд
        fig = sp.make_subplots(rows=1, cols=3, subplot_titles=('Фронтальная плоскость', 
                                                                'Сагиттальная плоскость',
                                                                'Аксиальная плоскость'))

        # Создаем графики для каждой плоскости
        trace1 = go.Scatter(x=df_term['y'], y=df_term['z'], mode='lines',
                            name='Фронтальная плоскость', showlegend=False)
        trace2 = go.Scatter(x=df_term['x'], y=df_term['z'], mode='lines',
                            name='Сагиттальная плоскость', showlegend=False)
        trace3 = go.Scatter(x=df_term['y'], y=df_term['x'], mode='lines',
                            name='Аксиальная плоскость', showlegend=False)
        
        # Добавляем графики в подокно
        fig.add_trace(trace1, row=1, col=1)
        fig.add_trace(trace2, row=1, col=2)
        fig.add_trace(trace3, row=1, col=3)

        # Установка общего заголовка
        title = f'{name_loop} петля'
        fig.update_layout(title_text=title, title_font_size=16, height=510, width=1300,)

        # Переворачиваем оси 
        fig.update_xaxes(autorange="reversed", row=1, col=2)  # Переворачиваем ось x для trace2
        fig.update_yaxes(autorange="reversed", row=1, col=3)   # Переворачиваем ось y для trace3

        # Отображение графика
        plotly_figures.append(fig)
    
    points = list(zip(df_term['y'], df_term['z']))
    area_inside_loop_1 = calculate_area(points)
    #print(f"Площадь петли {name_loop} во фронтальной плоскости:", area_inside_loop_1)

    points = list(zip(df_term['x'], df_term['z']))
    area_inside_loop_2 = calculate_area(points)
    #print(f"Площадь петли {name_loop} в сагиттальной плоскости:", area_inside_loop_2)

    points = list(zip(df_term['y'], df_term['x']))
    area_inside_loop_3 = calculate_area(points)
    #print(f"Площадь петли {name_loop} в аксиальной плоскости:", area_inside_loop_3)

    return area_inside_loop_1, area_inside_loop_2, area_inside_loop_3


def get_area(show, df, waves_peak, start, Fs_new, QRS, T, plotly_figures):
    # Выделяет области петель для дальнейшей обработки - подсчета угла QRST и площадей
    area = []
    # Уберем nan:
    waves_peak['ECG_Q_Peaks'] = [x for x in waves_peak['ECG_Q_Peaks'] if not math.isnan(x)]
    waves_peak['ECG_S_Peaks'] = [x for x in waves_peak['ECG_S_Peaks'] if not math.isnan(x)]
    waves_peak['ECG_T_Offsets'] = [x for x in waves_peak['ECG_T_Offsets'] if not math.isnan(x)]   

    # QRS петля
    # Ищем ближний пик к R пику
    closest_Q_peak = min(waves_peak['ECG_Q_Peaks'], key=lambda x: abs(x - start))
    closest_S_peak = min(waves_peak['ECG_S_Peaks'], key=lambda x: abs(x - start))


    df_new = df.copy()
    df_term = df_new.iloc[closest_Q_peak:closest_S_peak,:]
    df_row = df_new.iloc[closest_Q_peak:closest_Q_peak+1,:]
    df_term = pd.concat([df_term, df_row])
    mean_qrs = find_mean(df_term)
    if QRS:
        area = list(loop(df_term, name='QRS', plotly_figures=plotly_figures, show=show))

    ## ST-T петля
    # Ищем ближний пик к R пику
    closest_S_peak = min(waves_peak['ECG_S_Peaks'], key=lambda x: abs(x - start))
    # Ищем ближний пик к S пику
    closest_T_end = min(waves_peak['ECG_T_Offsets'], key=lambda x: abs(x - closest_S_peak))
    df_new = df.copy()
    df_term = df_new.iloc[closest_S_peak + int(0.025*Fs_new) : closest_T_end, :]
    df_row = df_new.iloc[closest_S_peak+int(0.025*Fs_new):closest_S_peak+int(0.025*Fs_new)+1,:]
    df_term = pd.concat([df_term, df_row])
    mean_t = find_mean(df_term)
    if T:
        area.extend(list(loop(df_term, name='T', plotly_figures=plotly_figures, show=show)))
    return area, mean_qrs, mean_t


def preprocessing_3d(list_coord):
    # Строит линии на 3D графике, отвечающие за вектора средних ЭДС петель
    A = np.array(list_coord)

    step = 0.025
    # Создаем массив точек от (0, 0, 0) до точки A с заданным шагом
    interpolated_points = []
    for t in np.arange(0, 1, step):
        interpolated_point = t * A
        interpolated_points.append(interpolated_point)

    # Добавляем точку A в конец массива
    interpolated_points.append(A)

    # Преобразуем список точек в numpy массив
    interpolated_points = np.array(interpolated_points)

    df = pd.DataFrame(interpolated_points, columns=['x', 'y', 'z'])
    df['s']=20 # задали размер для 3D отображения
    return df


def angle_3d_plot(df1, df2, df3):
    # Построение интерактивного графика логов вычисления угла QRST 
    fig = go.Figure()

    fig.add_trace(
        go.Scatter3d(
            x=df1['x'],
            y=df1['y'],
            z=df1['z'],
            mode='markers',
            marker=dict(size=df1['s'], sizemode='diameter', opacity=1),
            name='Средняя электродвижущая сила QRS'
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=df2['x'],
            y=df2['y'],
            z=df2['z'],
            mode='markers',
            marker=dict(size=df2['s'], sizemode='diameter', opacity=1),
            name='Средняя электродвижущая сила ST-T'
        )
    )
    df3['size'] = 10
    fig.add_trace(
        go.Scatter3d(
            x=df3['x'],
            y=df3['y'],
            z=df3['z'],
            mode='markers',
            marker=dict(size=df3['size'], sizemode='diameter', opacity=1),
            name='ВЭКГ'
        )
    )
    fig.update_layout(title_text="Угол QRST")
    fig.update_layout(height=800)
    return fig
  
 


def apply_filter_mean(column, window_size):
    # Фильтр среднего для сглаживания петли ST-T
    filtered_column = []

    for i in range(len(column)):
        if i < window_size // 2 or i >= len(column) - window_size // 2:
            filtered_column.append(column[i])
        else:
            window = column[i - window_size // 2:i + window_size // 2 + 1]
            filtered_value = np.mean(window)
            filtered_column.append(filtered_value)

    return filtered_column






#------------------------------------------ГЛАВНЫЙ КОД--------------------------------------#

def get_VECG(input_data: dict):
    # ------------------ ARG parse ------------------
    data_edf = input_data["data_edf"]
    n_term_start = input_data["n_term_start"]
    n_term_finish = input_data["n_term_finish"] 
    filt = input_data["filt"]
    f_sreza = input_data["f_sreza"]
    Fs_new = input_data["f_sampling"]
    show_ECG = input_data["show_ecg"]
    plot_3D = input_data["plot_3d"]
    QRS_loop_area = input_data["qrs_loop_area"]
    T_loop_area = input_data["t_loop_area"]
    count_qrst_angle = input_data["count_qrst_angle"]
    mean_filter = input_data["mean_filter"]
    predict_res = input_data["predict"]
    plot_projections = input_data["plot_projections"]
    logs = input_data["logs"]
    save_coord = input_data["save_coord"] 
    pr_delta = input_data["pr_delta"]
    show_XYZ = input_data["show_xyz"]
    show_loops = False
    show_angle = False
    show_detect_pqrst = False
  
    pio.templates.default = "plotly"  # Используем классический стиль Plotly

    if logs:
        show_loops = True
        show_angle = True
        show_detect_pqrst = True

    plotly_figures = []
    output_results = {}


    # Устанавливаем фильтр для игнорирования всех RuntimeWarning
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Отработка ошибок введения номеров периодов ЭКГ
    if n_term_finish != None:
        if n_term_finish < n_term_start:
            raise ValueError("Ошибка: n_term_finish должно быть >= n_term_start")
        else:
          n_term = [n_term_start, n_term_finish]  
    else:
        n_term = n_term_start

    # Приведение путей к posix формату
    if '\\' in data_edf:
        data_edf = convert_to_posix_path(data_edf)

    # Считывание edf данных:
    # Открываем EDF файл
    f = pyedflib.EdfReader(data_edf)

    # Получаем информацию о каналах
    num_channels = f.signals_in_file
    channels = f.getSignalLabels()

    # Читаем данные по каналам
    raw_data = []
    for i in range(num_channels):
        channel_data = f.readSignal(i)
        raw_data.append(channel_data)

    # Получаем частоту дискретизации
    fd = f.getSampleFrequency(0)

    # Закрываем файл EDF после чтения
    f.close()

    raw_data = np.array(raw_data)

    # Создаем DataFrame
    df = pd.DataFrame(data=raw_data.T,   
            index=range(raw_data.shape[1]), 
            columns=channels)  

    # Переименование столбцов при необходимости:
    if 'ECG I-Ref' in df.columns:
        df = rename_columns(df)
        channels = df.columns

    # Создание массива времени    
    Ts = 1/fd
    t = []
    for i in range(raw_data.shape[1]):
        t.append(i*Ts)

    # Ресемлинг:
    df_new = pd.DataFrame()
    for graph in channels:
        sig = np.array(df[graph])
        new_ecg, time_new = discrete_signal_resample(sig, t, Fs_new)
        df_new[graph] = pd.Series(new_ecg) 
    df = df_new.copy()

    # ФВЧ фильтрация артефактов дыхания:
    if filt == True:
        df_new = pd.DataFrame()
        for graph in channels:
            sig = np.array(df[graph])
            sos = scipy.signal.butter(1, f_sreza, 'hp', fs=Fs_new, output='sos')
            avg = np.mean(sig)
            filtered = scipy.signal.sosfilt(sos, sig)
            filtered += avg
            df_new[graph] = pd.Series(filtered)
        df = df_new.copy()
        
    # ФНЧ фильтрация (по желанию можно включить):
    filt_low_pass = False
    if filt_low_pass:
        df_new = pd.DataFrame()
        for graph in channels:
            sig = np.array(df[graph])
            sos = scipy.signal.butter(1, 150, 'lp', fs=Fs_new, output='sos')
            avg = np.mean(sig)
            filtered = scipy.signal.sosfilt(sos, sig)
            filtered += avg
            df_new[graph] = pd.Series(filtered)
        df = df_new.copy()
    
    df['time'] = time_new

    ## Поиск точек PQRST:
    n_otvedenie = 'I'
    signal = np.array(df['ECG I'])  

    # способ чистить сигнал перед поиском пиков:
    signal = nk.ecg_clean(signal, sampling_rate=Fs_new, method="neurokit") 

    # Поиск R зубцов:
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=Fs_new)

    # Проверка в случае отсутствия результатов и повторная попытка:
    if rpeaks['ECG_R_Peaks'].size <= 5:
        print("На I отведении не удалось детектировать R зубцы")
        print("Проводим детектирование по II отведению:")
        n_otvedenie = 'II'
        signal = np.array(df['ECG II'])  
        signal = nk.ecg_clean(signal, sampling_rate=Fs_new, method="neurokit") 
        _, rpeaks = nk.ecg_peaks(signal, sampling_rate=Fs_new)
        
        # При повторной проблеме выход из функции:
        if rpeaks['ECG_R_Peaks'].size <= 3:
            print('Сигналы ЭКГ слишком шумные для анализа')
            
            # Создаем подразделы для графиков
            num_channels = len(channels)
            rows = int(num_channels / 2)
            cols = 2
            fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, subplot_titles=channels)

            # Задаем общий интервал по оси X
            x_range = [1, 7]

            # Добавляем графики в subplot
            for i, graph in enumerate(channels):
                row = i // 2 + 1
                col = i % 2 + 1
                sig = df[graph]
                trace = go.Scatter(x=time_new, y=sig, mode='lines', name=graph,
                                   showlegend=False, line=dict(color='blue'))
                fig.add_trace(trace, row=row, col=col)
                fig.update_xaxes(row=row, col=col, range=x_range)

            # Настроим макет и отобразим графики
            fig.update_layout(title_text="Сигналы ЭКГ, которые не получилось обработать")
            output_results['text'] = 'too_noisy'
            output_results['charts'] = [fig]
            return output_results

    # Поиск медианного размера кардиоцикла для данного пациента (нужно для рассчета сдвига pr)
    dif_rr = np.diff(rpeaks['ECG_R_Peaks'])
    median_rr = np.median(dif_rr)

    # Поиск точек pqst:
    _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=Fs_new, method="peak")

    # Отображение PQST точек на сигнале первого отведения (или второго при ошибке на первом)
    if show_detect_pqrst:
        # Создаем график сигнала
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_new, y=signal, mode='lines', name='Signal', line=dict(color='black')))

        # Отображаем вертикальные линии для каждой точки
        colors = {'ECG_P_Peaks': 'red', 'ECG_Q_Peaks': 'green', 'ECG_S_Peaks': 'magenta', 'ECG_T_Peaks': 'blue'}
        for wave_type, peaks in waves_peak.items():
            if wave_type in ['ECG_P_Peaks', 'ECG_Q_Peaks', 'ECG_S_Peaks', 'ECG_T_Peaks']:
                wave_type_label = wave_type.split('_')[1]  # Извлекаем часть имени для метки графика
                for peak in peaks:
                    if not np.isnan(peak):  # Проверяем, что значение точки не является NaN
                        fig.add_shape(go.layout.Shape(
                            type='line',
                            x0=time_new[int(peak)],
                            x1=time_new[int(peak)],
                            y0=min(signal),
                            y1=max(signal),
                            line=dict(color=colors[wave_type], dash='dot'),
                            name=f'{wave_type_label} Peak'
                        ))

        # Настройка макета и отображение графика
        fig.update_layout(
            xaxis=dict(range=[2, 5], title='Time (seconds)'),
            yaxis=dict(title='Signal ECG'),
            title=f'Детекция PQRST на {n_otvedenie} отведении'
        )
        plotly_figures.append(fig)


    # Выбор исследуемого периода/периодов
    i = n_term
    if type(i) == list:
        #print(f"Запрошен диапазон с {i[0]} по {i[1]} период включительно")
        fin = i[1]
        beg = i[0]
    else:
        #print(f"Запрошен {i} период")
        fin = i
        beg = i

    if beg-1 < 0 or fin >= len(rpeaks['ECG_R_Peaks']):
        #print('Запрашиваемого перода/диапазона периодов не существует')
        output_results['text'] = 'no_this_period'
        output_results['charts'] = []
        return output_results
    
    start_r = rpeaks['ECG_R_Peaks'][beg-1]
    end_r = rpeaks['ECG_R_Peaks'][fin]

    # сдвиг pr:
    start = start_r - int(median_rr * pr_delta)
    end = end_r - int(median_rr * pr_delta)
    df_term = df.iloc[start:end,:]
    df_row = df.iloc[start:start+1,:]
    

    # Отображение многоканального ЭКГ 
    if show_ECG:
        # Создаем подразделы для графиков
        num_channels = len(channels)
        rows = num_channels
        cols = 1
        fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, subplot_titles=channels)

        # Задаем общий интервал по оси X
        x_range = [0.5, 9.5]

        fig_height = num_channels * 140

        # Добавляем графики в subplot
        for i, graph in enumerate(channels):
            row = i + 1

            trace1 = go.Scatter(x=df['time'], y=df[graph], mode='lines', name=graph,
                                line=dict(color='blue'), showlegend=False)
            trace2 = go.Scatter(x=df_term['time'], y=df_term[graph],
                                mode='lines', name='Term_' + graph,
                                line=dict(color='red'), showlegend=False)

            fig.add_trace(trace1, row=row, col=1)
            fig.add_trace(trace2, row=row, col=1)
            fig.update_xaxes(row=row, col=1, range=x_range)

        # Настроим макет и отобразим графики
        fig.update_layout(title_text="Графики ЭКГ отведений", height=fig_height)
        plotly_figures.append(fig)

    # Отображение отведений XYZ
    if show_XYZ:
        df = make_vecg(df)
        df_term_show = make_vecg(df_term.copy())
        # Создаем подразделы для графиков
        rows = 3
        cols = 1
        fig = make_subplots(rows=rows, cols=cols, shared_xaxes=True, subplot_titles=['X','Y','Z'])

        # Задаем общий интервал по оси X
        x_range = [0.5, 9.5]

        fig_height = 3 * 140

        # Добавляем графики в subplot
        for i, graph in enumerate(['x','y','z']):
            row = i + 1

            trace1 = go.Scatter(x=df['time'], y=df[graph], mode='lines', name=graph,
                                line=dict(color='blue'), showlegend=False)
            trace2 = go.Scatter(x=df_term_show['time'], y=df_term_show[graph],
                                mode='lines', name='Term_' + graph,
                                line=dict(color='red'), showlegend=False)

            fig.add_trace(trace1, row=row, col=1)
            fig.add_trace(trace2, row=row, col=1)
            fig.update_xaxes(row=row, col=1, range=x_range)

        # Настроим макет и отобразим графики
        fig.update_layout(title_text="Графики ВЭКГ отведений", height=fig_height)
        plotly_figures.append(fig)
  

    # Проверка на адекватность значений median_rr
    if (median_rr > Fs_new * 3) or (median_rr < Fs_new * 0.1):
            print('Медиана RR имеет неадекватные значения (ошибка детектирования R пиков)')
            output_results['text'] = 'too_noisy'
            output_results['charts'] = plotly_figures
            return output_results
    
    # Расчет ВЭКГ
    df_term = pd.concat([df_term, df_row])
    df_term = make_vecg(df_term)
    df_term['size'] = 100 # задание размера для 3D визуализации

    # Сглаживание петель
    if mean_filter:
        df = make_vecg(df)
        window = int(Fs_new * 0.02)
        df['x'] = apply_filter_mean(np.array(df['x']), window)
        df['y'] = apply_filter_mean(np.array(df['y']), window)
        df['z'] = apply_filter_mean(np.array(df['z']), window)
        df_term = df.iloc[start:end,:]
        df_row = df.iloc[start:start+1,:]
        df_term = pd.concat([df_term, df_row])
        df_term['size'] = 100 
        
    # Построение проекций ВЭКГ:
    if plot_projections:
        # Создаем подразделы для графиков
        fig = make_subplots(rows=1, cols=3, subplot_titles=['Фронтальная плоскость',
                                                            'Сагиттальная плоскость',
                                                            'Аксиальная плоскость'])

        # График фронтальной плоскости
        trace1 = go.Scatter(x=df_term['y'], y=df_term['z'], mode='lines', showlegend=False)
        fig.add_trace(trace1, row=1, col=1)
        fig.update_xaxes(title_text='Y', row=1, col=1)
        fig.update_yaxes(title_text='Z', row=1, col=1)

        # График сагиттальной плоскости
        trace2 = go.Scatter(x=df_term['x'], y=df_term['z'], mode='lines', showlegend=False)
        fig.add_trace(trace2, row=1, col=2)
        fig.update_xaxes(title_text='X', row=1, col=2)
        fig.update_yaxes(title_text='Z', row=1, col=2)

        # График аксиальной плоскости
        trace3 = go.Scatter(x=df_term['y'], y=df_term['x'], mode='lines', showlegend=False)
        fig.add_trace(trace3, row=1, col=3)
        fig.update_xaxes(title_text='Y', row=1, col=3)
        fig.update_yaxes(title_text='X', row=1, col=3)

        # Настроим макет и отобразим графики
        fig.update_layout(height=510, width=1300, title_text="Проекции ВЭКГ на главные плоскости")

        # Переворачиваем оси 
        fig.update_xaxes(autorange="reversed", row=1, col=2)  # Переворачиваем ось x для trace2
        fig.update_yaxes(autorange="reversed", row=1, col=3)   # Переворачиваем ось y для trace3

        plotly_figures.append(fig)


    # Интерактивное 3D отображение
    if plot_3D:
        fig = show_3d(df_term.x, df_term.y, df_term.z)
        plotly_figures.append(fig)
        

    # Работа при указании одного периода ЭКГ: 
    if  n_term_finish == None or n_term_finish == n_term_start:
        if save_coord:
            df_save = df_term[['x', 'y', 'z']]
            # Путь к файлу CSV для сохранения
            file_name_without_extension = os.path.splitext(os.path.basename(data_edf))[0]
            name = f'{file_name_without_extension}_period_{n_term_start}.csv'

            # Сохраняем выбранные столбцы в CSV файл
                # Создадим папки для записи если их еще нет:
            if not os.path.exists('point_cloud_dataset'):
                os.makedirs('point_cloud_dataset')
            df_save.to_csv('point_cloud_dataset/' + name, index=False)

            return df_save.shape[0]

        ## Масштабирование:
        # Поиск центра масс:
        x_center = df_term.x.mean()
        y_center = df_term.y.mean()
        z_center = df_term.z.mean()

        df_term['x_scaled'] = df_term.x - x_center
        df_term['y_scaled'] = df_term.y - y_center
        df_term['z_scaled'] = df_term.z - z_center

        # Нормирование на максимальное значение 
        max_value = max(df_term['x_scaled'].abs().max(),
                        df_term['y_scaled'].abs().max(),
                        df_term['z_scaled'].abs().max())
        df_term['x_scaled'] = df_term['x_scaled'] / max_value
        df_term['y_scaled'] = df_term['y_scaled'] / max_value
        df_term['z_scaled'] = df_term['z_scaled'] / max_value

       
        # СППР:
        # Инференс модели pointnet:
        message_predict = None
        if predict_res:
            point_cloud_array_innitial = df_term[['x', 'y', 'z']].values
            
            # Приведем к дискретизации 1000 Гц на котором обучалась сеть
            new_num_points = int(len(point_cloud_array_innitial) * 1000 / Fs_new)
            

            # Инициализируем новый массив
            point_cloud_array = np.zeros((new_num_points, 3))

            # Производим ресемплирование каждой координаты
            for i in range(3):
                point_cloud_array[:, i] = discrete_signal_resample_for_DL(point_cloud_array_innitial[:, i],
                                                                          Fs_new, 1000)

            # Трансформация входных данных
            val_transforms = transforms.Compose([
                        Normalize(),
                        PointSampler_weighted(512),
                        ToTensor()
                        ])
            inputs = val_transforms(point_cloud_array)
            inputs = torch.unsqueeze(inputs, 0)
            inputs = inputs.double()

            pointnet = PointNet().double()
            # Загрузка сохраненных весов модели
            pointnet.load_state_dict(torch.load('models_for_inference/pointnet.pth',
                                                map_location=torch.device('cpu')))
            pointnet.eval().to('cpu')

            # инференс:
            with torch.no_grad():
                outputs, __, __ = pointnet(inputs.transpose(1,2))

                softmax_outputs = torch.softmax(outputs, dim=1)
                probabilities, predicted_class = torch.max(softmax_outputs, 1)

            if predicted_class == 0:
                message_predict = f'Здоров (уверенность предсказания {probabilities.item() * 100:.2f}%)__'
            else:
                message_predict = f'Болен (уверенность предсказания {probabilities.item() * 100:.2f}%)__'
            #print(message_predict)

        # Задание ответов по умолчанию
        area_projections = None
        angle_qrst = None
        angle_qrst_front = None

        if count_qrst_angle or T_loop_area or QRS_loop_area:
            start = start_r # Считать надо для всех расчетов петель относительно реального R пика
            # Поиск площадей при задании на исследование одного периодка ЭКГ:
            area_projections , mean_qrs, mean_t = get_area(show=show_loops, df=df,
                                                        waves_peak=waves_peak, start=start,
                                                        Fs_new=Fs_new,  QRS=QRS_loop_area, 
                                                       T=T_loop_area, plotly_figures=plotly_figures)
        # Определение угла QRST:
        if count_qrst_angle:
            angle_qrst = find_qrst_angle(mean_qrs, mean_t)
            angle_qrst_front = find_qrst_angle(mean_qrs[1:], mean_t[1:],
                                               name='во фронтальной плоскости ')
            
            # Отображение трехмерного угла QRST
            if show_angle:
                df_qrs = preprocessing_3d(mean_qrs)
                df_t = preprocessing_3d(mean_t)
                fig = angle_3d_plot(df_qrs, df_t, df_term)
                plotly_figures.append(fig)
                
    
    output_results['text'] = (area_projections, angle_qrst, angle_qrst_front, message_predict)
    output_results['charts'] = plotly_figures
    
    return output_results


if __name__ == "__main__":
    input_data = {}
    input_data["data_edf"] = 'Data_for_testing/ECG_bad_1.edf'
    input_data["n_term_start"] = 3
    input_data["n_term_finish"] = None
    input_data["filt"] = True
    input_data["f_sreza"] = 0.7
    input_data["f_sampling"] = 2000
    input_data["show_ecg"] = True
    input_data["plot_3d"] = True
    input_data["qrs_loop_area"] = True
    input_data["t_loop_area"] = True
    input_data["show_log_loop_area"] = True
    input_data["count_qrst_angle"] = True
    input_data["mean_filter"] = True
    input_data["predict"] = True
    input_data["plot_projections"] = True
    input_data["logs"] = True
    save_coord = False
    pr_delta = 0.5
    show_XYZ = True
    get_VECG(input_data)