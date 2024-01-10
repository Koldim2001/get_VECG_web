import streamlit as st
import json
from functions import *
import tempfile


def main(config : dict):
    st.title("Получение ВЭКГ")

    uploaded_file = st.sidebar.file_uploader("Выберите файл .edf", type="edf")
    if uploaded_file:
        st.write(f"Выбран файл: {uploaded_file.name}")

    n_term_start = st.sidebar.number_input("Номер периода ЭКГ", value=config['n_term_start'], min_value=1,)
    button_pressed = st.sidebar.button(":red[Запуск]", key="launch_button",
                                       help="Нажмите, чтобы начать обработку данных", use_container_width=True)
    
    
    st.sidebar.markdown('---') 
    st.sidebar.markdown('### Выбор режимов:')
    plot_projections = st.sidebar.checkbox("Построение проекций ВЭКГ", value=config['plot_projections'])
    plot_3D = st.sidebar.checkbox("Построение 3D ВЭКГ", value=config['plot_3D'])
    show_ECG = st.sidebar.checkbox("Отображение ЭКГ сигналов", value=config['show_ECG'])
    predict_res = st.sidebar.checkbox("Результат СППР (болен/здоров)", value=config['predict_res'])
    count_qrst_angle = st.sidebar.checkbox("Расчет угла QRST", value=config['count_qrst_angle'])
    QRS_loop_area = st.sidebar.checkbox("Расчет площади QRS петли", value=config['QRS_loop_area'])
    T_loop_area = st.sidebar.checkbox("Расчет площади ST-T петли", value=config['T_loop_area'])

    st.sidebar.markdown('---') 
    st.sidebar.markdown('### Настройки:')
    mean_filter = st.sidebar.checkbox("Сглаживание петель", value=config['mean_filter'])
    filt = st.sidebar.checkbox("ФВЧ фильтрация ЭКГ сигналов", value=config['filt'])
    if filt:
        f_sreza = st.sidebar.number_input("Частота среза ФВЧ фильтра (в Гц)", value=config['f_sreza'])
    f_sampling = st.sidebar.number_input("Частота дискретизации (в Гц)", value=config['f_sampling'])
    if config["dev_mode"]:
        logs = st.sidebar.checkbox("Показ логов обработки", value=config['logs'])  # Показать только при dev_mode

    st.sidebar.markdown('---') 
    
    if button_pressed:
        #print(temp_file_path)
        if uploaded_file is not None:

            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name

                input_data = {
                    "data_edf": temp_file_path,
                    "n_term_start": n_term_start,
                    "filt": filt,
                    "f_sreza": f_sreza if filt else None,
                    "f_sampling": f_sampling,
                    "plot_projections": plot_projections,
                    "plot_3d": plot_3D,
                    "show_ecg": show_ECG,
                    "predict": predict_res,
                    "count_qrst_angle": count_qrst_angle,
                    "qrs_loop_area": QRS_loop_area,
                    "t_loop_area": T_loop_area,
                    "mean_filter": mean_filter,
                    "logs": logs,
                    "n_term_finish": None
                }

                # Получить ВЭКГ
                res = get_VECG(input_data)
                
                # Обработаем результаты программы, поместив в список предложения:
                message = []
                error = False
                if res == 'no_this_period':
                    st.error("Не найден такой период. Попробуйте ввести меньше значение")
                    error = True
                elif res == 'too_noisy':
                    st.error("Не получилось построить ВЭКГ, так как ЭКГ слишком шумный")
                    error = True
                elif len(res) == 4:
                    area_projections, angle_qrst, angle_qrst_front, message_predict = res
                    if input_data["predict"]:
                        message.append('__СППР: ' + message_predict)
                    if input_data["qrs_loop_area"]:
                        message.append(f'Площадь петли QRS во фронтальной плоскости: {"{:.3e}".format(area_projections[0])}')
                        message.append(f'Площадь петли QRS во сагиттальной плоскости: {"{:.3e}".format(area_projections[1])}')
                        message.append(f'Площадь петли QRS во аксиальной плоскости: {"{:.3e}".format(area_projections[2])}')
                    if input_data["qrs_loop_area"] and input_data["t_loop_area"]:
                        message.append(f'Площадь петли ST-T во фронтальной плоскости: {"{:.3e}".format(area_projections[3])}')
                        message.append(f'Площадь петли ST-T во сагиттальной плоскости: {"{:.3e}".format(area_projections[4])}')
                        message.append(f'Площадь петли ST-T во аксиальной плоскости: {"{:.3e}".format(area_projections[5])}')
                    elif input_data["t_loop_area"]:
                        message.append(f'Площадь петли ST-T во фронтальной плоскости: {"{:.3e}".format(area_projections[0])}')
                        message.append(f'Площадь петли ST-T во сагиттальной плоскости: {"{:.3e}".format(area_projections[1])}')
                        message.append(f'Площадь петли ST-T во аксиальной плоскости: {"{:.3e}".format(area_projections[2])}')
                    if input_data["count_qrst_angle"]:
                        message.append(f'Пространственный угол QRST равен {round(angle_qrst, 2)} градусов')

                # Проверка на факт наличия ошибок (для разработчика)
                if isinstance(res, str) and config["dev_mode"] and res not in ['no_this_period', 'too_noisy']: 
                    st.error(res)

                # Вывести результаты
                if not error:
                    if message != []:
                        st.markdown('---') 
                        for result in message:
                            st.markdown(result)

        else:
            st.warning("Пожалуйста, загрузите файл .edf для обработки.")


if __name__ == "__main__":
    # Загрузка конфигурации программы:
    with open('configs/config.json', 'r') as json_file:
        config = json.load(json_file)
    main(config)

